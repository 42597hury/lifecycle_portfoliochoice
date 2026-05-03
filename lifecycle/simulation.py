"""
simulation.py - Lifecycle simulation for the partitioned-VAR portfolio model.

Simulates household paths downstream of the backward solver in `solver.py`.
Policies are stored on a 4D grid (age, persistent income, joint financial
state, cash-on-hand) and evaluated continuously in (z, s, x). The joint
financial state is propagated continuously via the partitioned VAR; the
nearest bracket-grid snap is recorded as the `sim_state` diagnostic only
and is not used for policy evaluation.

Per-period sequence (matches CONVENTIONS.md section 1):

1. Enter period t alive with state (x_t, z_t, s_t).
2. Look up policies c_t, alpha_s_t, alpha_b_t. Interpolation order matches
   the solver's continuation-value lookup
   (solver.py:_interp_z_wealth + bracket_state_3d):
       - trilinear in s across the 8 bracket-grid corners,
       - PCHIP monotone cubic in z on interior intervals (linear fallback
         on first/last z intervals),
       - linear in x with flat extrapolation outside [wealth_min, wealth_max]
         for shares; consumption level is rescaled to preserve top-node c/x
         when x > wealth_max (homothetic asymptote of CRRA policy).
3. Savings a_t = x_t - c_t.
4. Draw next-period state innovation v^s ~ N(0, Sigma_ss) and propagate
   s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s. The continuous s_{t+1} is
   carried forward; the start-of-next-period bracket lookup re-projects it
   into the bracket-grid for trilinear interpolation.
5. Conditional return mean mu_r = const_r + A_r @ s_t + M @ v^s. Realize
   gross real returns via one of two modes:
   - "monte_carlo": residual ~ N(0, Sigma_r_cond) drawn via a factor matrix
     F such that F @ F' = Sigma_r_cond.
   - "quadrature":  residual drawn by sampling a return-quadrature node by
     ret_weights (used to verify the solver's Euler integration; the realized
     return distribution is not Gaussian).
   R_bill  = exp(mu_rtb + r_resid_0)
   R_stock = R_bill * exp(mu_xr  + r_resid_1)
   R_bond  = R_bill * exp(mu_xb  + r_resid_2)
6. Draw survival from psi[t, z_t].
7. If alive, draw next-period income and form x_{t+1} = a_t * R_port + Y_{t+1}.

Income process (DESIGN.md section 1.4 / CONVENTIONS.md section 8):
  - Working phase (t < retire_age_idx): both income innovations are drawn
    from their continuous mixtures (eta_nodes / eps_nodes are quadrature
    rules for the SOLVER, not sampling distributions):
      eta ~ pz N(mu_eta1, sigma_eta1^2) + (1-pz) N(mu_eta2_eff, sigma_eta2^2)
      eps ~ pe N(mu_eps1, sigma_eps1^2) + (1-pe) N(mu_eps2_eff, sigma_eps2^2)
    where mu_eta2_eff = -(pz/(1-pz))*mu_eta1 and
          mu_eps2_eff = -(pe/(1-pe))*mu_eps1 enforce E[eta] = E[eps] = 0.
    z_{t+1} = rho * z_t + eta;  Y_{t+1} = disposable_income(exp(f + z + eps)).
  - Retirement phase (t >= retire_age_idx): z frozen; pension is computed
    directly from continuous z via the PIA formula on AIME = exp(z)*avg_det.

At the terminal age all remaining households die. Their final estate
    estate_T = savings_T * R_port_T
is recorded as estate_at_death.

Initial cash-on-hand:
  - If initial_x is None, x_0 = initial_wealth_i + Y_{0,i}, where Y_{0,i}
    uses the same continuous-mixture eps draw as the per-period income.
  - If initial_x is supplied, sim_income[:, 0] still records a drawn income
    for bookkeeping, but that draw was not used to construct x_0.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Union

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range
    warnings.warn("Numba not available. Simulation will run in pure Python.")

from lifecycle.model import scalar_disposable_income
from lifecycle.numerics import (
    _pchip_slope_uniform,
    _pchip_eval_with_basis,
    _normal_bin_probs,
)

__all__ = ["simulate_lifecycle"]


def get_stationary_distribution(Pi: np.ndarray,
                                tol: float = 1e-12,
                                max_iter: int = 10_000) -> np.ndarray:
    """Compute the stationary distribution of a Markov chain via power iteration."""
    Pi = np.asarray(Pi, dtype=float)
    n = Pi.shape[0]
    pi = np.ones(n, dtype=float) / n

    for _ in range(max_iter):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new

    warnings.warn("Stationary distribution power iteration did not converge.")
    return pi


def initialize_states(n_simulations: int,
                      n_states: int,
                      Pi: np.ndarray,
                      method: Union[str, np.ndarray],
                      rng: np.random.Generator) -> np.ndarray:
    """
    Initialize discrete state indices for simulation.

    Parameters
    ----------
    method : {"median", "stationary"} or np.ndarray of int32
        "median"     -> all households start at the middle grid point
        "stationary" -> draw from the chain's stationary distribution
        array        -> use provided indices directly
    """
    if isinstance(method, str):
        if method == "median":
            return np.full(n_simulations, n_states // 2, dtype=np.int32)
        if method == "stationary":
            pi = get_stationary_distribution(Pi)
            return rng.choice(n_states, size=n_simulations, p=pi).astype(np.int32)
        raise ValueError(
            f"Unknown initialization method: '{method}'. "
            "Expected 'median', 'stationary', or np.ndarray."
        )

    arr = np.asarray(method, dtype=np.int32)
    if arr.shape[0] != n_simulations:
        raise ValueError(
            f"Initial state array has length {arr.shape[0]}, expected {n_simulations}."
        )
    return arr


def _resolve_initial_state_indices(n_simulations: int,
                                   pc,
                                   initial_state: Union[str, np.ndarray],
                                   rng: np.random.Generator) -> np.ndarray:
    """Initialize the joint financial state index for each simulated household."""
    if isinstance(initial_state, str) and initial_state == "stationary":
        probs = np.asarray(pc.state_stationary_probs, dtype=float)
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        return rng.choice(pc.N_state, size=n_simulations, p=probs).astype(np.int32)
    return initialize_states(n_simulations, pc.N_state, pc.Pi_state, initial_state, rng)


def _build_return_factor(Sigma_r_cond: np.ndarray) -> np.ndarray:
    """
    Build a factor matrix F such that F @ F' = Sigma_r_cond.

    Uses Cholesky to match the rest of the codebase (state grid, state
    quadrature, return quadrature all use Cholesky as of 2026-04-30).
    Sigma_r_cond is strictly positive definite for the production
    calibration; if a PSD-but-not-PD matrix were ever passed in,
    np.linalg.cholesky would raise, which is the desired behaviour
    (degenerate Sigma_r_cond is a model-spec problem to surface).
    """
    Sigma = np.asarray(Sigma_r_cond, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return np.linalg.cholesky(Sigma)


def _pad_simulation_state_inputs_to_3d(pc, model, init_state_coords: np.ndarray):
    """Pad lower-dimensional state inputs to the simulation core's 3-coordinate layout."""
    n_state = int(model.n_state)
    if n_state == 3:
        L_ss_arr = np.ascontiguousarray(np.linalg.cholesky(
            0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T)
        ))
        return {
            "initial_s": np.ascontiguousarray(init_state_coords, dtype=float),
            "state_grids_0": np.ascontiguousarray(pc.state_bracket_grids[0], dtype=float),
            "state_grids_1": np.ascontiguousarray(pc.state_bracket_grids[1], dtype=float),
            "state_grids_2": np.ascontiguousarray(pc.state_bracket_grids[2], dtype=float),
            "state_bracket_shift": np.ascontiguousarray(pc.state_bracket_shift, dtype=float),
            "state_bracket_L_inv": np.ascontiguousarray(pc.state_bracket_L_inv, dtype=float),
            "L_ss": L_ss_arr,
            "Phi_0_state": np.ascontiguousarray(model.Phi_0_state, dtype=float),
            "Phi_11": np.ascontiguousarray(model.Phi_11, dtype=float),
            "A_r": np.ascontiguousarray(pc.A_r, dtype=float),
            "M_matrix": np.ascontiguousarray(model.M, dtype=float),
        }

    init_s_pad = np.zeros((init_state_coords.shape[0], 3), dtype=float)
    init_s_pad[:, :n_state] = np.asarray(init_state_coords, dtype=float)

    grids = []
    for d in range(3):
        if d < n_state:
            grids.append(np.ascontiguousarray(np.asarray(pc.state_bracket_grids[d], dtype=float)))
        else:
            grids.append(np.zeros(1, dtype=float))

    shift_pad = np.zeros(3, dtype=float)
    shift_pad[:n_state] = np.asarray(pc.state_bracket_shift, dtype=float)

    L_inv_pad = np.zeros((3, 3), dtype=float)
    L_inv_pad[:n_state, :n_state] = np.asarray(pc.state_bracket_L_inv, dtype=float)

    L_ss_pad = np.zeros((3, 3), dtype=float)
    is_dummy_state = n_state == 1 and tuple(model.state_names) == ("dummy",)
    if not is_dummy_state:
        L_ss_pad[:n_state, :n_state] = np.linalg.cholesky(
            0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T)
        )

    Phi_0_state_pad = np.zeros(3, dtype=float)
    Phi_0_state_pad[:n_state] = np.asarray(model.Phi_0_state, dtype=float)

    Phi_11_pad = np.zeros((3, 3), dtype=float)
    Phi_11_pad[:n_state, :n_state] = np.asarray(model.Phi_11, dtype=float)

    A_r_pad = np.zeros((pc.A_r.shape[0], 3), dtype=float)
    A_r_pad[:, :n_state] = np.asarray(pc.A_r, dtype=float)

    M_pad = np.zeros((model.M.shape[0], 3), dtype=float)
    M_pad[:, :n_state] = np.asarray(model.M, dtype=float)

    return {
        "initial_s": np.ascontiguousarray(init_s_pad),
        "state_grids_0": grids[0],
        "state_grids_1": grids[1],
        "state_grids_2": grids[2],
        "state_bracket_shift": np.ascontiguousarray(shift_pad),
        "state_bracket_L_inv": np.ascontiguousarray(L_inv_pad),
        "L_ss": np.ascontiguousarray(L_ss_pad),
        "Phi_0_state": np.ascontiguousarray(Phi_0_state_pad),
        "Phi_11": np.ascontiguousarray(Phi_11_pad),
        "A_r": np.ascontiguousarray(A_r_pad),
        "M_matrix": np.ascontiguousarray(M_pad),
    }


def _initialize_initial_wealth(n_simulations: int,
                               initial_wealth: Optional[Union[float, np.ndarray]],
                               initial_wealth_distribution: Optional[str],
                               initial_wealth_normal_std: float,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Build the vector of pre-labor-income wealth at entry.

    This is interpreted as asset wealth households bring into working life
    before receiving their first labor-income draw. If `initial_x` is not
    supplied, period-0 cash-on-hand is constructed as
        x_0 = initial_wealth_i + Y_{0,i}.

    The default 0.1 corresponds to ~0.1 * AWI (~$5,400) in model units
    (CONVENTIONS.md section 4.1). Initial wealth is floored at zero — this
    is a clipped normal, not a truncated normal, so negative draws pile up
    at zero.
    """
    if initial_wealth_distribution is None:
        if initial_wealth is None:
            return np.full(n_simulations, 0.1, dtype=float)
        if np.isscalar(initial_wealth):
            return np.maximum(np.full(n_simulations, float(initial_wealth), dtype=float), 0.0)

        wealth_arr = np.asarray(initial_wealth, dtype=float)
        if wealth_arr.shape[0] != n_simulations:
            raise ValueError(
                f"initial_wealth array has length {wealth_arr.shape[0]}, expected {n_simulations}."
            )
        return np.maximum(wealth_arr, 0.0)

    if initial_wealth_distribution != "normal":
        raise ValueError(
            "initial_wealth_distribution must be None or 'normal', "
            f"got '{initial_wealth_distribution}'."
        )

    mean = 0.1 if initial_wealth is None else float(initial_wealth)
    if initial_wealth_normal_std <= 0.0:
        raise ValueError(
            "initial_wealth_normal_std must be strictly positive when "
            "initial_wealth_distribution='normal'."
        )
    wealth_draws = rng.normal(loc=mean, scale=initial_wealth_normal_std, size=n_simulations)
    return np.maximum(wealth_draws, 0.0)


@njit(fastmath=True)
def fast_interp_1d(x: float, x_grid: np.ndarray, y_grid: np.ndarray) -> float:
    """Linear interpolation on a sorted grid with FLAT extrapolation.

    Beyond the grid endpoints, returns the boundary y-value. Linear
    extrapolation of policies (especially unconstrained portfolio shares,
    which can have steep slopes near the wealth-grid edges) compounds across
    the lifecycle and produces blow-ups when households drift outside
    [wealth_min, wealth_max]. Flat extrapolation contains the simulation
    inside the solved domain; off-grid visits should be diagnosed by
    widening the wealth grid or capping leverage rather than by extrapolating.

    Simulator-internal: solver.py has its own `fast_interp_1d` that uses
    LINEAR extrapolation (needed for Newton convergence at the boundary).
    The two functions are deliberately divergent — do not "fix" either one
    to match the other.
    """
    n = len(x_grid)

    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[n - 1]:
        return y_grid[n - 1]

    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0, x1 = x_grid[lo], x_grid[hi]
    y0, y1 = y_grid[lo], y_grid[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@njit
def draw_discrete(probs: np.ndarray, u: float) -> int:
    """Inverse-CDF draw from a discrete distribution given a single uniform draw."""
    cumsum = 0.0
    for i in range(len(probs)):
        cumsum += probs[i]
        if u < cumsum:
            return i
    return len(probs) - 1


@njit(fastmath=True)
def _clean_constrained_shares(alpha_s: float, alpha_b: float) -> tuple:
    """
    Light-touch cleanup for tiny feasibility violations in interpolated
    constrained portfolio policies.
    """
    tol = 1e-10

    if -tol < alpha_s < 0.0:
        alpha_s = 0.0
    if -tol < alpha_b < 0.0:
        alpha_b = 0.0

    total = alpha_s + alpha_b
    if 1.0 < total < 1.0 + tol and total > 0.0:
        scale = 1.0 / total
        alpha_s *= scale
        alpha_b *= scale

    return alpha_s, alpha_b


# Backward-compatible alias — canonical version now in model.py
_scalar_disposable_income = scalar_disposable_income


@njit(fastmath=True)
def _interp_policy_zx(arr4d: np.ndarray,
                      t: int,
                      iz_lo: int,
                      frac_z: float,
                      j_s: int,
                      x_t: float,
                      wealth_grid: np.ndarray) -> float:
    """Interpolate a 4D policy array at one state-grid corner.

    arr4d has shape (n_age, n_z, N_state, n_w). Returns the value at age t,
    persistent income state interpolated between iz_lo and iz_lo+1 by frac_z,
    joint financial state j_s, and cash-on-hand x_t.

    z-interpolation order matches solver.py:_interp_z_wealth: PCHIP
    (Fritsch-Carlson monotone cubic Hermite) on interior intervals
    (`iz_lo >= 1 and iz_lo + 2 < n_z`); linear fallback on the first and
    last z intervals. Wealth-interpolation is linear with flat extrapolation
    via fast_interp_1d. The cubic branch wealth-blends the 4-point z stencil
    FIRST and then applies PCHIP — preserves shape because Fritsch-Carlson
    is non-linear in stencil values and does not commute with linear blends.
    """
    n_z = arr4d.shape[1]
    use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)
    if use_cubic:
        f2 = frac_z * frac_z
        f3 = f2 * frac_z
        h00 = 2.0 * f3 - 3.0 * f2 + 1.0
        h10 = f3 - 2.0 * f2 + frac_z
        h01 = -2.0 * f3 + 3.0 * f2
        h11 = f3 - f2
        b0 = fast_interp_1d(x_t, wealth_grid, arr4d[t, iz_lo - 1, j_s, :])
        b1 = fast_interp_1d(x_t, wealth_grid, arr4d[t, iz_lo,     j_s, :])
        b2 = fast_interp_1d(x_t, wealth_grid, arr4d[t, iz_lo + 1, j_s, :])
        b3 = fast_interp_1d(x_t, wealth_grid, arr4d[t, iz_lo + 2, j_s, :])
        return _pchip_eval_with_basis(b0, b1, b2, b3, h00, h10, h01, h11)
    v_lo = fast_interp_1d(x_t, wealth_grid, arr4d[t, iz_lo, j_s, :])
    v_hi = fast_interp_1d(x_t, wealth_grid, arr4d[t, iz_lo + 1, j_s, :])
    return (1.0 - frac_z) * v_lo + frac_z * v_hi


@njit(fastmath=True)
def _scalar_pension_after_tax(z_val: float, avg_det: float) -> float:
    """After-tax pension benefit for a single continuous z value.

    AIME = min(exp(z) * avg_det, 2.5), then PIA formula, then same
    progressive tax schedule. Matches model.compute_pension_after_tax.
    """
    aime = min(np.exp(z_val) * avg_det, 2.5)

    # PIA formula — Catherine eq. (19)
    b1, b2 = 0.21, 1.25
    r1, r2, r3 = 0.90, 0.32, 0.15

    if aime <= b1:
        pension = aime * r1
    elif aime <= b2:
        pension = r1 * b1 + r2 * (aime - b1)
    else:
        pension = r1 * b1 + r2 * (b2 - b1) + r3 * (aime - b2)

    # Same tax schedule
    if pension <= 0.18:
        tax = pension * 0.10
    elif pension <= 0.72:
        tax = 0.018 + (pension - 0.18) * 0.12
    elif pension <= 1.54:
        tax = 0.0828 + (pension - 0.72) * 0.22
    elif pension <= 2.94:
        tax = 0.2632 + (pension - 1.54) * 0.24
    elif pension <= 3.73:
        tax = 0.5992 + (pension - 2.94) * 0.32
    elif pension <= 9.32:
        tax = 0.8520 + (pension - 3.73) * 0.35
    else:
        tax = 2.8085 + (pension - 9.32) * 0.37

    return pension - tax


@njit(parallel=True)
def simulate_lifecycle_core(
    C_mat: np.ndarray,
    S_mat: np.ndarray,
    B_mat: np.ndarray,
    wealth_grid: np.ndarray,
    z_grid: np.ndarray,
    ret_nodes: np.ndarray,
    ret_weights: np.ndarray,
    ret_factor: np.ndarray,
    log_det_profile: np.ndarray,
    avg_det: float,
    pension_at_z_grid: np.ndarray,
    survival_probs_2d: np.ndarray,
    rho: float,
    pz: float,
    mu_eta1: float,
    sigma_eta1: float,
    sigma_eta2: float,
    mu_eta2_eff: float,
    pe: float,
    mu_eps1: float,
    sigma_eps1: float,
    sigma_eps2: float,
    mu_eps2_eff: float,
    start_age: int,
    retire_age_idx: int,
    constrained: bool,
    use_mc_returns: bool,
    initial_z: np.ndarray,
    initial_s: np.ndarray,
    initial_x: np.ndarray,
    initial_income: np.ndarray,
    uniform_draws: np.ndarray,
    normal_draws: np.ndarray,
    # --- Continuous state transition arrays ---
    state_grids_0: np.ndarray,
    state_grids_1: np.ndarray,
    state_grids_2: np.ndarray,
    state_bracket_shift: np.ndarray,
    state_bracket_L_inv: np.ndarray,
    L_ss: np.ndarray,
    Phi_0_state: np.ndarray,
    Phi_11: np.ndarray,
    const_r: np.ndarray,
    A_r: np.ndarray,
    M_matrix: np.ndarray,
) -> tuple:
    """
    Numba-parallel core simulation loop over households.

    z and the joint financial state s are tracked continuously per household.
    Policies are interpolated trilinearly in s × linearly in z × linearly in x,
    matching the solver's continuation-value lookup (solver.py:bracket_state_3d).
    Survival and pension also use linear z-interp on the same iz_lo/frac_z
    bracket so the simulator stays consistent with the solver's discrete-z
    evaluation. eta and eps are drawn from their continuous mixtures.

    `pension_at_z_grid` is `pc.pension_after_tax[retire_age_idx, :]` — the
    after-tax pension evaluated at every z_grid point. The retirement branch
    and the work→retirement boundary both linearly interpolate this table
    (matching solver.py's `pension_next_by_z` and the retirement step's
    `pension_table[t+1, z_i]`).

    `initial_s` has shape (n_simulations, 3) and seeds each household at an
    economic state vector — typically the grid point selected by
    `_resolve_initial_state_indices`. Subsequent s values are continuous.

    uniform_draws[:, t, :] slots (4 columns):
      [0] survival draw
      [1] eta mixture component selector (u < pz -> component 1)
      [2] eps mixture component selector (u < pe -> component 1)
      [3] return-node draw (quadrature mode only)

    normal_draws[:, t, :] slots (n_ret + 2 + n_state columns):
      [0..n_ret-1]                          return residuals (monte_carlo mode)
      [n_ret]                               eta standard normal
      [n_ret + 1]                           eps standard normal
      [n_ret + 2 .. n_ret + 1 + n_state]    state-innovation standard normals
    """
    n_simulations = len(initial_x)
    n_age = C_mat.shape[0]
    n_z = len(z_grid)
    n_ret = ret_nodes.shape[1]
    dz = z_grid[1] - z_grid[0]
    z_lo = z_grid[0]

    sim_x = np.zeros((n_simulations, n_age))
    sim_c = np.zeros((n_simulations, n_age))
    sim_savings = np.zeros((n_simulations, n_age))
    sim_alpha_s = np.zeros((n_simulations, n_age))
    sim_alpha_b = np.zeros((n_simulations, n_age))
    sim_R_port = np.zeros((n_simulations, n_age))
    sim_income = np.zeros((n_simulations, n_age))
    sim_estate = np.zeros((n_simulations, n_age))
    sim_z = np.zeros((n_simulations, n_age))
    sim_z_idx = np.zeros((n_simulations, n_age), dtype=np.int32)
    sim_state = np.zeros((n_simulations, n_age), dtype=np.int32)
    sim_state_coords = np.zeros((n_simulations, n_age, 3))
    sim_alive = np.zeros((n_simulations, n_age), dtype=np.bool_)

    death_age = np.full(n_simulations, -1, dtype=np.int32)
    estate_at_death = np.zeros(n_simulations)

    N0_s = len(state_grids_0); N1_s = len(state_grids_1); N2_s = len(state_grids_2)

    for i in prange(n_simulations):
        z_val = initial_z[i]
        s0_t = initial_s[i, 0]
        s1_t = initial_s[i, 1]
        s2_t = initial_s[i, 2]
        x_t = initial_x[i]
        income_t = initial_income[i]

        for t in range(n_age):
            sim_alive[i, t] = True
            sim_z[i, t] = z_val
            sim_x[i, t] = x_t
            sim_income[i, t] = income_t
            sim_state_coords[i, t, 0] = s0_t
            sim_state_coords[i, t, 1] = s1_t
            sim_state_coords[i, t, 2] = s2_t

            # --- z bracketing ---
            iz_lo = int((z_val - z_lo) / dz)
            iz_lo = max(0, min(iz_lo, n_z - 2))
            frac_z = (z_val - z_grid[iz_lo]) / dz
            frac_z = max(0.0, min(1.0, frac_z))
            z_idx_near = iz_lo if frac_z <= 0.5 else iz_lo + 1
            sim_z_idx[i, t] = z_idx_near

            # --- s bracketing in transformed (bracket) coordinates ---
            ds0 = s0_t - state_bracket_shift[0]
            ds1 = s1_t - state_bracket_shift[1]
            ds2 = s2_t - state_bracket_shift[2]
            b0 = state_bracket_L_inv[0, 0] * ds0 + state_bracket_L_inv[0, 1] * ds1 + state_bracket_L_inv[0, 2] * ds2
            b1 = state_bracket_L_inv[1, 0] * ds0 + state_bracket_L_inv[1, 1] * ds1 + state_bracket_L_inv[1, 2] * ds2
            b2 = state_bracket_L_inv[2, 0] * ds0 + state_bracket_L_inv[2, 1] * ds1 + state_bracket_L_inv[2, 2] * ds2

            # Bracket b0 in state_grids_0
            if N0_s == 1:
                lo0 = 0; f0 = 0.0
            elif b0 <= state_grids_0[0]:
                lo0 = 0; f0 = 0.0
            elif b0 >= state_grids_0[N0_s - 1]:
                lo0 = N0_s - 2; f0 = 1.0
            else:
                lo0 = 0
                for ii in range(N0_s - 1):
                    if state_grids_0[ii + 1] > b0:
                        lo0 = ii
                        break
                dg0 = state_grids_0[lo0 + 1] - state_grids_0[lo0]
                f0 = (b0 - state_grids_0[lo0]) / dg0 if dg0 > 1e-30 else 0.0
                f0 = max(0.0, min(1.0, f0))

            if N1_s == 1:
                lo1 = 0; f1 = 0.0
            elif b1 <= state_grids_1[0]:
                lo1 = 0; f1 = 0.0
            elif b1 >= state_grids_1[N1_s - 1]:
                lo1 = N1_s - 2; f1 = 1.0
            else:
                lo1 = 0
                for ii in range(N1_s - 1):
                    if state_grids_1[ii + 1] > b1:
                        lo1 = ii
                        break
                dg1 = state_grids_1[lo1 + 1] - state_grids_1[lo1]
                f1 = (b1 - state_grids_1[lo1]) / dg1 if dg1 > 1e-30 else 0.0
                f1 = max(0.0, min(1.0, f1))

            if N2_s == 1:
                lo2 = 0; f2 = 0.0
            elif b2 <= state_grids_2[0]:
                lo2 = 0; f2 = 0.0
            elif b2 >= state_grids_2[N2_s - 1]:
                lo2 = N2_s - 2; f2 = 1.0
            else:
                lo2 = 0
                for ii in range(N2_s - 1):
                    if state_grids_2[ii + 1] > b2:
                        lo2 = ii
                        break
                dg2 = state_grids_2[lo2 + 1] - state_grids_2[lo2]
                f2 = (b2 - state_grids_2[lo2]) / dg2 if dg2 > 1e-30 else 0.0
                f2 = max(0.0, min(1.0, f2))

            # Diagnostic: nearest grid index from current bracket fractions
            d0 = lo0 if f0 <= 0.5 else lo0 + 1
            d1 = lo1 if f1 <= 0.5 else lo1 + 1
            d2 = lo2 if f2 <= 0.5 else lo2 + 1
            sim_state[i, t] = d0 * N1_s * N2_s + d1 * N2_s + d2

            hi0 = lo0 if N0_s == 1 else lo0 + 1
            hi1 = lo1 if N1_s == 1 else lo1 + 1
            hi2 = lo2 if N2_s == 1 else lo2 + 1

            # 8 trilinear corners — same flat indexing as the solver
            j000 = lo0 * N1_s * N2_s + lo1 * N2_s + lo2
            j001 = lo0 * N1_s * N2_s + lo1 * N2_s + hi2
            j010 = lo0 * N1_s * N2_s + hi1 * N2_s + lo2
            j011 = lo0 * N1_s * N2_s + hi1 * N2_s + hi2
            j100 = hi0 * N1_s * N2_s + lo1 * N2_s + lo2
            j101 = hi0 * N1_s * N2_s + lo1 * N2_s + hi2
            j110 = hi0 * N1_s * N2_s + hi1 * N2_s + lo2
            j111 = hi0 * N1_s * N2_s + hi1 * N2_s + hi2

            w000 = (1.0 - f0) * (1.0 - f1) * (1.0 - f2)
            w001 = (1.0 - f0) * (1.0 - f1) * f2
            w010 = (1.0 - f0) * f1 * (1.0 - f2)
            w011 = (1.0 - f0) * f1 * f2
            w100 = f0 * (1.0 - f1) * (1.0 - f2)
            w101 = f0 * (1.0 - f1) * f2
            w110 = f0 * f1 * (1.0 - f2)
            w111 = f0 * f1 * f2

            # --- Policy lookup: trilinear in s × linear in z × linear in x ---
            c_t = (
                w000 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j000, x_t, wealth_grid)
                + w001 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j001, x_t, wealth_grid)
                + w010 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j010, x_t, wealth_grid)
                + w011 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j011, x_t, wealth_grid)
                + w100 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j100, x_t, wealth_grid)
                + w101 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j101, x_t, wealth_grid)
                + w110 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j110, x_t, wealth_grid)
                + w111 * _interp_policy_zx(C_mat, t, iz_lo, frac_z, j111, x_t, wealth_grid)
            )
            alpha_s_t = (
                w000 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j000, x_t, wealth_grid)
                + w001 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j001, x_t, wealth_grid)
                + w010 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j010, x_t, wealth_grid)
                + w011 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j011, x_t, wealth_grid)
                + w100 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j100, x_t, wealth_grid)
                + w101 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j101, x_t, wealth_grid)
                + w110 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j110, x_t, wealth_grid)
                + w111 * _interp_policy_zx(S_mat, t, iz_lo, frac_z, j111, x_t, wealth_grid)
            )
            alpha_b_t = (
                w000 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j000, x_t, wealth_grid)
                + w001 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j001, x_t, wealth_grid)
                + w010 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j010, x_t, wealth_grid)
                + w011 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j011, x_t, wealth_grid)
                + w100 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j100, x_t, wealth_grid)
                + w101 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j101, x_t, wealth_grid)
                + w110 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j110, x_t, wealth_grid)
                + w111 * _interp_policy_zx(B_mat, t, iz_lo, frac_z, j111, x_t, wealth_grid)
            )

            if constrained:
                alpha_s_t, alpha_b_t = _clean_constrained_shares(alpha_s_t, alpha_b_t)

            # Above wealth_max, fast_interp_1d flat-extrapolates the consumption
            # LEVEL, which forces c/x → 0 and creates extreme-saver pathology in
            # off-grid agents. Rescale to preserve the top-node consumption rate
            # (c/x), matching the asymptotic homotheticity of CRRA policies.
            # Shares are already in rate form so flat extrapolation is fine.
            wealth_max_grid = wealth_grid[len(wealth_grid) - 1]
            if x_t > wealth_max_grid:
                c_t = c_t * (x_t / wealth_max_grid)

            if x_t <= 0.0:
                c_t = 0.0
            else:
                if c_t < 0.0:
                    c_t = 0.0
                if c_t > x_t:
                    c_t = x_t

            savings_t = x_t - c_t

            sim_c[i, t] = c_t
            sim_savings[i, t] = savings_t
            sim_alpha_s[i, t] = alpha_s_t
            sim_alpha_b[i, t] = alpha_b_t

            # Draw state innovation v^s = L_ss @ z, z ~ N(0, I)
            n_state = L_ss.shape[0]
            v_s_0 = 0.0; v_s_1 = 0.0; v_s_2 = 0.0
            for kk in range(n_state):
                zz = normal_draws[i, t, n_ret + 2 + kk]
                v_s_0 += L_ss[0, kk] * zz
                v_s_1 += L_ss[1, kk] * zz
                v_s_2 += L_ss[2, kk] * zz

            # Propagate continuous state s_{t+1} = Phi_0 + Phi_11 @ s_t + v^s
            s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * s0_t + Phi_11[0, 1] * s1_t + Phi_11[0, 2] * s2_t + v_s_0
            s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * s0_t + Phi_11[1, 1] * s1_t + Phi_11[1, 2] * s2_t + v_s_1
            s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * s0_t + Phi_11[2, 1] * s1_t + Phi_11[2, 2] * s2_t + v_s_2

            # Conditional return mean given (s_t, v^s) — 3 returns: rtb, xr, xb
            mu_rtb = const_r[0] + A_r[0, 0] * s0_t + A_r[0, 1] * s1_t + A_r[0, 2] * s2_t + M_matrix[0, 0] * v_s_0 + M_matrix[0, 1] * v_s_1 + M_matrix[0, 2] * v_s_2
            mu_xr  = const_r[1] + A_r[1, 0] * s0_t + A_r[1, 1] * s1_t + A_r[1, 2] * s2_t + M_matrix[1, 0] * v_s_0 + M_matrix[1, 1] * v_s_1 + M_matrix[1, 2] * v_s_2
            mu_xb  = const_r[2] + A_r[2, 0] * s0_t + A_r[2, 1] * s1_t + A_r[2, 2] * s2_t + M_matrix[2, 0] * v_s_0 + M_matrix[2, 1] * v_s_1 + M_matrix[2, 2] * v_s_2

            if use_mc_returns:
                rtb_res = 0.0
                xr_res = 0.0
                xb_res = 0.0
                for k in range(n_ret):
                    shock_k = normal_draws[i, t, k]
                    rtb_res += ret_factor[0, k] * shock_k
                    xr_res += ret_factor[1, k] * shock_k
                    xb_res += ret_factor[2, k] * shock_k
                R_bill = np.exp(mu_rtb + rtb_res)
                R_stock = R_bill * np.exp(mu_xr + xr_res)
                R_bond = R_bill * np.exp(mu_xb + xb_res)
            else:
                ret_idx = draw_discrete(ret_weights, uniform_draws[i, t, 3])
                R_bill = np.exp(mu_rtb + ret_nodes[ret_idx, 0])
                R_stock = R_bill * np.exp(mu_xr + ret_nodes[ret_idx, 1])
                R_bond = R_bill * np.exp(mu_xb + ret_nodes[ret_idx, 2])

            alpha_bill_t = 1.0 - alpha_s_t - alpha_b_t
            R_port = alpha_s_t * R_stock + alpha_b_t * R_bond + alpha_bill_t * R_bill
            estate_t = max(savings_t * R_port, 0.0)

            sim_R_port[i, t] = R_port
            sim_estate[i, t] = estate_t

            age_t = start_age + t

            if t == n_age - 1:
                death_age[i] = age_t
                estate_at_death[i] = estate_t
                break

            # --- Survival (linear z-interp on the same iz_lo/frac_z used
            # for policy lookup; matches the rest of the continuous-z
            # treatment and avoids the step-function in nearest-neighbour). ---
            psi_t = (1.0 - frac_z) * survival_probs_2d[t, iz_lo] + frac_z * survival_probs_2d[t, iz_lo + 1]
            if uniform_draws[i, t, 0] > psi_t:
                death_age[i] = age_t
                estate_at_death[i] = estate_t
                break

            # --- z and income transition ---
            if t < retire_age_idx:
                # Draw eta from continuous mixture
                std_normal_eta = normal_draws[i, t, n_ret]
                if uniform_draws[i, t, 1] < pz:
                    eta = mu_eta1 + sigma_eta1 * std_normal_eta
                else:
                    eta = mu_eta2_eff + sigma_eta2 * std_normal_eta

                # z is propagated unclamped; downstream policy/survival lookup
                # bracket-clamps the policy axis but income uses raw z so tail
                # earnings (and hence pension) are not artificially truncated.
                z_next = rho * z_val + eta

                if t == retire_age_idx - 1:
                    # Work→retirement boundary. The solver replaces next-period
                    # labor income with linearly z-interpolated pension(z_next)
                    # at this step (see solver.py work-to-retirement block where
                    # use_pension_next=True). Match exactly so the simulator's
                    # x_{retire_age} is the same x_{retire_age} the solver
                    # optimised c, alpha_s, alpha_b for.
                    iz_lo_next = int((z_next - z_lo) / dz)
                    iz_lo_next = max(0, min(iz_lo_next, n_z - 2))
                    frac_z_next = (z_next - z_grid[iz_lo_next]) / dz
                    frac_z_next = max(0.0, min(1.0, frac_z_next))
                    income_next = ((1.0 - frac_z_next) * pension_at_z_grid[iz_lo_next]
                                   + frac_z_next * pension_at_z_grid[iz_lo_next + 1])
                else:
                    # Draw eps from continuous mixture
                    std_normal_eps = normal_draws[i, t, n_ret + 1]
                    if uniform_draws[i, t, 2] < pe:
                        eps_val = mu_eps1 + sigma_eps1 * std_normal_eps
                    else:
                        eps_val = mu_eps2_eff + sigma_eps2 * std_normal_eps

                    y_gross = np.exp(log_det_profile[t + 1] + z_next + eps_val)
                    income_next = _scalar_disposable_income(y_gross)
            else:
                # Retirement: z frozen. Pension uses linear z-interp on the
                # discrete-z pension table (same iz_lo/frac_z bracket used for
                # policy lookup at the top of the period), matching the solver's
                # pension_table[t+1, z_i] evaluation at retirement.
                z_next = z_val
                income_next = ((1.0 - frac_z) * pension_at_z_grid[iz_lo]
                               + frac_z * pension_at_z_grid[iz_lo + 1])

            x_t = estate_t + income_next
            income_t = income_next
            z_val = z_next
            s0_t = s_next_0
            s1_t = s_next_1
            s2_t = s_next_2

    return (
        sim_x,
        sim_c,
        sim_savings,
        sim_alpha_s,
        sim_alpha_b,
        sim_R_port,
        sim_income,
        sim_estate,
        sim_z,
        sim_z_idx,
        sim_state,
        sim_state_coords,
        sim_alive,
        death_age,
        estate_at_death,
    )


def _validate_policy_shapes(C_mat: np.ndarray,
                            S_mat: np.ndarray,
                            B_mat: np.ndarray,
                            pc) -> None:
    expected = (pc.n_age, pc.n_z, pc.N_state, pc.n_w)
    for name, arr in (("C_mat", C_mat), ("S_mat", S_mat), ("B_mat", B_mat)):
        if arr.shape != expected:
            raise ValueError(f"{name} has shape {arr.shape}, expected {expected}.")


def _wealth_offgrid_diagnostics(sim_x: np.ndarray,
                                sim_alive: np.ndarray,
                                wealth_grid: np.ndarray) -> dict:
    """Per-age fraction of alive households whose cash-on-hand is outside
    [wealth_min, wealth_max]. Off-grid households fall on the boundary policy
    via flat extrapolation, so a high off-grid share means the simulator is
    no longer reporting moments of the solved model. Standard practice in
    lifecycle simulation: widen the wealth grid until off-grid share is small.
    """
    wlo = float(wealth_grid[0])
    whi = float(wealth_grid[-1])
    n_age = sim_x.shape[1]

    n_alive = sim_alive.sum(axis=0).astype(float)              # (n_age,)
    safe = np.maximum(n_alive, 1.0)
    below_count = ((sim_x < wlo) & sim_alive).sum(axis=0)
    above_count = ((sim_x > whi) & sim_alive).sum(axis=0)
    below_frac = below_count / safe
    above_frac = above_count / safe
    off_frac = below_frac + above_frac

    # Where no households are alive, set to NaN so callers can mask cleanly.
    no_alive = n_alive == 0
    below_frac = np.where(no_alive, np.nan, below_frac)
    above_frac = np.where(no_alive, np.nan, above_frac)
    off_frac = np.where(no_alive, np.nan, off_frac)

    return {
        "wealth_min": wlo,
        "wealth_max": whi,
        "below_frac": below_frac,
        "above_frac": above_frac,
        "off_frac": off_frac,
        "max_off_frac": float(np.nanmax(off_frac)) if not np.all(no_alive) else float("nan"),
        "max_off_age_offset": int(np.nanargmax(off_frac)) if not np.all(no_alive) else -1,
        "n_age": int(n_age),
    }


def simulate_lifecycle(C_mat: np.ndarray,
                       S_mat: np.ndarray,
                       B_mat: np.ndarray,
                       pc,
                       model,
                       n_simulations: int = 10_000,
                       initial_x: Optional[Union[float, np.ndarray]] = None,
                       initial_wealth: Optional[Union[float, np.ndarray]] = 0.1,
                       initial_wealth_distribution: Optional[str] = None,
                       initial_wealth_normal_std: float = 0.0,
                       initial_z: Union[str, np.ndarray] = "stationary",
                       initial_z_normal_std: float = 0.652,
                       initial_state: Union[str, np.ndarray] = "median",
                       seed: int = 42,
                       return_draw_mode: str = "monte_carlo",
                       wealth_offgrid_warn_threshold: float = 0.05,
                       verbose: bool = True) -> dict:
    """
    Simulate lifecycle paths using the solved consumption and portfolio policies.

    Parameters
    ----------
    C_mat, S_mat, B_mat : np.ndarray
        Policy arrays from `solver.run_lifecycle_solver(...)`, each with shape
        `(n_age, n_z, N_state, n_w)`.
    pc : Precompute
        Precomputed grids, transitions, income tables, and survival probabilities.
    model : LifecyclePortfolioModel
        Economic model object.
    n_simulations : int
        Number of simulated households.
    initial_x : float, np.ndarray, or None
        Start-of-period cash-on-hand at age `start_age`.
        If supplied, this overrides the initial-wealth construction.
    initial_wealth : float, np.ndarray, or None
        Pre-labor-income wealth at entry. If `initial_x` is omitted, this is
        combined with period-0 labor income as `x_0 = initial_wealth + Y_0`.
        The default placeholder is `0.1` in model units.
    initial_wealth_distribution : {None, "normal"}
        Optional distribution for initial wealth. If `None`, `initial_wealth`
        is treated as a scalar or custom array. If `"normal"`, `initial_wealth`
        is interpreted as the mean and `initial_wealth_normal_std` as the
        standard deviation.
    initial_wealth_normal_std : float
        Standard deviation used when `initial_wealth_distribution="normal"`.
    initial_z : {"median", "stationary", "normal"} or np.ndarray
        How to initialize the persistent income state. Grid indices are drawn
        then converted to continuous z values from `pc.z_grid`. The `"normal"`
        option draws from N(0, initial_z_normal_std^2) binned onto the grid.
    initial_z_normal_std : float
        Standard deviation used when `initial_z="normal"`.
    initial_state : {"median", "stationary"} or np.ndarray
        How to initialize the financial state.
    seed : int
        Master random seed.
    return_draw_mode : {"monte_carlo", "quadrature"}
        How to simulate residual return shocks each period.
    wealth_offgrid_warn_threshold : float
        If the maximum (over alive ages) fraction of households whose cash-on-hand
        leaves [wealth_min, wealth_max] exceeds this, emit a UserWarning. Off-grid
        households fall on the boundary policy via flat extrapolation, so a high
        share means simulated moments no longer represent the solved model.
        Default 0.05 (5%); set to 1.0 to disable.
    verbose : bool
        Print progress and a simulation summary.

    Returns
    -------
    dict
        Future periods after death remain zero-filled in the panel arrays; use
        the `alive` mask to restrict to active periods.
    """
    _validate_policy_shapes(C_mat, S_mat, B_mat, pc)

    if return_draw_mode not in ("monte_carlo", "quadrature"):
        raise ValueError(
            "return_draw_mode must be 'monte_carlo' or 'quadrature', "
            f"got '{return_draw_mode}'."
        )

    retire_age_idx = model.retire_age - model.start_age
    n_age = pc.n_age
    n_ret = model.n_ret

    if verbose:
        print("=" * 66)
        print("SIMULATING LIFECYCLE PATHS")
        print("=" * 66)
        print(f"  Households:          {n_simulations:,}")
        print(f"  Ages:                {model.start_age} to {model.terminal_age} ({n_age} periods)")
        print(f"  Retirement age:      {model.retire_age} (index {retire_age_idx})")
        print(f"  Income x fin state:  {pc.n_z} z x {pc.N_state} s = {pc.n_z * pc.N_state:,} combined states")
        print(f"  Eta quad nodes:      {len(pc.eta_nodes)} (Judd-mixture, total = n_eta_nodes={pc.disc_config.n_eta_nodes})")
        print(f"  Eps quad nodes:      {len(pc.eps_nodes)} (Judd-mixture, total = n_eps_nodes={pc.disc_config.n_eps_nodes})")
        print(f"  State quad nodes:    {pc.n_state_quad} ({pc.disc_config.n_state_quad_nodes} per dim, solver only)")
        print(f"  Return quad nodes:   {pc.n_ret_quad} (K=({'x'.join(str(k) for k in pc.n_ret_nodes_1d)}) per dim)")
        print(f"  Return draw mode:    {return_draw_mode}")
        if return_draw_mode == "monte_carlo":
            diag = np.sqrt(np.clip(np.diag(model.Sigma_r_cond), 0.0, None))
            print(f"  sigma_resid (rtb,xr,xb): {diag[0]:.4f}, {diag[1]:.4f}, {diag[2]:.4f}")

    rng = np.random.default_rng(seed)

    # Mixture component-2 means derived to enforce E[eta] = E[eps] = 0
    # (model.mu_eta2 and model.mu_eps2 stored on the model object are
    # informational; quadrature and simulation both recompute these.)
    mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1
    mu_eps2_eff = -(model.pe / (1.0 - model.pe)) * model.mu_eps1

    # --- Initialize z as continuous values ---
    if isinstance(initial_z, str) and initial_z == "normal":
        init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=initial_z_normal_std)
        init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
    elif isinstance(initial_z, str) and initial_z == "stationary":
        # Unconditional distribution of z: N(0, sigma_z^2) with
        # sigma_z^2 = Var(eta) / (1 - rho^2). With E[eta]=0 enforced via
        # mu_eta2_eff, Var(eta) = pz*(sigma_eta1^2 + mu_eta1^2)
        #                       + (1-pz)*(sigma_eta2^2 + mu_eta2_eff^2).
        var_eta = (model.pz * (model.sigma_eta1**2 + model.mu_eta1**2)
                   + (1 - model.pz) * (model.sigma_eta2**2 + mu_eta2_eff**2))
        sigma_z = np.sqrt(var_eta / (1.0 - model.rho**2))
        init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=sigma_z)
        init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
    else:
        init_z_idx = initialize_states(n_simulations, pc.n_z, pc.Pi_z, initial_z, rng)
    # Convert grid indices to continuous z values
    init_z_val = pc.z_grid[init_z_idx].astype(np.float64)

    init_state_idx = _resolve_initial_state_indices(n_simulations, pc, initial_state, rng)

    # --- Initial income: continuous mixture eps draw, direct from continuous z ---
    from model import disposable_income_working as _disp_inc_vec
    from model import compute_pension_after_tax as _pension_vec
    if retire_age_idx > 0:
        init_eps_uniform = rng.uniform(size=n_simulations)
        init_eps_normal = rng.standard_normal(size=n_simulations)
        init_eps_val = np.where(
            init_eps_uniform < model.pe,
            model.mu_eps1 + model.sigma_eps1 * init_eps_normal,
            mu_eps2_eff + model.sigma_eps2 * init_eps_normal,
        )
        init_y_gross = np.exp(pc.log_det_profile[0] + init_z_val + init_eps_val)
        initial_income_arr = _disp_inc_vec(init_y_gross)
    else:
        initial_income_arr = _pension_vec(init_z_val, pc.avg_det)

    if initial_x is None:
        init_wealth_arr = _initialize_initial_wealth(
            n_simulations=n_simulations,
            initial_wealth=initial_wealth,
            initial_wealth_distribution=initial_wealth_distribution,
            initial_wealth_normal_std=initial_wealth_normal_std,
            rng=rng,
        )
        init_x = init_wealth_arr + initial_income_arr
        if verbose:
            if initial_wealth_distribution == "normal":
                mean = 0.1 if initial_wealth is None else float(initial_wealth)
                print(
                    f"\n  Initial wealth: N({mean:.3f}, {initial_wealth_normal_std:.3f}^2) before income"
                )
            elif np.isscalar(initial_wealth) or initial_wealth is None:
                wealth_level = 0.1 if initial_wealth is None else float(initial_wealth)
                print(f"\n  Initial wealth: {wealth_level:.3f} before income")
            else:
                print(
                    "\n  Initial wealth: custom array before income  |  "
                    f"mean={np.mean(init_wealth_arr):.3f}  "
                    f"p25={np.percentile(init_wealth_arr, 25):.3f}  "
                    f"p75={np.percentile(init_wealth_arr, 75):.3f}"
                )
            print(
                f"  Initial x = wealth + Y0  |  mean={np.mean(init_x):.3f}  "
                f"p25={np.percentile(init_x, 25):.3f}  "
                f"p75={np.percentile(init_x, 75):.3f}"
            )
    elif np.isscalar(initial_x):
        init_x = np.full(n_simulations, float(initial_x), dtype=float)
        if verbose:
            print(f"\n  Initial x:  {float(initial_x):.4f} (constant)")
    else:
        init_x = np.asarray(initial_x, dtype=float)
        if init_x.shape[0] != n_simulations:
            raise ValueError(
                f"initial_x array has length {init_x.shape[0]}, expected {n_simulations}."
            )
        if verbose:
            print(
                "\n  Initial x:  custom array  |  "
                f"mean={np.mean(init_x):.3f}  "
                f"p25={np.percentile(init_x, 25):.3f}  "
                f"p75={np.percentile(init_x, 75):.3f}"
            )

    if np.any(init_x < 0.0):
        raise ValueError(
            "Constructed initial_x must be non-negative. Check the initial wealth "
            "specification and the period-0 income draw."
        )

    if verbose:
        z_label = initial_z if isinstance(initial_z, str) else "custom array"
        s_label = initial_state if isinstance(initial_state, str) else "custom array"
        print(f"  Initial z state:     {z_label}")
        if isinstance(initial_z, str) and initial_z == "normal":
            print(f"  Initial z sigma:     {initial_z_normal_std:.3f}")
        print(f"  Initial fin state:   {s_label}")

    # uniform_draws columns: [survival, eta-mix, eps-mix, ret-node]
    uniform_draws = rng.uniform(size=(n_simulations, n_age, 4))

    # normal_draws columns: n_ret return residuals + 1 eta + 1 eps + 3 padded
    # state innovations. Lower-dimensional systems use zeros in the omitted
    # state-transition rows/cols of the padded Cholesky factor.
    n_state = int(model.n_state)
    n_normal_cols = n_ret + 2 + 3
    normal_draws = rng.standard_normal(size=(n_simulations, n_age, n_normal_cols))
    if return_draw_mode == "monte_carlo":
        ret_factor_arr = _build_return_factor(model.Sigma_r_cond)
        use_mc_returns = True
    else:
        ret_factor_arr = np.zeros((n_ret, n_ret), dtype=float)
        use_mc_returns = False

    if verbose:
        print("\n  Running simulation...", flush=True)

    t_start = time.perf_counter()

    # Resolve the integer state index (still used to seed each household at
    # an on-grid economic state vector) and convert to continuous s coords.
    init_s_coords = np.ascontiguousarray(pc.state_grid[init_state_idx, :])
    state_inputs = _pad_simulation_state_inputs_to_3d(pc, model, init_s_coords)
    const_r_arr = np.ascontiguousarray(pc.const_r)

    results = simulate_lifecycle_core(
        C_mat=np.ascontiguousarray(C_mat),
        S_mat=np.ascontiguousarray(S_mat),
        B_mat=np.ascontiguousarray(B_mat),
        wealth_grid=np.ascontiguousarray(pc.wealth_grid),
        z_grid=np.ascontiguousarray(pc.z_grid),
        ret_nodes=np.ascontiguousarray(pc.ret_nodes),
        ret_weights=np.ascontiguousarray(pc.ret_weights),
        ret_factor=np.ascontiguousarray(ret_factor_arr),
        log_det_profile=np.ascontiguousarray(pc.log_det_profile),
        avg_det=float(pc.avg_det),
        pension_at_z_grid=np.ascontiguousarray(pc.pension_after_tax[retire_age_idx, :]),
        survival_probs_2d=np.ascontiguousarray(pc.survival_probs_2d),
        rho=float(model.rho),
        pz=float(model.pz),
        mu_eta1=float(model.mu_eta1),
        sigma_eta1=float(model.sigma_eta1),
        sigma_eta2=float(model.sigma_eta2),
        mu_eta2_eff=float(mu_eta2_eff),
        pe=float(model.pe),
        mu_eps1=float(model.mu_eps1),
        sigma_eps1=float(model.sigma_eps1),
        sigma_eps2=float(model.sigma_eps2),
        mu_eps2_eff=float(mu_eps2_eff),
        start_age=int(model.start_age),
        retire_age_idx=int(retire_age_idx),
        constrained=bool(model.constrained),
        use_mc_returns=use_mc_returns,
        initial_z=np.ascontiguousarray(init_z_val),
        initial_s=state_inputs["initial_s"],
        initial_x=np.ascontiguousarray(init_x),
        initial_income=np.ascontiguousarray(initial_income_arr),
        uniform_draws=np.ascontiguousarray(uniform_draws),
        normal_draws=np.ascontiguousarray(normal_draws),
        state_grids_0=state_inputs["state_grids_0"],
        state_grids_1=state_inputs["state_grids_1"],
        state_grids_2=state_inputs["state_grids_2"],
        state_bracket_shift=state_inputs["state_bracket_shift"],
        state_bracket_L_inv=state_inputs["state_bracket_L_inv"],
        L_ss=state_inputs["L_ss"],
        Phi_0_state=state_inputs["Phi_0_state"],
        Phi_11=state_inputs["Phi_11"],
        const_r=const_r_arr,
        A_r=state_inputs["A_r"],
        M_matrix=state_inputs["M_matrix"],
    )

    elapsed = time.perf_counter() - t_start

    (
        sim_x,
        sim_c,
        sim_savings,
        sim_alpha_s,
        sim_alpha_b,
        sim_R_port,
        sim_income,
        sim_estate,
        sim_z,
        sim_z_idx,
        sim_state,
        sim_state_coords,
        sim_alive,
        death_age,
        estate_at_death,
    ) = results

    sim_alpha_bill = 1.0 - sim_alpha_s - sim_alpha_b

    offgrid = _wealth_offgrid_diagnostics(sim_x, sim_alive, pc.wealth_grid)
    if offgrid["max_off_frac"] > wealth_offgrid_warn_threshold:
        peak_age = model.start_age + offgrid["max_off_age_offset"]
        warnings.warn(
            f"simulate_lifecycle: {offgrid['max_off_frac']:.1%} of alive households "
            f"are outside the solved wealth grid [{offgrid['wealth_min']:.2e}, "
            f"{offgrid['wealth_max']:.2f}] at age {peak_age} (threshold "
            f"{wealth_offgrid_warn_threshold:.1%}). Off-grid households fall on "
            "the boundary policy via flat extrapolation, so simulated moments "
            "no longer represent the solved model. Widen wealth_max (and "
            "potentially raise n_wealth) before relying on results.",
            stacklevel=2,
        )

    if verbose:
        _print_simulation_summary(
            model=model,
            pc=pc,
            sim_x=sim_x,
            sim_c=sim_c,
            sim_alpha_s=sim_alpha_s,
            sim_alpha_b=sim_alpha_b,
            sim_R_port=sim_R_port,
            sim_alive=sim_alive,
            death_age=death_age,
            estate_at_death=estate_at_death,
            retire_age_idx=retire_age_idx,
            elapsed=elapsed,
        )

    return {
        "x": sim_x,
        "c": sim_c,
        "savings": sim_savings,
        "alpha_s": sim_alpha_s,
        "alpha_b": sim_alpha_b,
        "alpha_bill": sim_alpha_bill,
        "R_port": sim_R_port,
        "income": sim_income,
        "estate": sim_estate,
        "estate_at_death": estate_at_death,
        "z": sim_z,
        "z_idx": sim_z_idx,
        "state_idx": sim_state,
        "state_coords": sim_state_coords[:, :, :n_state],
        "alive": sim_alive,
        "death_age": death_age,
        "ages": pc.ages.copy(),
        "wealth_offgrid": offgrid,
    }


def _print_simulation_summary(model,
                              pc,
                              sim_x,
                              sim_c,
                              sim_alpha_s,
                              sim_alpha_b,
                              sim_R_port,
                              sim_alive,
                              death_age,
                              estate_at_death,
                              retire_age_idx,
                              elapsed) -> None:
    """Print a short post-simulation summary."""
    n_age = pc.n_age
    start_age = model.start_age
    retire_age = model.retire_age
    terminal_age = model.terminal_age

    def _quantiles(arr, mask):
        vals = arr[mask]
        if vals.size == 0:
            return np.nan, np.nan, np.nan
        return tuple(np.quantile(vals, [0.25, 0.50, 0.75]))

    alive_terminal = sim_alive[:, -1]
    surv_rate = np.mean(alive_terminal)
    median_death_age = np.median(death_age[death_age >= 0])

    print(f"\n  Simulation complete: {elapsed:.2f}s")
    print(f"  Survival to age {terminal_age}: {surv_rate:.1%}")
    print(f"  Median death age: {median_death_age:.0f}")

    phases = [
        ("Start", 0, start_age),
        ("Mid-work", retire_age_idx // 2, start_age + retire_age_idx // 2),
        ("Pre-retire", max(0, retire_age_idx - 1), retire_age - 1),
        ("Retire", min(retire_age_idx, n_age - 1), retire_age),
    ]

    age75_t = 75 - start_age
    if 0 <= age75_t < n_age:
        phases.append(("Age 75", age75_t, 75))

    phases.append(("Terminal", n_age - 1, terminal_age))

    print("\n  Cash-on-hand by phase")
    print(f"  {'Phase':<20} {'age':>3}  {'alive%':>6}  {'p25':>7}  {'p50':>7}  {'p75':>7}")
    for label, t_idx, age in phases:
        if t_idx < 0 or t_idx >= n_age:
            continue
        alive_t = sim_alive[:, t_idx]
        surv_t = np.mean(alive_t)
        if np.any(alive_t):
            p25, p50, p75 = _quantiles(sim_x[:, t_idx], alive_t)
            print(f"  {label:<20} {age:>3}  {surv_t:>5.1%}  {p25:>7.3f}  {p50:>7.3f}  {p75:>7.3f}")

    print("\n  Consumption share c/x by phase")
    print(f"  {'Phase':<20} {'age':>3}  {'mean c/x':>8}  {'p50 c/x':>8}")
    for label, t_idx, age in phases:
        if t_idx < 0 or t_idx >= n_age:
            continue
        alive_t = sim_alive[:, t_idx]
        if np.any(alive_t):
            x_t = sim_x[alive_t, t_idx]
            c_t = sim_c[alive_t, t_idx]
            ratio = np.where(x_t > 0, c_t / x_t, np.nan)
            valid = ratio[np.isfinite(ratio)]
            if valid.size > 0:
                print(f"  {label:<20} {age:>3}  {np.mean(valid):>8.3f}  {np.median(valid):>8.3f}")

    print("\n  Portfolio shares by phase")
    print(f"  {'Phase':<20} {'age':>3}  {'a_stock':>8}  {'a_bond':>8}  {'a_bill':>8}")
    for label, t_idx, age in phases[:5]:
        if t_idx < 0 or t_idx >= n_age:
            continue
        alive_t = sim_alive[:, t_idx]
        if np.any(alive_t):
            s_mean = np.mean(sim_alpha_s[alive_t, t_idx])
            b_mean = np.mean(sim_alpha_b[alive_t, t_idx])
            bill_mean = 1.0 - s_mean - b_mean
            print(f"  {label:<20} {age:>3}  {s_mean:>8.3f}  {b_mean:>8.3f}  {bill_mean:>8.3f}")

    R_port_alive = sim_R_port[sim_alive]
    if R_port_alive.size > 0:
        rp_mean = np.mean(R_port_alive)
        rp_std = np.std(R_port_alive)
        rp_p10, rp_p50, rp_p90 = np.quantile(R_port_alive, [0.10, 0.50, 0.90])
        print("\n  Realized gross portfolio return (all alive periods)")
        print(f"  Mean={rp_mean:.4f}  Std={rp_std:.4f}  p10={rp_p10:.4f}  p50={rp_p50:.4f}  p90={rp_p90:.4f}")

    pos_estate = estate_at_death[estate_at_death > 0]
    e_mean = np.mean(estate_at_death)
    e_med = np.median(pos_estate) if pos_estate.size > 0 else np.nan
    e_p90 = np.percentile(pos_estate, 90) if pos_estate.size > 0 else np.nan
    print("\n  Estate at death")
    print(f"  Mean={e_mean:.4f}  Median(>0)={e_med:.4f}  p90(>0)={e_p90:.4f}")
    print("=" * 66)
