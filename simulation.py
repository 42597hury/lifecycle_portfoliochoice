"""
simulation.py - Lifecycle simulation for the refactored portfolio model.

This module simulates household paths downstream of the backward solver in
`solver.py`. Policies are defined on start-of-period cash-on-hand and the
simulation follows the same timing as the Bellman problem:

1. Enter period t alive with state (x_t, z_t, s_t).
2. Look up policies c_t, alpha_s_t, alpha_b_t on the solved grids.
3. Savings a_t = x_t - c_t.
4. Propagate the next financial state continuously and map it back to the
   nearest policy-grid index for next period's lookup.
5. Realize end-of-period wealth a_t * R_port via one of two modes:
   - "monte_carlo": residual return drawn from N(0, Sigma_r_cond) via a
     precomputed factor matrix F such that F @ F' = Sigma_r_cond.
   - "quadrature": residual drawn by sampling a Gauss-Hermite node using
     ret_weights.
6. Draw survival from psi[t, z_t].
7. If alive, draw next-period income and form x_{t+1} = a_t * R_port + Y_{t+1}.

Income timing (MODEL_DESIGN Section 1.3):
  - Working phase (t < retire_age_idx):
      z transitions continuously: z' = rho*z + eta, where eta is drawn from
      the mixture distribution. Income is computed directly from continuous z:
      Y = disposable_income(exp(f(age) + z + eps)).
  - Last working period (t = retire_age_idx - 1):
      z transitions one final time; realized z at age 67 determines pension.
  - Retirement phase (t >= retire_age_idx, age 67+):
      z frozen; pension computed directly from continuous z via PIA formula.

At the terminal age, all remaining households die. Their final estate
    estate_T = savings_T * R_port_T
is recorded as estate_at_death.

Initial income consistency:
  - If initial_x is None, each household starts at
      x_0 = initial_wealth_i + Y_{0,i}
    where initial_wealth_i is pre-labor-income asset wealth at entry and Y_{0,i}
    is a drawn income realization at age 0.
  - If initial_x is supplied, sim_income[:, 0] still records a drawn income
    for bookkeeping, but that draw was not used to construct x_0.
"""

from __future__ import annotations

import time
import warnings
from math import erf, sqrt
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

from model import scalar_disposable_income

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
    """Mode-aware initialization for the financial state."""
    if isinstance(initial_state, str) and initial_state == "stationary":
        probs = getattr(pc, "state_stationary_probs", None)
        if probs is not None:
            probs = np.asarray(probs, dtype=float)
            probs = np.clip(probs, 0.0, None)
            probs = probs / np.maximum(probs.sum(), 1e-300)
            return rng.choice(pc.N_state, size=n_simulations, p=probs).astype(np.int32)
    return initialize_states(n_simulations, pc.N_state, pc.Pi_state, initial_state, rng)


def _draw_discrete_vectorized(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Vectorized inverse-CDF draw from a fixed discrete distribution."""
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, uniforms, side="left")
    return np.clip(indices, 0, len(weights) - 1).astype(np.int32)


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """Scalar normal CDF using the error function."""
    if x == np.inf:
        return 1.0
    if x == -np.inf:
        return 0.0
    z = (x - mean) / (std * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def _normal_bin_probs(grid: np.ndarray,
                      mean: float = 0.0,
                      std: float = 0.652) -> np.ndarray:
    """
    Probability mass of N(mean, std^2) over the bins implied by a sorted grid.

    Bin edges are the midpoints between neighboring grid points, with
    (-inf, +inf) at the tails.
    """
    grid = np.asarray(grid, dtype=float)
    n = grid.shape[0]

    if n == 0:
        raise ValueError("grid must contain at least one point.")
    if std <= 0.0:
        raise ValueError("initial_z_normal_std must be strictly positive.")
    if n == 1:
        return np.ones(1, dtype=float)

    edges = np.empty(n + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])

    probs = np.empty(n, dtype=float)
    for j in range(n):
        probs[j] = _normal_cdf(edges[j + 1], mean, std) - _normal_cdf(edges[j], mean, std)

    probs = np.clip(probs, 0.0, None)
    mass = probs.sum()
    if mass <= 0.0:
        raise ValueError("Normal bin probabilities sum to zero; check the grid and std.")
    return probs / mass


def _build_return_factor(Sigma_r_cond: np.ndarray) -> np.ndarray:
    """
    Build a factor matrix F such that F @ F' = Sigma_r_cond.

    Uses an eigen decomposition instead of Cholesky so the code remains robust
    if Sigma_r_cond is positive semidefinite but not strictly positive definite.
    """
    Sigma = np.asarray(Sigma_r_cond, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    if np.any(eigvals < -1e-12):
        raise ValueError("Sigma_r_cond must be positive semidefinite.")
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def _initialize_initial_wealth(n_simulations: int,
                               initial_wealth: Optional[Union[float, np.ndarray]],
                               initial_wealth_distribution: Optional[str],
                               initial_wealth_normal_std: float,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Build the vector of pre-labor-income wealth at entry.

    This is interpreted as asset wealth households bring into working life
    before receiving their first labor-income draw. If `initial_x` is not
    supplied, period-0 cash-on-hand is constructed as:

        x_0 = initial_wealth_i + Y_{0,i}

    TODO: clarify the mapping between Catherine's "0.1 x national wage index"
    calibration and the model's internal income units. For now, `0.1` is
    treated as a direct model-unit placeholder.

    Initial wealth is floored at zero. This is a clipped normal, not a
    truncated normal, so negative draws pile up at zero.
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
    """Linear interpolation on a sorted grid with linear extrapolation."""
    n = len(x_grid)

    if x <= x_grid[0]:
        dx = x_grid[1] - x_grid[0] + 1e-30
        return y_grid[0] + (y_grid[1] - y_grid[0]) * (x - x_grid[0]) / dx
    if x >= x_grid[n - 1]:
        dx = x_grid[n - 1] - x_grid[n - 2] + 1e-30
        return y_grid[n - 1] + (y_grid[n - 1] - y_grid[n - 2]) * (x - x_grid[n - 1]) / dx

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
    Pi_state: np.ndarray,
    mu_r: np.ndarray,
    ret_nodes: np.ndarray,
    ret_weights: np.ndarray,
    ret_factor: np.ndarray,
    log_det_profile: np.ndarray,
    avg_det: float,
    eps_nodes: np.ndarray,
    eps_weights: np.ndarray,
    survival_probs_2d: np.ndarray,
    rho: float,
    pz: float,
    mu_eta1: float,
    sigma_eta1: float,
    sigma_eta2: float,
    mu_eta2_eff: float,
    start_age: int,
    retire_age_idx: int,
    constrained: bool,
    use_mc_returns: bool,
    initial_z: np.ndarray,
    initial_state: np.ndarray,
    initial_x: np.ndarray,
    initial_income: np.ndarray,
    uniform_draws: np.ndarray,
    normal_draws: np.ndarray,
    # --- Continuous state transition arrays ---
    use_continuous_state: bool,
    state_grid: np.ndarray,
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

    z is tracked as a continuous float and policies are interpolated in z,
    consistent with the solver's Gauss-Hermite quadrature treatment.
    Income and pension are computed directly from continuous z (no table
    interpolation).

    uniform_draws[:, t, :] slots:
      [0] survival draw
      [1] mixture component draw (u < pz -> component 1)
      [2] financial state transition draw (legacy discrete branch only)
      [3] return-node draw (quadrature mode only)
      [4] transitory income shock draw

    normal_draws[:, t, :] slots:
      [0..n_ret-1] return shocks (monte_carlo mode)
      [n_ret]      eta standard normal for persistent income innovation
      [n_ret+1..n_ret+n_state] state innovation standard normals (when use_continuous_state=True)
    """
    n_simulations = len(initial_x)
    n_age = C_mat.shape[0]
    n_z = len(z_grid)
    n_ret = mu_r.shape[2]
    dz = z_grid[1] - z_grid[0]
    z_lo = z_grid[0]
    z_hi = z_grid[n_z - 1]

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
    sim_alive = np.zeros((n_simulations, n_age), dtype=np.bool_)

    death_age = np.full(n_simulations, -1, dtype=np.int32)
    estate_at_death = np.zeros(n_simulations)

    for i in prange(n_simulations):
        z_val = initial_z[i]
        state_idx = initial_state[i]
        x_t = initial_x[i]
        income_t = initial_income[i]

        for t in range(n_age):
            sim_alive[i, t] = True
            sim_z[i, t] = z_val
            sim_state[i, t] = state_idx
            sim_x[i, t] = x_t
            sim_income[i, t] = income_t

            # --- z interpolation indices for policy lookup ---
            iz_lo = int((z_val - z_lo) / dz)
            iz_lo = max(0, min(iz_lo, n_z - 2))
            frac_z = (z_val - z_grid[iz_lo]) / dz
            frac_z = max(0.0, min(1.0, frac_z))

            # nearest grid index (for survival lookup and diagnostics)
            z_idx_near = iz_lo if frac_z <= 0.5 else iz_lo + 1
            sim_z_idx[i, t] = z_idx_near

            # --- Policy lookup: interpolate in z, then in wealth ---
            c_lo = fast_interp_1d(x_t, wealth_grid, C_mat[t, iz_lo, state_idx, :])
            c_hi = fast_interp_1d(x_t, wealth_grid, C_mat[t, iz_lo + 1, state_idx, :])
            c_t = (1.0 - frac_z) * c_lo + frac_z * c_hi

            as_lo = fast_interp_1d(x_t, wealth_grid, S_mat[t, iz_lo, state_idx, :])
            as_hi = fast_interp_1d(x_t, wealth_grid, S_mat[t, iz_lo + 1, state_idx, :])
            alpha_s_t = (1.0 - frac_z) * as_lo + frac_z * as_hi

            ab_lo = fast_interp_1d(x_t, wealth_grid, B_mat[t, iz_lo, state_idx, :])
            ab_hi = fast_interp_1d(x_t, wealth_grid, B_mat[t, iz_lo + 1, state_idx, :])
            alpha_b_t = (1.0 - frac_z) * ab_lo + frac_z * ab_hi

            if constrained:
                alpha_s_t, alpha_b_t = _clean_constrained_shares(alpha_s_t, alpha_b_t)

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

            if use_continuous_state:
                # Draw v^s = L_ss @ z where z ~ N(0, I)
                n_state = L_ss.shape[0]
                v_s_0 = 0.0; v_s_1 = 0.0; v_s_2 = 0.0
                for kk in range(n_state):
                    zz = normal_draws[i, t, n_ret + 1 + kk]
                    v_s_0 += L_ss[0, kk] * zz
                    v_s_1 += L_ss[1, kk] * zz
                    v_s_2 += L_ss[2, kk] * zz

                # Propagate state
                s_cur = state_grid[state_idx]
                s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * s_cur[0] + Phi_11[0, 1] * s_cur[1] + Phi_11[0, 2] * s_cur[2] + v_s_0
                s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * s_cur[0] + Phi_11[1, 1] * s_cur[1] + Phi_11[1, 2] * s_cur[2] + v_s_1
                s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * s_cur[0] + Phi_11[2, 1] * s_cur[1] + Phi_11[2, 2] * s_cur[2] + v_s_2

                # Conditional return mean from continuous v^s (3 returns: rtb, xr, xb)
                mu_rtb = const_r[0] + A_r[0, 0] * s_cur[0] + A_r[0, 1] * s_cur[1] + A_r[0, 2] * s_cur[2] + M_matrix[0, 0] * v_s_0 + M_matrix[0, 1] * v_s_1 + M_matrix[0, 2] * v_s_2
                mu_xr  = const_r[1] + A_r[1, 0] * s_cur[0] + A_r[1, 1] * s_cur[1] + A_r[1, 2] * s_cur[2] + M_matrix[1, 0] * v_s_0 + M_matrix[1, 1] * v_s_1 + M_matrix[1, 2] * v_s_2
                mu_xb  = const_r[2] + A_r[2, 0] * s_cur[0] + A_r[2, 1] * s_cur[1] + A_r[2, 2] * s_cur[2] + M_matrix[2, 0] * v_s_0 + M_matrix[2, 1] * v_s_1 + M_matrix[2, 2] * v_s_2

                ds0 = s_next_0 - state_bracket_shift[0]
                ds1 = s_next_1 - state_bracket_shift[1]
                ds2 = s_next_2 - state_bracket_shift[2]
                b_next_0 = state_bracket_L_inv[0, 0] * ds0 + state_bracket_L_inv[0, 1] * ds1 + state_bracket_L_inv[0, 2] * ds2
                b_next_1 = state_bracket_L_inv[1, 0] * ds0 + state_bracket_L_inv[1, 1] * ds1 + state_bracket_L_inv[1, 2] * ds2
                b_next_2 = state_bracket_L_inv[2, 0] * ds0 + state_bracket_L_inv[2, 1] * ds1 + state_bracket_L_inv[2, 2] * ds2

                # Find nearest grid point for next period's policy lookup
                N1_s = len(state_grids_1); N2_s = len(state_grids_2)
                best_d0 = 0; best_dist = abs(b_next_0 - state_grids_0[0])
                for dd in range(1, len(state_grids_0)):
                    d = abs(b_next_0 - state_grids_0[dd])
                    if d < best_dist:
                        best_dist = d; best_d0 = dd
                best_d1 = 0; best_dist = abs(b_next_1 - state_grids_1[0])
                for dd in range(1, N1_s):
                    d = abs(b_next_1 - state_grids_1[dd])
                    if d < best_dist:
                        best_dist = d; best_d1 = dd
                best_d2 = 0; best_dist = abs(b_next_2 - state_grids_2[0])
                for dd in range(1, N2_s):
                    d = abs(b_next_2 - state_grids_2[dd])
                    if d < best_dist:
                        best_dist = d; best_d2 = dd
                next_state_idx = best_d0 * N1_s * N2_s + best_d1 * N2_s + best_d2
            else:
                next_state_idx = draw_discrete(Pi_state[state_idx, :], uniform_draws[i, t, 2])
                mu_rtb = mu_r[state_idx, next_state_idx, 0]
                mu_xr  = mu_r[state_idx, next_state_idx, 1]
                mu_xb  = mu_r[state_idx, next_state_idx, 2]

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
            estate_t = savings_t * R_port

            sim_R_port[i, t] = R_port
            sim_estate[i, t] = estate_t

            age_t = start_age + t

            if t == n_age - 1:
                death_age[i] = age_t
                estate_at_death[i] = estate_t
                break

            # --- Survival (nearest-neighbor z lookup) ---
            if uniform_draws[i, t, 0] > survival_probs_2d[t, z_idx_near]:
                death_age[i] = age_t
                estate_at_death[i] = estate_t
                break

            # --- z and income transition ---
            if t < retire_age_idx:
                # Draw eta from mixture: component selection + normal draw
                std_normal_eta = normal_draws[i, t, n_ret]
                if uniform_draws[i, t, 1] < pz:
                    eta = mu_eta1 + sigma_eta1 * std_normal_eta
                else:
                    eta = mu_eta2_eff + sigma_eta2 * std_normal_eta

                z_next = rho * z_val + eta
                # Clamp to grid bounds
                z_next = max(z_lo, min(z_hi, z_next))

                # Transitory shock (discrete quadrature draw)
                eps_idx = draw_discrete(eps_weights, uniform_draws[i, t, 4])

                # Direct income computation
                y_gross = np.exp(log_det_profile[t + 1] + z_next + eps_nodes[eps_idx])
                income_next = _scalar_disposable_income(y_gross)
            else:
                # Retirement: z frozen, pension from continuous z
                z_next = z_val
                income_next = _scalar_pension_after_tax(z_val, avg_det)

            x_t = estate_t + income_next
            income_t = income_next
            z_val = z_next
            state_idx = next_state_idx

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
        print(f"  Return quad nodes:   {pc.n_ret_quad} (K=({'x'.join(str(k) for k in pc.n_ret_nodes_1d)}) per dim)")
        print(f"  Return draw mode:    {return_draw_mode}")
        if return_draw_mode == "monte_carlo":
            diag = np.sqrt(np.clip(np.diag(model.Sigma_r_cond), 0.0, None))
            print(f"  sigma_resid (xr,xb): {diag[0]:.4f}, {diag[1]:.4f}")

    rng = np.random.default_rng(seed)

    # --- Initialize z as continuous values ---
    if isinstance(initial_z, str) and initial_z == "normal":
        init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=initial_z_normal_std)
        init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
    elif isinstance(initial_z, str) and initial_z == "stationary":
        # Unconditional distribution of z: N(0, sigma_z^2) where sigma_z^2 = Var(eta) / (1 - rho^2)
        var_eta = (model.pz * (model.sigma_eta1**2 + model.mu_eta1**2)
                   + (1 - model.pz) * (model.sigma_eta2**2 + model.mu_eta2**2))
        sigma_z = np.sqrt(var_eta / (1.0 - model.rho**2))
        init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=sigma_z)
        init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
    else:
        init_z_idx = initialize_states(n_simulations, pc.n_z, pc.Pi_z, initial_z, rng)
    # Convert grid indices to continuous z values
    init_z_val = pc.z_grid[init_z_idx].astype(np.float64)

    init_state_idx = _resolve_initial_state_indices(n_simulations, pc, initial_state, rng)

    # --- Initial income: compute directly from continuous z ---
    # Use the vectorized model functions (outside the Numba core loop)
    from model import disposable_income_working as _disp_inc_vec
    from model import compute_pension_after_tax as _pension_vec
    init_eps_raw = rng.uniform(size=n_simulations)
    if retire_age_idx > 0:
        init_eps_idx = _draw_discrete_vectorized(pc.eps_weights, init_eps_raw)
        init_y_gross = np.exp(pc.log_det_profile[0] + init_z_val + pc.eps_nodes[init_eps_idx])
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

    uniform_draws = rng.uniform(size=(n_simulations, n_age, 5))

    # normal_draws: n_ret columns for return shocks + 1 column for eta + n_state for state innovations
    n_state = int(model.n_state)
    use_continuous_state = hasattr(pc, 'v_nodes')  # True if state quadrature was set up
    n_normal_cols = n_ret + 1 + (n_state if use_continuous_state else 0)
    if return_draw_mode == "monte_carlo":
        ret_factor_arr = _build_return_factor(model.Sigma_r_cond)
        normal_draws = rng.standard_normal(size=(n_simulations, n_age, n_normal_cols))
        use_mc_returns = True
    else:
        ret_factor_arr = np.zeros((n_ret, n_ret), dtype=float)
        normal_draws = rng.standard_normal(size=(n_simulations, n_age, n_normal_cols))
        use_mc_returns = False

    if verbose:
        print("\n  Running simulation...", flush=True)

    # Mixture parameters for continuous eta draws
    mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1

    t_start = time.perf_counter()

    # Prepare continuous state transition arrays
    if use_continuous_state:
        L_ss_arr = np.ascontiguousarray(np.linalg.cholesky(
            0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T)))
        state_grid_arr = np.ascontiguousarray(pc.state_grid)
        sg0 = np.ascontiguousarray(pc.state_bracket_grids[0])
        sg1 = np.ascontiguousarray(pc.state_bracket_grids[1])
        sg2 = np.ascontiguousarray(pc.state_bracket_grids[2])
        state_bracket_shift_arr = np.ascontiguousarray(pc.state_bracket_shift)
        state_bracket_L_inv_arr = np.ascontiguousarray(pc.state_bracket_L_inv)
        Phi_0_state_arr = np.ascontiguousarray(model.Phi_0_state)
        Phi_11_arr = np.ascontiguousarray(model.Phi_11)
        const_r_arr = np.ascontiguousarray(pc.const_r)
        A_r_arr = np.ascontiguousarray(pc.A_r)
        M_matrix_arr = np.ascontiguousarray(model.M)
    else:
        L_ss_arr = np.empty((0, 0), dtype=float)
        state_grid_arr = np.empty((0, 0), dtype=float)
        sg0 = np.empty(0, dtype=float)
        sg1 = np.empty(0, dtype=float)
        sg2 = np.empty(0, dtype=float)
        state_bracket_shift_arr = np.empty(0, dtype=float)
        state_bracket_L_inv_arr = np.empty((0, 0), dtype=float)
        Phi_0_state_arr = np.empty(0, dtype=float)
        Phi_11_arr = np.empty((0, 0), dtype=float)
        const_r_arr = np.empty(0, dtype=float)
        A_r_arr = np.empty((0, 0), dtype=float)
        M_matrix_arr = np.empty((0, 0), dtype=float)

    results = simulate_lifecycle_core(
        C_mat=np.ascontiguousarray(C_mat),
        S_mat=np.ascontiguousarray(S_mat),
        B_mat=np.ascontiguousarray(B_mat),
        wealth_grid=np.ascontiguousarray(pc.wealth_grid),
        z_grid=np.ascontiguousarray(pc.z_grid),
        Pi_state=np.ascontiguousarray(pc.Pi_state),
        mu_r=np.ascontiguousarray(pc.mu_r),
        ret_nodes=np.ascontiguousarray(pc.ret_nodes),
        ret_weights=np.ascontiguousarray(pc.ret_weights),
        ret_factor=np.ascontiguousarray(ret_factor_arr),
        log_det_profile=np.ascontiguousarray(pc.log_det_profile),
        avg_det=float(pc.avg_det),
        eps_nodes=np.ascontiguousarray(pc.eps_nodes),
        eps_weights=np.ascontiguousarray(pc.eps_weights),
        survival_probs_2d=np.ascontiguousarray(pc.survival_probs_2d),
        rho=float(model.rho),
        pz=float(model.pz),
        mu_eta1=float(model.mu_eta1),
        sigma_eta1=float(model.sigma_eta1),
        sigma_eta2=float(model.sigma_eta2),
        mu_eta2_eff=float(mu_eta2_eff),
        start_age=int(model.start_age),
        retire_age_idx=int(retire_age_idx),
        constrained=bool(model.constrained),
        use_mc_returns=use_mc_returns,
        initial_z=np.ascontiguousarray(init_z_val),
        initial_state=np.ascontiguousarray(init_state_idx),
        initial_x=np.ascontiguousarray(init_x),
        initial_income=np.ascontiguousarray(initial_income_arr),
        uniform_draws=np.ascontiguousarray(uniform_draws),
        normal_draws=np.ascontiguousarray(normal_draws),
        use_continuous_state=use_continuous_state,
        state_grid=state_grid_arr,
        state_grids_0=sg0,
        state_grids_1=sg1,
        state_grids_2=sg2,
        state_bracket_shift=state_bracket_shift_arr,
        state_bracket_L_inv=state_bracket_L_inv_arr,
        L_ss=L_ss_arr,
        Phi_0_state=Phi_0_state_arr,
        Phi_11=Phi_11_arr,
        const_r=const_r_arr,
        A_r=A_r_arr,
        M_matrix=M_matrix_arr,
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
        sim_alive,
        death_age,
        estate_at_death,
    ) = results

    sim_alpha_bill = 1.0 - sim_alpha_s - sim_alpha_b

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
        "alive": sim_alive,
        "death_age": death_age,
        "ages": pc.ages.copy(),
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
