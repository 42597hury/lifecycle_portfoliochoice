"""simulation.py — JAX-pure forward simulation of lifecycle paths.

Hot path: vmap over households, lax.scan over time. Per-household carry holds
``(z, s_3vec, x, alive, income, t_idx)``; the scan output is the per-period
panel slice. Death is a carried boolean — no Python ``break``, just
``alive_next = alive & (uniform <= psi)``. Future periods after death stay
zero-filled and are masked off via the ``alive`` panel.

RNG strategy: NumPy ``rng.uniform`` / ``rng.standard_normal`` are pre-generated
on the host and shipped to device as ``(n_sim, n_age, n_cols)`` arrays. This
keeps the kernel pure and preserves bit-equivalence with the Numba simulator
when policies are unchanged. (Migrating to ``jax.random.PRNGKey`` is a
follow-up — see handoff 3 deferred-work list.)

Wealth-extrapolation policy:
    Inside the wealth grid: linear interp.
    Outside: flat extrap (fast_interp_1d's flat-tail variant).
    Above wealth_max only: rescale c_t by x_t / wealth_max so the consumption
    rate (c/x) is preserved at the upper edge — matches CRRA homotheticity.

Wealth-dynamics in the simulated economy:
    Arithmetic ``R_p = a_s*R_stock + a_b*R_bond + a_bill*R_bill`` —
    INTENTIONALLY divergent from the solver's CCV log spec. The CCV form is
    a value-function approximation; the simulator uses the truth. See the
    handoff-3 spec for the rationale (don't "harmonise" them).

Bankruptcy: NO clamp on ``estate_t``. With leverage uncapped, catastrophic
portfolio realisations can drive ``x_{t+1} < 0``. The bracket-clamp on
consumption (``c_t = 0`` when ``x_t <= 0``) keeps the household alive on
negative wealth; ``_wealth_offgrid_diagnostics`` reports the share of
households below ``wealth_min`` so this is visible without breaking the run.
"""

from __future__ import annotations

import time
import warnings
from functools import partial
from typing import Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from lifecycle.model import disposable_income_working, compute_pension_after_tax
from lifecycle.numerics import _normal_bin_probs

__all__ = ["simulate_lifecycle"]


# =============================================================================
# Host-side initialisation helpers (untouched from pre-rewrite)
# =============================================================================

def get_stationary_distribution(Pi, tol=1e-12, max_iter=10_000):
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


def initialize_states(n_simulations, n_states, Pi, method, rng):
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


def _resolve_initial_state_indices(n_simulations, pc, initial_state, rng):
    if isinstance(initial_state, str) and initial_state == "stationary":
        probs = np.asarray(pc.state_stationary_probs, dtype=float)
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        return rng.choice(pc.N_state, size=n_simulations, p=probs).astype(np.int32)
    return initialize_states(n_simulations, pc.N_state, pc.Pi_state, initial_state, rng)


def _build_return_factor(Sigma_r_cond):
    """Cholesky factor of the conditional return covariance — matches the
    rest of the codebase's convention (state grid, state quadrature, return
    quadrature all Cholesky as of 2026-04-30).
    """
    Sigma = np.asarray(Sigma_r_cond, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return np.linalg.cholesky(Sigma)


def _initialize_initial_wealth(n_simulations, initial_wealth,
                                initial_wealth_distribution,
                                initial_wealth_normal_std, rng):
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


# =============================================================================
# JAX hot-path helpers
# =============================================================================

def _flat_interp_1d(x, x_grid, y_grid):
    """Linear interpolation on a sorted grid with FLAT extrapolation outside.

    Off-grid households should be diagnosed (via ``_wealth_offgrid_diagnostics``)
    rather than given an extrapolated policy that may diverge. The solver's
    ``interp_1d_lin_extrap`` is deliberately divergent — Newton needs a valid
    gradient at the wealth-grid edge; the simulator just needs a stable value.
    """
    n = x_grid.shape[0]
    iw = jnp.clip(jnp.searchsorted(x_grid, x, side="right") - 1, 0, n - 2)
    x0 = x_grid[iw]
    x1 = x_grid[iw + 1]
    y0 = y_grid[iw]
    y1 = y_grid[iw + 1]
    inside = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return jnp.where(
        x < x_grid[0], y_grid[0],
        jnp.where(x > x_grid[-1], y_grid[-1], inside),
    )


def _bracket_uniform(z, z_lo, dz, n_z):
    iz = jnp.clip(((z - z_lo) / dz).astype(jnp.int32), 0, n_z - 2)
    z0 = z_lo + dz * iz
    frac = jnp.clip((z - z0) / dz, 0.0, 1.0)
    return iz, frac


def _bracket_axis(grid, val):
    """Bracket ``val`` in a sorted (possibly non-uniform) axis grid.

    Mirrors solver._bracket_axis. Each axis must have at least 2 points;
    degenerate n=1 axes are not supported.
    """
    n = grid.shape[0]
    lo = jnp.clip(jnp.searchsorted(grid, val, side="right") - 1, 0, n - 2)
    denom = grid[lo + 1] - grid[lo]
    frac = jnp.clip((val - grid[lo]) / denom, 0.0, 1.0)
    return lo, frac


def _draw_discrete(probs, u):
    """Inverse-CDF sample from a discrete distribution given a single uniform draw."""
    cum = jnp.cumsum(probs)
    idx = jnp.searchsorted(cum, u, side="right")
    return jnp.clip(idx, 0, probs.shape[0] - 1)


def jax_disposable_income(y_gross):
    """JAX port of ``model.scalar_disposable_income`` (7 progressive brackets).

    Bit-identical (to ~1e-15) with model.scalar_disposable_income for any
    finite scalar ``y_gross``. Uses ``jnp.where`` rather than a Python if/elif
    tower so it traces under jit + vmap.
    """
    payroll_tax = 0.106 * jnp.minimum(y_gross, 2.5)
    taxable = jnp.maximum(0.0, y_gross - payroll_tax)
    tax = jnp.where(
        taxable <= 0.18, taxable * 0.10,
        jnp.where(
            taxable <= 0.72, 0.018 + (taxable - 0.18) * 0.12,
            jnp.where(
                taxable <= 1.54, 0.0828 + (taxable - 0.72) * 0.22,
                jnp.where(
                    taxable <= 2.94, 0.2632 + (taxable - 1.54) * 0.24,
                    jnp.where(
                        taxable <= 3.73, 0.5992 + (taxable - 2.94) * 0.32,
                        jnp.where(
                            taxable <= 9.32, 0.8520 + (taxable - 3.73) * 0.35,
                            2.8085 + (taxable - 9.32) * 0.37,
                        ),
                    ),
                ),
            ),
        ),
    )
    return taxable - tax


def _interp_policy_zw_at_corner(arr4d, t_idx, iz_lo, frac_z, j_s, x_t, wealth_grid):
    """Linear-in-z, linear-in-x, flat-extrap-in-x at one state-grid corner."""
    v_lo = _flat_interp_1d(x_t, wealth_grid, arr4d[t_idx, iz_lo,     j_s, :])
    v_hi = _flat_interp_1d(x_t, wealth_grid, arr4d[t_idx, iz_lo + 1, j_s, :])
    return (1.0 - frac_z) * v_lo + frac_z * v_hi


def _trilinear_z_w_lookup(arr4d, t_idx, iz_lo, frac_z,
                           j_corners, w_corners, x_t, wealth_grid):
    """Sum-reduce ``w_corners[c] * interp_at_corner_c`` for c in 0..7."""
    def per_corner(j, w):
        return w * _interp_policy_zw_at_corner(arr4d, t_idx, iz_lo, frac_z,
                                                j, x_t, wealth_grid)
    vals = vmap(per_corner)(j_corners, w_corners)
    return jnp.sum(vals)


# =============================================================================
# Hot-path JAX kernel
# =============================================================================

def _build_simulate_kernel(
    C_mat, S_mat, B_mat,
    wealth_grid, z_grid, dz, z_lo,
    axis_grids, axis_sizes, corner_offsets, strides,
    state_bracket_shift, state_bracket_L_inv, L_ss,
    Phi_0_state, Phi_11, const_r, A_r, M_matrix,
    ret_nodes, ret_weights, ret_factor,
    pension_at_z_grid, log_det_next_per_age,
    survival_per_age,                # (n_age, n_z)
    is_working_per_age,              # (n_age,) bool
    is_pre_retire_boundary_per_age,  # (n_age,) bool
    rho, pz, mu_eta1, sigma_eta1, sigma_eta2, mu_eta2_eff,
    pe, mu_eps1, sigma_eps1, sigma_eps2, mu_eps2_eff,
    n_age, n_z, n_ret, n_state,
    rtb_idx, xr_pos, xb_pos,
    use_mc_returns,
):
    """Return a jit'd kernel that simulates one batch of households.

    All arrays close over the kernel scope so the per-household scan body
    can index them by ``t_idx`` directly. ``use_mc_returns`` and ``n_age``
    are baked in at trace time. Generic over n_state via the static
    ``corner_offsets`` table and ``strides`` array.

    Post rtb-as-state migration: realised ``log_R_bill`` ( = rtb_{t+1}) is
    read from the next-period state vector at ``rtb_idx``; the return block
    carries (xr, xb) only.
    """
    n_corners = 1 << int(n_state)

    def step_fn(carry, draws_t):
        z_val, s_t, x_t, alive, income_t, t_idx = carry
        u = draws_t["u"]   # (4,)
        n = draws_t["n"]   # (n_normal_cols,)

        # ----- shared per-period lookups (closures + dynamic index) -----
        psi_t = survival_per_age[t_idx]                         # (n_z,)
        log_det_next_t = log_det_next_per_age[t_idx]            # scalar
        is_working_t = is_working_per_age[t_idx]                # bool
        is_pre_retire_boundary_t = is_pre_retire_boundary_per_age[t_idx]

        # ----- z bracket for policy lookup + survival linear-interp -----
        iz_lo, frac_z = _bracket_uniform(z_val, z_lo, dz, n_z)
        z_idx_near = jnp.where(frac_z <= 0.5, iz_lo, iz_lo + 1)

        # ----- state bracket on transformed coords (Cholesky-decorrelated) -----
        b_vec = state_bracket_L_inv @ (s_t - state_bracket_shift)
        lo_list = []
        frac_list = []
        for d in range(n_state):
            lo_d, f_d = _bracket_axis(axis_grids[d], b_vec[d])
            lo_list.append(lo_d)
            frac_list.append(f_d)
        lo = jnp.stack(lo_list)                              # (n_state,)
        frac = jnp.stack(frac_list)                          # (n_state,)

        # 2^n_state multilinear corners, flat indexing matching solver layout
        per_axis = jnp.where(corner_offsets > 0, frac[None, :], 1.0 - frac[None, :])
        w_corners = jnp.prod(per_axis, axis=1)               # (2^n_state,)
        idx_per_axis = lo[None, :] + corner_offsets          # (2^n_state, n_state)
        j_corners = jnp.sum(idx_per_axis * strides[None, :], axis=1).astype(jnp.int32)

        # Diagnostic state index: nearest-corner flatten — kept for the
        # backwards-compatible panel layout.
        nearest_offsets = (frac > 0.5).astype(corner_offsets.dtype)
        state_idx_near = jnp.sum((lo + nearest_offsets) * strides).astype(jnp.int32)

        # ----- Policy lookup: multilinear in s × bilinear in (z, x) -----
        c_t = _trilinear_z_w_lookup(C_mat, t_idx, iz_lo, frac_z,
                                     j_corners, w_corners, x_t, wealth_grid)
        a_s_t = _trilinear_z_w_lookup(S_mat, t_idx, iz_lo, frac_z,
                                       j_corners, w_corners, x_t, wealth_grid)
        a_b_t = _trilinear_z_w_lookup(B_mat, t_idx, iz_lo, frac_z,
                                       j_corners, w_corners, x_t, wealth_grid)

        # Above wealth_max: flat extrap forces c -> wealth_grid[-1] level,
        # which sends c/x -> 0 for high-wealth agents. Rescale so c/x is
        # preserved at the upper edge — matches CRRA homotheticity.
        wealth_max_grid = wealth_grid[-1]
        c_t = jnp.where(x_t > wealth_max_grid, c_t * (x_t / wealth_max_grid), c_t)

        # Bracket-clamp on consumption: 0 <= c <= max(x, 0).
        c_t = jnp.maximum(c_t, 0.0)
        c_t = jnp.minimum(c_t, jnp.maximum(x_t, 0.0))
        savings_t = x_t - c_t

        # ----- State innovation v^s = L_ss @ standard_normals -----
        z_innov = lax.dynamic_slice(n, (n_ret + 2,), (n_state,))
        v_s = L_ss @ z_innov

        # ----- Next-period state vector (carries the realised rtb) -----
        s_next = Phi_0_state + Phi_11 @ s_t + v_s              # (n_state,)
        log_R_bill = s_next[rtb_idx]                           # rtb_{t+1}

        # ----- Conditional return mean for (xr, xb) given (s_t, v^s) -----
        base_mu_r = const_r + A_r @ s_t                        # (n_ret,)
        mu_r = base_mu_r + M_matrix @ v_s                      # (n_ret,)

        if use_mc_returns:
            # Continuous return residuals: factor @ standard_normals.
            ret_resid = ret_factor @ lax.dynamic_slice(n, (0,), (n_ret,))
            log_x_s = mu_r[xr_pos] + ret_resid[xr_pos]
            log_x_b = mu_r[xb_pos] + ret_resid[xb_pos]
        else:
            ret_idx = _draw_discrete(ret_weights, u[3])
            log_x_s = mu_r[xr_pos] + ret_nodes[ret_idx, xr_pos]
            log_x_b = mu_r[xb_pos] + ret_nodes[ret_idx, xb_pos]

        R_bill = jnp.exp(log_R_bill)
        R_stock = R_bill * jnp.exp(log_x_s)
        R_bond = R_bill * jnp.exp(log_x_b)

        a_bill_t = 1.0 - a_s_t - a_b_t
        R_port = a_s_t * R_stock + a_b_t * R_bond + a_bill_t * R_bill
        # No clamp on estate — the simulator reflects the truth, including
        # rare catastrophic realisations under uncapped leverage. The
        # offgrid diagnostic surfaces these post-hoc.
        estate_t = savings_t * R_port

        # ----- Survival (linear z-interp on the policy's iz_lo / frac_z) -----
        psi_eff = (1.0 - frac_z) * psi_t[iz_lo] + frac_z * psi_t[iz_lo + 1]
        survives = u[0] <= psi_eff
        alive_next = jnp.logical_and(alive, survives)

        # ----- z + income transition -----
        # Always compute both branches; jnp.where selects.
        std_eta = n[n_ret]
        eta = jnp.where(
            u[1] < pz,
            mu_eta1 + sigma_eta1 * std_eta,
            mu_eta2_eff + sigma_eta2 * std_eta,
        )
        z_next_working = rho * z_val + eta
        z_next = jnp.where(is_working_t, z_next_working, z_val)

        # Working income at z_next
        std_eps = n[n_ret + 1]
        eps_val = jnp.where(
            u[2] < pe,
            mu_eps1 + sigma_eps1 * std_eps,
            mu_eps2_eff + sigma_eps2 * std_eps,
        )
        y_gross = jnp.exp(log_det_next_t + z_next + eps_val)
        income_working = jax_disposable_income(y_gross)

        # Pension at z_next (work->retirement boundary)
        iz_lo_next, frac_z_next = _bracket_uniform(z_next, z_lo, dz, n_z)
        pension_at_znext = (
            (1.0 - frac_z_next) * pension_at_z_grid[iz_lo_next]
            + frac_z_next * pension_at_z_grid[iz_lo_next + 1]
        )
        # Pension at current z (retired branch — z frozen, use the same
        # iz_lo/frac_z that policy lookup used).
        pension_at_z = (
            (1.0 - frac_z) * pension_at_z_grid[iz_lo]
            + frac_z * pension_at_z_grid[iz_lo + 1]
        )

        income_next = jnp.where(
            is_working_t,
            jnp.where(is_pre_retire_boundary_t, pension_at_znext, income_working),
            pension_at_z,
        )
        x_next = estate_t + income_next

        # ----- Per-period output (panel slice; gated post-scan via alive) -----
        out = {
            "z": z_val,
            "x": x_t,
            "c": c_t,
            "savings": savings_t,
            "alpha_s": a_s_t,
            "alpha_b": a_b_t,
            "R_port": R_port,
            "income": income_t,
            "estate": estate_t,
            "alive": alive,
            "z_idx": z_idx_near,
            "state_idx": state_idx_near,
            "state_coords": s_t,
        }

        # Carry forward — s_next was already computed above (rtb-as-state)
        carry_next = (z_next, s_next, x_next, alive_next, income_next, t_idx + 1)
        return carry_next, out

    @jit
    def per_household_scan(init_carry, hh_draws):
        # hh_draws is a pytree with leading axis n_age.
        return lax.scan(step_fn, init_carry, hh_draws)

    return per_household_scan


# =============================================================================
# Public entrypoint
# =============================================================================

def _validate_policy_shapes(C_mat, S_mat, B_mat, pc):
    expected = (pc.n_age, pc.n_z, pc.N_state, pc.n_w)
    for name, arr in (("C_mat", C_mat), ("S_mat", S_mat), ("B_mat", B_mat)):
        if arr.shape != expected:
            raise ValueError(f"{name} has shape {arr.shape}, expected {expected}.")


def _wealth_offgrid_diagnostics(sim_x, sim_alive, wealth_grid):
    """Per-age fraction of alive households whose cash-on-hand is outside the grid."""
    wlo = float(wealth_grid[0])
    whi = float(wealth_grid[-1])
    n_age = sim_x.shape[1]

    n_alive = sim_alive.sum(axis=0).astype(float)
    safe = np.maximum(n_alive, 1.0)
    below_count = ((sim_x < wlo) & sim_alive).sum(axis=0)
    above_count = ((sim_x > whi) & sim_alive).sum(axis=0)
    negative_count = ((sim_x < 0.0) & sim_alive).sum(axis=0)
    below_frac = below_count / safe
    above_frac = above_count / safe
    negative_frac = negative_count / safe
    off_frac = below_frac + above_frac

    no_alive = n_alive == 0
    below_frac = np.where(no_alive, np.nan, below_frac)
    above_frac = np.where(no_alive, np.nan, above_frac)
    off_frac = np.where(no_alive, np.nan, off_frac)
    negative_frac = np.where(no_alive, np.nan, negative_frac)

    return {
        "wealth_min": wlo,
        "wealth_max": whi,
        "below_frac": below_frac,
        "above_frac": above_frac,
        "off_frac": off_frac,
        "negative_frac": negative_frac,                  # NEW: x_t < 0 cells
        "max_off_frac": float(np.nanmax(off_frac)) if not np.all(no_alive) else float("nan"),
        "max_off_age_offset": int(np.nanargmax(off_frac)) if not np.all(no_alive) else -1,
        "max_negative_frac": float(np.nanmax(negative_frac)) if not np.all(no_alive) else float("nan"),
        "n_age": int(n_age),
    }


def simulate_lifecycle(
    C_mat, S_mat, B_mat, pc, model,
    n_simulations=10_000,
    initial_x=None,
    initial_wealth=0.1,
    initial_wealth_distribution=None,
    initial_wealth_normal_std=0.0,
    initial_z="stationary",
    initial_z_normal_std=0.652,
    initial_state="median",
    seed=42,
    return_draw_mode="monte_carlo",
    wealth_offgrid_warn_threshold=0.05,
    wealth_dynamics_spec="ccv_log",   # accepted for compat; simulator always uses arithmetic returns
    verbose=True,
):
    """Simulate lifecycle paths using the JAX kernel.

    Parameters
    ----------
    C_mat, S_mat, B_mat : (n_age, n_z, N_state, n_w) np.ndarray
    pc, model : Precompute / LifecyclePortfolioModel
    n_simulations : int
    initial_x : float | array | None
    initial_wealth : float | array | None
    initial_wealth_distribution : None or "normal"
    initial_wealth_normal_std : float
    initial_z : "median" | "stationary" | "normal" | array of indices
    initial_z_normal_std : float
    initial_state : "median" | "stationary" | array of indices
    seed : int
    return_draw_mode : "monte_carlo" | "quadrature"
    wealth_offgrid_warn_threshold : float (0–1)
    wealth_dynamics_spec : kept for signature compat — simulator uses
        arithmetic ``R_p = a_s*R_stock + a_b*R_bond + a_bill*R_bill``.
    verbose : bool

    Returns
    -------
    dict with keys: x, c, savings, alpha_s, alpha_b, alpha_bill, R_port,
    income, estate, estate_at_death, z, z_idx, state_idx, state_coords,
    alive, death_age, ages, wealth_offgrid.
    """
    _validate_policy_shapes(C_mat, S_mat, B_mat, pc)
    if return_draw_mode not in ("monte_carlo", "quadrature"):
        raise ValueError(
            f"return_draw_mode must be 'monte_carlo' or 'quadrature', got '{return_draw_mode}'."
        )
    n_state_int = int(model.n_state)
    if n_state_int < 1 or n_state_int > 4:
        raise NotImplementedError(
            f"JAX simulator supports n_state in {{1, 2, 3, 4}} (got {n_state_int})."
        )

    retire_age_idx = model.retire_age - model.start_age
    n_age = pc.n_age
    n_ret = model.n_ret

    if verbose:
        print("=" * 66)
        print("SIMULATING LIFECYCLE PATHS (JAX)")
        print("=" * 66)
        print(f"  Households:         {n_simulations:,}")
        print(f"  Ages:               {model.start_age} to {model.terminal_age} ({n_age} periods)")
        print(f"  Retirement age:     {model.retire_age} (index {retire_age_idx})")
        print(f"  Devices:            {len(jax.devices())}")
        print(f"  Return draw mode:   {return_draw_mode}")
        if return_draw_mode == "monte_carlo":
            diag = np.sqrt(np.clip(np.diag(model.Sigma_r_cond), 0.0, None))
            label = ",".join(model.ret_names)
            diag_str = ", ".join(f"{v:.4f}" for v in diag)
            print(f"  sigma_resid ({label}): {diag_str}")

    rng = np.random.default_rng(seed)

    mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1
    mu_eps2_eff = -(model.pe / (1.0 - model.pe)) * model.mu_eps1

    # --- Initialise z (continuous) ---
    if isinstance(initial_z, str) and initial_z == "normal":
        init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=initial_z_normal_std)
        init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
    elif isinstance(initial_z, str) and initial_z == "stationary":
        var_eta = (
            model.pz * (model.sigma_eta1 ** 2 + model.mu_eta1 ** 2)
            + (1 - model.pz) * (model.sigma_eta2 ** 2 + mu_eta2_eff ** 2)
        )
        sigma_z = np.sqrt(var_eta / (1.0 - model.rho ** 2))
        init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=sigma_z)
        init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
    else:
        init_z_idx = initialize_states(n_simulations, pc.n_z, pc.Pi_z, initial_z, rng)
    init_z_val = pc.z_grid[init_z_idx].astype(np.float64)

    init_state_idx = _resolve_initial_state_indices(n_simulations, pc, initial_state, rng)

    # --- Initial income (host NumPy) ---
    if retire_age_idx > 0:
        init_eps_uniform = rng.uniform(size=n_simulations)
        init_eps_normal = rng.standard_normal(size=n_simulations)
        init_eps_val = np.where(
            init_eps_uniform < model.pe,
            model.mu_eps1 + model.sigma_eps1 * init_eps_normal,
            mu_eps2_eff + model.sigma_eps2 * init_eps_normal,
        )
        init_y_gross = np.exp(pc.log_det_profile[0] + init_z_val + init_eps_val)
        initial_income_arr = disposable_income_working(init_y_gross)
    else:
        initial_income_arr = compute_pension_after_tax(init_z_val, pc.avg_det)

    # --- Initial wealth / cash-on-hand ---
    if initial_x is None:
        init_wealth_arr = _initialize_initial_wealth(
            n_simulations, initial_wealth, initial_wealth_distribution,
            initial_wealth_normal_std, rng,
        )
        init_x = init_wealth_arr + initial_income_arr
    elif np.isscalar(initial_x):
        init_x = np.full(n_simulations, float(initial_x), dtype=float)
    else:
        init_x = np.asarray(initial_x, dtype=float)
        if init_x.shape[0] != n_simulations:
            raise ValueError(
                f"initial_x array has length {init_x.shape[0]}, expected {n_simulations}."
            )
    if np.any(init_x < 0.0):
        raise ValueError(
            "Constructed initial_x must be non-negative. Check the initial wealth "
            "specification and the period-0 income draw."
        )

    if verbose:
        print(f"  Initial x mean: {np.mean(init_x):.3f}  p25={np.percentile(init_x,25):.3f}  p75={np.percentile(init_x,75):.3f}")

    # --- Pre-generate RNG draws (NumPy, shipped to device) ---
    # n_normal_cols = n_ret (return residuals) + 2 (eta, eps standard normals)
    #                 + n_state (state-innovation standard normals).
    uniform_draws = rng.uniform(size=(n_simulations, n_age, 4))
    n_normal_cols = n_ret + 2 + n_state_int
    normal_draws = rng.standard_normal(size=(n_simulations, n_age, n_normal_cols))

    if return_draw_mode == "monte_carlo":
        ret_factor_arr = _build_return_factor(model.Sigma_r_cond)
        use_mc_returns = True
    else:
        ret_factor_arr = np.zeros((n_ret, n_ret), dtype=float)
        use_mc_returns = False

    # --- Per-period shared arrays ---
    is_working = np.arange(n_age) < retire_age_idx
    is_pre_retire_boundary = (np.arange(n_age) == retire_age_idx - 1) & is_working
    log_det_next_per_age = np.zeros(n_age, dtype=np.float64)
    log_det_next_per_age[:-1] = pc.log_det_profile[1:]   # last entry unused

    # The retirement pension table: pension_after_tax is constant across
    # retirement ages, so any retired-age slice is fine.
    pension_at_z_grid = np.asarray(pc.pension_after_tax[retire_age_idx, :])

    # --- Pack arrays as jnp ---
    init_s_coords = np.ascontiguousarray(pc.state_grid[init_state_idx, :])
    L_ss = np.linalg.cholesky(0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T))

    j = jnp.asarray
    # Build the n_state-generic axis and corner arrays. Mirrors solver._pc_to_jnp.
    axis_grids = tuple(j(np.asarray(g)) for g in pc.state_bracket_grids)
    axis_sizes = tuple(int(g.shape[0]) for g in pc.state_bracket_grids)
    n_corners = 1 << n_state_int
    corner_offsets_np = np.zeros((n_corners, n_state_int), dtype=np.int32)
    for c in range(n_corners):
        for d in range(n_state_int):
            corner_offsets_np[c, d] = (c >> (n_state_int - 1 - d)) & 1
    strides_np = np.empty(n_state_int, dtype=np.int32)
    s = 1
    for d in range(n_state_int - 1, -1, -1):
        strides_np[d] = s
        s *= axis_sizes[d]

    rtb_idx = int(model.rtb_index_in_state)
    xr_pos = int(model.ret_names.index("xr"))
    xb_pos = int(model.ret_names.index("xb"))

    kernel = _build_simulate_kernel(
        C_mat=j(C_mat), S_mat=j(S_mat), B_mat=j(B_mat),
        wealth_grid=j(pc.wealth_grid),
        z_grid=j(pc.z_grid), dz=jnp.float64(pc.dz),
        z_lo=jnp.float64(pc.z_grid[0]),
        axis_grids=axis_grids,
        axis_sizes=axis_sizes,
        corner_offsets=j(corner_offsets_np),
        strides=j(strides_np),
        state_bracket_shift=j(pc.state_bracket_shift),
        state_bracket_L_inv=j(pc.state_bracket_L_inv),
        L_ss=j(L_ss),
        Phi_0_state=j(np.asarray(model.Phi_0_state, dtype=np.float64)),
        Phi_11=j(np.asarray(model.Phi_11, dtype=np.float64)),
        const_r=j(pc.const_r), A_r=j(pc.A_r),
        M_matrix=j(np.asarray(model.M, dtype=np.float64)),
        ret_nodes=j(pc.ret_nodes), ret_weights=j(pc.ret_weights),
        ret_factor=j(ret_factor_arr),
        pension_at_z_grid=j(pension_at_z_grid),
        log_det_next_per_age=j(log_det_next_per_age),
        survival_per_age=j(np.asarray(pc.survival_probs_2d)),
        is_working_per_age=j(is_working),
        is_pre_retire_boundary_per_age=j(is_pre_retire_boundary),
        rho=jnp.float64(model.rho),
        pz=jnp.float64(model.pz),
        mu_eta1=jnp.float64(model.mu_eta1),
        sigma_eta1=jnp.float64(model.sigma_eta1),
        sigma_eta2=jnp.float64(model.sigma_eta2),
        mu_eta2_eff=jnp.float64(mu_eta2_eff),
        pe=jnp.float64(model.pe),
        mu_eps1=jnp.float64(model.mu_eps1),
        sigma_eps1=jnp.float64(model.sigma_eps1),
        sigma_eps2=jnp.float64(model.sigma_eps2),
        mu_eps2_eff=jnp.float64(mu_eps2_eff),
        n_age=int(n_age), n_z=int(pc.n_z), n_ret=int(n_ret),
        n_state=n_state_int,
        rtb_idx=rtb_idx, xr_pos=xr_pos, xb_pos=xb_pos,
        use_mc_returns=use_mc_returns,
    )

    # --- Build per-household pytree of draws + initial carry ---
    init_carry_each = (
        j(init_z_val),
        j(init_s_coords),
        j(init_x),
        jnp.ones(n_simulations, dtype=bool),
        j(initial_income_arr),
        jnp.zeros(n_simulations, dtype=jnp.int32),
    )
    hh_draws_pytree = {
        "u": j(uniform_draws),    # (n_sim, n_age, 4)
        "n": j(normal_draws),      # (n_sim, n_age, n_normal_cols)
    }

    if verbose:
        print("\n  Running JAX simulation kernel...", flush=True)
    t_start = time.perf_counter()

    # vmap over households: each household runs its own scan.
    # Per-household carry has axis 0 = n_sim; per-household draws have axis 0 = n_sim.
    final_carry, panel = vmap(kernel, in_axes=(0, 0))(init_carry_each, hh_draws_pytree)
    # Block until ready so timing is real wall time.
    panel["x"].block_until_ready()
    elapsed = time.perf_counter() - t_start

    # --- Convert outputs to NumPy ---
    sim_x = np.asarray(panel["x"])
    sim_c = np.asarray(panel["c"])
    sim_savings = np.asarray(panel["savings"])
    sim_alpha_s = np.asarray(panel["alpha_s"])
    sim_alpha_b = np.asarray(panel["alpha_b"])
    sim_R_port = np.asarray(panel["R_port"])
    sim_income = np.asarray(panel["income"])
    sim_estate = np.asarray(panel["estate"])
    sim_z = np.asarray(panel["z"])
    sim_z_idx = np.asarray(panel["z_idx"])
    sim_state = np.asarray(panel["state_idx"])
    sim_state_coords = np.asarray(panel["state_coords"])
    sim_alive = np.asarray(panel["alive"])

    sim_alpha_bill = 1.0 - sim_alpha_s - sim_alpha_b

    # --- Death age + estate-at-death (NumPy post-hoc) ---
    # death_age[i]: age at which alive becomes False, or terminal_age if alive throughout.
    # The carried alive mask: alive at slot t is the survival status as of the
    # START of period t. So the FIRST t where alive[i,t] is False marks death
    # (the household's "last alive period" is t-1).
    n_sim = sim_x.shape[0]
    death_age = np.full(n_sim, model.terminal_age, dtype=np.int32)
    estate_at_death = np.zeros(n_sim)
    for i in range(n_sim):
        first_dead = -1
        for t in range(n_age):
            if not sim_alive[i, t]:
                first_dead = t
                break
        if first_dead == -1:
            # Alive at terminal age — bequest at terminal estate.
            death_age[i] = model.terminal_age
            estate_at_death[i] = sim_estate[i, n_age - 1]
        else:
            # Died at age start_age + first_dead; estate is the previous period's.
            death_age[i] = model.start_age + first_dead - 1
            estate_at_death[i] = sim_estate[i, first_dead - 1] if first_dead > 0 else 0.0

    # --- Off-grid diagnostic ---
    offgrid = _wealth_offgrid_diagnostics(sim_x, sim_alive, pc.wealth_grid)
    if offgrid["max_off_frac"] > wealth_offgrid_warn_threshold:
        peak_age = model.start_age + offgrid["max_off_age_offset"]
        warnings.warn(
            f"simulate_lifecycle: {offgrid['max_off_frac']:.1%} of alive households "
            f"are outside the solved wealth grid [{offgrid['wealth_min']:.2e}, "
            f"{offgrid['wealth_max']:.2f}] at age {peak_age} (threshold "
            f"{wealth_offgrid_warn_threshold:.1%}). Off-grid households fall on "
            "the boundary policy via flat extrapolation.",
            stacklevel=2,
        )
    if offgrid["max_negative_frac"] > 0.0:
        warnings.warn(
            f"simulate_lifecycle: {offgrid['max_negative_frac']:.2%} of alive "
            "households have x_t < 0 at some age — uncapped leverage produced "
            "catastrophic portfolio realisations. The simulation continues these "
            "households on negative wealth (c_t = 0). Consider tightening the "
            "leverage cap or widening wealth_min.",
            stacklevel=2,
        )

    if verbose:
        _print_simulation_summary(
            model=model, pc=pc,
            sim_x=sim_x, sim_c=sim_c,
            sim_alpha_s=sim_alpha_s, sim_alpha_b=sim_alpha_b,
            sim_R_port=sim_R_port, sim_alive=sim_alive,
            death_age=death_age, estate_at_death=estate_at_death,
            retire_age_idx=retire_age_idx, elapsed=elapsed,
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
        "state_coords": sim_state_coords,
        "alive": sim_alive,
        "death_age": death_age,
        "ages": np.asarray(pc.ages).copy(),
        "wealth_offgrid": offgrid,
    }


# =============================================================================
# Print summary (NumPy, untouched)
# =============================================================================

def _print_simulation_summary(model, pc, sim_x, sim_c, sim_alpha_s, sim_alpha_b,
                               sim_R_port, sim_alive, death_age, estate_at_death,
                               retire_age_idx, elapsed):
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
