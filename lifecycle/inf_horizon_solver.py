"""inf_horizon_solver.py — Standalone infinite-horizon benchmark.

Built directly on top of solver._build_per_age_retirement_kernel: the
retirement step is the infinite-horizon Bellman operator (no mortality, no
pension, no labour income). We iterate the JAX retirement kernel to a
stationary fixed point in a Python while loop, with optional damping and a
state-dependent Markowitz cold start.

Public API:
  run_infinite_horizon_solver(model, pc, ...) -> (C, S, B, diagnostics)
  compile_inner_kernel_smoke_test(model, pc, ...) -> dict
  extract_policy_at_point(C, S, B, i_z, i_s, i_w) -> dict
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lifecycle.model import DELTA_BEQUEST, SolverConfig
from lifecycle.solver import (
    ModelParams,
    _build_per_age_retirement_kernel,
    _pc_to_jnp,
    _precompute_per_is_tensors,
)


DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 500
DEFAULT_DAMPING = 1.0
DEFAULT_TRIM_WEALTH_POINTS = 5
COLD_START_COV_RIDGE = 1e-10


# =============================================================================
# Progress printing
# =============================================================================

def _print_progress_line(
    iter_idx: int,
    xi_err: float,
    share_err: float,
    stop_err: float,
    show_probe: bool,
    probe_w: float,
    probe_c_over_w: float,
    probe_alpha_s: float,
    probe_alpha_b: float,
    probe_alpha_bill: float,
) -> None:
    """Render one compact progress line in-place for long notebook/script runs.

    The Newton-failure column from the Numba era is gone — the JAX kernel does
    not expose per-cell exit codes. The diagnostics dict still reports
    ``total_newton_failures = 0`` for downstream API compatibility.
    """
    line = (
        f"\rih iter {iter_idx:4d} | xi {xi_err:.2e} | share {share_err:.2e} | stop {stop_err:.2e}"
    )
    if show_probe:
        line += (
            f" | W {probe_w:.2f} | c/W {probe_c_over_w:.2e}"
            f" | s {probe_alpha_s:.3f} b {probe_alpha_b:.3f} bill {probe_alpha_bill:.3f}"
        )
    print(line, end="", flush=True)


# =============================================================================
# Initial-policy preparation (NumPy, untouched from pre-rewrite)
# =============================================================================

def _coerce_policy_array(
    name: str,
    arr: np.ndarray | None,
    expected_shape: tuple[int, int, int],
    pc,
    retire_age: int,
) -> np.ndarray | None:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 4:
        # Lifecycle policy: take the slice corresponding to model.retire_age.
        ages = np.asarray(pc.ages)
        idx = np.flatnonzero(ages == int(retire_age))
        if idx.size == 0:
            raise ValueError(f"{name}: retire_age {retire_age} not in pc.ages")
        out = out[int(idx[0])]
    if out.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape} or lifecycle shape "
            f"(n_age, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), got {out.shape}"
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(out.copy())


def _project_simplex_nonnegative(alpha_raw: np.ndarray) -> np.ndarray:
    alpha_clipped = np.maximum(alpha_raw, 0.0)
    total = float(np.sum(alpha_clipped))
    if total > 0.0:
        return alpha_clipped / total
    return np.full(3, 1.0 / 3.0, dtype=np.float64)


def _markowitz_cold_start(model, pc):
    """State-dependent Markowitz cold start (NumPy).

    Computes a no-risk-free-asset myopic portfolio at each financial state
    using the same return quadrature as the solver kernel. Pairs this with a
    log-utility-limit consumption rule ``xi = 1 - beta``.
    """
    n_z = pc.n_z
    N_state = pc.N_state
    n_w = pc.n_w
    gamma = float(model.gamma)
    beta = float(model.beta)

    ones3 = np.ones(3, dtype=np.float64)
    identity3 = np.eye(3, dtype=np.float64)

    alpha_s_per_state = np.empty(N_state, dtype=np.float64)
    alpha_b_per_state = np.empty(N_state, dtype=np.float64)

    raw_weights = np.outer(
        np.asarray(pc.v_weights, dtype=np.float64),
        np.asarray(pc.ret_weights, dtype=np.float64),
    )
    weight_vec = raw_weights.reshape(-1)
    weight_sum = float(np.sum(weight_vec))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Quadrature weights must sum to a positive finite number")
    weight_vec = weight_vec / weight_sum

    rtb_idx = int(model.rtb_index_in_state)
    xr_pos = int(model.ret_names.index("xr"))
    xb_pos = int(model.ret_names.index("xb"))
    Phi_0_state = np.asarray(model.Phi_0_state, dtype=np.float64)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
    v_nodes = np.asarray(pc.v_nodes, dtype=np.float64)

    for i_s in range(N_state):
        base_mu_r = pc.const_r + pc.A_r @ pc.state_grid[i_s]
        # rtb realisation per state-quadrature node lives in s_next:
        s_next_kv = Phi_0_state[None, :] + pc.state_grid[i_s] @ Phi_11.T + v_nodes

        returns = np.empty((weight_vec.size, 3), dtype=np.float64)
        row = 0
        for k_v in range(len(pc.v_weights)):
            mu_stock = base_mu_r[xr_pos] + pc.M_v_nodes[k_v, xr_pos]
            mu_bond = base_mu_r[xb_pos] + pc.M_v_nodes[k_v, xb_pos]
            log_R_bill_kv = float(s_next_kv[k_v, rtb_idx])
            R_bill = np.exp(log_R_bill_kv)
            exp_mu_stock = np.exp(mu_stock)
            exp_mu_bond = np.exp(mu_bond)
            for k_r in range(len(pc.ret_weights)):
                R_stock = R_bill * exp_mu_stock * pc.exp_ret_stock[k_r]
                R_bond = R_bill * exp_mu_bond * pc.exp_ret_bond[k_r]
                returns[row, 0] = R_bill
                returns[row, 1] = R_stock
                returns[row, 2] = R_bond
                row += 1

        Rbar = np.sum(returns * weight_vec[:, None], axis=0)
        centered = returns - Rbar[None, :]
        Sigma = centered.T @ (centered * weight_vec[:, None])
        Sigma_reg = Sigma + COLD_START_COV_RIDGE * identity3

        try:
            x = np.linalg.solve(Sigma_reg, np.column_stack((Rbar, ones3)))
            x_R = x[:, 0]
            x_1 = x[:, 1]
            denom = float(ones3 @ x_1)
            if abs(denom) <= 1e-14:
                raise np.linalg.LinAlgError("Degenerate Markowitz denominator")
            lam = (1.0 - (1.0 / gamma) * float(ones3 @ x_R)) / denom
            alpha_raw = (1.0 / gamma) * x_R + lam * x_1
            if not np.all(np.isfinite(alpha_raw)):
                raise np.linalg.LinAlgError("Non-finite Markowitz solution")
        except np.linalg.LinAlgError:
            alpha_raw = np.full(3, 1.0 / 3.0, dtype=np.float64)

        alpha_proj = _project_simplex_nonnegative(np.asarray(alpha_raw, dtype=np.float64))
        alpha_s_per_state[i_s] = alpha_proj[1]
        alpha_b_per_state[i_s] = alpha_proj[2]

    xi_init = 1.0 - beta
    expected_shape = (n_z, N_state, n_w)
    C_init = np.broadcast_to(
        (xi_init * pc.wealth_grid).reshape(1, 1, -1), expected_shape,
    ).astype(np.float64, copy=True)
    S_init = np.broadcast_to(
        alpha_s_per_state.reshape(1, N_state, 1), expected_shape,
    ).astype(np.float64, copy=True)
    B_init = np.broadcast_to(
        alpha_b_per_state.reshape(1, N_state, 1), expected_shape,
    ).astype(np.float64, copy=True)
    return (
        np.ascontiguousarray(C_init),
        np.ascontiguousarray(S_init),
        np.ascontiguousarray(B_init),
    )


def _prepare_initial_policies(
    model, pc, retire_age, solver_config,
    warm_start_c, warm_start_s, warm_start_b,
):
    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    C_old = _coerce_policy_array("warm_start_c", warm_start_c, expected_shape, pc, retire_age)
    S_old = _coerce_policy_array("warm_start_s", warm_start_s, expected_shape, pc, retire_age)
    B_old = _coerce_policy_array("warm_start_b", warm_start_b, expected_shape, pc, retire_age)

    all_cold = (warm_start_c is None) and (warm_start_s is None) and (warm_start_b is None)
    if all_cold:
        return _markowitz_cold_start(model, pc)

    if C_old is None:
        C_old = np.broadcast_to(
            pc.wealth_grid.reshape(1, 1, -1), expected_shape,
        ).astype(np.float64, copy=True)
    if S_old is None:
        S_old = np.full(expected_shape, float(solver_config.init_alpha_s), dtype=np.float64)
    if B_old is None:
        B_old = np.full(expected_shape, float(solver_config.init_alpha_b), dtype=np.float64)
    return (
        np.ascontiguousarray(C_old),
        np.ascontiguousarray(S_old),
        np.ascontiguousarray(B_old),
    )


# =============================================================================
# Validation + progress probe resolution
# =============================================================================

def _validate_runtime_options(pc, tol, max_iter, damping, trim_wealth_points):
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1]")
    if trim_wealth_points < 0:
        raise ValueError("trim_wealth_points must be non-negative")
    if trim_wealth_points >= pc.n_w:
        raise ValueError("trim_wealth_points must be strictly smaller than pc.n_w")


def _resolve_progress_probe_indices(
    pc, progress_probe_wealth, progress_probe_state_idx, progress_probe_z_idx,
):
    if progress_probe_wealth is None:
        return False, 0, 0, 0
    i_z = pc.n_z // 2 if progress_probe_z_idx is None else int(progress_probe_z_idx)
    i_s = pc.N_state // 2 if progress_probe_state_idx is None else int(progress_probe_state_idx)
    if i_z < 0 or i_z >= pc.n_z:
        raise ValueError(f"progress_probe_z_idx must lie in [0, {pc.n_z - 1}]")
    if i_s < 0 or i_s >= pc.N_state:
        raise ValueError(f"progress_probe_state_idx must lie in [0, {pc.N_state - 1}]")
    target_w = float(progress_probe_wealth)
    i_w = int(np.argmin(np.abs(pc.wealth_grid - target_w)))
    return True, i_z, i_s, i_w


# =============================================================================
# Post-hoc diagnostics (NumPy, untouched)
# =============================================================================

def _compute_z_invariance(C, S, B):
    return {
        "max_z_slice_diff_c": float(np.max(np.abs(C - C[0:1]))),
        "max_z_slice_diff_s": float(np.max(np.abs(S - S[0:1]))),
        "max_z_slice_diff_b": float(np.max(np.abs(B - B[0:1]))),
    }


def _compute_wealth_homogeneity(C, S, B, wealth_grid, trim_wealth_points):
    w = np.asarray(wealth_grid[trim_wealth_points:], dtype=np.float64)
    xi = C[:, :, trim_wealth_points:] / w.reshape(1, 1, -1)
    xi_spread = np.max(np.max(xi, axis=2) - np.min(xi, axis=2))
    stock_spread = np.max(
        np.max(S[:, :, trim_wealth_points:], axis=2)
        - np.min(S[:, :, trim_wealth_points:], axis=2)
    )
    bond_spread = np.max(
        np.max(B[:, :, trim_wealth_points:], axis=2)
        - np.min(B[:, :, trim_wealth_points:], axis=2)
    )
    return {
        "max_xi_spread_across_w": float(xi_spread),
        "max_stock_spread_across_w": float(stock_spread),
        "max_bond_spread_across_w": float(bond_spread),
        "max_share_spread_across_w": float(max(stock_spread, bond_spread)),
    }


def _compute_stability_proxy(model, pc, solver_config, S, B, trim_wealth_points):
    """Bound the contraction-mapping proxy ``beta * E[exp((1-gamma) r_p)]``.

    Computed against the CCV log-return formula since the JAX solver uses
    only that wealth-dynamics specification.
    """
    if pc.n_w == 0:
        return float("nan")
    i_z = min(pc.n_z // 2, pc.n_z - 1)
    i_w = min(max(trim_wealth_points, pc.n_w // 2), pc.n_w - 1)
    max_proxy = 0.0

    sigma2_xr = float(pc.sigma2_xr)
    sigma2_xb = float(pc.sigma2_xb)
    sigma_xrxb = float(pc.sigma_xrxb)

    rtb_idx = int(model.rtb_index_in_state)
    xr_pos = int(model.ret_names.index("xr"))
    xb_pos = int(model.ret_names.index("xb"))
    Phi_0_state = np.asarray(model.Phi_0_state, dtype=np.float64)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
    v_nodes = np.asarray(pc.v_nodes, dtype=np.float64)

    for i_s in range(pc.N_state):
        s_i = pc.state_grid[i_s]
        base_mu_r_i = pc.const_r + pc.A_r @ s_i
        s_next_kv = Phi_0_state[None, :] + s_i @ Phi_11.T + v_nodes
        alpha_s = float(S[i_z, i_s, i_w])
        alpha_b = float(B[i_z, i_s, i_w])

        expected_rpow = 0.0
        for k_v, w_v in enumerate(pc.v_weights):
            mu_r_stock = base_mu_r_i[xr_pos] + pc.M_v_nodes[k_v, xr_pos]
            mu_r_bond = base_mu_r_i[xb_pos] + pc.M_v_nodes[k_v, xb_pos]
            log_R_bill = float(s_next_kv[k_v, rtb_idx])
            for k_r, p_ret in enumerate(pc.ret_weights):
                log_x_s = mu_r_stock + pc.ret_nodes[k_r, xr_pos]
                log_x_b = mu_r_bond + pc.ret_nodes[k_r, xb_pos]
                r_p = (
                    log_R_bill
                    + alpha_s * log_x_s + alpha_b * log_x_b
                    + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
                    - 0.5 * (
                        alpha_s * alpha_s * sigma2_xr
                        + 2.0 * alpha_s * alpha_b * sigma_xrxb
                        + alpha_b * alpha_b * sigma2_xb
                    )
                )
                expected_rpow += float(w_v) * float(p_ret) * np.exp((1.0 - model.gamma) * r_p)

        proxy_i = float(model.beta) * expected_rpow
        if proxy_i > max_proxy:
            max_proxy = proxy_i
    return float(max_proxy)


def _build_diagnostics(
    model, pc, solver_config, C, S, B,
    converged, n_iter_done, total_newton_failures,
    policy_supnorm_history, xi_supnorm_history, share_supnorm_history,
    tol, damping, trim_wealth_points, used_warm_start,
):
    policy_hist = np.asarray(policy_supnorm_history[:n_iter_done], dtype=np.float64).copy()
    xi_hist = np.asarray(xi_supnorm_history[:n_iter_done], dtype=np.float64).copy()
    share_hist = np.asarray(share_supnorm_history[:n_iter_done], dtype=np.float64).copy()

    diagnostics: dict[str, Any] = {
        "converged": bool(converged),
        "n_iter": int(n_iter_done),
        "tol": float(tol),
        "used_damping": bool(damping != 1.0),
        "damping_lambda": float(damping),
        "trim_wealth_points": int(trim_wealth_points),
        "used_warm_start": bool(used_warm_start),
        "policy_supnorm_history": policy_hist,
        "xi_supnorm_history": xi_hist,
        "share_supnorm_history": share_hist,
        "final_policy_supnorm": float(policy_hist[-1]) if n_iter_done else float("nan"),
        "final_xi_supnorm": float(xi_hist[-1]) if n_iter_done else float("nan"),
        "final_share_supnorm": float(share_hist[-1]) if n_iter_done else float("nan"),
        "final_stopping_supnorm": float(max(xi_hist[-1], share_hist[-1])) if n_iter_done else float("nan"),
        "total_newton_failures": int(total_newton_failures),
    }
    diagnostics.update(_compute_z_invariance(C, S, B))
    diagnostics.update(_compute_wealth_homogeneity(C, S, B, pc.wealth_grid, trim_wealth_points))
    diagnostics["stability_proxy"] = _compute_stability_proxy(
        model, pc, solver_config, S, B, trim_wealth_points
    )
    return diagnostics


# =============================================================================
# Iteration metrics
# =============================================================================

def _compute_metrics_numpy(C_old, C_new, S_old, S_new, B_old, B_new,
                            wealth_grid, trim_wealth_points):
    """Sup-norm convergence metrics (NumPy)."""
    w = np.asarray(wealth_grid[trim_wealth_points:], dtype=np.float64)
    xi_old = C_old[:, :, trim_wealth_points:] / w.reshape(1, 1, -1)
    xi_new = C_new[:, :, trim_wealth_points:] / w.reshape(1, 1, -1)
    xi_err = float(np.max(np.abs(xi_new - xi_old)))
    share_err = float(max(np.max(np.abs(S_new - S_old)),
                          np.max(np.abs(B_new - B_old))))
    policy_err = float(max(np.max(np.abs(C_new - C_old)), share_err))
    return policy_err, xi_err, share_err


# =============================================================================
# Public entrypoint
# =============================================================================

def run_infinite_horizon_solver(
    model, pc,
    solver_config: SolverConfig | None = None,
    warm_start_c=None, warm_start_s=None, warm_start_b=None,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    damping: float = DEFAULT_DAMPING,
    trim_wealth_points: int = DEFAULT_TRIM_WEALTH_POINTS,
    constrained: bool | None = None,    # ignored — JAX solver is unconstrained-only
    run_smoke_test: bool = False,
    verbose: bool = True,
    show_progress: bool = False,
    progress_every: int = 1,
    progress_probe_wealth: float | None = None,
    progress_probe_state_idx: int | None = None,
    progress_probe_z_idx: int | None = None,
):
    """Solve the stationary no-income, no-mortality benchmark by fixed-point
    iteration of the JAX retirement kernel.

    With all three ``warm_start_*`` arguments None the solver cold-starts from
    a state-dependent Markowitz no-risk-free portfolio paired with the
    log-utility-limit consumption rule ``xi = 1 - beta``.
    """
    if solver_config is None:
        solver_config = SolverConfig()
    if constrained:
        warnings.warn(
            "constrained=True is ignored — the JAX infinite-horizon solver "
            "is unconstrained-only. Returning unconstrained solution.",
            stacklevel=2,
        )

    _validate_runtime_options(pc, tol, max_iter, damping, trim_wealth_points)
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")
    show_progress_probe, probe_z, probe_s, probe_w = _resolve_progress_probe_indices(
        pc, progress_probe_wealth, progress_probe_state_idx, progress_probe_z_idx,
    )

    if verbose:
        print("Preparing infinite-horizon benchmark solve (JAX)...")

    # ---- Initial policies (host NumPy) ----
    C_old, S_old, B_old = _prepare_initial_policies(
        model, pc, model.retire_age, solver_config,
        warm_start_c, warm_start_s, warm_start_b,
    )
    used_warm_start = any(x is not None for x in (warm_start_c, warm_start_s, warm_start_b))

    # ---- Build the JAX retirement kernel once ----
    delta = solver_config.delta_bequest if solver_config.delta_bequest >= 0 else DELTA_BEQUEST
    pcj = _pc_to_jnp(pc, delta)
    mp = ModelParams(
        gamma=jnp.float64(model.gamma),
        beta=jnp.float64(model.beta),
        b_bar=jnp.float64(0.0),                # no bequest in the benchmark
        delta=jnp.float64(delta),
        rho=jnp.float64(model.rho),
    )
    n_dev = len(jax.devices())
    per_is_tensors = _precompute_per_is_tensors(pcj)
    retirement_kernel = _build_per_age_retirement_kernel(
        pcj, mp, solver_config, n_dev, pc.n_z, pc.N_state, per_is_tensors,
    )

    pension_zero = jnp.zeros(pc.n_z, dtype=jnp.float64)
    psi_one = jnp.ones(pc.n_z, dtype=jnp.float64)

    if run_smoke_test:
        # One warm-up call to JIT-compile the kernel before timing the loop.
        # Seeds the Newton init from the prepared (warm) S_old/B_old, matching
        # the convention used inside the iteration loop below.
        _c, _s, _b, _ni, _nb = retirement_kernel(
            jnp.asarray(C_old), pension_zero, psi_one,
            jnp.asarray(S_old), jnp.asarray(B_old),
        )
        np.asarray(_c)  # block until ready

    # ---- Iteration ----
    policy_supnorm_history = []
    xi_supnorm_history = []
    share_supnorm_history = []
    converged = False
    n_iter_done = 0

    if verbose:
        print(f"Running JAX infinite-horizon fixed-point loop "
              f"(devices={n_dev}, max_iter={max_iter})...")

    t_start = time.time()
    for it in range(max_iter):
        c_old_jnp = jnp.asarray(C_old)
        # Newton init at each cell is gathered from the previous iteration's
        # converged share policy at mid-wealth (the kernel's convention). This
        # mirrors run_lifecycle_solver's use_backward_age_warm_start=True
        # behavior in the time dimension. Cost: 2x extra device upload per
        # iteration; gain: typically 3-8 Newton iters/cell once near the fixed
        # point, vs much higher under cold init.
        s_old_jnp = jnp.asarray(S_old)
        b_old_jnp = jnp.asarray(B_old)
        c_new_jnp, s_new_jnp, b_new_jnp, _ni_jnp, _nb_jnp = retirement_kernel(
            c_old_jnp, pension_zero, psi_one, s_old_jnp, b_old_jnp,
        )
        C_new = np.asarray(c_new_jnp)
        S_new = np.asarray(s_new_jnp)
        B_new = np.asarray(b_new_jnp)

        # Damped update.
        if damping == 1.0:
            C_next, S_next, B_next = C_new, S_new, B_new
        else:
            C_next = damping * C_new + (1.0 - damping) * C_old
            S_next = damping * S_new + (1.0 - damping) * S_old
            B_next = damping * B_new + (1.0 - damping) * B_old

        policy_err, xi_err, share_err = _compute_metrics_numpy(
            C_old, C_next, S_old, S_next, B_old, B_next, pc.wealth_grid, trim_wealth_points,
        )
        policy_supnorm_history.append(policy_err)
        xi_supnorm_history.append(xi_err)
        share_supnorm_history.append(share_err)

        C_old, S_old, B_old = C_next, S_next, B_next
        n_iter_done = it + 1
        stop_err = max(xi_err, share_err)

        if show_progress and progress_every > 0 and ((it + 1) % progress_every == 0):
            if show_progress_probe:
                probe_W = float(pc.wealth_grid[probe_w])
                probe_c = float(C_old[probe_z, probe_s, probe_w])
                probe_s_val = float(S_old[probe_z, probe_s, probe_w])
                probe_b_val = float(B_old[probe_z, probe_s, probe_w])
                _print_progress_line(
                    it + 1, xi_err, share_err, stop_err,
                    True, probe_W, probe_c / probe_W if probe_W > 0 else float("nan"),
                    probe_s_val, probe_b_val, 1.0 - probe_s_val - probe_b_val,
                )
            else:
                _print_progress_line(
                    it + 1, xi_err, share_err, stop_err,
                    False, 0.0, 0.0, 0.0, 0.0, 0.0,
                )

        if it > 0 and stop_err < tol:
            converged = True
            break

    if show_progress and n_iter_done > 0:
        print()

    diagnostics = _build_diagnostics(
        model, pc, solver_config, C_old, S_old, B_old,
        converged, n_iter_done,
        0,                              # total_newton_failures: kernel doesn't expose per-cell exits
        np.asarray(policy_supnorm_history),
        np.asarray(xi_supnorm_history),
        np.asarray(share_supnorm_history),
        tol, damping, trim_wealth_points, used_warm_start=used_warm_start,
    )

    if verbose:
        status = "converged" if converged else "hit max_iter"
        print(
            f"Infinite-horizon solve {status} after {n_iter_done} iterations; "
            f"final stopping error = {diagnostics['final_stopping_supnorm']:.3e}; "
            f"wall {time.time() - t_start:.1f}s"
        )
    return C_old, S_old, B_old, diagnostics


# =============================================================================
# Smoke test (JAX warm-up)
# =============================================================================

def compile_inner_kernel_smoke_test(
    model, pc,
    solver_config: SolverConfig | None = None,
    constrained: bool | None = None,
    verbose: bool = True,
):
    """Force a one-iteration call so the JAX retirement kernel is compiled.

    The Numba-era version performed AOT compilation via objmode and timed it.
    For JAX, calling the kernel once on representative arrays warms it up.
    Returns a small dict so callers that key off the result keep working.
    """
    if solver_config is None:
        solver_config = SolverConfig()
    if constrained:
        warnings.warn(
            "constrained=True is ignored — JAX inf-horizon is unconstrained-only.",
            stacklevel=2,
        )

    delta = solver_config.delta_bequest if solver_config.delta_bequest >= 0 else DELTA_BEQUEST
    pcj = _pc_to_jnp(pc, delta)
    mp = ModelParams(
        gamma=jnp.float64(model.gamma),
        beta=jnp.float64(model.beta),
        b_bar=jnp.float64(0.0),
        delta=jnp.float64(delta),
        rho=jnp.float64(model.rho),
    )
    n_dev = len(jax.devices())
    per_is_tensors = _precompute_per_is_tensors(pcj)
    retirement_kernel = _build_per_age_retirement_kernel(
        pcj, mp, solver_config, n_dev, pc.n_z, pc.N_state, per_is_tensors,
    )

    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    C_old = np.broadcast_to(
        pc.wealth_grid.reshape(1, 1, -1), expected_shape,
    ).astype(np.float64, copy=True)
    pension_zero = jnp.zeros(pc.n_z, dtype=jnp.float64)
    psi_one = jnp.ones(pc.n_z, dtype=jnp.float64)
    # Smoke-test has no policy in hand to warm-start from, so seed from the
    # canonical cold scalars (mirrors run_lifecycle_solver's
    # use_backward_age_warm_start=False path).
    init_a_s_arr = jnp.full(expected_shape, float(solver_config.init_alpha_s), dtype=jnp.float64)
    init_a_b_arr = jnp.full(expected_shape, float(solver_config.init_alpha_b), dtype=jnp.float64)

    if verbose:
        print("Compiling JAX retirement kernel for the inf-horizon benchmark...")
    t0 = time.time()
    c_new, s_new, b_new, _ni, _nb = retirement_kernel(
        jnp.asarray(C_old), pension_zero, psi_one,
        init_a_s_arr, init_a_b_arr,
    )
    np.asarray(c_new)  # force completion
    elapsed = time.time() - t0
    if verbose:
        print(f"  done in {elapsed:.1f}s (devices={n_dev})")
    return {
        "elapsed_sec": elapsed,
        "n_devices": n_dev,
    }


# =============================================================================
# Convenience accessor
# =============================================================================

def extract_policy_at_point(C, S, B, i_z: int, i_s: int, i_w: int):
    """Convenience helper for figure overlays at a single state-space point."""
    return {
        "consumption": float(C[i_z, i_s, i_w]),
        "alpha_stock": float(S[i_z, i_s, i_w]),
        "alpha_bond": float(B[i_z, i_s, i_w]),
        "alpha_bill": float(1.0 - S[i_z, i_s, i_w] - B[i_z, i_s, i_w]),
    }
