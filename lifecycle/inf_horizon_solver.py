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
    EC_INTERIOR,
    ModelParams,
    _build_per_age_retirement_kernel,
    _fixup_failed_cells,
    _pc_to_jnp,
    _precompute_per_is_tensors,
)
from lifecycle.wealth_grid import wealth_grid_hash


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

    The Newton-failure column from the Numba era is gone — the per-iteration
    strict-tolerance-miss count is recorded in the diagnostics dict instead
    (``newton_strict_tol_misses_per_iter``); ``total_newton_strict_tol_misses``
    is the sum. (Aliased as ``newton_failures_per_iter`` /
    ``total_newton_failures`` for backwards compatibility — see
    ``_build_diagnostics`` for the rename rationale.)
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

    if model.y_1_index_in_state is None:
        raise ValueError(
            "inf_horizon cold-start requires y_1_index_in_state on the "
            "state grid (real-yields bill anchor)."
        )
    y_1_idx = int(model.y_1_index_in_state)
    xr_pos = int(model.ret_names.index("xr"))
    xb_pos = int(model.ret_names.index("xb"))

    for i_s in range(N_state):
        base_mu_r = pc.const_r + pc.A_r @ pc.state_grid[i_s]
        # Real-yields bill: log_R_bill_{t+1} = state_t[y_1_idx], constant
        # across the state-innovation quadrature node k_v.
        log_R_bill_i = float(pc.state_grid[i_s, y_1_idx])
        R_bill = np.exp(log_R_bill_i)

        returns = np.empty((weight_vec.size, 3), dtype=np.float64)
        row = 0
        for k_v in range(len(pc.v_weights)):
            mu_stock = base_mu_r[xr_pos] + pc.M_v_nodes[k_v, xr_pos]
            mu_bond = base_mu_r[xb_pos] + pc.M_v_nodes[k_v, xb_pos]
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
    if progress_probe_state_idx is None:
        # Per-axis midpoint, not flat-index midpoint. flat // 2 unravels to
        # (n0//2, 0, 0, ..., 0) in C-order — center on axis 0 but axis-zero
        # tails on the rest. Use ravel_multi_index of (n_k//2,) per axis to
        # actually probe the state-grid center.
        state_axis_sizes = tuple(g.shape[0] for g in pc.state_bracket_grids)
        mid_per_axis = tuple(n // 2 for n in state_axis_sizes)
        i_s = int(np.ravel_multi_index(mid_per_axis, state_axis_sizes))
    else:
        i_s = int(progress_probe_state_idx)
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

    if model.y_1_index_in_state is None:
        raise ValueError(
            "inf_horizon Z-proxy requires y_1_index_in_state on the "
            "state grid (real-yields bill anchor)."
        )
    y_1_idx = int(model.y_1_index_in_state)
    xr_pos = int(model.ret_names.index("xr"))
    xb_pos = int(model.ret_names.index("xb"))

    for i_s in range(pc.N_state):
        s_i = pc.state_grid[i_s]
        base_mu_r_i = pc.const_r + pc.A_r @ s_i
        # Real-yields bill: deterministic given current state.
        log_R_bill = float(s_i[y_1_idx])
        alpha_s = float(S[i_z, i_s, i_w])
        alpha_b = float(B[i_z, i_s, i_w])

        expected_rpow = 0.0
        for k_v, w_v in enumerate(pc.v_weights):
            mu_r_stock = base_mu_r_i[xr_pos] + pc.M_v_nodes[k_v, xr_pos]
            mu_r_bond = base_mu_r_i[xb_pos] + pc.M_v_nodes[k_v, xb_pos]
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


def _build_iter_histograms_per_iter(newton_iter_per_iter, backtrack_iter_per_iter):
    """Per-iteration analog of solver._build_iter_histograms.

    Each entry is a NumPy array of shape ``(n_z, N_state)`` from one
    fixed-point iteration's kernel call. Mirrors the lifecycle solver's
    histogram dict shape (p50/p95/p99/max + per-iter slices) so downstream
    consumers can read both diagnostics surfaces with the same code, modulo
    the per_age_* -> per_iter_* key rename.
    """
    def _stats_for(per_iter_list):
        per_iter_p99 = []
        per_iter_max = []
        flat_chunks = []
        for arr in per_iter_list:
            if arr is None:
                continue
            arr_flat = np.asarray(arr).ravel()
            if arr_flat.size == 0:
                continue
            flat_chunks.append(arr_flat)
            per_iter_p99.append(float(np.percentile(arr_flat, 99)))
            per_iter_max.append(int(arr_flat.max()))
        if not flat_chunks:
            return {
                "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0,
                "per_iter_p99": [], "per_iter_max": [], "n_cells": 0,
            }
        all_flat = np.concatenate(flat_chunks)
        return {
            "p50": float(np.median(all_flat)),
            "p95": float(np.percentile(all_flat, 95)),
            "p99": float(np.percentile(all_flat, 99)),
            "max": int(all_flat.max()),
            "per_iter_p99": per_iter_p99,
            "per_iter_max": per_iter_max,
            "n_cells": int(all_flat.size),
        }

    return _stats_for(newton_iter_per_iter), _stats_for(backtrack_iter_per_iter)


def _build_diagnostics(
    model, pc, solver_config, C, S, B,
    converged, n_iter_done,
    policy_supnorm_history, xi_supnorm_history, share_supnorm_history,
    newton_iter_per_iter, backtrack_iter_per_iter,
    newton_failures_per_iter,
    tol, damping, trim_wealth_points, used_warm_start,
):
    policy_hist = np.asarray(policy_supnorm_history[:n_iter_done], dtype=np.float64).copy()
    xi_hist = np.asarray(xi_supnorm_history[:n_iter_done], dtype=np.float64).copy()
    share_hist = np.asarray(share_supnorm_history[:n_iter_done], dtype=np.float64).copy()

    ni_hist, nb_hist = _build_iter_histograms_per_iter(
        newton_iter_per_iter[:n_iter_done],
        backtrack_iter_per_iter[:n_iter_done],
    )

    # ``newton_failures_per_iter`` counts cells whose Newton ``exit_code !=
    # EC_INTERIOR`` — i.e., did not satisfy ``err < tol * scale`` by the time
    # the FORI Newton loop wrapped at ``solver_config.max_iter``. Despite the
    # historical name, this is NOT a divergence count: the
    # ``newton_iter_histogram`` (correct) routinely shows p99=2 max=10 while
    # this counter saturates near ``N_state * IH_max_iter``, because the
    # FORI exit-code is set purely from the final-residual check (see
    # ``_newton_fori`` / ``_newton_while`` in lifecycle/solver.py — both
    # collapse "ls_failed early-out" and "max_iter cap miss" into the same
    # ``EC_NEWTON_FAIL``). Real divergence almost never happens; what this
    # tally measures is "FORI strict-tolerance misses".
    #
    # New canonical key: ``total_newton_strict_tol_misses`` /
    # ``newton_strict_tol_misses_per_iter``. The old keys
    # ``total_newton_failures`` / ``newton_failures_per_iter`` are retained
    # as aliases for backwards compatibility with saved bundles, analysis
    # scripts, and tests (~80 consumers across the repo). Prefer the new
    # keys in new code.
    #
    # TODO(option-B): tease apart the two failure modes by introducing a
    # distinct exit code for "ls_failed early-out" vs. "max_iter cap miss",
    # then narrow this counter to genuine divergence. Out of scope for this
    # rename pass — see fix(diagnostics) commit on
    # fix/newton-failure-counter-semantics.
    misses_per_iter_arr = np.asarray(
        newton_failures_per_iter[:n_iter_done], dtype=np.int64
    ).copy()
    total_strict_tol_misses = int(misses_per_iter_arr.sum()) if misses_per_iter_arr.size else 0

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
        # Canonical (post-rename) keys.
        "total_newton_strict_tol_misses": total_strict_tol_misses,
        "newton_strict_tol_misses_per_iter": misses_per_iter_arr,
        # Deprecated aliases — same values, kept for backwards compatibility.
        "total_newton_failures": total_strict_tol_misses,
        "newton_failures_per_iter": misses_per_iter_arr,
        "newton_iter_histogram": ni_hist,
        "backtrack_iter_histogram": nb_hist,
        "newton_iter_p99": float(ni_hist["p99"]),
        "n_backtrack_total_p99": float(nb_hist["p99"]),
        "wealth_grid_hash": getattr(pc, "wealth_grid_hash", wealth_grid_hash(pc.wealth_grid)),
        "wealth_grid_kind": getattr(pc, "wealth_grid_kind", "log1p"),
        "wealth_grid_source": getattr(pc, "wealth_grid_source", None),
        "wealth_grid_min": float(np.asarray(pc.wealth_grid)[0]),
        "wealth_grid_max": float(np.asarray(pc.wealth_grid)[-1]),
        "wealth_grid_n": int(np.asarray(pc.wealth_grid).size),
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

    # Per-savings backward-iter warm-start buffer (mirrors run_lifecycle_solver's
    # ``as_grid_prev``/``ab_grid_prev``). For iteration 0 we have no per-savings
    # α-grid — the prepared ``S_old``/``B_old`` policies live on the *wealth*
    # grid. Mid-wealth gather + broadcast across the savings axis gives the same
    # iter-0 init the previous (Variant A) code used; iter 1+ then rolls forward
    # the kernel's per-savings α-grid output (Variant B).
    w_ref_idx = pc.n_w // 2
    init_a_s_arr = jnp.broadcast_to(
        jnp.asarray(S_old[:, :, w_ref_idx])[:, :, None],
        (pc.n_z, pc.N_state, pc.n_s),
    )
    init_a_b_arr = jnp.broadcast_to(
        jnp.asarray(B_old[:, :, w_ref_idx])[:, :, None],
        (pc.n_z, pc.N_state, pc.n_s),
    )

    if run_smoke_test:
        # One warm-up call to JIT-compile the kernel before timing the loop.
        # Same init source as iter 0 below — mid-wealth gather of S_old/B_old.
        (_c, _s, _b,
         _as_grid, _ab_grid,
         _ni, _nb, _ec) = retirement_kernel(
            jnp.asarray(C_old), pension_zero, psi_one,
            init_a_s_arr, init_a_b_arr,
        )
        np.asarray(_c)  # block until ready

    # ---- Iteration ----
    policy_supnorm_history = []
    xi_supnorm_history = []
    share_supnorm_history = []
    # Per-iteration Newton-iter and backtrack-iter counts, one (n_z, N_state)
    # array per iteration. Aggregated in _build_diagnostics into a histogram
    # dict keyed parallel to run_lifecycle_solver's diagnostics["newton_iter_
    # histogram"] (modulo per_age_* -> per_iter_* key rename).
    newton_iter_per_iter = []
    backtrack_iter_per_iter = []
    # Per-iteration Newton strict-tolerance-miss count (cells whose FORI Newton
    # exit_code != EC_INTERIOR — i.e., did not hit ``err < tol * scale`` by the
    # FORI cap; this conflates "ls_failed early-out" and "max_iter cap miss",
    # see _build_diagnostics for the semantic note). Summed into
    # ``total_newton_strict_tol_misses`` (and aliased as
    # ``total_newton_failures``) in the diagnostics dict.
    newton_failures_per_iter: list[int] = []
    converged = False
    n_iter_done = 0

    if verbose:
        print(f"Running JAX infinite-horizon fixed-point loop "
              f"(devices={n_dev}, max_iter={max_iter})...")

    t_start = time.time()
    interrupted = False
    # Mirrors lifecycle/solver.py:2642 / 2755-2758: on Ctrl-C, fall through to
    # _build_diagnostics + return so the caller can save a partial bundle. The
    # last fully-committed (C_old, S_old, B_old) and n_iter_done stay in sync
    # because both update at the *end* of each iteration body.
    try:
        for it in range(max_iter):
            c_old_jnp = jnp.asarray(C_old)
            # Per-savings backward-iter warm-start: iter 0 starts from a
            # mid-wealth-gather broadcast (Variant A — see pre-loop setup);
            # iter 1+ uses the previous iteration's converged per-savings α-grid
            # (Variant B). Mirrors run_lifecycle_solver's per-savings warm-start
            # in the time dimension.
            (c_new_jnp, s_new_jnp, b_new_jnp,
             as_grid_new_jnp, ab_grid_new_jnp,
             ni_jnp, nb_jnp, ec_jnp) = retirement_kernel(
                c_old_jnp, pension_zero, psi_one, init_a_s_arr, init_a_b_arr,
            )
            C_new = np.asarray(c_new_jnp)
            S_new = np.asarray(s_new_jnp)
            B_new = np.asarray(b_new_jnp)
            newton_iter_per_iter.append(np.asarray(ni_jnp))
            backtrack_iter_per_iter.append(np.asarray(nb_jnp))
            ec_iter = np.asarray(ec_jnp)
            newton_failures_per_iter.append(int(np.sum(ec_iter != EC_INTERIOR)))

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
            # Replace failed cells' stored α with a converged neighbor's α
            # before threading the per-savings buffer to the next iter's
            # Variant B init. Cells that fail at iter k get neighbor-seeded
            # for iter k+1 — over many iterations the neighbor's α diffuses
            # through the failure region, breaking the cold-init cascade.
            if solver_config.failure_seed_from_neighbor:
                as_grid_new_jnp, ab_grid_new_jnp = _fixup_failed_cells(
                    as_grid_new_jnp, ab_grid_new_jnp, ec_jnp,
                    solver_config.init_alpha_s, solver_config.init_alpha_b,
                )
            # Roll the per-savings α-grid forward as the next iter's warm-start.
            # Reassigning the bindings releases the previous arrays.
            init_a_s_arr = as_grid_new_jnp
            init_a_b_arr = ab_grid_new_jnp
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
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print(
                f"\n  Inf-horizon solve interrupted at iter {n_iter_done}. "
                f"Returning partial output.",
                flush=True,
            )

    if show_progress and n_iter_done > 0:
        print()

    diagnostics = _build_diagnostics(
        model, pc, solver_config, C_old, S_old, B_old,
        converged, n_iter_done,
        np.asarray(policy_supnorm_history),
        np.asarray(xi_supnorm_history),
        np.asarray(share_supnorm_history),
        newton_iter_per_iter,
        backtrack_iter_per_iter,
        newton_failures_per_iter,
        tol, damping, trim_wealth_points, used_warm_start=used_warm_start,
    )

    if verbose:
        if interrupted:
            status = "interrupted"
        elif converged:
            status = "converged"
        else:
            status = "hit max_iter"
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

    c_shape = (pc.n_z, pc.N_state, pc.n_w)
    init_arr_shape = (pc.n_z, pc.N_state, pc.n_s)
    C_old = np.broadcast_to(
        pc.wealth_grid.reshape(1, 1, -1), c_shape,
    ).astype(np.float64, copy=True)
    pension_zero = jnp.zeros(pc.n_z, dtype=jnp.float64)
    psi_one = jnp.ones(pc.n_z, dtype=jnp.float64)
    # Smoke-test has no policy in hand to warm-start from, so seed from the
    # canonical cold scalars (mirrors run_lifecycle_solver's
    # use_backward_age_warm_start=False path).
    init_a_s_arr = jnp.full(init_arr_shape, float(solver_config.init_alpha_s), dtype=jnp.float64)
    init_a_b_arr = jnp.full(init_arr_shape, float(solver_config.init_alpha_b), dtype=jnp.float64)

    if verbose:
        print("Compiling JAX retirement kernel for the inf-horizon benchmark...")
    t0 = time.time()
    (c_new, s_new, b_new,
     _as_grid, _ab_grid,
     _ni, _nb, _ec) = retirement_kernel(
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
