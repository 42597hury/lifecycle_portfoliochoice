"""
Standalone infinite-horizon benchmark solver built on the retirement kernel.

This module is intentionally separate from ``solver.py`` so the lifecycle code
path remains unchanged. The benchmark shuts off labor income, pension, and
mortality economically, then iterates the retirement-period policy operator to
its stationary fixed point.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from model import SolverConfig
from solver import DI_NEWTON_FAIL, _solve_retirement_step_quad_jit


DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 500
DEFAULT_DAMPING = 1.0
DEFAULT_TRIM_WEALTH_POINTS = 5


@njit(cache=True)
def _smoke_call_inner_kernel(
    wealth_grid,
    savings_grid,
    z_grid,
    N_state,
    c_next_full,
    pension_1d,
    annuity_factors,
    state_grid,
    grids_0,
    grids_1,
    grids_2,
    state_bracket_shift,
    state_bracket_L_inv,
    v_nodes,
    v_weights,
    M_v_nodes,
    const_r,
    A_r,
    Phi_0_state,
    Phi_11,
    exp_ret_bill,
    exp_ret_stock,
    exp_ret_bond,
    ret_weights,
    gamma,
    psi_vec,
    beta,
    b_bar,
    constrained,
    solver_config,
    out_c,
    out_s,
    out_b,
):
    _solve_retirement_step_quad_jit(
        wealth_grid,
        savings_grid,
        z_grid,
        N_state,
        c_next_full,
        pension_1d,
        annuity_factors,
        state_grid,
        grids_0,
        grids_1,
        grids_2,
        state_bracket_shift,
        state_bracket_L_inv,
        v_nodes,
        v_weights,
        M_v_nodes,
        const_r,
        A_r,
        Phi_0_state,
        Phi_11,
        exp_ret_bill,
        exp_ret_stock,
        exp_ret_bond,
        ret_weights,
        gamma,
        psi_vec,
        beta,
        b_bar,
        constrained,
        solver_config,
        out_c,
        out_s,
        out_b,
    )
    return 0


@njit(cache=True)
def _count_diag_column(diag_int, column_idx):
    total = 0
    for i_s in range(diag_int.shape[0]):
        total += diag_int[i_s, column_idx]
    return total


@njit(cache=True)
def _apply_update_in_place(C_old, S_old, B_old, out_c, out_s, out_b, damping):
    n_z, N_state, n_w = C_old.shape
    if damping == 1.0:
        for iz in range(n_z):
            for is_ in range(N_state):
                for iw in range(n_w):
                    C_old[iz, is_, iw] = out_c[iz, is_, iw]
                    S_old[iz, is_, iw] = out_s[iz, is_, iw]
                    B_old[iz, is_, iw] = out_b[iz, is_, iw]
    else:
        one_minus = 1.0 - damping
        for iz in range(n_z):
            for is_ in range(N_state):
                for iw in range(n_w):
                    C_old[iz, is_, iw] = (
                        damping * out_c[iz, is_, iw] + one_minus * C_old[iz, is_, iw]
                    )
                    S_old[iz, is_, iw] = (
                        damping * out_s[iz, is_, iw] + one_minus * S_old[iz, is_, iw]
                    )
                    B_old[iz, is_, iw] = (
                        damping * out_b[iz, is_, iw] + one_minus * B_old[iz, is_, iw]
                    )


@njit(cache=True)
def _compute_ih_metrics_jit(
    C_old,
    out_c,
    S_old,
    out_s,
    B_old,
    out_b,
    wealth_grid,
    trim_wealth_points,
):
    n_z, N_state, n_w = C_old.shape

    policy_err = 0.0
    xi_err = 0.0
    share_err = 0.0

    for iz in range(n_z):
        for is_ in range(N_state):
            for iw in range(n_w):
                dc = out_c[iz, is_, iw] - C_old[iz, is_, iw]
                if dc < 0.0:
                    dc = -dc
                if dc > policy_err:
                    policy_err = dc

                ds = out_s[iz, is_, iw] - S_old[iz, is_, iw]
                if ds < 0.0:
                    ds = -ds
                if ds > share_err:
                    share_err = ds
                if ds > policy_err:
                    policy_err = ds

                db = out_b[iz, is_, iw] - B_old[iz, is_, iw]
                if db < 0.0:
                    db = -db
                if db > share_err:
                    share_err = db
                if db > policy_err:
                    policy_err = db

            for iw in range(trim_wealth_points, n_w):
                w = wealth_grid[iw]
                vold = C_old[iz, is_, iw] / w
                vnew = out_c[iz, is_, iw] / w
                dxi = vnew - vold
                if dxi < 0.0:
                    dxi = -dxi
                if dxi > xi_err:
                    xi_err = dxi

    return policy_err, xi_err, share_err


@njit(cache=True)
def _run_infinite_horizon_core_jit(
    wealth_grid,
    savings_grid,
    z_grid,
    N_state,
    annuity_factors,
    state_grid,
    grids_0,
    grids_1,
    grids_2,
    state_bracket_shift,
    state_bracket_L_inv,
    v_nodes,
    v_weights,
    M_v_nodes,
    const_r,
    A_r,
    Phi_0_state,
    Phi_11,
    exp_ret_bill,
    exp_ret_stock,
    exp_ret_bond,
    ret_weights,
    gamma,
    beta,
    psi_vec,
    pension_1d,
    b_bar,
    constrained,
    solver_config,
    C_old,
    S_old,
    B_old,
    out_c,
    out_s,
    out_b,
    policy_supnorm_history,
    xi_supnorm_history,
    share_supnorm_history,
    tol,
    max_iter,
    damping,
    trim_wealth_points,
):
    total_newton_failures = 0
    converged = False
    n_iter_done = 0

    for it in range(max_iter):
        diag_int, _ = _solve_retirement_step_quad_jit(
            wealth_grid,
            savings_grid,
            z_grid,
            N_state,
            C_old,
            pension_1d,
            annuity_factors,
            state_grid,
            grids_0,
            grids_1,
            grids_2,
            state_bracket_shift,
            state_bracket_L_inv,
            v_nodes,
            v_weights,
            M_v_nodes,
            const_r,
            A_r,
            Phi_0_state,
            Phi_11,
            exp_ret_bill,
            exp_ret_stock,
            exp_ret_bond,
            ret_weights,
            gamma,
            psi_vec,
            beta,
            b_bar,
            constrained,
            solver_config,
            out_c,
            out_s,
            out_b,
        )

        total_newton_failures += _count_diag_column(diag_int, DI_NEWTON_FAIL)

        policy_err, xi_err, share_err = _compute_ih_metrics_jit(
            C_old,
            out_c,
            S_old,
            out_s,
            B_old,
            out_b,
            wealth_grid,
            trim_wealth_points,
        )
        policy_supnorm_history[it] = policy_err
        xi_supnorm_history[it] = xi_err
        share_supnorm_history[it] = share_err

        _apply_update_in_place(C_old, S_old, B_old, out_c, out_s, out_b, damping)

        n_iter_done = it + 1
        if it > 0 and max(xi_err, share_err) < tol:
            converged = True
            break

    return converged, n_iter_done, total_newton_failures


def _retirement_entry_index(pc, retire_age: int) -> int:
    matches = np.where(np.asarray(pc.ages) == int(retire_age))[0]
    if len(matches) == 0:
        raise ValueError(f"retire_age={retire_age} not found in pc.ages")
    return int(matches[0])


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
        out = out[_retirement_entry_index(pc, retire_age)]
    if out.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape} or lifecycle shape (n_age, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(out.copy())


def _prepare_initial_policies(
    pc,
    retire_age: int,
    solver_config: SolverConfig,
    warm_start_c,
    warm_start_s,
    warm_start_b,
):
    expected_shape = (pc.n_z, pc.N_state, pc.n_w)

    C_old = _coerce_policy_array("warm_start_c", warm_start_c, expected_shape, pc, retire_age)
    S_old = _coerce_policy_array("warm_start_s", warm_start_s, expected_shape, pc, retire_age)
    B_old = _coerce_policy_array("warm_start_b", warm_start_b, expected_shape, pc, retire_age)

    if C_old is None:
        C_old = np.broadcast_to(
            pc.wealth_grid.reshape(1, 1, -1),
            expected_shape,
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
        np.max(S[:, :, trim_wealth_points:], axis=2) - np.min(S[:, :, trim_wealth_points:], axis=2)
    )
    bond_spread = np.max(
        np.max(B[:, :, trim_wealth_points:], axis=2) - np.min(B[:, :, trim_wealth_points:], axis=2)
    )

    return {
        "max_xi_spread_across_w": float(xi_spread),
        "max_stock_spread_across_w": float(stock_spread),
        "max_bond_spread_across_w": float(bond_spread),
        "max_share_spread_across_w": float(max(stock_spread, bond_spread)),
    }


def _compute_stability_proxy(model, pc, solver_config: SolverConfig, S, B, trim_wealth_points):
    if pc.n_w == 0:
        return float("nan")

    i_z = min(pc.n_z // 2, pc.n_z - 1)
    i_w = min(max(trim_wealth_points, pc.n_w // 2), pc.n_w - 1)
    max_proxy = 0.0

    for i_s in range(pc.N_state):
        s_i = pc.state_grid[i_s]
        base_mu_r_i = np.empty(3, dtype=np.float64)
        base_mu_r_i[0] = pc.const_r[0] + pc.A_r[0, 0] * s_i[0] + pc.A_r[0, 1] * s_i[1] + pc.A_r[0, 2] * s_i[2]
        base_mu_r_i[1] = pc.const_r[1] + pc.A_r[1, 0] * s_i[0] + pc.A_r[1, 1] * s_i[1] + pc.A_r[1, 2] * s_i[2]
        base_mu_r_i[2] = pc.const_r[2] + pc.A_r[2, 0] * s_i[0] + pc.A_r[2, 1] * s_i[1] + pc.A_r[2, 2] * s_i[2]

        alpha_s = float(S[i_z, i_s, i_w])
        alpha_b = float(B[i_z, i_s, i_w])
        alpha_bill = 1.0 - alpha_s - alpha_b

        expected_rpow = 0.0
        for k_v, w_v in enumerate(pc.v_weights):
            mu_r_bill = base_mu_r_i[0] + pc.M_v_nodes[k_v, 0]
            mu_r_stock = base_mu_r_i[1] + pc.M_v_nodes[k_v, 1]
            mu_r_bond = base_mu_r_i[2] + pc.M_v_nodes[k_v, 2]
            exp_mu_bill = np.exp(mu_r_bill)
            exp_mu_stock = np.exp(mu_r_stock)
            exp_mu_bond = np.exp(mu_r_bond)

            for k_r, p_ret in enumerate(pc.ret_weights):
                R_bill = exp_mu_bill * pc.exp_ret_bill[k_r]
                R_stock = R_bill * exp_mu_stock * pc.exp_ret_stock[k_r]
                R_bond = R_bill * exp_mu_bond * pc.exp_ret_bond[k_r]
                R_p = alpha_s * R_stock + alpha_b * R_bond + alpha_bill * R_bill
                if R_p <= 0.0:
                    return float("inf")
                expected_rpow += float(w_v) * float(p_ret) * (R_p ** (1.0 - model.gamma))

        proxy_i = float(model.beta) * expected_rpow
        if proxy_i > max_proxy:
            max_proxy = proxy_i

    return float(max_proxy)


def _build_diagnostics(
    model,
    pc,
    solver_config: SolverConfig,
    C,
    S,
    B,
    converged: bool,
    n_iter_done: int,
    total_newton_failures: int,
    policy_supnorm_history,
    xi_supnorm_history,
    share_supnorm_history,
    tol: float,
    damping: float,
    trim_wealth_points: int,
    used_warm_start: bool,
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
    diagnostics["stability_proxy"] = _compute_stability_proxy(model, pc, solver_config, S, B, trim_wealth_points)

    return diagnostics


def compile_inner_kernel_smoke_test(
    model,
    pc,
    solver_config: SolverConfig | None = None,
    constrained: bool | None = None,
    verbose: bool = True,
):
    """Compile and call the nested JIT solve path once on a one-iteration benchmark."""
    if solver_config is None:
        solver_config = SolverConfig()
    if constrained is None:
        constrained = bool(model.constrained)

    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    c_next_full = np.broadcast_to(
        pc.wealth_grid.reshape(1, 1, -1),
        expected_shape,
    ).astype(np.float64, copy=True)
    s_next_full = np.full(expected_shape, float(solver_config.init_alpha_s), dtype=np.float64)
    b_next_full = np.full(expected_shape, float(solver_config.init_alpha_b), dtype=np.float64)
    out_c = np.empty(expected_shape, dtype=np.float64)
    out_s = np.empty(expected_shape, dtype=np.float64)
    out_b = np.empty(expected_shape, dtype=np.float64)
    psi_vec = np.ones(pc.n_z, dtype=np.float64)
    pension_1d = np.zeros(pc.n_z, dtype=np.float64)
    b_bar = 0.0
    policy_supnorm_history = np.empty(1, dtype=np.float64)
    xi_supnorm_history = np.empty(1, dtype=np.float64)
    share_supnorm_history = np.empty(1, dtype=np.float64)

    if verbose:
        print("Compiling inner-kernel smoke test for inf_horizon_solver...")

    try:
        _run_infinite_horizon_core_jit(
            pc.wealth_grid,
            pc.s_grid,
            pc.z_grid,
            pc.N_state,
            pc.annuity_factors,
            pc.state_grid,
            pc.state_bracket_grids[0],
            pc.state_bracket_grids[1],
            pc.state_bracket_grids[2],
            pc.state_bracket_shift,
            pc.state_bracket_L_inv,
            pc.v_nodes,
            pc.v_weights,
            pc.M_v_nodes,
            pc.const_r,
            pc.A_r,
            model.Phi_0_state,
            model.Phi_11,
            pc.exp_ret_bill,
            pc.exp_ret_stock,
            pc.exp_ret_bond,
            pc.ret_weights,
            model.gamma,
            model.beta,
            psi_vec,
            pension_1d,
            b_bar,
            constrained,
            solver_config,
            c_next_full,
            s_next_full,
            b_next_full,
            out_c,
            out_s,
            out_b,
            policy_supnorm_history,
            xi_supnorm_history,
            share_supnorm_history,
            1e-4,
            1,
            1.0,
            0,
        )
    except Exception as exc:  # pragma: no cover - exercised only on failure path
        raise RuntimeError(
            "Inner-kernel smoke test failed. The first thing to inspect is whether "
            "SolverConfig remains Numba-resolvable in nested JIT calls."
        ) from exc

    return {
        "compiled": True,
        "shape": expected_shape,
    }


def run_infinite_horizon_solver(
    model,
    pc,
    solver_config: SolverConfig | None = None,
    warm_start_c=None,
    warm_start_s=None,
    warm_start_b=None,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    damping: float = DEFAULT_DAMPING,
    trim_wealth_points: int = DEFAULT_TRIM_WEALTH_POINTS,
    constrained: bool | None = None,
    run_smoke_test: bool = False,
    verbose: bool = True,
):
    """
    Solve the stationary no-income, no-mortality benchmark by fixed-point iteration.

    Warm-start arrays may be passed either as 3D arrays with shape
    ``(n_z, N_state, n_w)`` or as lifecycle arrays with shape
    ``(n_age, n_z, N_state, n_w)``. In the latter case the retirement-entry
    slice is extracted automatically.
    """
    if solver_config is None:
        solver_config = SolverConfig()
    if constrained is None:
        constrained = bool(model.constrained)

    _validate_runtime_options(pc, tol, max_iter, damping, trim_wealth_points)

    if verbose:
        print("Preparing infinite-horizon benchmark solve...")

    if run_smoke_test:
        compile_inner_kernel_smoke_test(
            model,
            pc,
            solver_config=solver_config,
            constrained=constrained,
            verbose=verbose,
        )

    C_old, S_old, B_old = _prepare_initial_policies(
        pc,
        model.retire_age,
        solver_config,
        warm_start_c,
        warm_start_s,
        warm_start_b,
    )

    used_warm_start = any(x is not None for x in (warm_start_c, warm_start_s, warm_start_b))

    out_c = np.empty_like(C_old)
    out_s = np.empty_like(S_old)
    out_b = np.empty_like(B_old)

    policy_supnorm_history = np.empty(max_iter, dtype=np.float64)
    xi_supnorm_history = np.empty(max_iter, dtype=np.float64)
    share_supnorm_history = np.empty(max_iter, dtype=np.float64)

    psi_vec = np.ones(pc.n_z, dtype=np.float64)
    pension_1d = np.zeros(pc.n_z, dtype=np.float64)
    b_bar = 0.0

    if verbose:
        print("Running JIT-compiled infinite-horizon fixed-point loop...")

    converged, n_iter_done, total_newton_failures = _run_infinite_horizon_core_jit(
        pc.wealth_grid,
        pc.s_grid,
        pc.z_grid,
        pc.N_state,
        pc.annuity_factors,
        pc.state_grid,
        pc.state_bracket_grids[0],
        pc.state_bracket_grids[1],
        pc.state_bracket_grids[2],
        pc.state_bracket_shift,
        pc.state_bracket_L_inv,
        pc.v_nodes,
        pc.v_weights,
        pc.M_v_nodes,
        pc.const_r,
        pc.A_r,
        model.Phi_0_state,
        model.Phi_11,
        pc.exp_ret_bill,
        pc.exp_ret_stock,
        pc.exp_ret_bond,
        pc.ret_weights,
        model.gamma,
        model.beta,
        psi_vec,
        pension_1d,
        b_bar,
        constrained,
        solver_config,
        C_old,
        S_old,
        B_old,
        out_c,
        out_s,
        out_b,
        policy_supnorm_history,
        xi_supnorm_history,
        share_supnorm_history,
        tol,
        max_iter,
        damping,
        trim_wealth_points,
    )

    diagnostics = _build_diagnostics(
        model,
        pc,
        solver_config,
        C_old,
        S_old,
        B_old,
        converged,
        n_iter_done,
        total_newton_failures,
        policy_supnorm_history,
        xi_supnorm_history,
        share_supnorm_history,
        tol,
        damping,
        trim_wealth_points,
        used_warm_start=used_warm_start,
    )

    if verbose:
        status = "converged" if converged else "hit max_iter"
        print(
            f"Infinite-horizon solve {status} after {n_iter_done} iterations; "
            f"final stopping error = {diagnostics['final_stopping_supnorm']:.3e}"
        )

    return C_old, S_old, B_old, diagnostics


def extract_policy_at_point(C, S, B, i_z: int, i_s: int, i_w: int):
    """Convenience helper for figure overlays at a single state-space point."""
    return {
        "consumption": float(C[i_z, i_s, i_w]),
        "alpha_stock": float(S[i_z, i_s, i_w]),
        "alpha_bond": float(B[i_z, i_s, i_w]),
        "alpha_bill": float(1.0 - S[i_z, i_s, i_w] - B[i_z, i_s, i_w]),
    }
