"""
_diag_euler_errors.py -- Simulation-path Euler-error diagnostics for saved bundles.

This script adds the referee-facing part of the convergence battery:

1. Simulate a saved policy bundle over its solved age window
2. Re-evaluate the Euler equation at the simulated states with an
   independent, finer expectation rule
3. Report age-by-age and phase summaries of log10|EE|

The implementation is window-aware, so it can be used immediately on partial
bundles such as the current age-65-to-99 convergence sweep. For partial
bundles the simulation starts at the first solved age and the report labels
that clearly. Once a full lifecycle bundle is available, the same script can
be rerun end-to-end for the paper-facing table.

Usage
-----
python -m scripts.diagnostics._diag_euler_errors ^
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 ^
  saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import DiscretizationConfig, disposable_income_working, scalar_disposable_income
from policy_io import load_policy_bundle
from precompute import Precompute
from scripts.diagnostics._diag_policy_convergence import (
    _build_disc_config,
    _build_model_from_bundle,
    _extract_disc_config,
    _infer_solved_mask,
    _summary,
)
from simulation import (
    _build_return_factor,
    _initialize_initial_wealth,
    _normal_bin_probs,
    _resolve_initial_state_indices,
    initialize_states,
    simulate_lifecycle,
    simulate_lifecycle_core,
)
from solver import _interp_z_wealth, find_bracket


@dataclass
class EulerBundleContext:
    path: Path
    label: str
    C: np.ndarray
    S: np.ndarray
    B: np.ndarray
    metadata: dict[str, Any]
    summary: dict[str, Any]
    model: Any
    ages: np.ndarray
    solved_mask: np.ndarray
    disc_policy: DiscretizationConfig
    disc_eval: DiscretizationConfig
    pc_policy: Precompute
    pc_eval: Precompute
    first_solved_t: int
    last_solved_t: int


def _coerce_tuple(value: Any, n: int) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),) * int(n)
    return tuple(int(v) for v in value)


def _build_eval_disc(base: DiscretizationConfig,
                     n_state: int,
                     n_ret: int,
                     mode: str,
                     ret_override: tuple[int, ...] | None,
                     state_override: tuple[int, ...] | None,
                     eta_override: int | None,
                     eps_override: int | None) -> DiscretizationConfig:
    ret_base = _coerce_tuple(base.n_ret_nodes_1d, n_ret)
    state_base = _coerce_tuple(base.n_state_quad_nodes, n_state)

    if mode == "same":
        ret_eval = ret_base
        state_eval = state_base
        eta_eval = int(base.n_eta_nodes)
        eps_eval = int(base.n_eps_nodes)
    elif mode == "double":
        ret_eval = tuple(max(1, 2 * k) for k in ret_base)
        state_eval = tuple(max(1, 2 * k) for k in state_base)
        eta_eval = max(1, 2 * int(base.n_eta_nodes))
        eps_eval = max(1, 2 * int(base.n_eps_nodes))
    elif mode == "next_finer":
        ret_eval = tuple(max(1, k + 2) for k in ret_base)
        state_eval = tuple(max(1, k + 1) for k in state_base)
        eta_eval = max(1, int(base.n_eta_nodes) + 2)
        eps_eval = max(1, int(base.n_eps_nodes) + 2)
    else:
        raise ValueError(f"Unknown eval mode: {mode}")

    if ret_override is not None:
        ret_eval = tuple(int(v) for v in ret_override)
    if state_override is not None:
        state_eval = tuple(int(v) for v in state_override)
    if eta_override is not None:
        eta_eval = int(eta_override)
    if eps_override is not None:
        eps_eval = int(eps_override)

    return DiscretizationConfig(
        n_wealth=int(base.n_wealth),
        wealth_min=float(base.wealth_min),
        wealth_max=float(base.wealth_max),
        n_savings=int(base.n_savings),
        savings_min=float(base.savings_min),
        savings_max=base.savings_max,
        state_grid_sizes=tuple(int(v) for v in base.state_grid_sizes),
        state_grid_mode=base.state_grid_mode,
        state_n_stds=base.state_n_stds,
        n_z=int(base.n_z),
        n_stds=float(base.n_stds),
        n_eps_nodes=int(eps_eval),
        n_eta_nodes=int(eta_eval),
        n_ret_nodes_1d=tuple(int(v) for v in ret_eval),
        n_state_quad_nodes=tuple(int(v) for v in state_eval),
    )


def _load_bundle_context(bundle_path: Path,
                         model_bundle: Path,
                         eval_mode: str,
                         ret_override: tuple[int, ...] | None,
                         state_override: tuple[int, ...] | None,
                         eta_override: int | None,
                         eps_override: int | None) -> EulerBundleContext:
    model, ages = _build_model_from_bundle(model_bundle)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    summary = _summary(metadata)
    disc_raw = _extract_disc_config(metadata)
    if not disc_raw:
        raise ValueError(f"Bundle '{bundle_path}' does not expose a discretization config.")

    disc_policy = _build_disc_config(disc_raw)
    disc_eval = _build_eval_disc(
        disc_policy,
        n_state=int(model.n_state),
        n_ret=int(model.n_ret),
        mode=eval_mode,
        ret_override=ret_override,
        state_override=state_override,
        eta_override=eta_override,
        eps_override=eps_override,
    )

    pc_policy = Precompute(model, disc_policy, verbose=False)
    pc_eval = Precompute(model, disc_eval, verbose=False)

    solved_mask = _infer_solved_mask(C, metadata, ages)
    solved_idx = np.flatnonzero(solved_mask)
    if solved_idx.size == 0:
        raise ValueError(f"Bundle '{bundle_path}' has no solved ages.")

    if not np.all(np.diff(solved_idx) == 1):
        raise ValueError(
            f"Bundle '{bundle_path}' has a non-contiguous solved age window; "
            "the Euler diagnostic currently expects one contiguous block."
        )

    if pc_policy.n_w != pc_eval.n_w or pc_policy.n_z != pc_eval.n_z or pc_policy.N_state != pc_eval.N_state:
        raise ValueError("Evaluation precompute must preserve the policy bundle's grids.")

    return EulerBundleContext(
        path=bundle_path,
        label=bundle_path.name,
        C=C,
        S=S,
        B=B,
        metadata=metadata,
        summary=summary,
        model=model,
        ages=ages,
        solved_mask=solved_mask,
        disc_policy=disc_policy,
        disc_eval=disc_eval,
        pc_policy=pc_policy,
        pc_eval=pc_eval,
        first_solved_t=int(solved_idx[0]),
        last_solved_t=int(solved_idx[-1]),
    )


@njit(fastmath=True)
def _annuity_factor_scalar(y_1: float, spr: float, b_bar: int) -> float:
    out = 0.0
    for k in range(1, b_bar + 1):
        frac = min(k - 1, 19) / 19.0
        y_k = y_1 + spr * frac
        out += (1.0 + y_k) ** (-k)
    return out


@njit(fastmath=True)
def _interp_z_row_value(z_val: float, z_grid: np.ndarray, row: np.ndarray) -> float:
    n_z = len(z_grid)
    if z_val <= z_grid[0]:
        return float(row[0])
    if z_val >= z_grid[n_z - 1]:
        return float(row[n_z - 1])
    dz = z_grid[1] - z_grid[0]
    iz_lo = int((z_val - z_grid[0]) / dz)
    iz_lo = max(0, min(iz_lo, n_z - 2))
    frac_z = (z_val - z_grid[iz_lo]) / dz
    frac_z = max(0.0, min(1.0, frac_z))
    return (1.0 - frac_z) * row[iz_lo] + frac_z * row[iz_lo + 1]


@njit(fastmath=True)
def _interp_continuation_c_mpc(
    c_next_full: np.ndarray,
    z_next: float,
    s_next_0: float,
    s_next_1: float,
    s_next_2: float,
    x_next: float,
    wealth_grid: np.ndarray,
    z_grid: np.ndarray,
    dz: float,
    state_bracket_shift: np.ndarray,
    state_bracket_L_inv: np.ndarray,
    grids_0: np.ndarray,
    grids_1: np.ndarray,
    grids_2: np.ndarray,
    N1: int,
    N2: int,
    min_consumption: float,
) -> tuple[float, float]:
    ds0 = s_next_0 - state_bracket_shift[0]
    ds1 = s_next_1 - state_bracket_shift[1]
    ds2 = s_next_2 - state_bracket_shift[2]
    b0 = state_bracket_L_inv[0, 0] * ds0 + state_bracket_L_inv[0, 1] * ds1 + state_bracket_L_inv[0, 2] * ds2
    b1 = state_bracket_L_inv[1, 0] * ds0 + state_bracket_L_inv[1, 1] * ds1 + state_bracket_L_inv[1, 2] * ds2
    b2 = state_bracket_L_inv[2, 0] * ds0 + state_bracket_L_inv[2, 1] * ds1 + state_bracket_L_inv[2, 2] * ds2

    if b0 <= grids_0[0]:
        lo0 = 0
        f0 = 0.0
    elif b0 >= grids_0[len(grids_0) - 1]:
        lo0 = len(grids_0) - 2
        f0 = 1.0
    else:
        lo0 = 0
        for ii in range(len(grids_0) - 1):
            if grids_0[ii + 1] > b0:
                lo0 = ii
                break
        dg0 = grids_0[lo0 + 1] - grids_0[lo0]
        f0 = (b0 - grids_0[lo0]) / dg0 if dg0 > 1e-30 else 0.0
        f0 = max(0.0, min(1.0, f0))

    if b1 <= grids_1[0]:
        lo1 = 0
        f1 = 0.0
    elif b1 >= grids_1[len(grids_1) - 1]:
        lo1 = len(grids_1) - 2
        f1 = 1.0
    else:
        lo1 = 0
        for ii in range(len(grids_1) - 1):
            if grids_1[ii + 1] > b1:
                lo1 = ii
                break
        dg1 = grids_1[lo1 + 1] - grids_1[lo1]
        f1 = (b1 - grids_1[lo1]) / dg1 if dg1 > 1e-30 else 0.0
        f1 = max(0.0, min(1.0, f1))

    if b2 <= grids_2[0]:
        lo2 = 0
        f2 = 0.0
    elif b2 >= grids_2[len(grids_2) - 1]:
        lo2 = len(grids_2) - 2
        f2 = 1.0
    else:
        lo2 = 0
        for ii in range(len(grids_2) - 1):
            if grids_2[ii + 1] > b2:
                lo2 = ii
                break
        dg2 = grids_2[lo2 + 1] - grids_2[lo2]
        f2 = (b2 - grids_2[lo2]) / dg2 if dg2 > 1e-30 else 0.0
        f2 = max(0.0, min(1.0, f2))

    w000 = (1.0 - f0) * (1.0 - f1) * (1.0 - f2)
    w001 = (1.0 - f0) * (1.0 - f1) * f2
    w010 = (1.0 - f0) * f1 * (1.0 - f2)
    w011 = (1.0 - f0) * f1 * f2
    w100 = f0 * (1.0 - f1) * (1.0 - f2)
    w101 = f0 * (1.0 - f1) * f2
    w110 = f0 * f1 * (1.0 - f2)
    w111 = f0 * f1 * f2

    j000 = lo0 * N1 * N2 + lo1 * N2 + lo2
    j001 = lo0 * N1 * N2 + lo1 * N2 + (lo2 + 1)
    j010 = lo0 * N1 * N2 + (lo1 + 1) * N2 + lo2
    j011 = lo0 * N1 * N2 + (lo1 + 1) * N2 + (lo2 + 1)
    j100 = (lo0 + 1) * N1 * N2 + lo1 * N2 + lo2
    j101 = (lo0 + 1) * N1 * N2 + lo1 * N2 + (lo2 + 1)
    j110 = (lo0 + 1) * N1 * N2 + (lo1 + 1) * N2 + lo2
    j111 = (lo0 + 1) * N1 * N2 + (lo1 + 1) * N2 + (lo2 + 1)

    iz_lo = int((z_next - z_grid[0]) / dz)
    iz_lo = max(0, min(iz_lo, len(z_grid) - 2))
    frac_z = (z_next - z_grid[iz_lo]) / dz
    frac_z = max(0.0, min(1.0, frac_z))
    use_cubic = (iz_lo >= 1) and (iz_lo + 2 < len(z_grid))

    iw, frac_w, inv_dw = find_bracket(x_next, wealth_grid)

    c000, mpc000 = _interp_z_wealth(c_next_full, j000, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c001, mpc001 = _interp_z_wealth(c_next_full, j001, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c010, mpc010 = _interp_z_wealth(c_next_full, j010, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c011, mpc011 = _interp_z_wealth(c_next_full, j011, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c100, mpc100 = _interp_z_wealth(c_next_full, j100, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c101, mpc101 = _interp_z_wealth(c_next_full, j101, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c110, mpc110 = _interp_z_wealth(c_next_full, j110, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)
    c111, mpc111 = _interp_z_wealth(c_next_full, j111, iz_lo, frac_z, iw, frac_w, inv_dw, len(z_grid), use_cubic, min_consumption)

    c_next = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011
              + w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)
    c_next = max(c_next, min_consumption)

    mpc = (w000 * mpc000 + w001 * mpc001 + w010 * mpc010 + w011 * mpc011
           + w100 * mpc100 + w101 * mpc101 + w110 * mpc110 + w111 * mpc111)
    mpc = max(0.0, min(1.0, mpc))

    return c_next, mpc


@njit(fastmath=True)
def _compute_euler_sum_retirement_continuous(
    alpha_s: float,
    alpha_b: float,
    s_val: float,
    z_cur: float,
    state_cur: np.ndarray,
    c_next_full: np.ndarray,
    wealth_grid: np.ndarray,
    z_grid: np.ndarray,
    dz: float,
    pension_row: np.ndarray,
    survival_row: np.ndarray,
    v_nodes: np.ndarray,
    v_weights: np.ndarray,
    M_v_nodes: np.ndarray,
    const_r: np.ndarray,
    A_r: np.ndarray,
    Phi_0_state: np.ndarray,
    Phi_11: np.ndarray,
    state_bracket_shift: np.ndarray,
    state_bracket_L_inv: np.ndarray,
    grids_0: np.ndarray,
    grids_1: np.ndarray,
    grids_2: np.ndarray,
    N1: int,
    N2: int,
    exp_ret_bill: np.ndarray,
    exp_ret_stock: np.ndarray,
    exp_ret_bond: np.ndarray,
    ret_weights: np.ndarray,
    gamma: float,
    b_bar: int,
    y_1_idx: int,
    spr_idx: int,
    min_wealth_inv: float = 1e-10,
    min_consumption: float = 1e-10,
    prob_skip: float = 1e-12,
) -> float:
    a_bill = 1.0 - alpha_s - alpha_b
    psi = _interp_z_row_value(z_cur, z_grid, survival_row)
    prob_death = 1.0 - psi
    pension_next = _interp_z_row_value(z_cur, z_grid, pension_row)

    annuity_factor_cur = _annuity_factor_scalar(
        float(state_cur[y_1_idx]),
        float(state_cur[spr_idx]),
        b_bar,
    )
    base_mu_r_0 = const_r[0] + A_r[0, 0] * state_cur[0] + A_r[0, 1] * state_cur[1] + A_r[0, 2] * state_cur[2]
    base_mu_r_1 = const_r[1] + A_r[1, 0] * state_cur[0] + A_r[1, 1] * state_cur[1] + A_r[1, 2] * state_cur[2]
    base_mu_r_2 = const_r[2] + A_r[2, 0] * state_cur[0] + A_r[2, 1] * state_cur[1] + A_r[2, 2] * state_cur[2]

    euler_sum = 0.0
    for k_v in range(len(v_weights)):
        w_v = v_weights[k_v]
        if w_v < prob_skip:
            continue

        s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * state_cur[0] + Phi_11[0, 1] * state_cur[1] + Phi_11[0, 2] * state_cur[2] + v_nodes[k_v, 0]
        s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * state_cur[0] + Phi_11[1, 1] * state_cur[1] + Phi_11[1, 2] * state_cur[2] + v_nodes[k_v, 1]
        s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * state_cur[0] + Phi_11[2, 1] * state_cur[1] + Phi_11[2, 2] * state_cur[2] + v_nodes[k_v, 2]

        exp_mu_bill = math.exp(base_mu_r_0 + M_v_nodes[k_v, 0])
        exp_mu_s = math.exp(base_mu_r_1 + M_v_nodes[k_v, 1])
        exp_mu_b = math.exp(base_mu_r_2 + M_v_nodes[k_v, 2])

        for k_r in range(len(ret_weights)):
            weight = w_v * ret_weights[k_r]
            if weight < prob_skip:
                continue

            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            sR_p = s_val * R_p
            if sR_p > 0.0:
                x_next = sR_p + pension_next
            else:
                x_next = pension_next

            c_next, _mpc = _interp_continuation_c_mpc(
                c_next_full,
                z_cur,
                s_next_0,
                s_next_1,
                s_next_2,
                x_next,
                wealth_grid,
                z_grid,
                dz,
                state_bracket_shift,
                state_bracket_L_inv,
                grids_0,
                grids_1,
                grids_2,
                N1,
                N2,
                min_consumption,
            )

            mu_alive = c_next ** (-gamma)
            if sR_p > 0.0:
                w_A = sR_p / annuity_factor_cur
                mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_cur
            else:
                mu_bequest = 0.0
            mu_comb = psi * mu_alive + prob_death * mu_bequest
            euler_sum += weight * mu_comb * R_p

    return euler_sum


@njit(fastmath=True)
def _compute_euler_sum_working_continuous(
    alpha_s: float,
    alpha_b: float,
    s_val: float,
    z_cur: float,
    state_cur: np.ndarray,
    c_next_full: np.ndarray,
    wealth_grid: np.ndarray,
    z_grid: np.ndarray,
    dz: float,
    log_det_next: float,
    pension_row: np.ndarray,
    survival_row: np.ndarray,
    rho: float,
    eta_nodes: np.ndarray,
    eta_weights: np.ndarray,
    eps_nodes: np.ndarray,
    eps_weights: np.ndarray,
    v_nodes: np.ndarray,
    v_weights: np.ndarray,
    M_v_nodes: np.ndarray,
    const_r: np.ndarray,
    A_r: np.ndarray,
    Phi_0_state: np.ndarray,
    Phi_11: np.ndarray,
    state_bracket_shift: np.ndarray,
    state_bracket_L_inv: np.ndarray,
    grids_0: np.ndarray,
    grids_1: np.ndarray,
    grids_2: np.ndarray,
    N1: int,
    N2: int,
    exp_ret_bill: np.ndarray,
    exp_ret_stock: np.ndarray,
    exp_ret_bond: np.ndarray,
    ret_weights: np.ndarray,
    gamma: float,
    b_bar: int,
    y_1_idx: int,
    spr_idx: int,
    use_pension_next: bool,
    min_wealth_inv: float = 1e-10,
    min_consumption: float = 1e-10,
    prob_skip: float = 1e-12,
) -> float:
    a_bill = 1.0 - alpha_s - alpha_b
    psi = _interp_z_row_value(z_cur, z_grid, survival_row)
    prob_death = 1.0 - psi

    annuity_factor_cur = _annuity_factor_scalar(
        float(state_cur[y_1_idx]),
        float(state_cur[spr_idx]),
        b_bar,
    )
    base_mu_r_0 = const_r[0] + A_r[0, 0] * state_cur[0] + A_r[0, 1] * state_cur[1] + A_r[0, 2] * state_cur[2]
    base_mu_r_1 = const_r[1] + A_r[1, 0] * state_cur[0] + A_r[1, 1] * state_cur[1] + A_r[1, 2] * state_cur[2]
    base_mu_r_2 = const_r[2] + A_r[2, 0] * state_cur[0] + A_r[2, 1] * state_cur[1] + A_r[2, 2] * state_cur[2]

    exp_eps = np.empty(len(eps_nodes))
    for i_e in range(len(eps_nodes)):
        exp_eps[i_e] = math.exp(eps_nodes[i_e])

    exp_eta = np.empty(len(eta_nodes))
    for k_eta in range(len(eta_nodes)):
        exp_eta[k_eta] = math.exp(eta_nodes[k_eta])

    base_det_z = math.exp(log_det_next + rho * z_cur)
    euler_sum = 0.0

    for k_v in range(len(v_weights)):
        w_v = v_weights[k_v]
        if w_v < prob_skip:
            continue

        s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * state_cur[0] + Phi_11[0, 1] * state_cur[1] + Phi_11[0, 2] * state_cur[2] + v_nodes[k_v, 0]
        s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * state_cur[0] + Phi_11[1, 1] * state_cur[1] + Phi_11[1, 2] * state_cur[2] + v_nodes[k_v, 1]
        s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * state_cur[0] + Phi_11[2, 1] * state_cur[1] + Phi_11[2, 2] * state_cur[2] + v_nodes[k_v, 2]

        exp_mu_bill = math.exp(base_mu_r_0 + M_v_nodes[k_v, 0])
        exp_mu_s = math.exp(base_mu_r_1 + M_v_nodes[k_v, 1])
        exp_mu_b = math.exp(base_mu_r_2 + M_v_nodes[k_v, 2])

        for k_r in range(len(ret_weights)):
            p_state_ret = w_v * ret_weights[k_r]
            if p_state_ret < prob_skip:
                continue

            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            sR_p = s_val * R_p
            if sR_p > 0.0:
                w_inv = sR_p
                w_A = w_inv / annuity_factor_cur
                mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_cur
                euler_sum += p_state_ret * prob_death * mu_bequest * R_p
            else:
                # Bankruptcy: heirs inherit nothing; alive branch uses w_inv = 0.
                w_inv = 0.0

            for k_eta in range(len(eta_nodes)):
                w_eta = eta_weights[k_eta]
                if w_eta < prob_skip:
                    continue

                z_next = rho * z_cur + eta_nodes[k_eta]
                p_out_base = p_state_ret * w_eta
                det_z_eta = base_det_z * exp_eta[k_eta]

                if use_pension_next:
                    income_next_const = _interp_z_row_value(z_next, z_grid, pension_row)
                else:
                    income_next_const = 0.0

                for i_e in range(len(eps_nodes)):
                    weight = p_out_base * eps_weights[i_e]
                    if weight < prob_skip:
                        continue

                    if use_pension_next:
                        income_next = income_next_const
                    else:
                        y_gross_next = det_z_eta * exp_eps[i_e]
                        income_next = scalar_disposable_income(y_gross_next)

                    x_next = w_inv + income_next
                    c_next, _mpc = _interp_continuation_c_mpc(
                        c_next_full,
                        z_next,
                        s_next_0,
                        s_next_1,
                        s_next_2,
                        x_next,
                        wealth_grid,
                        z_grid,
                        dz,
                        state_bracket_shift,
                        state_bracket_L_inv,
                        grids_0,
                        grids_1,
                        grids_2,
                        N1,
                        N2,
                        min_consumption,
                    )
                    mu_alive = c_next ** (-gamma)
                    euler_sum += weight * psi * mu_alive * R_p

    return euler_sum


@njit(parallel=True, fastmath=True)
def _evaluate_age_errors(
    household_idx: np.ndarray,
    z_age: np.ndarray,
    state_coords_age: np.ndarray,
    c_age: np.ndarray,
    alpha_s_age: np.ndarray,
    alpha_b_age: np.ndarray,
    savings_age: np.ndarray,
    age: int,
    retire_age: int,
    c_next_full: np.ndarray,
    wealth_grid: np.ndarray,
    z_grid: np.ndarray,
    dz: float,
    log_det_next: float,
    pension_row: np.ndarray,
    survival_row: np.ndarray,
    rho: float,
    eta_nodes: np.ndarray,
    eta_weights: np.ndarray,
    eps_nodes: np.ndarray,
    eps_weights: np.ndarray,
    v_nodes: np.ndarray,
    v_weights: np.ndarray,
    M_v_nodes: np.ndarray,
    const_r: np.ndarray,
    A_r: np.ndarray,
    Phi_0_state: np.ndarray,
    Phi_11: np.ndarray,
    state_bracket_shift: np.ndarray,
    state_bracket_L_inv: np.ndarray,
    grids_0: np.ndarray,
    grids_1: np.ndarray,
    grids_2: np.ndarray,
    N1: int,
    N2: int,
    exp_ret_bill: np.ndarray,
    exp_ret_stock: np.ndarray,
    exp_ret_bond: np.ndarray,
    ret_weights: np.ndarray,
    gamma: float,
    beta: float,
    b_bar: int,
    y_1_idx: int,
    spr_idx: int,
    kink_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(household_idx)
    ee = np.empty(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    is_constrained = np.zeros(n, dtype=np.bool_)
    use_pension_next = age == retire_age - 1

    for j in prange(n):
        i = household_idx[j]
        c_t = c_age[i]
        s_t = savings_age[i]
        x_t = c_t + s_t
        # Constraint kink: savings≈0 means the borrowing limit (savings>=0) binds and
        # the FOC has KKT slack rather than holding with equality. Detected as a
        # share-of-cash-on-hand below kink_tol so the test is scale-free.
        if x_t > 1e-12 and s_t / x_t < kink_tol:
            is_constrained[j] = True
        if c_t <= 0.0:
            ee[j] = np.nan
            valid[j] = False
            continue

        state_cur = state_coords_age[i]
        z_cur = z_age[i]
        if age >= retire_age:
            e_sum = _compute_euler_sum_retirement_continuous(
                alpha_s_age[i],
                alpha_b_age[i],
                savings_age[i],
                z_cur,
                state_cur,
                c_next_full,
                wealth_grid,
                z_grid,
                dz,
                pension_row,
                survival_row,
                v_nodes,
                v_weights,
                M_v_nodes,
                const_r,
                A_r,
                Phi_0_state,
                Phi_11,
                state_bracket_shift,
                state_bracket_L_inv,
                grids_0,
                grids_1,
                grids_2,
                N1,
                N2,
                exp_ret_bill,
                exp_ret_stock,
                exp_ret_bond,
                ret_weights,
                gamma,
                b_bar,
                y_1_idx,
                spr_idx,
            )
        else:
            e_sum = _compute_euler_sum_working_continuous(
                alpha_s_age[i],
                alpha_b_age[i],
                savings_age[i],
                z_cur,
                state_cur,
                c_next_full,
                wealth_grid,
                z_grid,
                dz,
                log_det_next,
                pension_row,
                survival_row,
                rho,
                eta_nodes,
                eta_weights,
                eps_nodes,
                eps_weights,
                v_nodes,
                v_weights,
                M_v_nodes,
                const_r,
                A_r,
                Phi_0_state,
                Phi_11,
                state_bracket_shift,
                state_bracket_L_inv,
                grids_0,
                grids_1,
                grids_2,
                N1,
                N2,
                exp_ret_bill,
                exp_ret_stock,
                exp_ret_bond,
                ret_weights,
                gamma,
                b_bar,
                y_1_idx,
                spr_idx,
                use_pension_next,
            )

        if beta * e_sum <= 0.0:
            ee[j] = np.nan
            valid[j] = False
        else:
            c_implied = (beta * e_sum) ** (-1.0 / gamma)
            ee[j] = 1.0 - c_implied / c_t
            valid[j] = True

    return ee, valid, is_constrained


def _simulate_bundle_window(ctx: EulerBundleContext,
                            n_simulations: int,
                            seed: int,
                            return_draw_mode: str,
                            initial_x: float | None,
                            initial_wealth: float | None,
                            initial_wealth_distribution: str | None,
                            initial_wealth_normal_std: float,
                            initial_z: str,
                            initial_z_normal_std: float,
                            initial_state: str,
                            warm_start: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    model = ctx.model
    pc = ctx.pc_policy
    t0 = ctx.first_solved_t
    t1 = ctx.last_solved_t
    start_age = int(ctx.ages[t0])
    ages_win = ctx.ages[t0:t1 + 1]
    C_win = np.ascontiguousarray(ctx.C[t0:t1 + 1])
    S_win = np.ascontiguousarray(ctx.S[t0:t1 + 1])
    B_win = np.ascontiguousarray(ctx.B[t0:t1 + 1])
    log_det_win = np.ascontiguousarray(pc.log_det_profile[t0:t1 + 1])
    survival_win = np.ascontiguousarray(pc.survival_probs_2d[t0:t1 + 1])
    retire_age_idx_model = int(model.retire_age - model.start_age)
    retire_age_idx_win = int(model.retire_age - start_age)

    if return_draw_mode not in ("monte_carlo", "quadrature"):
        raise ValueError("return_draw_mode must be 'monte_carlo' or 'quadrature'")

    rng = np.random.default_rng(seed)

    mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1
    mu_eps2_eff = -(model.pe / (1.0 - model.pe)) * model.mu_eps1

    pension_row = np.asarray(pc.pension_after_tax[retire_age_idx_model, :], dtype=float)

    if warm_start is not None:
        init_z_val = np.asarray(warm_start["z"], dtype=np.float64)
        init_s_coords = np.ascontiguousarray(warm_start["state_coords"], dtype=np.float64)
        init_x = np.asarray(warm_start["x"], dtype=np.float64)
        initial_income_arr = np.asarray(warm_start["income"], dtype=np.float64)
        if init_z_val.shape[0] != n_simulations or init_s_coords.shape[0] != n_simulations or init_x.shape[0] != n_simulations:
            raise ValueError("Warm-start arrays must all have length n_simulations.")
    else:
        if initial_z == "normal":
            init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=initial_z_normal_std)
            init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
        elif initial_z == "stationary":
            var_eta = (model.pz * (model.sigma_eta1**2 + model.mu_eta1**2)
                       + (1.0 - model.pz) * (model.sigma_eta2**2 + mu_eta2_eff**2))
            sigma_z = np.sqrt(var_eta / (1.0 - model.rho**2))
            init_z_probs = _normal_bin_probs(pc.z_grid, mean=0.0, std=sigma_z)
            init_z_idx = rng.choice(pc.n_z, size=n_simulations, p=init_z_probs).astype(np.int32)
        else:
            init_z_idx = initialize_states(n_simulations, pc.n_z, pc.Pi_z, initial_z, rng)
        init_z_val = pc.z_grid[init_z_idx].astype(np.float64)

        init_state_idx = _resolve_initial_state_indices(n_simulations, pc, initial_state, rng)
        if start_age < model.retire_age:
            init_eps_uniform = rng.uniform(size=n_simulations)
            init_eps_normal = rng.standard_normal(size=n_simulations)
            init_eps_val = np.where(
                init_eps_uniform < model.pe,
                model.mu_eps1 + model.sigma_eps1 * init_eps_normal,
                mu_eps2_eff + model.sigma_eps2 * init_eps_normal,
            )
            init_y_gross = np.exp(pc.log_det_profile[t0] + init_z_val + init_eps_val)
            initial_income_arr = disposable_income_working(init_y_gross)
        else:
            initial_income_arr = np.interp(init_z_val, pc.z_grid, pension_row)

        if initial_x is None:
            init_wealth_arr = _initialize_initial_wealth(
                n_simulations=n_simulations,
                initial_wealth=initial_wealth,
                initial_wealth_distribution=initial_wealth_distribution,
                initial_wealth_normal_std=initial_wealth_normal_std,
                rng=rng,
            )
            init_x = init_wealth_arr + initial_income_arr
        else:
            init_x = np.full(n_simulations, float(initial_x), dtype=float)

        init_s_coords = np.ascontiguousarray(pc.state_grid[init_state_idx, :])

    if np.any(init_x < 0.0):
        raise ValueError("Constructed initial_x must be non-negative.")

    uniform_draws = rng.uniform(size=(n_simulations, len(ages_win), 4))
    n_state = int(model.n_state)
    n_ret = int(model.n_ret)
    normal_draws = rng.standard_normal(size=(n_simulations, len(ages_win), n_ret + 2 + n_state))

    if return_draw_mode == "monte_carlo":
        ret_factor_arr = _build_return_factor(model.Sigma_r_cond)
        use_mc_returns = True
    else:
        ret_factor_arr = np.zeros((n_ret, n_ret), dtype=float)
        use_mc_returns = False

    results = simulate_lifecycle_core(
        C_mat=C_win,
        S_mat=S_win,
        B_mat=B_win,
        wealth_grid=np.ascontiguousarray(pc.wealth_grid),
        z_grid=np.ascontiguousarray(pc.z_grid),
        ret_nodes=np.ascontiguousarray(pc.ret_nodes),
        ret_weights=np.ascontiguousarray(pc.ret_weights),
        ret_factor=np.ascontiguousarray(ret_factor_arr),
        log_det_profile=log_det_win,
        avg_det=float(pc.avg_det),
        pension_at_z_grid=np.ascontiguousarray(pension_row),
        survival_probs_2d=survival_win,
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
        start_age=start_age,
        retire_age_idx=retire_age_idx_win,
        constrained=bool(model.constrained),
        use_mc_returns=use_mc_returns,
        initial_z=np.ascontiguousarray(init_z_val),
        initial_s=init_s_coords,
        initial_x=np.ascontiguousarray(init_x),
        initial_income=np.ascontiguousarray(initial_income_arr),
        uniform_draws=np.ascontiguousarray(uniform_draws),
        normal_draws=np.ascontiguousarray(normal_draws),
        state_grids_0=np.ascontiguousarray(pc.state_bracket_grids[0]),
        state_grids_1=np.ascontiguousarray(pc.state_bracket_grids[1]),
        state_grids_2=np.ascontiguousarray(pc.state_bracket_grids[2]),
        state_bracket_shift=np.ascontiguousarray(pc.state_bracket_shift),
        state_bracket_L_inv=np.ascontiguousarray(pc.state_bracket_L_inv),
        L_ss=np.ascontiguousarray(np.linalg.cholesky(0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T))),
        Phi_0_state=np.ascontiguousarray(model.Phi_0_state),
        Phi_11=np.ascontiguousarray(model.Phi_11),
        const_r=np.ascontiguousarray(pc.const_r),
        A_r=np.ascontiguousarray(pc.A_r),
        M_matrix=np.ascontiguousarray(model.M),
    )

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
        sim_state_idx,
        sim_state_coords,
        sim_alive,
        death_age,
        estate_at_death,
    ) = results

    return {
        "x": sim_x,
        "c": sim_c,
        "savings": sim_savings,
        "alpha_s": sim_alpha_s,
        "alpha_b": sim_alpha_b,
        "R_port": sim_R_port,
        "income": sim_income,
        "estate": sim_estate,
        "z": sim_z,
        "z_idx": sim_z_idx,
        "state_idx": sim_state_idx,
        "state_coords": sim_state_coords,
        "alive": sim_alive,
        "death_age": death_age,
        "estate_at_death": estate_at_death,
        "ages": ages_win.copy(),
    }


def _age_sample_indices(alive_mask: np.ndarray,
                        max_households: int,
                        rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(alive_mask)
    if idx.size <= max_households:
        return idx.astype(np.int64, copy=False)
    pick = rng.choice(idx, size=max_households, replace=False)
    pick.sort()
    return pick.astype(np.int64, copy=False)


def _maybe_build_warm_start(ctx: EulerBundleContext, args: argparse.Namespace) -> tuple[dict[str, np.ndarray] | None, str | None]:
    start_age = int(ctx.ages[ctx.first_solved_t])
    if start_age <= int(ctx.model.start_age):
        return None, None

    source_path = Path(args.model_bundle)
    if not source_path.exists():
        return None, None

    C_src, S_src, B_src, _diag_src, meta_src = load_policy_bundle(source_path)
    solved_src = _infer_solved_mask(C_src, meta_src, ctx.ages)
    warm_t = start_age - int(ctx.model.start_age)
    if warm_t < 0 or warm_t >= C_src.shape[0] or not bool(solved_src[warm_t]):
        return None, None

    disc_raw = _extract_disc_config(meta_src)
    if not disc_raw:
        return None, None

    pc_src = Precompute(ctx.model, _build_disc_config(disc_raw), verbose=False)
    sim_src = simulate_lifecycle(
        C_src,
        S_src,
        B_src,
        pc_src,
        ctx.model,
        n_simulations=args.n_simulations,
        initial_x=args.initial_x,
        initial_wealth=args.initial_wealth,
        initial_wealth_distribution=args.initial_wealth_distribution,
        initial_wealth_normal_std=args.initial_wealth_normal_std,
        initial_z=args.initial_z,
        initial_z_normal_std=args.initial_z_normal_std,
        initial_state=args.initial_state,
        seed=args.seed,
        return_draw_mode=args.return_draw_mode,
        verbose=False,
    )

    alive_idx = np.flatnonzero(sim_src["alive"][:, warm_t])
    if alive_idx.size == 0:
        return None, None

    rng = np.random.default_rng(args.seed + 2)
    replace = alive_idx.size < args.n_simulations
    chosen = rng.choice(alive_idx, size=args.n_simulations, replace=replace)

    warm = {
        "x": np.asarray(sim_src["x"][chosen, warm_t], dtype=float),
        "z": np.asarray(sim_src["z"][chosen, warm_t], dtype=float),
        "income": np.asarray(sim_src["income"][chosen, warm_t], dtype=float),
        "state_coords": np.asarray(sim_src["state_coords"][chosen, warm_t, :], dtype=float),
    }
    return warm, source_path.name


def _summarize_ee(age: int,
                  phase: str,
                  ee: np.ndarray,
                  valid: np.ndarray,
                  is_constrained: np.ndarray,
                  n_alive: int,
                  n_eval_target: int) -> dict[str, Any]:
    vals = ee[valid]
    out: dict[str, Any] = {
        "age": age,
        "phase": phase,
        "n_alive": int(n_alive),
        "n_eval_target": int(n_eval_target),
        "n_eval": int(ee.size),
        "n_valid": int(vals.size),
        "n_invalid": int(ee.size - vals.size),
        "n_constrained": int(np.sum(is_constrained)),
        "n_constrained_valid": int(np.sum(is_constrained & valid)),
    }
    nan_block = {
        "mean_log10_abs_ee": np.nan,
        "p99_log10_abs_ee": np.nan,
        "max_log10_abs_ee": np.nan,
        "mean_abs_ee_pct": np.nan,
        "p95_abs_ee_pct": np.nan,
        "max_abs_ee_pct": np.nan,
    }
    if vals.size == 0:
        out.update(nan_block)
        out.update({k + "_unc": np.nan for k in nan_block})
        out["n_unconstrained_valid"] = 0
        return out

    def _stats(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {k: np.nan for k in nan_block}
        abs_ee = np.abs(arr)
        log_abs = np.log10(np.maximum(abs_ee, 1e-16))
        return {
            "mean_log10_abs_ee": float(np.mean(log_abs)),
            "p99_log10_abs_ee": float(np.percentile(log_abs, 99.0)),
            "max_log10_abs_ee": float(np.max(log_abs)),
            "mean_abs_ee_pct": float(100.0 * np.mean(abs_ee)),
            "p95_abs_ee_pct": float(100.0 * np.percentile(abs_ee, 95.0)),
            "max_abs_ee_pct": float(100.0 * np.max(abs_ee)),
        }

    out.update(_stats(vals))
    unc_mask = valid & ~is_constrained
    vals_unc = ee[unc_mask]
    out["n_unconstrained_valid"] = int(vals_unc.size)
    out.update({k + "_unc": v for k, v in _stats(vals_unc).items()})
    return out


def _phase_gate(rows: list[dict[str, Any]], phase: str, grade: str,
                cell_set: str = "all") -> dict[str, Any]:
    """Compute pass/fail for a phase × grade.

    cell_set controls which subset of cells the gate sees:
      - "all": every valid cell (current behaviour, includes constraint-binding cells
        whose EE reflects KKT slack rather than discretization error).
      - "unconstrained": only cells where savings/x >= kink_tol (the FOC-interior
        subset). This is the convention HARK and AFV-RR (2006) use because the
        constrained branch has a slack term that is not a discretization error.
    """
    if cell_set == "all":
        valid_key = "n_valid"
        mean_key, max_key = "mean_log10_abs_ee", "max_log10_abs_ee"
    elif cell_set == "unconstrained":
        valid_key = "n_unconstrained_valid"
        mean_key, max_key = "mean_log10_abs_ee_unc", "max_log10_abs_ee_unc"
    else:
        raise ValueError(f"Unknown cell_set: {cell_set}")

    phase_rows = [r for r in rows if r["phase"] == phase and r.get(valid_key, 0) > 0]
    if not phase_rows:
        return {
            "phase": phase,
            "grade": grade,
            "cell_set": cell_set,
            "passes": False,
            "mean_log10_abs_ee": np.nan,
            "max_log10_abs_ee": np.nan,
            "mean_gate": np.nan,
            "max_gate": np.nan,
        }

    mean_log = float(np.nanmean([r[mean_key] for r in phase_rows]))
    max_log = float(np.nanmax([r[max_key] for r in phase_rows]))

    if phase == "working":
        mean_gate = -4.0 if grade == "publication" else -5.0
    else:
        mean_gate = -4.5 if grade == "publication" else -5.5
    max_gate = -3.0 if grade == "publication" else -4.0

    return {
        "phase": phase,
        "grade": grade,
        "cell_set": cell_set,
        "passes": bool(mean_log < mean_gate and max_log < max_gate),
        "mean_log10_abs_ee": mean_log,
        "max_log10_abs_ee": max_log,
        "mean_gate": mean_gate,
        "max_gate": max_gate,
    }


def _markdown_report(ctx: EulerBundleContext,
                     sim: dict[str, Any],
                     rows: list[dict[str, Any]],
                     gates: list[dict[str, Any]],
                     args: argparse.Namespace) -> str:
    solved_ages = ctx.ages[ctx.solved_mask]
    age_start = int(solved_ages[0])
    age_end = int(solved_ages[-1])
    eval_disc = ctx.disc_eval
    policy_disc = ctx.disc_policy

    lines: list[str] = []
    lines.append(f"# Euler Error Report: `{ctx.label}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Bundle: `{ctx.path}`")
    lines.append(f"- Solved window: ages `{age_start}-{age_end}`")
    lines.append(f"- Simulation households: `{args.n_simulations}`")
    lines.append(f"- Evaluated households per age cap: `{args.eval_households_per_age}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- Return draw mode: `{args.return_draw_mode}`")
    lines.append(f"- Initial z: `{args.initial_z}`")
    lines.append(f"- Initial state: `{args.initial_state}`")
    lines.append(f"- Partial init mode: `{args.partial_init_mode}`")
    if sim.get("warm_start_source"):
        lines.append(f"- Warm-start source bundle: `{sim['warm_start_source']}`")
    lines.append(f"- Eval mode: `{args.eval_mode}`")
    lines.append(f"- Policy quadrature: `ret={policy_disc.n_ret_nodes_1d}`, `state={policy_disc.n_state_quad_nodes}`, `eta={policy_disc.n_eta_nodes}`, `eps={policy_disc.n_eps_nodes}`")
    lines.append(f"- Eval quadrature: `ret={eval_disc.n_ret_nodes_1d}`, `state={eval_disc.n_state_quad_nodes}`, `eta={eval_disc.n_eta_nodes}`, `eps={eval_disc.n_eps_nodes}`")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("`cell_set=all` includes constraint-binding cells (savings≈0). Their EE reflects KKT slack on the no-borrowing constraint, not discretization error. `cell_set=unconstrained` is the FOC-interior subset (savings/x ≥ kink_tol) — this is the convention HARK and AFV-RR (2006) use to gate solver accuracy.")
    lines.append("")
    lines.append("| cell_set | phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for g in gates:
        lines.append(
            f"| {g['cell_set']} | {g['phase']} | {g['grade']} | {'yes' if g['passes'] else 'no'} | "
            f"{g['mean_log10_abs_ee']:.3f} | < {g['mean_gate']:.1f} | "
            f"{g['max_log10_abs_ee']:.3f} | < {g['max_gate']:.1f} |"
        )
    lines.append("")

    lines.append("## By Age")
    lines.append("")
    lines.append(f"`*_unc` columns restrict to the unconstrained-cell subset (savings/x ≥ kink_tol = {args.kink_tol:g}). `n_cstr` = number of constraint-binding cells in the age sample.")
    lines.append("")
    lines.append("| age | phase | alive | eval | valid | n_cstr | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean log10|EE| (unc) | p99 log10|EE| (unc) | max log10|EE| (unc) | max |EE| % | max |EE| % (unc) |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['age']} | {r['phase']} | {r['n_alive']} | {r['n_eval']} | {r['n_valid']} | {r['n_constrained']} | "
            f"{r['mean_log10_abs_ee']:.3f} | {r['p99_log10_abs_ee']:.3f} | {r['max_log10_abs_ee']:.3f} | "
            f"{r['mean_log10_abs_ee_unc']:.3f} | {r['p99_log10_abs_ee_unc']:.3f} | {r['max_log10_abs_ee_unc']:.3f} | "
            f"{r['max_abs_ee_pct']:.4f} | {r['max_abs_ee_pct_unc']:.4f} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    if age_start > int(ctx.model.start_age):
        lines.append(
            f"- This is a partial-window simulation starting at age `{age_start}` because the bundle only solves ages `{age_start}-{age_end}`."
        )
    lines.append(
        "- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule."
    )
    lines.append(
        f"- A cell is flagged `is_constrained` when `savings/x < kink_tol = {args.kink_tol:g}`. At such cells the FOC has KKT slack (the agent would want savings<0 but the savings≥0 constraint binds), so `1 - c_implied/c_policy` measures slack, not discretization error."
    )
    lines.append(
        "- The working/retirement publication-grade gates follow `docs/GRID_CONVERGENCE_CRITERIA.md`. The `unconstrained` cell-set rows are the gate the literature reports (HARK, AFV-RR 2006); the `all` rows are kept for backwards compatibility."
    )
    return "\n".join(lines) + "\n"


def run_euler_diagnostic(args: argparse.Namespace) -> tuple[EulerBundleContext, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ctx = _load_bundle_context(
        bundle_path=Path(args.bundle),
        model_bundle=Path(args.model_bundle),
        eval_mode=args.eval_mode,
        ret_override=tuple(args.eval_ret_nodes) if args.eval_ret_nodes else None,
        state_override=tuple(args.eval_state_nodes) if args.eval_state_nodes else None,
        eta_override=args.eval_eta_nodes,
        eps_override=args.eval_eps_nodes,
    )

    warm_start = None
    warm_start_source = None
    if args.partial_init_mode == "warm_start":
        warm_start, warm_start_source = _maybe_build_warm_start(ctx, args)
    sim = _simulate_bundle_window(
        ctx=ctx,
        n_simulations=args.n_simulations,
        seed=args.seed,
        return_draw_mode=args.return_draw_mode,
        initial_x=args.initial_x,
        initial_wealth=args.initial_wealth,
        initial_wealth_distribution=args.initial_wealth_distribution,
        initial_wealth_normal_std=args.initial_wealth_normal_std,
        initial_z=args.initial_z,
        initial_z_normal_std=args.initial_z_normal_std,
        initial_state=args.initial_state,
        warm_start=warm_start,
    )
    sim["warm_start_source"] = warm_start_source

    pc_eval = ctx.pc_eval
    model = ctx.model
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 1)

    pension_row = np.ascontiguousarray(
        pc_eval.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    N1 = len(pc_eval.state_bracket_grids[1])
    N2 = len(pc_eval.state_bracket_grids[2])

    ages_win = sim["ages"]
    for t in range(len(ages_win) - 1):
        age = int(ages_win[t])
        alive_mask = sim["alive"][:, t]
        sample_idx = _age_sample_indices(alive_mask, args.eval_households_per_age, rng)
        phase = "retirement" if age >= model.retire_age else "working"

        ee, valid, is_constrained = _evaluate_age_errors(
            sample_idx,
            np.ascontiguousarray(sim["z"][:, t]),
            np.ascontiguousarray(sim["state_coords"][:, t, :]),
            np.ascontiguousarray(sim["c"][:, t]),
            np.ascontiguousarray(sim["alpha_s"][:, t]),
            np.ascontiguousarray(sim["alpha_b"][:, t]),
            np.ascontiguousarray(sim["savings"][:, t]),
            age,
            int(model.retire_age),
            np.ascontiguousarray(ctx.C[ctx.first_solved_t + t + 1]),
            np.ascontiguousarray(pc_eval.wealth_grid),
            np.ascontiguousarray(pc_eval.z_grid),
            float(pc_eval.dz),
            float(pc_eval.log_det_profile[ctx.first_solved_t + t + 1]),
            pension_row,
            np.ascontiguousarray(pc_eval.survival_probs_2d[ctx.first_solved_t + t, :]),
            float(model.rho),
            np.ascontiguousarray(pc_eval.eta_nodes),
            np.ascontiguousarray(pc_eval.eta_weights),
            np.ascontiguousarray(pc_eval.eps_nodes),
            np.ascontiguousarray(pc_eval.eps_weights),
            np.ascontiguousarray(pc_eval.v_nodes),
            np.ascontiguousarray(pc_eval.v_weights),
            np.ascontiguousarray(pc_eval.M_v_nodes),
            np.ascontiguousarray(pc_eval.const_r),
            np.ascontiguousarray(pc_eval.A_r),
            np.ascontiguousarray(model.Phi_0_state),
            np.ascontiguousarray(model.Phi_11),
            np.ascontiguousarray(pc_eval.state_bracket_shift),
            np.ascontiguousarray(pc_eval.state_bracket_L_inv),
            np.ascontiguousarray(pc_eval.state_bracket_grids[0]),
            np.ascontiguousarray(pc_eval.state_bracket_grids[1]),
            np.ascontiguousarray(pc_eval.state_bracket_grids[2]),
            int(N1),
            int(N2),
            np.ascontiguousarray(pc_eval.exp_ret_bill),
            np.ascontiguousarray(pc_eval.exp_ret_stock),
            np.ascontiguousarray(pc_eval.exp_ret_bond),
            np.ascontiguousarray(pc_eval.ret_weights),
            float(model.gamma),
            float(model.beta),
            int(model.b_bar),
            int(model.y_1_index_in_state),
            int(model.spr_index_in_state),
            float(args.kink_tol),
        )

        rows.append(_summarize_ee(
            age=age,
            phase=phase,
            ee=ee,
            valid=valid,
            is_constrained=is_constrained,
            n_alive=int(np.sum(alive_mask)),
            n_eval_target=int(args.eval_households_per_age),
        ))

    gates = [
        _phase_gate(rows, "working", "publication", cell_set="all"),
        _phase_gate(rows, "retirement", "publication", cell_set="all"),
        _phase_gate(rows, "working", "welfare", cell_set="all"),
        _phase_gate(rows, "retirement", "welfare", cell_set="all"),
        _phase_gate(rows, "working", "publication", cell_set="unconstrained"),
        _phase_gate(rows, "retirement", "publication", cell_set="unconstrained"),
        _phase_gate(rows, "working", "welfare", cell_set="unconstrained"),
        _phase_gate(rows, "retirement", "welfare", cell_set="unconstrained"),
    ]
    return ctx, sim, rows, gates


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", help="Policy bundle to evaluate.")
    p.add_argument(
        "--model-bundle",
        required=True,
        help="Bundle with full run_config metadata used to reconstruct the model.",
    )
    p.add_argument(
        "--eval-mode",
        choices=("same", "next_finer", "double"),
        default="next_finer",
        help="How to build the independent evaluation quadrature.",
    )
    p.add_argument("--eval-ret-nodes", nargs="+", type=int, default=None)
    p.add_argument("--eval-state-nodes", nargs="+", type=int, default=None)
    p.add_argument("--eval-eta-nodes", type=int, default=None)
    p.add_argument("--eval-eps-nodes", type=int, default=None)
    p.add_argument("--n-simulations", type=int, default=1000)
    p.add_argument("--eval-households-per-age", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--return-draw-mode",
        choices=("monte_carlo", "quadrature"),
        default="monte_carlo",
    )
    p.add_argument("--initial-x", type=float, default=None)
    p.add_argument("--initial-wealth", type=float, default=0.1)
    p.add_argument(
        "--initial-wealth-distribution",
        choices=("normal",),
        default=None,
    )
    p.add_argument("--initial-wealth-normal-std", type=float, default=0.0)
    p.add_argument(
        "--initial-z",
        choices=("median", "stationary", "normal"),
        default="median",
    )
    p.add_argument("--initial-z-normal-std", type=float, default=0.652)
    p.add_argument(
        "--initial-state",
        choices=("median", "stationary"),
        default="median",
    )
    p.add_argument(
        "--partial-init-mode",
        choices=("centered", "warm_start"),
        default="centered",
        help=(
            "For partial bundles, either start directly from the specified age-window "
            "initial condition (`centered`) or warm-start from the model bundle's "
            "simulated cross-section at the first solved age (`warm_start`)."
        ),
    )
    p.add_argument("--markdown-out", type=str, default=None)
    p.add_argument(
        "--kink-tol",
        type=float,
        default=1e-3,
        help=(
            "Threshold on savings/x below which a cell is treated as constraint-binding "
            "and excluded from the `unconstrained` cell-set. EE there reflects KKT slack "
            "rather than discretization error; default 1e-3 matches HARK's convention."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    ctx, sim, rows, gates = run_euler_diagnostic(args)

    print(f"Bundle: {ctx.label}")
    print(f"Solved window: {int(ctx.ages[ctx.first_solved_t])}-{int(ctx.ages[ctx.last_solved_t])}")
    print(f"Simulated households: {args.n_simulations}")
    print(f"Evaluated households per age cap: {args.eval_households_per_age}")
    print("")
    for g in gates:
        print(
            f"{g['cell_set']:>13} | {g['phase']:>10} | {g['grade']:<11} | pass={str(g['passes']).lower():<5} "
            f"| mean log10|EE|={g['mean_log10_abs_ee']:.3f} (gate {g['mean_gate']:.1f}) "
            f"| max log10|EE|={g['max_log10_abs_ee']:.3f} (gate {g['max_gate']:.1f})"
        )

    if args.markdown_out:
        report = _markdown_report(ctx, sim, rows, gates, args)
        out_path = Path(args.markdown_out)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nWrote markdown report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
