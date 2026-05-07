"""verify/ee_simpath.py -- Sim-path consumption-Euler diagnostic for saved bundles.

Ports scripts/diagnostics/_diag_euler_errors.py from the Numba `main` branch to
the JAX branch with the right economics:
  - 4-D state vector (cy, spr, rtb, y_1)
  - rtb-as-state (log_R_bill = s_next[rtb_idx], not from return draws)
  - CCV log-portfolio returns (matches solver._ccv_log_return_and_grad)
  - Unbounded leverage (no alpha_s/alpha_b clipping anywhere)

Pipeline:
  1. Load saved bundle; rehydrate model + solver-side precompute (pc_solver).
  2. Build eval-rule precompute (pc_eval) with bumped quadrature per --eval-mode.
  3. Run simulate_lifecycle with the bundle's policy on pc_solver to get a
     household panel of (z, state_coords, c, alpha_s, alpha_b, savings, alive).
  4. Per age t in the solved window, subsample alive households (default 256).
  5. JIT'd per-age kernel re-evaluates the FOC at sim-path inputs against
     pc_eval's quadrature; reads e_sum (6th return of retirement_foc_jac_ccv /
     working_foc_jac_ccv) which is the combined (alive+bequest) Euler sum:
         e_sum = E[ (psi*c'^(-gamma) + (1-psi)*b'(W')) * R_p ]
  6. Compute consumption EE: c_implied = (beta * e_sum) ** (-1/gamma); then
     EE = 1 - c_implied / c_t. Floor log10|EE| at -16 before aggregating.
  7. Aggregate per-age, per-phase; apply pass/fail gates; write JSON + markdown.

Usage
-----
    python verify/ee_simpath.py <bundle> [--eval-mode {same,next_finer,double}] \
        [--n-simulations 5000] [--eval-households-per-age 256] [--seed 42]

Output
------
    <bundle>/ee_simpath.json
    <bundle>/ee_simpath.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap

from lifecycle.model import (
    DELTA_BEQUEST,
    DiscretizationConfig,
    SolverConfig,
    annuity_factor,
)
from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.simulation import simulate_lifecycle
from lifecycle.solver import (
    _build_step_log_returns,
    _build_step_state_brackets,
    _pc_to_jnp,
    bracket_uniform,
    retirement_foc_jac_ccv,
    working_foc_jac_ccv,
)
from lifecycle.var import build_nominal_system1_var_config_hardcoded


# Floor for log10|EE| aggregation: c_implied ~= c gives -inf which would
# poison means/medians. Matches main's np.log10(np.maximum(abs_ee, 1e-16)).
LOG10_EE_FLOOR = -16.0


# =============================================================================
# Bundle loading + config rehydration (mirrors verify/ee_residuals.py)
# =============================================================================

def _resolve_bundle_path(bundle_arg: str) -> Path:
    p = Path(bundle_arg)
    if p.is_dir():
        return p
    p2 = Path("saved_runs") / bundle_arg
    if p2.is_dir():
        return p2
    raise FileNotFoundError(f"Bundle not found. Tried: {p}, {p2}")


def _list_to_tuple_recursive(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_list_to_tuple_recursive(x) for x in v)
    return v


def _rehydrate_disc_config(d: dict) -> DiscretizationConfig:
    tuple_fields = {
        "state_grid_sizes",
        "n_state_quad_nodes",
        "n_ret_nodes_1d",
        "state_n_stds",
        "ret_lobatto_Z",
        "state_lobatto_Z",
    }
    valid = set(DiscretizationConfig._fields)
    kwargs = {}
    for k, v in d.items():
        if k not in valid:
            continue
        kwargs[k] = _list_to_tuple_recursive(v) if k in tuple_fields else v
    return DiscretizationConfig(**kwargs)


def _rehydrate_solver_config(d: dict | None) -> SolverConfig:
    if d is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    return SolverConfig(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# Eval-rule construction (mirrors main's _build_eval_disc, axis-agnostic)
# =============================================================================

def _coerce_K_per_axis(value: Any, n_axes: int) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n_axes
    seq = list(value)
    if len(seq) != n_axes:
        raise ValueError(f"Expected length-{n_axes} sequence, got {seq!r}")
    return tuple(int(v) for v in seq)


def _normalize_lobatto_axes(value: Any, n_axes: int) -> tuple[Any, ...]:
    """Return length-n_axes tuple of (None | float). Accepts None / scalar /
    sequence per the canonical convention."""
    if value is None:
        return (None,) * n_axes
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value),) * n_axes
    seq = list(value)
    if len(seq) != n_axes:
        raise ValueError(f"lobatto_Z length {len(seq)} != {n_axes}")
    return tuple(None if v is None else float(v) for v in seq)


def _build_eval_disc(base: DiscretizationConfig, model, mode: str) -> DiscretizationConfig:
    """Build an eval-rule DiscretizationConfig from a solver-config base.

    Bumps quadrature K's per `mode`; preserves grid sizes (state_grid_sizes,
    n_z, n_wealth, n_savings) and grid bounds. Lobatto-Z axes are propagated
    pass-through (Z unchanged); the K-vs-Z compatibility logic from main's
    _adjust_lobatto_axis is omitted for now since the JAX-branch's Lobatto
    implementation accepts any K >= 3 for a given Z (the (3,5,7) restriction
    in main was a Numba-side closed-form ceiling).
    """
    n_state = int(model.n_state)
    n_ret = int(model.n_ret)
    ret_base = _coerce_K_per_axis(base.n_ret_nodes_1d, n_ret)
    state_base = _coerce_K_per_axis(base.n_state_quad_nodes, n_state)
    eta_base = int(base.n_eta_nodes)
    eps_base = int(base.n_eps_nodes)

    if mode == "same":
        ret_eval = ret_base
        state_eval = state_base
        eta_eval = eta_base
        eps_eval = eps_base
    elif mode == "next_finer":
        ret_eval = tuple(max(1, k + 2) for k in ret_base)
        state_eval = tuple(max(1, k + 1) for k in state_base)
        eta_eval = max(1, eta_base + 2)
        eps_eval = max(1, eps_base + 2)
    elif mode == "double":
        ret_eval = tuple(max(1, 2 * k) for k in ret_base)
        state_eval = tuple(max(1, 2 * k) for k in state_base)
        eta_eval = max(1, 2 * eta_base)
        eps_eval = max(1, 2 * eps_base)
    else:
        raise ValueError(f"Unknown eval mode: {mode!r}")

    return base._replace(
        n_ret_nodes_1d=ret_eval,
        n_state_quad_nodes=state_eval,
        n_eta_nodes=eta_eval,
        n_eps_nodes=eps_eval,
        # ret_lobatto_Z / state_lobatto_Z carried through unchanged
    )


# =============================================================================
# Per-household EE evaluator (retirement)
# =============================================================================

def _build_retirement_ee_kernel(pcj_solver, pcj_eval, model, sc, delta):
    """JIT'd per-age retirement EE kernel.

    Vmaps over a sample of households. Each household:
      - has continuous (z_cur, state_coords) from the simulation
      - reads its (alpha_s, alpha_b, c_t, savings_t) from sim panel
      - the FOC is re-evaluated against pc_eval's quadrature

    Returns ee, valid per household.
    """
    gamma = jnp.float64(model.gamma)
    beta = jnp.float64(model.beta)
    b_bar = jnp.float64(model.b_bar)
    delta_j = jnp.float64(delta)
    min_consumption = jnp.float64(sc.min_consumption)

    @jit
    def per_age(
        z_arr, state_coords_arr,        # (K,), (K, n_state)
        c_arr, alpha_s_arr, alpha_b_arr, savings_arr,
        annuity_arr,                    # (K,)  precomputed by caller
        psi_z_solver_arr,               # (n_z,) survival[t, :]
        pension_z_next_solver_arr,      # (n_z,) pension[t+1, :]
        C_next,                         # (n_z, N_state, n_w) policy at t+1
    ):
        def per_hh(z_cur, state_coords, c_t, alpha_s, alpha_b, savings_t, A_is):
            # --- z bracket against solver's z_grid ---
            iz_lo, frac_z = bracket_uniform(
                z_cur, pcj_solver.z_grid[0], pcj_solver.dz,
                pcj_solver.z_grid.shape[0],
            )
            iz_hi = iz_lo + 1
            psi_z = (1.0 - frac_z) * psi_z_solver_arr[iz_lo] \
                  + frac_z * psi_z_solver_arr[iz_hi]
            pension_next_z = (1.0 - frac_z) * pension_z_next_solver_arr[iz_lo] \
                  + frac_z * pension_z_next_solver_arr[iz_hi]

            # --- state brackets at sim coords against solver state grid,
            #     using EVAL-rule v_nodes for the inner-period state quadrature ---
            s_next, j_corners, w_corners = _build_step_state_brackets(
                state_coords,
                pcj_solver.Phi_0_state,        # model-level (shared)
                pcj_solver.Phi_11,
                pcj_eval.v_nodes,              # EVAL-rule
                pcj_solver.axis_grids,         # solver grid (where policy lives)
                pcj_solver.axis_sizes,
                pcj_solver.corner_offsets,
                pcj_solver.strides,
                pcj_solver.state_bracket_shift,
                pcj_solver.state_bracket_L_inv,
            )
            log_R_bill, log_x_s, log_x_b = _build_step_log_returns(
                state_coords,
                pcj_eval.M_v_nodes,            # EVAL-rule
                pcj_eval.ret_nodes,            # EVAL-rule
                pcj_eval.const_r,              # model-level
                pcj_eval.A_r,
                s_next,
                pcj_solver.rtb_idx,
                pcj_solver.xr_pos,
                pcj_solver.xb_pos,
            )

            # --- c_corners at z: linear blend across solver's z bracket ---
            # j_corners: (n_state_quad_eval, 2^n_state) flat indices into N_state.
            c_lo = C_next[iz_lo, j_corners, :]   # (n_state_quad_eval, n_corners, n_w)
            c_hi = C_next[iz_hi, j_corners, :]
            c_corners_at_z = (1.0 - frac_z) * c_lo + frac_z * c_hi

            # --- Call the FOC; only need 6th output (e_sum, mu-combined Euler sum) ---
            _, _, _, _, _, e_sum = retirement_foc_jac_ccv(
                alpha_s, alpha_b, savings_t, psi_z,
                log_R_bill, log_x_s, log_x_b, pcj_eval.weight_kv_kr,
                w_corners, c_corners_at_z, pcj_solver.wealth_grid,
                pension_next_z, A_is,
                pcj_eval.sigma2_xr, pcj_eval.sigma2_xb, pcj_eval.sigma_xrxb,
                gamma, b_bar, delta_j, min_consumption,
            )

            # --- Consumption EE ---
            # c_implied = (beta * e_sum)^(-1/gamma); ee = 1 - c_implied/c
            beta_e = beta * e_sum
            valid = jnp.logical_and(beta_e > 0.0, c_t > 0.0)
            c_implied = jnp.where(
                valid,
                jnp.power(jnp.where(valid, beta_e, 1.0), -1.0 / gamma),
                jnp.nan,
            )
            ee = jnp.where(valid, 1.0 - c_implied / c_t, jnp.nan)
            return ee, valid

        return vmap(per_hh)(
            z_arr, state_coords_arr, c_arr, alpha_s_arr, alpha_b_arr, savings_arr,
            annuity_arr,
        )

    return per_age


# =============================================================================
# Per-household EE evaluator (working & boundary)
# =============================================================================

def _build_working_ee_kernel(pcj_solver, pcj_eval, model, sc, delta,
                              use_pension_next: bool):
    """JIT'd per-age working / boundary EE kernel.

    use_pension_next=True selects the work->retire boundary (income_next is the
    pension at z_next, broadcast across eps). Otherwise income_next is the
    eval-rule's working_income_next table linearly blended in z_cur.
    """
    gamma = jnp.float64(model.gamma)
    beta = jnp.float64(model.beta)
    b_bar = jnp.float64(model.b_bar)
    delta_j = jnp.float64(delta)
    rho = jnp.float64(model.rho)
    min_consumption = jnp.float64(sc.min_consumption)

    @jit
    def per_age(
        z_arr, state_coords_arr,
        c_arr, alpha_s_arr, alpha_b_arr, savings_arr,
        annuity_arr,
        psi_z_solver_arr,
        pension_z_next_solver_arr,
        income_next_table_eval,         # (n_z, n_eta_eval, n_eps_eval) at age t+1, sourced from pc_eval
        C_next,
    ):
        def per_hh(z_cur, state_coords, c_t, alpha_s, alpha_b, savings_t, A_is):
            # z bracket
            iz_lo, frac_z = bracket_uniform(
                z_cur, pcj_solver.z_grid[0], pcj_solver.dz,
                pcj_solver.z_grid.shape[0],
            )
            iz_hi = iz_lo + 1
            psi_z = (1.0 - frac_z) * psi_z_solver_arr[iz_lo] \
                  + frac_z * psi_z_solver_arr[iz_hi]

            # z_next bracket at eval-rule eta nodes (for c_next interp inside FOC)
            z_next = rho * z_cur + pcj_eval.eta_nodes
            eta_iz_lo, eta_frac_z = vmap(
                bracket_uniform, in_axes=(0, None, None, None),
            )(z_next, pcj_solver.z_grid[0], pcj_solver.dz, pcj_solver.z_grid.shape[0])

            if use_pension_next:
                pension_at_eta = (
                    (1.0 - eta_frac_z) * pension_z_next_solver_arr[eta_iz_lo]
                    + eta_frac_z * pension_z_next_solver_arr[eta_iz_lo + 1]
                )
                income_table_z = (
                    pension_at_eta[:, None]
                    * jnp.ones_like(pcj_eval.eps_weights)[None, :]
                )
            else:
                # income_next_table_eval: (n_z, n_eta_eval, n_eps_eval).
                # Linearly blend in lagged-z (continuous z_cur) between iz_lo and iz_hi.
                inc_lo = income_next_table_eval[iz_lo]
                inc_hi = income_next_table_eval[iz_hi]
                income_table_z = (1.0 - frac_z) * inc_lo + frac_z * inc_hi

            # State brackets at eval-rule v_nodes
            s_next, j_corners, w_corners = _build_step_state_brackets(
                state_coords,
                pcj_solver.Phi_0_state, pcj_solver.Phi_11,
                pcj_eval.v_nodes,
                pcj_solver.axis_grids, pcj_solver.axis_sizes,
                pcj_solver.corner_offsets, pcj_solver.strides,
                pcj_solver.state_bracket_shift, pcj_solver.state_bracket_L_inv,
            )
            log_R_bill, log_x_s, log_x_b = _build_step_log_returns(
                state_coords,
                pcj_eval.M_v_nodes, pcj_eval.ret_nodes,
                pcj_eval.const_r, pcj_eval.A_r,
                s_next,
                pcj_solver.rtb_idx, pcj_solver.xr_pos, pcj_solver.xb_pos,
            )

            # c_corners_T: (n_state_quad_eval, n_z, n_corners, n_w)
            # Pre-gather at j_corners for ALL z (FOC needs the full z-axis since
            # eta_iz_lo varies per k_eta inside the kernel).
            c_corners = C_next[:, j_corners, :]                 # (n_z, n_state_quad_eval, n_corners, n_w)
            c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))

            # FOC; only need 6th output (e_sum, mu-combined Euler sum)
            _, _, _, _, _, e_sum = working_foc_jac_ccv(
                alpha_s, alpha_b, savings_t, psi_z,
                log_R_bill, log_x_s, log_x_b, pcj_eval.weight_kv_kr,
                w_corners,
                c_corners_T, pcj_solver.wealth_grid,
                income_table_z,
                eta_iz_lo, eta_frac_z,
                pcj_eval.eta_weights, pcj_eval.eps_weights,
                A_is,
                pcj_eval.sigma2_xr, pcj_eval.sigma2_xb, pcj_eval.sigma_xrxb,
                gamma, b_bar, delta_j, min_consumption,
            )

            beta_e = beta * e_sum
            valid = jnp.logical_and(beta_e > 0.0, c_t > 0.0)
            c_implied = jnp.where(
                valid,
                jnp.power(jnp.where(valid, beta_e, 1.0), -1.0 / gamma),
                jnp.nan,
            )
            ee = jnp.where(valid, 1.0 - c_implied / c_t, jnp.nan)
            return ee, valid

        return vmap(per_hh)(
            z_arr, state_coords_arr, c_arr, alpha_s_arr, alpha_b_arr, savings_arr,
            annuity_arr,
        )

    return per_age


# =============================================================================
# Annuity factor at sim-path state coords (NumPy host-side, cheap)
# =============================================================================

def _annuity_factors_per_household(model, state_coords_age: np.ndarray) -> np.ndarray:
    """Mirror of main's _annuity_factors_per_household. state_coords_age has
    shape (K, n_state). Falls back to model's scalar for ablation systems
    where y_1 / spr aren't grid axes."""
    n = state_coords_age.shape[0]
    y_1_idx = model.y_1_index_in_state
    spr_idx = model.spr_index_in_state
    y_1 = (
        np.asarray(state_coords_age[:, int(y_1_idx)], dtype=float)
        if y_1_idx is not None
        else np.full(n, float(model.y_1_scalar_fallback), dtype=float)
    )
    spr = (
        np.asarray(state_coords_age[:, int(spr_idx)], dtype=float)
        if spr_idx is not None
        else np.full(n, float(model.spr_scalar_fallback), dtype=float)
    )
    return np.ascontiguousarray(annuity_factor(y_1, spr, int(model.b_bar)), dtype=float)


# =============================================================================
# Sampling + aggregation
# =============================================================================

def _age_sample_indices(alive_mask: np.ndarray, max_households: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Mirror of main's _age_sample_indices. Subsample only alive slots."""
    idx = np.flatnonzero(alive_mask)
    if max_households is None or idx.size <= max_households:
        return idx.astype(np.int64, copy=False)
    pick = rng.choice(idx, size=max_households, replace=False)
    pick.sort()
    return pick.astype(np.int64, copy=False)


def _summarize_age(age: int, phase: str, ee: np.ndarray, valid: np.ndarray,
                   is_constrained: np.ndarray, n_alive: int) -> dict:
    """Per-age summary. Floors log10|EE| at LOG10_EE_FLOOR before aggregating.

    Returns BOTH 'all' (every valid cell) and 'unconstrained' (cells with
    s_t/x_t > kink_tol — KKT-slack cells excluded). The unconstrained subset
    is the gate the publication-grade test uses.
    """
    n_eval = int(ee.size)
    n_valid = int(np.sum(valid))
    n_constrained = int(np.sum(is_constrained & valid))

    def _stats(mask):
        if not np.any(mask):
            return {
                "count": 0,
                "mean_log10_abs_ee": None,
                "p95_log10_abs_ee": None,
                "p99_log10_abs_ee": None,
                "max_log10_abs_ee": None,
            }
        abs_ee = np.abs(ee[mask])
        log_abs = np.log10(np.maximum(abs_ee, 10.0 ** LOG10_EE_FLOOR))
        return {
            "count": int(mask.sum()),
            "mean_log10_abs_ee": float(np.mean(log_abs)),
            "p95_log10_abs_ee": float(np.percentile(log_abs, 95.0)),
            "p99_log10_abs_ee": float(np.percentile(log_abs, 99.0)),
            "max_log10_abs_ee": float(np.max(log_abs)),
        }

    valid_mask = valid
    unc_mask = valid & (~is_constrained)

    return {
        "age": int(age),
        "phase": phase,
        "n_alive_at_age": int(n_alive),
        "n_evaluated": n_eval,
        "n_valid": n_valid,
        "n_constrained": n_constrained,
        "all": _stats(valid_mask),
        "unconstrained": _stats(unc_mask),
    }


# =============================================================================
# Phase-level aggregation + gates
# =============================================================================

# Gate thresholds: mean / max log10|EE|. Per docs/agents/EE_DIAGNOSTIC_WORKFLOW.md
# §2: publication-grade is mean ≤ -5 to -6, max ≤ -3 to -4. Welfare-grade is laxer.
GATES = {
    "publication": {"mean_gate": -5.0, "max_gate": -3.0},
    "welfare":     {"mean_gate": -4.0, "max_gate": -2.0},
}


def _phase_aggregate(rows: list[dict], phase: str, cell_set: str) -> dict:
    """Pool per-age stats into a single phase-level summary.

    cell_set = "all" or "unconstrained". Pools log_abs values per the count
    field; we don't have raw arrays here so use the mean-of-means with weights.
    For percentile pooling, max-of-maxes is exact for max; p95/p99/mean are
    approximated by count-weighted aggregates of the per-age stats.
    """
    rows_phase = [r for r in rows if r["phase"] == phase]
    counts = np.array([r[cell_set]["count"] for r in rows_phase], dtype=float)
    means = np.array(
        [r[cell_set]["mean_log10_abs_ee"] for r in rows_phase],
        dtype=float,
    )
    maxes = np.array(
        [r[cell_set]["max_log10_abs_ee"] for r in rows_phase
         if r[cell_set]["max_log10_abs_ee"] is not None],
        dtype=float,
    )
    p95s = np.array(
        [r[cell_set]["p95_log10_abs_ee"] for r in rows_phase
         if r[cell_set]["p95_log10_abs_ee"] is not None],
        dtype=float,
    )
    p99s = np.array(
        [r[cell_set]["p99_log10_abs_ee"] for r in rows_phase
         if r[cell_set]["p99_log10_abs_ee"] is not None],
        dtype=float,
    )
    valid_age_mask = np.array(
        [r[cell_set]["mean_log10_abs_ee"] is not None for r in rows_phase],
        dtype=bool,
    )
    if not np.any(valid_age_mask):
        return {
            "phase": phase, "cell_set": cell_set, "n_ages": len(rows_phase),
            "total_count": 0,
            "mean_log10_abs_ee": None, "max_log10_abs_ee": None,
            "p95_log10_abs_ee": None, "p99_log10_abs_ee": None,
        }
    w = counts[valid_age_mask]
    pooled_mean = float(np.average(means[valid_age_mask], weights=w)) if w.sum() > 0 else None
    pooled_max = float(np.max(maxes)) if maxes.size else None
    # p95/p99 pooled as count-weighted average across ages (approximation).
    pooled_p95 = float(np.average(p95s, weights=w)) if w.sum() > 0 else None
    pooled_p99 = float(np.average(p99s, weights=w)) if w.sum() > 0 else None
    return {
        "phase": phase,
        "cell_set": cell_set,
        "n_ages": int(len(rows_phase)),
        "total_count": int(counts.sum()),
        "mean_log10_abs_ee": pooled_mean,
        "p95_log10_abs_ee": pooled_p95,
        "p99_log10_abs_ee": pooled_p99,
        "max_log10_abs_ee": pooled_max,
    }


def _build_gates(phase_stats: list[dict]) -> list[dict]:
    out = []
    for ps in phase_stats:
        for grade, thr in GATES.items():
            mean_v = ps["mean_log10_abs_ee"]
            max_v = ps["max_log10_abs_ee"]
            if mean_v is None or max_v is None:
                passed = None
            else:
                passed = (mean_v <= thr["mean_gate"]) and (max_v <= thr["max_gate"])
            out.append({
                "phase": ps["phase"],
                "cell_set": ps["cell_set"],
                "grade": grade,
                "mean_log10_abs_ee": mean_v,
                "max_log10_abs_ee": max_v,
                "mean_gate": thr["mean_gate"],
                "max_gate": thr["max_gate"],
                "pass": passed,
            })
    return out


# =============================================================================
# Markdown report
# =============================================================================

def _format_md(summary: dict) -> str:
    lines = []
    lines.append(f"# Sim-path EE: `{Path(summary['bundle_path']).name}`")
    lines.append("")
    lines.append(f"- Bundle: `{summary['bundle_path']}`")
    lines.append(f"- Eval mode: `{summary['eval_mode']}`")
    lines.append(f"- n_simulations: `{summary['n_simulations']}`")
    lines.append(f"- eval-households-per-age: `{summary['eval_households_per_age']}`")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Initial: z={summary['initial_z']!r}, state={summary['initial_state']!r}, "
                 f"wealth={summary['initial_wealth']}")
    lines.append(f"- Solver tol: `{summary['tolerance_used_at_solve']:.1e}`")
    eval_disc = summary["eval_disc_config"]
    solver_disc = summary["solver_disc_config"]
    lines.append(f"- Solver quadrature: ret={solver_disc['n_ret_nodes_1d']}, "
                 f"state={solver_disc['n_state_quad_nodes']}, "
                 f"eta={solver_disc['n_eta_nodes']}, eps={solver_disc['n_eps_nodes']}")
    lines.append(f"- Eval quadrature:   ret={eval_disc['n_ret_nodes_1d']}, "
                 f"state={eval_disc['n_state_quad_nodes']}, "
                 f"eta={eval_disc['n_eta_nodes']}, eps={eval_disc['n_eps_nodes']}")
    lines.append(f"- log10|EE| floor: `{LOG10_EE_FLOOR}`")
    lines.append("")

    g = summary
    lines.append("## Global summary (pooled over all evaluated cells)")
    lines.append("")
    def _fmt(x):
        return "n/a" if x is None else f"{x:.4f}"
    lines.append(f"- Mean log10|EE|: `{_fmt(g['global_mean_log10_abs_ee'])}`")
    lines.append(f"- P95 log10|EE| : `{_fmt(g['global_p95_log10_abs_ee'])}`")
    lines.append(f"- P99 log10|EE| : `{_fmt(g['global_p99_log10_abs_ee'])}`")
    lines.append(f"- Max log10|EE| : `{_fmt(g['global_max_log10_abs_ee'])}`")
    lines.append(f"- Total cells   : `{g['global_n_cells']}`")
    lines.append(f"- NaN/invalid   : `{g['global_n_invalid']}`")
    lines.append("")

    lines.append("## Phase summaries")
    lines.append("")
    lines.append("| phase | cell_set | n_ages | n_cells | mean | p95 | p99 | max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ps in summary["phase_summary"]:
        m = "-" if ps["mean_log10_abs_ee"] is None else f"{ps['mean_log10_abs_ee']:.3f}"
        p95 = "-" if ps["p95_log10_abs_ee"] is None else f"{ps['p95_log10_abs_ee']:.3f}"
        p99 = "-" if ps["p99_log10_abs_ee"] is None else f"{ps['p99_log10_abs_ee']:.3f}"
        mx = "-" if ps["max_log10_abs_ee"] is None else f"{ps['max_log10_abs_ee']:.3f}"
        lines.append(f"| {ps['phase']} | {ps['cell_set']} | {ps['n_ages']} | "
                     f"{ps['total_count']} | {m} | {p95} | {p99} | {mx} |")
    lines.append("")

    lines.append("## Pass / fail gates")
    lines.append("")
    lines.append("| phase | cell_set | grade | mean | max | mean_gate | max_gate | pass |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | :---: |")
    for g in summary["gates"]:
        m = "-" if g["mean_log10_abs_ee"] is None else f"{g['mean_log10_abs_ee']:.3f}"
        mx = "-" if g["max_log10_abs_ee"] is None else f"{g['max_log10_abs_ee']:.3f}"
        passed = "-" if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
        lines.append(f"| {g['phase']} | {g['cell_set']} | {g['grade']} | "
                     f"{m} | {mx} | {g['mean_gate']:.1f} | {g['max_gate']:.1f} | {passed} |")
    lines.append("")

    lines.append("## Per-age summary (cell_set = all)")
    lines.append("")
    lines.append("| age | phase | n_alive | n_eval | n_valid | n_constrained | mean | p95 | p99 | max |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary["per_age"]:
        s = r["all"]
        m = "-" if s["mean_log10_abs_ee"] is None else f"{s['mean_log10_abs_ee']:.3f}"
        p95 = "-" if s["p95_log10_abs_ee"] is None else f"{s['p95_log10_abs_ee']:.3f}"
        p99 = "-" if s["p99_log10_abs_ee"] is None else f"{s['p99_log10_abs_ee']:.3f}"
        mx = "-" if s["max_log10_abs_ee"] is None else f"{s['max_log10_abs_ee']:.3f}"
        lines.append(f"| {r['age']} | {r['phase']} | {r['n_alive_at_age']} | "
                     f"{r['n_evaluated']} | {r['n_valid']} | {r['n_constrained']} | "
                     f"{m} | {p95} | {p99} | {mx} |")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("bundle", help="Bundle dir or bare name (looked up under saved_runs/).")
    parser.add_argument("--eval-mode", choices=("same", "next_finer", "double"),
                        default="next_finer",
                        help="Quadrature for the FOC re-evaluation. Default next_finer.")
    parser.add_argument("--n-simulations", type=int, default=5000)
    parser.add_argument("--eval-households-per-age", default="256",
                        help="Subsample size per age. Pass 'all' to use every alive household.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-z", choices=("median", "stationary", "normal"),
                        default="stationary",
                        help="Initial-z distribution; default 'stationary' matches simulator.")
    parser.add_argument("--initial-z-normal-std", type=float, default=0.652)
    parser.add_argument("--initial-state", choices=("median", "stationary"),
                        default="stationary")
    parser.add_argument("--initial-wealth", type=float, default=0.1)
    parser.add_argument("--initial-x", type=float, default=None)
    parser.add_argument("--kink-tol", type=float, default=1e-3,
                        help="Cells with savings/x < this are flagged constrained (KKT slack).")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--out-suffix", default="",
                        help="Extra suffix for output files (e.g. '_same' or '_kall').")
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle_path}", flush=True)

    # eval_households_per_age may be 'all' or an int.
    if isinstance(args.eval_households_per_age, str) and args.eval_households_per_age.lower() == "all":
        K_per_age: int | None = None
    else:
        K_per_age = int(args.eval_households_per_age)

    # ---- Load bundle ----
    print("Loading bundle...", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    print(f"  Policy shape: C={C.shape}, dtype={C.dtype}", flush=True)
    spec = metadata.get("wealth_dynamics_spec", "?")
    if spec != "ccv_log":
        raise ValueError(
            f"Bundle wealth_dynamics_spec={spec!r}; this diagnostic requires ccv_log."
        )

    # ---- Rebuild model + pc_solver from saved config ----
    run_config = metadata.get("run_config")
    if run_config is None:
        raise ValueError(
            "Bundle has no run_config; cannot rebuild precompute. "
            "Re-solve with verify/benchmark_bundle.py."
        )
    base_config = run_config["base_config"]
    disc_solver = _rehydrate_disc_config(run_config["discretization_config"])
    solver_config = _rehydrate_solver_config(run_config.get("solver_config"))
    delta = solver_config.delta_bequest if solver_config.delta_bequest >= 0.0 else DELTA_BEQUEST

    print("Rebuilding model + solver-side precompute...", flush=True)
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=False)
    pc_solver = build_precompute(model, disc_solver, verbose=False)

    expected_shape = (pc_solver.n_age, pc_solver.n_z, pc_solver.N_state, pc_solver.n_w)
    if C.shape != expected_shape:
        raise RuntimeError(
            f"Policy shape {C.shape} != rebuilt {expected_shape}; bundle/config mismatch."
        )

    # ---- Build pc_eval (quadrature bumped per --eval-mode) ----
    print(f"Building eval-rule precompute (mode={args.eval_mode})...", flush=True)
    disc_eval = _build_eval_disc(disc_solver, model, args.eval_mode)
    pc_eval = build_precompute(model, disc_eval, verbose=False)
    print(
        f"  Solver q: ret={disc_solver.n_ret_nodes_1d} state={disc_solver.n_state_quad_nodes} "
        f"eta={disc_solver.n_eta_nodes} eps={disc_solver.n_eps_nodes}",
        flush=True,
    )
    print(
        f"  Eval q  : ret={disc_eval.n_ret_nodes_1d} state={disc_eval.n_state_quad_nodes} "
        f"eta={disc_eval.n_eta_nodes} eps={disc_eval.n_eps_nodes}",
        flush=True,
    )

    pcj_solver = _pc_to_jnp(pc_solver, delta)
    pcj_eval = _pc_to_jnp(pc_eval, delta)

    # ---- Solved-age mask (needed before simulating so we can backfill) ----
    diag_summary = metadata.get("diagnostics_summary", {}) or {}
    sma = diag_summary.get("solved_age_mask") if isinstance(diag_summary, dict) else None
    solved_mask = None
    if isinstance(sma, dict) and "values" in sma:
        solved_mask = np.asarray(sma["values"], dtype=bool)
    if solved_mask is None or solved_mask.size != pc_solver.n_age:
        solved_mask = np.array(
            [not np.isnan(C[t]).any() for t in range(pc_solver.n_age)], dtype=bool,
        )

    # Backfill unsolved early ages with the first solved age's policy. The JAX
    # simulator runs the full lifecycle from model.start_age; a retirement-only
    # bundle (e.g. youngest_age_to_solve=67) leaves ages 22..66 with NaN
    # policies, which NaN-poison the state by the time it reaches the solved
    # window. We replicate the first solved age's policy backward so the
    # simulator gets a finite (if economically arbitrary) trajectory through
    # the unsolved ages. The EE diagnostic only evaluates at solved ages, so
    # the trajectory through the unsolved window doesn't bias the residuals
    # (only the resulting age-T0 state distribution does).
    first_solved_t = int(np.argmax(solved_mask)) if solved_mask.any() else 0
    if first_solved_t > 0:
        print(
            f"\n  Bundle is partial: ages 0..{first_solved_t-1} unsolved "
            f"(start_age={model.start_age}, first solved age={int(np.asarray(pc_solver.ages)[first_solved_t])}).",
            flush=True,
        )
        print(
            f"  Backfilling those slabs with the first-solved-age policy so the simulator "
            f"can produce a finite trajectory. EE eval skips unsolved ages anyway.",
            flush=True,
        )
        # Don't mutate caller's arrays.
        C = C.copy()
        S = S.copy()
        B = B.copy()
        for t in range(first_solved_t):
            C[t] = C[first_solved_t]
            S[t] = S[first_solved_t]
            B[t] = B[first_solved_t]

    # ---- Simulate ----
    print(f"\nSimulating {args.n_simulations} households...", flush=True)
    t_sim = time.time()
    sim = simulate_lifecycle(
        C, S, B, pc_solver, model,
        n_simulations=int(args.n_simulations),
        initial_x=args.initial_x,
        initial_wealth=float(args.initial_wealth),
        initial_z=args.initial_z,
        initial_z_normal_std=float(args.initial_z_normal_std),
        initial_state=args.initial_state,
        seed=int(args.seed),
        return_draw_mode="monte_carlo",
        wealth_dynamics_spec="ccv_log",
        verbose=True,
    )
    print(f"  Sim wall: {time.time() - t_sim:.1f}s", flush=True)

    ages_arr = np.asarray(pc_solver.ages)
    retire_age = model.retire_age
    terminal_age = model.terminal_age

    n_solved = int(np.sum(solved_mask))
    print(f"\nSolved ages: {n_solved}/{pc_solver.n_age}", flush=True)

    eligible_t = []
    for t in range(pc_solver.n_age - 1):
        if not solved_mask[t]:
            continue
        if not solved_mask[t + 1]:
            continue
        if int(ages_arr[t]) >= int(terminal_age):
            continue
        eligible_t.append(t)
    if not eligible_t:
        print("\nNo ages eligible for sim-path EE. Exiting.", flush=True)
        return 1
    print(
        f"Evaluating sim-path EE at {len(eligible_t)} age(s): "
        f"ages {int(ages_arr[eligible_t[0]])}..{int(ages_arr[eligible_t[-1]])}",
        flush=True,
    )

    # ---- Build per-age kernels ----
    print("\nBuilding per-age JIT kernels...", flush=True)
    retire_kernel = _build_retirement_ee_kernel(pcj_solver, pcj_eval, model, solver_config, delta)
    working_kernel = _build_working_ee_kernel(
        pcj_solver, pcj_eval, model, solver_config, delta, use_pension_next=False,
    )
    boundary_kernel = _build_working_ee_kernel(
        pcj_solver, pcj_eval, model, solver_config, delta, use_pension_next=True,
    )

    # ---- Pre-stage device-side arrays ----
    psi_tab = jnp.asarray(pc_solver.survival_probs_2d)              # (n_age, n_z)
    pension_tab = jnp.asarray(pc_solver.pension_after_tax)          # (n_age, n_z)
    income_next_tab_eval = jnp.asarray(pc_eval.working_income_next) # (n_age, n_z, n_eta_eval, n_eps_eval)

    # Need C[t+1] on device per age — simplest: stage all needed slabs once.
    needed_t1 = sorted({t + 1 for t in eligible_t})
    C_jnp_by_age = {tt: jnp.asarray(C[tt]) for tt in needed_t1}

    # ---- Per-age sweep ----
    rng = np.random.default_rng(int(args.seed) + 1)
    sim_z = sim["z"]                       # (n_sim, n_age) continuous
    sim_state = sim["state_coords"]        # (n_sim, n_age, n_state)
    sim_c = sim["c"]
    sim_alpha_s = sim["alpha_s"]
    sim_alpha_b = sim["alpha_b"]
    sim_savings = sim["savings"]
    sim_x = sim["x"]
    sim_alive = sim["alive"]               # (n_sim, n_age) bool

    rows: list[dict] = []
    raw_log_abs_arrays: list[np.ndarray] = []     # for global percentile aggregation
    n_invalid_total = 0
    n_cells_total = 0
    print("\nPer-age sweep:", flush=True)
    t_sweep = time.time()
    for t in eligible_t:
        age = int(ages_arr[t])
        alive_t = sim_alive[:, t].astype(bool)
        sample_idx = _age_sample_indices(alive_t, K_per_age, rng)
        if sample_idx.size == 0:
            print(f"  age {age:3d}: 0 alive, skipping", flush=True)
            continue

        z_arr = jnp.asarray(sim_z[sample_idx, t])
        state_arr = jnp.asarray(sim_state[sample_idx, t, :])
        c_arr = jnp.asarray(sim_c[sample_idx, t])
        as_arr = jnp.asarray(sim_alpha_s[sample_idx, t])
        ab_arr = jnp.asarray(sim_alpha_b[sample_idx, t])
        sv_arr = jnp.asarray(sim_savings[sample_idx, t])
        annuity_np = _annuity_factors_per_household(model, sim_state[sample_idx, t, :])
        annuity_arr = jnp.asarray(annuity_np)

        psi_z_arr = psi_tab[t, :]
        pension_next_z_arr = pension_tab[t + 1, :]

        if age >= retire_age:
            ee_j, valid_j = retire_kernel(
                z_arr, state_arr, c_arr, as_arr, ab_arr, sv_arr, annuity_arr,
                psi_z_arr, pension_next_z_arr, C_jnp_by_age[t + 1],
            )
            phase = "retirement"
        elif age == retire_age - 1:
            inc_table = income_next_tab_eval[t + 1]   # ignored (use_pension_next=True)
            ee_j, valid_j = boundary_kernel(
                z_arr, state_arr, c_arr, as_arr, ab_arr, sv_arr, annuity_arr,
                psi_z_arr, pension_next_z_arr,
                inc_table,
                C_jnp_by_age[t + 1],
            )
            phase = "boundary"
        else:
            inc_table = income_next_tab_eval[t + 1]
            dummy_pension = jnp.zeros(pc_solver.n_z)
            ee_j, valid_j = working_kernel(
                z_arr, state_arr, c_arr, as_arr, ab_arr, sv_arr, annuity_arr,
                psi_z_arr, dummy_pension,
                inc_table,
                C_jnp_by_age[t + 1],
            )
            phase = "working"

        ee_np = np.asarray(jax.device_get(ee_j))
        valid_np = np.asarray(jax.device_get(valid_j))
        # KKT-slack flag: cells with s/x < kink_tol
        x_np = np.asarray(sim_x[sample_idx, t])
        s_np = np.asarray(sim_savings[sample_idx, t])
        is_constrained = np.zeros_like(valid_np)
        positive_x = x_np > 1e-12
        is_constrained[positive_x] = (s_np[positive_x] / x_np[positive_x]) < args.kink_tol

        n_alive = int(alive_t.sum())
        row = _summarize_age(age, phase, ee_np, valid_np, is_constrained, n_alive)
        rows.append(row)

        # Stash raw log|EE| for accurate global percentile aggregation later.
        n_cells_total += int(ee_np.size)
        n_invalid_total += int((~valid_np).sum())
        valid_ee = ee_np[valid_np]
        if valid_ee.size:
            raw_log_abs_arrays.append(
                np.log10(np.maximum(np.abs(valid_ee), 10.0 ** LOG10_EE_FLOOR))
            )

        s = row["all"]
        m = f"{s['mean_log10_abs_ee']:.3f}" if s["mean_log10_abs_ee"] is not None else "—"
        p99 = f"{s['p99_log10_abs_ee']:.3f}" if s["p99_log10_abs_ee"] is not None else "—"
        mx = f"{s['max_log10_abs_ee']:.3f}" if s["max_log10_abs_ee"] is not None else "—"
        print(
            f"  age {age:3d} {phase:11s}  alive={n_alive:5d}  eval={int(sample_idx.size):4d}  "
            f"valid={row['n_valid']:4d}  cstr={row['n_constrained']:3d}  "
            f"mean={m}  p99={p99}  max={mx}",
            flush=True,
        )

    total = time.time() - t_sweep
    print(f"\nTotal sweep wall: {total:.1f}s", flush=True)

    # ---- Phase-level aggregation + gates ----
    phase_stats = []
    for phase in ("working", "boundary", "retirement"):
        for cell_set in ("all", "unconstrained"):
            phase_stats.append(_phase_aggregate(rows, phase, cell_set))
    gates = _build_gates(phase_stats)

    # ---- Global aggregation (across all cells, all ages, cell_set=all) ----
    if raw_log_abs_arrays:
        all_log = np.concatenate(raw_log_abs_arrays)
        global_mean = float(np.mean(all_log))
        global_p95 = float(np.percentile(all_log, 95.0))
        global_p99 = float(np.percentile(all_log, 99.0))
        global_max = float(np.max(all_log))
    else:
        global_mean = global_p95 = global_p99 = global_max = None

    summary = {
        "bundle_path": str(bundle_path),
        "wealth_dynamics_spec": metadata.get("wealth_dynamics_spec"),
        "tolerance_used_at_solve": float(solver_config.tol),
        "max_iter_used_at_solve": int(solver_config.max_iter),
        "delta_bequest_used": float(delta),
        "eval_mode": args.eval_mode,
        "n_simulations": int(args.n_simulations),
        "eval_households_per_age": (None if K_per_age is None else int(K_per_age)),
        "seed": int(args.seed),
        "initial_z": args.initial_z,
        "initial_state": args.initial_state,
        "initial_wealth": float(args.initial_wealth),
        "kink_tol": float(args.kink_tol),
        "log10_ee_floor": float(LOG10_EE_FLOOR),
        "solver_disc_config": disc_solver._asdict(),
        "eval_disc_config": disc_eval._asdict(),
        "n_ages_evaluated": len(rows),
        "global_n_cells": int(n_cells_total),
        "global_n_invalid": int(n_invalid_total),
        "global_mean_log10_abs_ee": global_mean,
        "global_p95_log10_abs_ee": global_p95,
        "global_p99_log10_abs_ee": global_p99,
        "global_max_log10_abs_ee": global_max,
        "phase_summary": phase_stats,
        "gates": gates,
        "per_age": rows,
    }

    # Tuple/list normalisation for JSON
    def _stringify(v):
        if isinstance(v, tuple):
            return [_stringify(x) for x in v]
        if isinstance(v, dict):
            return {k: _stringify(x) for k, x in v.items()}
        if isinstance(v, (list,)):
            return [_stringify(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    summary["solver_disc_config"] = _stringify(summary["solver_disc_config"])
    summary["eval_disc_config"] = _stringify(summary["eval_disc_config"])

    # ---- stdout summary ----
    print("\n" + "=" * 72, flush=True)
    print("Sim-path EE summary", flush=True)
    print("=" * 72, flush=True)
    print(f"  Bundle             : {bundle_path}", flush=True)
    print(f"  Eval mode          : {args.eval_mode}", flush=True)
    print(f"  Eval households/age: {K_per_age if K_per_age is not None else 'all'}", flush=True)
    print(f"  Total cells        : {n_cells_total}", flush=True)
    print(f"  NaN/invalid        : {n_invalid_total}", flush=True)
    if global_mean is not None:
        print(f"  Global mean log10|EE|: {global_mean:.4f}", flush=True)
        print(f"  Global p95           : {global_p95:.4f}", flush=True)
        print(f"  Global p99           : {global_p99:.4f}", flush=True)
        print(f"  Global max           : {global_max:.4f}", flush=True)
    print("\n  Phase summary (cell_set=all):", flush=True)
    for ps in phase_stats:
        if ps["cell_set"] != "all":
            continue
        if ps["mean_log10_abs_ee"] is None:
            continue
        print(
            f"    {ps['phase']:11s}  n={ps['total_count']:6d}  mean={ps['mean_log10_abs_ee']:.3f}  "
            f"p95={ps['p95_log10_abs_ee']:.3f}  p99={ps['p99_log10_abs_ee']:.3f}  "
            f"max={ps['max_log10_abs_ee']:.3f}",
            flush=True,
        )
    print("\n  Gates:", flush=True)
    for g in gates:
        if g["pass"] is None:
            continue
        verdict = "PASS" if g["pass"] else "FAIL"
        print(
            f"    {g['phase']:11s} {g['cell_set']:14s} {g['grade']:11s}  "
            f"mean={g['mean_log10_abs_ee']:.3f} (gate {g['mean_gate']:.1f})  "
            f"max={g['max_log10_abs_ee']:.3f} (gate {g['max_gate']:.1f})  -> {verdict}",
            flush=True,
        )
    print("=" * 72, flush=True)

    # ---- Save outputs ----
    suffix = args.out_suffix
    if not args.no_save:
        out_path = bundle_path / f"ee_simpath{suffix}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved JSON: {out_path}", flush=True)
    if not args.no_markdown:
        md_path = bundle_path / f"ee_simpath{suffix}.md"
        md_path.write_text(_format_md(summary), encoding="utf-8")
        print(f"Saved markdown: {md_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
