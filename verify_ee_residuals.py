"""verify_ee_residuals.py -- Euler-equation residual diagnostic for saved policy bundles.

Loads a saved policy bundle, rebuilds the precompute the solver used, then re-evaluates
the FOC at every solved cell using the same FOC kernels (retirement_foc_jac_ccv /
working_foc_jac_ccv) the solver used. Reports per-age residual statistics and saves a
JSON summary alongside the bundle at ``<bundle>/ee_residuals.json``.

Usage
-----
    python verify_ee_residuals.py <bundle-name-or-path>

Examples
--------
    # Bundle name resolved under ./saved_runs/
    python verify_ee_residuals.py system_iv_full_var_unconstrained_cholesky_grid9x9x9x9_nz11_jax_benchmark

    # Or full path
    python verify_ee_residuals.py saved_runs/system_iv_full_var_..._jax_benchmark

Output
------
    ./<bundle>/ee_residuals.json
    Plus per-age summary printed to stdout. Pass / fail criteria are documented in
    docs/handoff/HANDOFF_PORT_EE_DIAGNOSTIC.md.

Scope
-----
Grid-based diagnostic only. Skips the terminal age (it has its own diagnostic) and any
age whose successor age is unsolved (residual needs C[t+1]). Working-age cells use
``working_foc_jac_ccv`` with eta-bracketed z_next; the work->retirement boundary uses the
pension table at z_next.

Cells flagged invalid (NaN'd in the output) when:
  - C[t, ...] is not finite (solver wrote NaN)
  - savings = wealth - c <= sc.tiny_savings (solver fell back to the tiny-savings branch)
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
import jax.lax as lax
import jax.numpy as jnp
from jax import jit, vmap

from lifecycle.model import (
    DELTA_BEQUEST,
    DiscretizationConfig,
    SolverConfig,
)
from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import (
    _pc_to_jnp,
    _precompute_per_is_tensors,
    bracket_uniform,
    retirement_foc_jac_ccv,
    working_foc_jac_ccv,
)
from lifecycle.var import build_nominal_system1_var_config_hardcoded


# =============================================================================
# Bundle loading + config rehydration
# =============================================================================

def _resolve_bundle_path(bundle_arg: str) -> Path:
    """Resolve a bundle path. Accepts a full path or a bare bundle name (looked
    up under ./saved_runs/<name>/)."""
    p = Path(bundle_arg)
    if p.is_dir():
        return p
    p2 = Path("saved_runs") / bundle_arg
    if p2.is_dir():
        return p2
    raise FileNotFoundError(
        f"Bundle directory not found. Tried: {p}, {p2}"
    )


def _list_to_tuple_recursive(v: Any) -> Any:
    """Convert lists in a config dict back to tuples. The JSON serialiser in
    policy_io._to_jsonable converts tuples -> lists; DiscretizationConfig fields
    like state_grid_sizes / n_state_quad_nodes are typed as tuples, so we
    convert back here. Leaves dicts / scalars / None alone."""
    if isinstance(v, list):
        return tuple(_list_to_tuple_recursive(x) for x in v)
    return v


def _rehydrate_disc_config(d: dict) -> DiscretizationConfig:
    """Reconstruct a DiscretizationConfig from a saved dict.

    Tuple-typed fields (state_grid_sizes, n_state_quad_nodes, n_ret_nodes_1d,
    state_n_stds, ret_lobatto_Z, state_lobatto_Z) are normalised back to
    tuples. Unknown keys are ignored to keep the script forward-compatible
    with config additions.
    """
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
        if k in tuple_fields:
            kwargs[k] = _list_to_tuple_recursive(v)
        else:
            kwargs[k] = v
    return DiscretizationConfig(**kwargs)


def _rehydrate_solver_config(d: dict | None) -> SolverConfig:
    """Reconstruct a SolverConfig from a saved dict (or default if absent).
    We need ``min_consumption``, ``tiny_savings``, ``delta_bequest``, ``tol``
    for the residual calc and the scaling.
    """
    if d is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    kwargs = {k: v for k, v in d.items() if k in valid}
    return SolverConfig(**kwargs)


def _rebuild_model_and_pc(metadata: dict, verbose: bool = False):
    """Rebuild model + precompute from bundle metadata."""
    run_config = metadata.get("run_config")
    if run_config is None:
        raise ValueError(
            "Bundle metadata has no 'run_config'. Cannot rebuild the precompute. "
            "This bundle was saved without configuration metadata (older format). "
            "Re-solve with the current verify_benchmark_bundle.py."
        )
    base_config = run_config.get("base_config")
    if base_config is None:
        raise ValueError("Bundle run_config missing 'base_config'.")
    disc_config_dict = run_config.get("discretization_config")
    if disc_config_dict is None:
        raise ValueError("Bundle run_config missing 'discretization_config'.")

    disc_config = _rehydrate_disc_config(disc_config_dict)
    solver_config = _rehydrate_solver_config(run_config.get("solver_config"))

    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=verbose)
    pc = build_precompute(model, disc_config, verbose=verbose)
    return model, pc, solver_config, run_config


def _ages_to_evaluate(metadata: dict, model, pc) -> np.ndarray:
    """Pick which age indices (t into pc.ages) to evaluate the EE residual at.

    Skips:
      - the terminal age (no successor; covered by diagnose_terminal_portfolio_states)
      - any age whose t+1 successor is unsolved (residual needs C[t+1])

    Solved-mask source: diagnostics_summary['solved_age_mask'] if present, else
    diagnostics_summary->shape match. Falls back to checking C for NaN slabs.
    """
    n_age = pc.n_age
    diag_summary = metadata.get("diagnostics_summary", {})
    solved_mask_meta = None
    if isinstance(diag_summary, dict):
        sma = diag_summary.get("solved_age_mask")
        if isinstance(sma, dict) and "values" in sma:
            solved_mask_meta = np.asarray(sma["values"], dtype=bool)
        elif isinstance(sma, list):
            solved_mask_meta = np.asarray(sma, dtype=bool)
    if solved_mask_meta is None or solved_mask_meta.size != n_age:
        return None  # caller infers from C
    return solved_mask_meta


def _infer_solved_mask_from_C(C: np.ndarray) -> np.ndarray:
    """Fallback when metadata doesn't carry solved_age_mask: an age is solved
    iff its slab has no NaNs (the solver fills unsolved ages with NaN)."""
    n_age = C.shape[0]
    return np.array(
        [not np.isnan(C[t]).any() for t in range(n_age)],
        dtype=bool,
    )


# =============================================================================
# Cell-axis chunking helpers
# =============================================================================

def _build_padded_cell_indices(n_z: int, N_state: int, n_chunks: int):
    """Pad the (z_idx, i_s) cell index arrays so each chunk has fixed size.

    Mirrors solver.py's ``_build_chunked_index_arrays`` (out of @jit so the
    padded arrays are jnp constants by trace time). Last cell is repeated to
    fill the padding; results sliced off later via ``[:n_cells]``.
    """
    n_cells = n_z * N_state
    chunk_size = (n_cells + n_chunks - 1) // n_chunks
    n_cells_padded = chunk_size * n_chunks

    cell_idx = np.arange(n_cells, dtype=np.int64)
    z_idx_np = (cell_idx // N_state).astype(np.int64)
    is_idx_np = (cell_idx % N_state).astype(np.int64)

    pad_count = n_cells_padded - n_cells
    if pad_count > 0:
        z_idx_np = np.concatenate(
            [z_idx_np, np.full(pad_count, z_idx_np[-1], dtype=np.int64)]
        )
        is_idx_np = np.concatenate(
            [is_idx_np, np.full(pad_count, is_idx_np[-1], dtype=np.int64)]
        )
    return jnp.asarray(z_idx_np), jnp.asarray(is_idx_np), n_cells, chunk_size


def _make_chunk_runner(jit_chunk_fn, z_idx_padded, is_idx_padded,
                       n_cells: int, chunk_size: int, n_chunks: int,
                       n_z: int, N_state: int, n_outputs: int = 3):
    """Return ``runner(*kernel_args) -> tuple of (n_z, N_state, n_w) arrays``.

    Calls a JIT'd per-chunk vmap kernel ``n_chunks`` times in pure Python and
    concatenates the outputs.

    Why outside @jit: with the chunk loop inside @jit, XLA fuses all chunks
    into one HLO graph and may schedule their per-cell intermediates
    concurrently — defeating the memory bound. Running each chunk as a
    separate JIT call forces the previous chunk's buffers to be freed before
    the next one runs, so peak HBM is one chunk's worth.

    The per-chunk kernel is traced once for ``chunk_size`` and reused.
    """
    def runner(*kernel_args):
        chunk_results: list[tuple] = []
        for i in range(n_chunks):
            start = i * chunk_size
            z_chunk = jax.device_put(z_idx_padded[start:start + chunk_size])
            is_chunk = jax.device_put(is_idx_padded[start:start + chunk_size])
            outs = jit_chunk_fn(*kernel_args, z_chunk, is_chunk)
            # Force chunk completion before scheduling the next one — bounds
            # peak HBM at one chunk's worth of intermediates.
            outs[0].block_until_ready()
            chunk_results.append(outs)
        full = tuple(
            jnp.concatenate([r[k] for r in chunk_results], axis=0)[:n_cells]
            for k in range(n_outputs)
        )
        n_w = full[0].shape[-1]
        return tuple(arr.reshape(n_z, N_state, n_w) for arr in full)

    return runner


# =============================================================================
# Per-age residual kernels
# =============================================================================

def _build_retirement_residual_kernel(pcj, model, sc, delta, per_is_tensors,
                                      n_z: int, N_state: int, n_chunks: int):
    """JIT'd per-age residual kernel for retirement ages (age >= retire_age).

    Returns ``per_age(C_t, S_t, B_t, C_next, pension_next_z_arr, psi_z_arr)
    -> (residual_abs, residual_rel, ee_cons)`` each of shape ``(n_z, N_state, n_w)``.

    Three metrics computed per cell:
      - ``residual_abs``: ``||(FOC_alpha_s, FOC_alpha_b)||`` (raw portfolio FOC norm)
      - ``residual_rel``: ``residual_abs / |V_dot at alpha=0|`` (Newton scale; what tol grades)
      - ``ee_cons``: ``1 - (beta * e_sum)^(-1/gamma) / c`` (consumption Euler residual,
        the metric main's *_same.md grid reports use)

    Vmaps over (z_idx, i_s) cells and (per cell) over wealth indices.
    c_corners_at_z is gathered once per (z_idx, i_s) cell and shared across all
    n_w wealth points in that cell — XLA streams the corner reads through the
    inner FOC kernel (same gather pattern the solver uses).
    """
    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    gamma = jnp.float64(model.gamma)
    beta = jnp.float64(model.beta)
    b_bar = jnp.float64(model.b_bar)
    delta_j = jnp.float64(delta)
    min_consumption = jnp.float64(sc.min_consumption)
    tiny_savings = jnp.float64(sc.tiny_savings)

    z_idx_padded, is_idx_padded, n_cells, chunk_size = _build_padded_cell_indices(
        n_z, N_state, n_chunks,
    )

    def per_cell(z_idx, i_s, C_t, S_t, B_t, C_next,
                 pension_next_z_arr, psi_z_arr):
        log_R_bill_i = log_R_bill_all[i_s]
        log_x_s_i = log_x_s_all[i_s]
        log_x_b_i = log_x_b_all[i_s]
        j_corners_i = j_corners_all[i_s]
        w_corners_i = w_corners_all[i_s]

        A_is = pcj.annuity_factors[i_s]
        psi_z = psi_z_arr[z_idx]
        pension_next_z = pension_next_z_arr[z_idx]

        c_corners_at_z = C_next[z_idx, j_corners_i, :]

        c_arr = C_t[z_idx, i_s, :]
        s_arr = S_t[z_idx, i_s, :]
        b_arr = B_t[z_idx, i_s, :]
        wealth_arr = pcj.wealth_grid

        def at_w(c, alpha_s, alpha_b, wealth):
            savings = wealth - c
            savings_safe = jnp.where(savings > 0.0, savings, 1e-12)

            fs, fb, _, _, _, e_sum = retirement_foc_jac_ccv(
                alpha_s, alpha_b, savings_safe, psi_z,
                log_R_bill_i, log_x_s_i, log_x_b_i, pcj.weight_kv_kr,
                w_corners_i, c_corners_at_z, pcj.wealth_grid,
                pension_next_z, A_is,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                gamma, b_bar, delta_j, min_consumption,
            )
            # Newton scale = |V_dot| at (a_s=0, a_b=0) — same scale the
            # solver's Newton uses to grade convergence (err < tol * scale).
            _, _, _, _, _, e0 = retirement_foc_jac_ccv(
                jnp.float64(0.0), jnp.float64(0.0), savings_safe, psi_z,
                log_R_bill_i, log_x_s_i, log_x_b_i, pcj.weight_kv_kr,
                w_corners_i, c_corners_at_z, pcj.wealth_grid,
                pension_next_z, A_is,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                gamma, b_bar, delta_j, min_consumption,
            )
            scale = jnp.maximum(jnp.abs(e0), 1e-30)
            residual = jnp.sqrt(fs * fs + fb * fb)
            rel_residual = residual / scale

            # Consumption Euler residual (matches main's `log10|EE|` metric)
            beta_e = beta * e_sum
            ee_valid = jnp.logical_and(beta_e > 0.0, c > 0.0)
            c_implied = jnp.power(jnp.where(ee_valid, beta_e, 1.0), -1.0 / gamma)
            ee_cons = jnp.where(ee_valid, 1.0 - c_implied / jnp.where(c > 0.0, c, 1.0),
                                jnp.nan)

            invalid = jnp.logical_or(
                jnp.logical_not(jnp.isfinite(c)),
                savings <= tiny_savings,
            )
            nan = jnp.float64(jnp.nan)
            return (
                jnp.where(invalid, nan, residual),
                jnp.where(invalid, nan, rel_residual),
                jnp.where(invalid, nan, ee_cons),
            )

        return vmap(at_w)(c_arr, s_arr, b_arr, wealth_arr)

    @jit
    def per_chunk(C_t, S_t, B_t, C_next, pension_next_z_arr, psi_z_arr,
                  z_chunk, is_chunk):
        return vmap(
            per_cell, in_axes=(0, 0, None, None, None, None, None, None),
        )(z_chunk, is_chunk, C_t, S_t, B_t, C_next, pension_next_z_arr, psi_z_arr)

    runner = _make_chunk_runner(
        per_chunk, z_idx_padded, is_idx_padded,
        n_cells, chunk_size, n_chunks, n_z, N_state,
    )

    def per_age(C_t, S_t, B_t, C_next, pension_next_z_arr, psi_z_arr):
        return runner(C_t, S_t, B_t, C_next, pension_next_z_arr, psi_z_arr)

    return per_age


def _build_working_residual_kernel(
    pcj, model, sc, delta, per_is_tensors,
    n_z: int, N_state: int, n_chunks: int,
    use_pension_next: bool,
):
    """JIT'd per-age residual kernel for working ages.

    ``use_pension_next=True`` selects the work->retirement boundary: income_next
    is the pension at bracketed z_next, broadcast across the eps axis. Otherwise
    income_next is the precomputed working_income_next table at this age.
    """
    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    gamma = jnp.float64(model.gamma)
    beta = jnp.float64(model.beta)
    b_bar = jnp.float64(model.b_bar)
    delta_j = jnp.float64(delta)
    rho = jnp.float64(model.rho)
    min_consumption = jnp.float64(sc.min_consumption)
    tiny_savings = jnp.float64(sc.tiny_savings)

    z_idx_padded, is_idx_padded, n_cells, chunk_size = _build_padded_cell_indices(
        n_z, N_state, n_chunks,
    )

    def per_cell(z_idx, i_s, C_t, S_t, B_t, C_next,
                 income_next_table, pension_next_z_arr, psi_z_arr):
        log_R_bill_i = log_R_bill_all[i_s]
        log_x_s_i = log_x_s_all[i_s]
        log_x_b_i = log_x_b_all[i_s]
        j_corners_i = j_corners_all[i_s]
        w_corners_i = w_corners_all[i_s]

        A_is = pcj.annuity_factors[i_s]
        psi_z = psi_z_arr[z_idx]

        # z_next bracket: same pattern as solver's working kernel
        z_now = pcj.z_grid[z_idx]
        z_next = rho * z_now + pcj.eta_nodes
        iz_lo, frac_z = vmap(bracket_uniform, in_axes=(0, None, None, None))(
            z_next, pcj.z_grid[0], pcj.dz, pcj.z_grid.shape[0]
        )

        if use_pension_next:
            pension_at_eta = (
                (1.0 - frac_z) * pension_next_z_arr[iz_lo]
                + frac_z * pension_next_z_arr[iz_lo + 1]
            )
            income_table_z = (
                pension_at_eta[:, None]
                * jnp.ones_like(pcj.eps_weights)[None, :]
            )
        else:
            income_table_z = income_next_table[z_idx]

        # c_corners pre-gather at multilinear-state corners, transposed so
        # k_v leads (matches the working FOC's outer vmap layout).
        c_corners = C_next[:, j_corners_i, :]
        c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))

        c_arr = C_t[z_idx, i_s, :]
        s_arr = S_t[z_idx, i_s, :]
        b_arr = B_t[z_idx, i_s, :]
        wealth_arr = pcj.wealth_grid

        def at_w(c, alpha_s, alpha_b, wealth):
            savings = wealth - c
            savings_safe = jnp.where(savings > 0.0, savings, 1e-12)

            fs, fb, _, _, _, e_sum = working_foc_jac_ccv(
                alpha_s, alpha_b, savings_safe, psi_z,
                log_R_bill_i, log_x_s_i, log_x_b_i, pcj.weight_kv_kr,
                w_corners_i,
                c_corners_T, pcj.wealth_grid,
                income_table_z,
                iz_lo, frac_z,
                pcj.eta_weights, pcj.eps_weights,
                A_is,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                gamma, b_bar, delta_j, min_consumption,
            )
            _, _, _, _, _, e0 = working_foc_jac_ccv(
                jnp.float64(0.0), jnp.float64(0.0), savings_safe, psi_z,
                log_R_bill_i, log_x_s_i, log_x_b_i, pcj.weight_kv_kr,
                w_corners_i,
                c_corners_T, pcj.wealth_grid,
                income_table_z,
                iz_lo, frac_z,
                pcj.eta_weights, pcj.eps_weights,
                A_is,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                gamma, b_bar, delta_j, min_consumption,
            )
            scale = jnp.maximum(jnp.abs(e0), 1e-30)
            residual = jnp.sqrt(fs * fs + fb * fb)
            rel_residual = residual / scale

            beta_e = beta * e_sum
            ee_valid = jnp.logical_and(beta_e > 0.0, c > 0.0)
            c_implied = jnp.power(jnp.where(ee_valid, beta_e, 1.0), -1.0 / gamma)
            ee_cons = jnp.where(ee_valid, 1.0 - c_implied / jnp.where(c > 0.0, c, 1.0),
                                jnp.nan)

            invalid = jnp.logical_or(
                jnp.logical_not(jnp.isfinite(c)),
                savings <= tiny_savings,
            )
            nan = jnp.float64(jnp.nan)
            return (
                jnp.where(invalid, nan, residual),
                jnp.where(invalid, nan, rel_residual),
                jnp.where(invalid, nan, ee_cons),
            )

        return vmap(at_w)(c_arr, s_arr, b_arr, wealth_arr)

    @jit
    def per_chunk(C_t, S_t, B_t, C_next,
                  income_next_table, pension_next_z_arr, psi_z_arr,
                  z_chunk, is_chunk):
        return vmap(
            per_cell, in_axes=(0, 0, None, None, None, None, None, None, None),
        )(
            z_chunk, is_chunk,
            C_t, S_t, B_t, C_next,
            income_next_table, pension_next_z_arr, psi_z_arr,
        )

    runner = _make_chunk_runner(
        per_chunk, z_idx_padded, is_idx_padded,
        n_cells, chunk_size, n_chunks, n_z, N_state,
    )

    def per_age(C_t, S_t, B_t, C_next,
                income_next_table, pension_next_z_arr, psi_z_arr):
        return runner(
            C_t, S_t, B_t, C_next,
            income_next_table, pension_next_z_arr, psi_z_arr,
        )

    return per_age


# =============================================================================
# Aggregation + reporting
# =============================================================================

# Floor for log10|EE| aggregation: c_implied ~= c gives -inf which would poison
# means/medians. Matches main's np.log10(np.maximum(abs_ee, 1e-16)).
LOG10_EE_FLOOR = -16.0


def _per_age_stats_residual(residual_arr: np.ndarray, age: int) -> dict:
    """Stats for a portfolio-FOC residual array (linear scale; threshold gates
    at 1e-5/1e-4/1e-2 follow the handoff's pass/fail thresholds)."""
    finite = residual_arr[np.isfinite(residual_arr)]
    n_cells = int(residual_arr.size)
    n_finite = int(finite.size)
    n_nan = n_cells - n_finite
    if n_finite == 0:
        return {
            "age": int(age),
            "n_cells": n_cells,
            "n_nan": n_nan,
            "median": None, "p95": None, "p99": None, "max": None,
            "frac_above_1e-5": None, "frac_above_1e-4": None, "frac_above_1e-2": None,
        }
    return {
        "age": int(age),
        "n_cells": n_cells,
        "n_nan": n_nan,
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
        "frac_above_1e-5": float(np.sum(finite > 1e-5) / n_finite),
        "frac_above_1e-4": float(np.sum(finite > 1e-4) / n_finite),
        "frac_above_1e-2": float(np.sum(finite > 1e-2) / n_finite),
    }


def _per_age_stats_ee(ee_arr: np.ndarray, age: int) -> dict:
    """Stats for the consumption-EE array. Reports on log10|EE| scale (matches
    main's *_same.md / *_nextfiner.md gridpoint reports). Floors at LOG10_EE_FLOOR.
    """
    finite_ee = ee_arr[np.isfinite(ee_arr)]
    n_cells = int(ee_arr.size)
    n_finite = int(finite_ee.size)
    n_nan = n_cells - n_finite
    if n_finite == 0:
        return {
            "age": int(age),
            "n_cells": n_cells,
            "n_nan": n_nan,
            "mean_log10_abs_ee": None, "median_log10_abs_ee": None,
            "p95_log10_abs_ee": None,  "p99_log10_abs_ee": None,
            "max_log10_abs_ee": None,
            "frac_below_neg6": None, "frac_below_neg5": None, "frac_below_neg4": None,
        }
    log_abs = np.log10(np.maximum(np.abs(finite_ee), 10.0 ** LOG10_EE_FLOOR))
    return {
        "age": int(age),
        "n_cells": n_cells,
        "n_nan": n_nan,
        "mean_log10_abs_ee":   float(np.mean(log_abs)),
        "median_log10_abs_ee": float(np.median(log_abs)),
        "p95_log10_abs_ee":    float(np.percentile(log_abs, 95.0)),
        "p99_log10_abs_ee":    float(np.percentile(log_abs, 99.0)),
        "max_log10_abs_ee":    float(np.max(log_abs)),
        "frac_below_neg6": float(np.sum(log_abs < -6.0) / n_finite),
        "frac_below_neg5": float(np.sum(log_abs < -5.0) / n_finite),
        "frac_below_neg4": float(np.sum(log_abs < -4.0) / n_finite),
    }


def _format_age_line_residual(s: dict) -> str:
    if s["median"] is None:
        return f"Age {s['age']:3d}: all-NaN ({s['n_nan']}/{s['n_cells']})"
    return (
        f"Age {s['age']:3d}: "
        f"median={s['median']:.2e}  p99={s['p99']:.2e}  max={s['max']:.2e}  "
        f">1e-5: {s['frac_above_1e-5']*100:5.1f}%  >1e-2: {s['frac_above_1e-2']*100:5.2f}%  "
        f"NaN: {s['n_nan']}/{s['n_cells']}"
    )


def _format_age_line_ee(s: dict) -> str:
    if s["mean_log10_abs_ee"] is None:
        return f"Age {s['age']:3d}: all-NaN ({s['n_nan']}/{s['n_cells']})"
    return (
        f"Age {s['age']:3d}: "
        f"mean={s['mean_log10_abs_ee']:6.3f}  p95={s['p95_log10_abs_ee']:6.3f}  "
        f"p99={s['p99_log10_abs_ee']:6.3f}  max={s['max_log10_abs_ee']:6.3f}  "
        f"<-6: {s['frac_below_neg6']*100:5.1f}%  <-4: {s['frac_below_neg4']*100:5.1f}%  "
        f"NaN: {s['n_nan']}/{s['n_cells']}"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "bundle",
        help="Bundle directory or bare bundle name (looked up under ./saved_runs/<name>/).",
    )
    parser.add_argument(
        "--metric",
        choices=("portfolio_abs", "portfolio_rel", "consumption_ee"),
        default="consumption_ee",
        help="Which metric drives stdout per-age trace and verdict. JSON always "
             "carries stats for all three. consumption_ee is the metric main's "
             "*_same.md / *_nextfiner.md gridpoint reports use (recommended for "
             "comparison against historical Numba reports).",
    )
    parser.add_argument(
        "--use-relative",
        action="store_true",
        help="DEPRECATED. Maps to --metric portfolio_rel for backwards compat.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing ee_residuals.json (useful for ad hoc spot checks).",
    )
    parser.add_argument(
        "--cell-chunks",
        type=int,
        default=None,
        help="Number of fixed-size chunks the per-age vmap is split into. "
             "Bounds peak HBM at ~(per-cell working set * n_z*N_state / n_chunks). "
             "Defaults to 1 for small grids and ~ceil((n_z*N_state)/2048) above that.",
    )
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle_path}", flush=True)

    print("Loading bundle...", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    print(
        f"  Policy shape: C={C.shape}, dtype={C.dtype}",
        flush=True,
    )

    spec = metadata.get("wealth_dynamics_spec", "?")
    if spec != "ccv_log":
        print(
            f"  WARNING: bundle wealth_dynamics_spec={spec!r}, "
            f"but the FOC kernels evaluate the CCV log specification. "
            f"Residuals will not be meaningful for non-ccv_log bundles.",
            flush=True,
        )

    print("\nRebuilding model + precompute from bundle metadata...", flush=True)
    t0 = time.time()
    model, pc, solver_config, run_config = _rebuild_model_and_pc(metadata, verbose=False)
    delta = solver_config.delta_bequest if solver_config.delta_bequest >= 0.0 else DELTA_BEQUEST
    print(
        f"  Setup wall: {time.time() - t0:.1f}s; "
        f"N_state={pc.N_state}, n_z={pc.n_z}, n_w={pc.n_w}, "
        f"n_state_quad={pc.n_state_quad}, n_ret_quad={pc.n_ret_quad}",
        flush=True,
    )

    expected_shape = (pc.n_age, pc.n_z, pc.N_state, pc.n_w)
    if C.shape != expected_shape:
        raise RuntimeError(
            f"Policy shape mismatch: bundle has {C.shape}, "
            f"rebuilt precompute expects {expected_shape}. "
            f"Bundle was probably solved at a different config."
        )

    pcj = _pc_to_jnp(pc, delta)
    per_is_tensors = _precompute_per_is_tensors(pcj)

    # Pre-stage tables on device.
    pension_table_jnp = jnp.asarray(pc.pension_after_tax)             # (n_age, n_z)
    survival_jnp = jnp.asarray(pc.survival_probs_2d)                  # (n_age, n_z)
    working_income_next_jnp = jnp.asarray(pc.working_income_next)     # (n_age, n_z, n_eta, n_eps)
    wealth_dummy_pension = jnp.zeros(pc.n_z, dtype=jnp.float64)

    # Cell-axis chunking. n_cells = n_z * N_state can reach 72k at canonical
    # 9⁴ x 11; the per-cell c_corners gather is ~3.3 MB (retirement) or ~36 MB
    # (working — extra n_z factor), and JAX/XLA materialises a chunk_size-sized
    # batch of per-cell FOC intermediates (the inner vmap doesn't have a scan
    # to force serialisation). 5⁴ x 11 = 6875 cells with chunk_size=2048 OOM'd
    # at ~68 GB on the first run, so the default keeps per-chunk cells <= 256.
    # Override with --cell-chunks if you have headroom and want fewer dispatches.
    n_cells_total = pc.n_z * pc.N_state
    if args.cell_chunks is not None:
        n_chunks = max(1, int(args.cell_chunks))
    else:
        n_chunks = max(1, (n_cells_total + 255) // 256)
    chunk_size_dbg = (n_cells_total + n_chunks - 1) // n_chunks
    print(
        f"  Cell-chunk split: {n_chunks} chunk(s) over {n_cells_total} cells "
        f"(~{chunk_size_dbg} cells/chunk)",
        flush=True,
    )

    # Build kernels.
    retire_kernel = _build_retirement_residual_kernel(
        pcj, model, solver_config, delta, per_is_tensors,
        pc.n_z, pc.N_state, n_chunks,
    )
    working_kernel = _build_working_residual_kernel(
        pcj, model, solver_config, delta, per_is_tensors,
        pc.n_z, pc.N_state, n_chunks, use_pension_next=False,
    )
    boundary_kernel = _build_working_residual_kernel(
        pcj, model, solver_config, delta, per_is_tensors,
        pc.n_z, pc.N_state, n_chunks, use_pension_next=True,
    )

    ages = np.asarray(pc.ages)
    retire_age = model.retire_age
    terminal_age = model.terminal_age

    # Decide which ages have a usable t+1 successor.
    solved_mask = _ages_to_evaluate(metadata, model, pc)
    if solved_mask is None:
        print("  (no solved_age_mask in metadata; inferring from C NaN slabs)", flush=True)
        solved_mask = _infer_solved_mask_from_C(np.asarray(C))
    n_solved = int(np.sum(solved_mask))
    print(f"  Solved ages: {n_solved}/{pc.n_age}", flush=True)

    eligible_t = []
    for t in range(pc.n_age - 1):
        # Skip terminal (t == n_age - 1) and any age whose t+1 successor is unsolved.
        if not solved_mask[t]:
            continue
        if not solved_mask[t + 1]:
            continue
        if int(ages[t]) >= int(terminal_age):
            continue
        eligible_t.append(t)

    if not eligible_t:
        print("\nNo ages eligible for EE residual computation "
              "(need solved t and solved t+1, t < terminal). Exiting.", flush=True)
        return 1

    # --use-relative is deprecated; map to --metric portfolio_rel.
    metric = "portfolio_rel" if args.use_relative else args.metric

    print(
        f"\nEvaluating EE residual at {len(eligible_t)} age(s): "
        f"ages {int(ages[eligible_t[0]])}..{int(ages[eligible_t[-1]])}",
        flush=True,
    )
    metric_descr = {
        "portfolio_abs":  "ABSOLUTE portfolio FOC residual = ||(FOC_a_s, FOC_a_b)||",
        "portfolio_rel":  "RELATIVE portfolio FOC residual = ||FOC|| / |V_dot at alpha=0|",
        "consumption_ee": "CONSUMPTION Euler residual = 1 - (beta * E[mu' R])^(-1/gamma) / c "
                          "(reported on log10|EE| scale, matches main's *_same.md reports)",
    }
    print(f"  (stdout metric: {metric_descr[metric]})", flush=True)
    print("  (JSON carries all three metrics)", flush=True)

    # Pre-stage policy slabs on device.
    C_jnp_by_age = {}
    S_jnp_by_age = {}
    B_jnp_by_age = {}
    needed_slab_idx = set(eligible_t) | {t + 1 for t in eligible_t}
    for t in needed_slab_idx:
        C_jnp_by_age[t] = jnp.asarray(C[t])
        S_jnp_by_age[t] = jnp.asarray(S[t])
        B_jnp_by_age[t] = jnp.asarray(B[t])

    # Per-age stats arrays — one list per metric.
    stats_abs:  list[dict] = []
    stats_rel:  list[dict] = []
    stats_ee:   list[dict] = []
    global_nan_total = 0
    t_sweep = time.time()

    for t in eligible_t:
        age = int(ages[t])
        psi_z = survival_jnp[t, :]

        if age >= retire_age:
            pension_next_z = pension_table_jnp[t + 1, :]
            res, rel, ee = retire_kernel(
                C_jnp_by_age[t], S_jnp_by_age[t], B_jnp_by_age[t],
                C_jnp_by_age[t + 1],
                pension_next_z, psi_z,
            )
            phase = "RETIRE"
        elif age == retire_age - 1:
            pension_next_z = pension_table_jnp[t + 1, :]
            income_table = jnp.zeros((pc.n_z, pc.n_eta, pc.n_eps), dtype=jnp.float64)
            res, rel, ee = boundary_kernel(
                C_jnp_by_age[t], S_jnp_by_age[t], B_jnp_by_age[t],
                C_jnp_by_age[t + 1],
                income_table, pension_next_z, psi_z,
            )
            phase = "BOUND "
        else:
            income_table = working_income_next_jnp[t + 1]
            res, rel, ee = working_kernel(
                C_jnp_by_age[t], S_jnp_by_age[t], B_jnp_by_age[t],
                C_jnp_by_age[t + 1],
                income_table, wealth_dummy_pension, psi_z,
            )
            phase = "WORK  "

        res_np, rel_np, ee_np = (
            np.asarray(jax.device_get(res)),
            np.asarray(jax.device_get(rel)),
            np.asarray(jax.device_get(ee)),
        )
        stats_abs.append(_per_age_stats_residual(res_np, age))
        stats_rel.append(_per_age_stats_residual(rel_np, age))
        stats_ee.append (_per_age_stats_ee     (ee_np,  age))
        global_nan_total += stats_abs[-1]["n_nan"]

        elapsed = time.time() - t_sweep
        if metric == "consumption_ee":
            line = _format_age_line_ee(stats_ee[-1])
        elif metric == "portfolio_rel":
            line = _format_age_line_residual(stats_rel[-1])
        else:
            line = _format_age_line_residual(stats_abs[-1])
        print(f" {phase} t={t:3d} age={age:3d}  ({elapsed:6.1f}s)  {line}", flush=True)

    total_sweep = time.time() - t_sweep
    print(f"\nTotal residual-sweep wall: {total_sweep:.1f}s", flush=True)

    # Sort by age for stable JSON output.
    stats_abs.sort(key=lambda s: s["age"])
    stats_rel.sort(key=lambda s: s["age"])
    stats_ee.sort (key=lambda s: s["age"])

    # Globals
    abs_max = max((s["max"] for s in stats_abs if s["max"] is not None), default=None)
    rel_max = max((s["max"] for s in stats_rel if s["max"] is not None), default=None)
    ee_means = [s["mean_log10_abs_ee"] for s in stats_ee if s["mean_log10_abs_ee"] is not None]
    ee_p95s  = [s["p95_log10_abs_ee"]  for s in stats_ee if s["p95_log10_abs_ee"]  is not None]
    ee_p99s  = [s["p99_log10_abs_ee"]  for s in stats_ee if s["p99_log10_abs_ee"]  is not None]
    ee_maxes = [s["max_log10_abs_ee"]  for s in stats_ee if s["max_log10_abs_ee"]  is not None]
    ee_global = {
        # mean: weighted across ages by cell count (all ages have the same n_cells = n_z*N_state*n_w
        # so unweighted mean is also fine; keep simple)
        "mean": float(np.mean(ee_means)) if ee_means else None,
        "p95":  float(np.mean(ee_p95s))  if ee_p95s  else None,
        "p99":  float(np.mean(ee_p99s))  if ee_p99s  else None,
        "max":  float(np.max(ee_maxes))  if ee_maxes else None,
    }

    summary = {
        "bundle_path": str(bundle_path),
        "wealth_dynamics_spec": metadata.get("wealth_dynamics_spec"),
        "tolerance_used_at_solve": float(solver_config.tol),
        "max_iter_used_at_solve": int(solver_config.max_iter),
        "delta_bequest_used": float(delta),
        "metric_for_verdict": metric,
        "log10_ee_floor": float(LOG10_EE_FLOOR),
        "n_ages_evaluated": len(stats_abs),
        "global_nan_count": int(global_nan_total),
        # Portfolio FOC residual (linear scale)
        "portfolio_abs": {
            "global_max": float(abs_max) if abs_max is not None else None,
            "per_age": stats_abs,
        },
        "portfolio_rel": {
            "global_max": float(rel_max) if rel_max is not None else None,
            "per_age": stats_rel,
        },
        # Consumption EE (log10 scale, matches main)
        "consumption_ee": {
            "global_mean_log10_abs_ee": ee_global["mean"],
            "global_p95_log10_abs_ee":  ee_global["p95"],
            "global_p99_log10_abs_ee":  ee_global["p99"],
            "global_max_log10_abs_ee":  ee_global["max"],
            "per_age": stats_ee,
        },
    }

    # ---- Stdout summary ----
    print("\n" + "=" * 72, flush=True)
    print("EE residuals summary", flush=True)
    print("=" * 72, flush=True)
    print(f"  Bundle              : {bundle_path}", flush=True)
    print(f"  Spec                : {summary['wealth_dynamics_spec']}", flush=True)
    print(f"  Solver tol          : {summary['tolerance_used_at_solve']:.1e}", flush=True)
    print(f"  Verdict metric      : {metric}", flush=True)
    print(f"  Ages evaluated      : {summary['n_ages_evaluated']}", flush=True)
    print(f"  Global NaN cells    : {summary['global_nan_count']}", flush=True)
    print(f"  Portfolio abs max   : {abs_max:.3e}" if abs_max is not None else "  Portfolio abs max   : n/a",
          flush=True)
    print(f"  Portfolio rel max   : {rel_max:.3e}" if rel_max is not None else "  Portfolio rel max   : n/a",
          flush=True)
    if ee_global["mean"] is not None:
        print(f"  Cons. EE mean       : log10|EE|={ee_global['mean']:.3f}", flush=True)
        print(f"  Cons. EE p95        : log10|EE|={ee_global['p95']:.3f}", flush=True)
        print(f"  Cons. EE p99        : log10|EE|={ee_global['p99']:.3f}", flush=True)
        print(f"  Cons. EE max        : log10|EE|={ee_global['max']:.3f}", flush=True)

    # Verdict — depends on selected metric.
    verdict = _verdict(metric, summary)
    summary["verdict"] = verdict.split()[0]
    print(f"  Verdict             : {verdict}", flush=True)
    print("=" * 72, flush=True)

    if not args.no_save:
        out_path = bundle_path / "ee_residuals.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path}", flush=True)
    else:
        print("\n(--no-save: skipping JSON write)", flush=True)

    return 0


# Verdict logic (per metric)
# - portfolio_abs / portfolio_rel: handoff's pass/fail thresholds (1e-2 / 1e-3 / 1e-5)
# - consumption_ee: main's gate thresholds (mean <= -5 publication, mean <= -4 welfare)
def _verdict(metric: str, summary: dict) -> str:
    if metric == "consumption_ee":
        ee = summary["consumption_ee"]
        m, mx = ee["global_mean_log10_abs_ee"], ee["global_max_log10_abs_ee"]
        if m is None or mx is None:
            return "n/a (no valid cells)"
        if m <= -5.0 and mx <= -3.0:
            return f"PASS_PUBLICATION  (mean {m:.2f} <= -5, max {mx:.2f} <= -3)"
        if m <= -4.0 and mx <= -2.0:
            return f"PASS_WELFARE     (mean {m:.2f} <= -4, max {mx:.2f} <= -2)"
        return f"FAIL              (mean {m:.2f}, max {mx:.2f})"

    # Portfolio FOC verdict (handoff thresholds)
    block = summary["portfolio_rel"] if metric == "portfolio_rel" else summary["portfolio_abs"]
    g_max = block["global_max"]
    per_age = block["per_age"]
    if g_max is None:
        return "n/a (no valid cells)"
    fail = any(
        (s["frac_above_1e-2"] is not None and s["frac_above_1e-2"] > 0.0)
        for s in per_age
    ) or summary["global_nan_count"] > 0
    concerning = (
        not fail
        and (g_max > 1e-3 or any(
            s["frac_above_1e-5"] is not None and s["frac_above_1e-5"] > 0.01
            for s in per_age
        ))
    )
    if fail:
        return f"FAIL              (max {g_max:.2e}; some age has frac_above_1e-2 > 0 or NaN)"
    if concerning:
        return f"CONCERNING        (max {g_max:.2e}; >1% of cells exceed 1e-5)"
    return f"PASS              (max {g_max:.2e}; <=1% above 1e-5; no NaN)"


if __name__ == "__main__":
    raise SystemExit(main())
