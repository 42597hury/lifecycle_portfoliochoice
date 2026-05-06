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
                       n_z: int, N_state: int):
    """Return ``runner(*kernel_args) -> (residual, rel_residual)`` that calls
    a JIT'd per-chunk vmap kernel ``n_chunks`` times in pure Python.

    Why outside @jit: with the chunk loop inside @jit, XLA fuses all chunks
    into one HLO graph and may schedule their per-cell intermediates
    concurrently — defeating the memory bound. Running each chunk as a
    separate JIT call forces the previous chunk's buffers to be freed before
    the next one runs, so peak HBM is one chunk's worth.

    The per-chunk kernel is traced once for ``chunk_size`` and reused.
    """
    def runner(*kernel_args):
        a_parts = []
        b_parts = []
        for i in range(n_chunks):
            start = i * chunk_size
            z_chunk = jax.device_put(z_idx_padded[start:start + chunk_size])
            is_chunk = jax.device_put(is_idx_padded[start:start + chunk_size])
            a_i, b_i = jit_chunk_fn(*kernel_args, z_chunk, is_chunk)
            # Force chunk completion before scheduling the next one — bounds
            # peak HBM at one chunk's worth of intermediates.
            a_i.block_until_ready()
            a_parts.append(a_i)
            b_parts.append(b_i)
        a_full = jnp.concatenate(a_parts, axis=0)[:n_cells]
        b_full = jnp.concatenate(b_parts, axis=0)[:n_cells]
        n_w = a_full.shape[-1]
        return (
            a_full.reshape(n_z, N_state, n_w),
            b_full.reshape(n_z, N_state, n_w),
        )

    return runner


# =============================================================================
# Per-age residual kernels
# =============================================================================

def _build_retirement_residual_kernel(pcj, model, sc, delta, per_is_tensors,
                                      n_z: int, N_state: int, n_chunks: int):
    """JIT'd per-age residual kernel for retirement ages (age >= retire_age).

    Returns ``per_age(C_t, S_t, B_t, C_next, pension_next_z_arr, psi_z_arr)
    -> (residual, rel_residual)`` each of shape ``(n_z, N_state, n_w)``.

    Vmaps over (z_idx, i_s) cells and (per cell) over wealth indices.
    c_corners_at_z is gathered once per (z_idx, i_s) cell and shared across all
    n_w wealth points in that cell — XLA streams the corner reads through the
    inner FOC kernel (same gather pattern the solver uses).
    """
    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    gamma = jnp.float64(model.gamma)
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

            fs, fb, _, _, _, _ = retirement_foc_jac_ccv(
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

            invalid = jnp.logical_or(
                jnp.logical_not(jnp.isfinite(c)),
                savings <= tiny_savings,
            )
            nan = jnp.float64(jnp.nan)
            return (
                jnp.where(invalid, nan, residual),
                jnp.where(invalid, nan, rel_residual),
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

            fs, fb, _, _, _, _ = working_foc_jac_ccv(
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

            invalid = jnp.logical_or(
                jnp.logical_not(jnp.isfinite(c)),
                savings <= tiny_savings,
            )
            nan = jnp.float64(jnp.nan)
            return (
                jnp.where(invalid, nan, residual),
                jnp.where(invalid, nan, rel_residual),
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

def _per_age_stats(residual_arr: np.ndarray, age: int) -> dict:
    finite = residual_arr[np.isfinite(residual_arr)]
    n_cells = int(residual_arr.size)
    n_finite = int(finite.size)
    n_nan = n_cells - n_finite
    if n_finite == 0:
        return {
            "age": int(age),
            "n_cells": n_cells,
            "n_nan": n_nan,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "frac_above_1e-5": None,
            "frac_above_1e-4": None,
            "frac_above_1e-2": None,
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


def _format_age_line(s: dict) -> str:
    if s["median"] is None:
        return f"Age {s['age']:3d}: all-NaN ({s['n_nan']}/{s['n_cells']})"
    return (
        f"Age {s['age']:3d}: "
        f"median={s['median']:.2e}  "
        f"p99={s['p99']:.2e}  "
        f"max={s['max']:.2e}  "
        f">1e-5: {s['frac_above_1e-5']*100:5.1f}%  "
        f">1e-2: {s['frac_above_1e-2']*100:5.2f}%  "
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
        "--use-relative",
        action="store_true",
        help="Aggregate the relative residual (||FOC|| / |V_dot at zero|) "
             "instead of the absolute residual. Relative is what the Newton's "
             "tol applies to.",
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

    print(
        f"\nEvaluating EE residual at {len(eligible_t)} age(s): "
        f"ages {int(ages[eligible_t[0]])}..{int(ages[eligible_t[-1]])}",
        flush=True,
    )
    if args.use_relative:
        print("  (reporting RELATIVE residual = ||FOC|| / |V_dot at zero|)", flush=True)
    else:
        print("  (reporting ABSOLUTE residual = ||FOC||)", flush=True)

    # Pre-stage policy slabs on device (one slab per age — keep it simple,
    # device transfer is cheap relative to the residual sweep).
    C_jnp_by_age = {}
    S_jnp_by_age = {}
    B_jnp_by_age = {}
    needed_slab_idx = set(eligible_t) | {t + 1 for t in eligible_t}
    for t in needed_slab_idx:
        C_jnp_by_age[t] = jnp.asarray(C[t])
        S_jnp_by_age[t] = jnp.asarray(S[t])
        B_jnp_by_age[t] = jnp.asarray(B[t])

    per_age_stats: list[dict] = []
    global_max = 0.0
    global_nan = 0
    t_sweep = time.time()

    for t in eligible_t:
        age = int(ages[t])
        psi_z = survival_jnp[t, :]

        if age >= retire_age:
            pension_next_z = pension_table_jnp[t + 1, :]
            res, rel = retire_kernel(
                C_jnp_by_age[t], S_jnp_by_age[t], B_jnp_by_age[t],
                C_jnp_by_age[t + 1],
                pension_next_z, psi_z,
            )
            phase = "RETIRE"
        elif age == retire_age - 1:
            pension_next_z = pension_table_jnp[t + 1, :]
            income_table = jnp.zeros((pc.n_z, pc.n_eta, pc.n_eps), dtype=jnp.float64)
            res, rel = boundary_kernel(
                C_jnp_by_age[t], S_jnp_by_age[t], B_jnp_by_age[t],
                C_jnp_by_age[t + 1],
                income_table, pension_next_z, psi_z,
            )
            phase = "BOUND "
        else:
            income_table = working_income_next_jnp[t + 1]
            res, rel = working_kernel(
                C_jnp_by_age[t], S_jnp_by_age[t], B_jnp_by_age[t],
                C_jnp_by_age[t + 1],
                income_table, wealth_dummy_pension, psi_z,
            )
            phase = "WORK  "

        chosen = rel if args.use_relative else res
        chosen_np = np.asarray(jax.device_get(chosen))
        stats = _per_age_stats(chosen_np, age)
        per_age_stats.append(stats)

        if stats["max"] is not None and stats["max"] > global_max:
            global_max = stats["max"]
        global_nan += stats["n_nan"]

        elapsed = time.time() - t_sweep
        print(
            f" {phase} t={t:3d} age={age:3d}  ({elapsed:6.1f}s)  "
            + _format_age_line(stats),
            flush=True,
        )

    total_sweep = time.time() - t_sweep
    print(f"\nTotal residual-sweep wall: {total_sweep:.1f}s", flush=True)

    # Sort per_age_stats by age for stable JSON output.
    per_age_stats.sort(key=lambda s: s["age"])

    summary = {
        "bundle_path": str(bundle_path),
        "wealth_dynamics_spec": metadata.get("wealth_dynamics_spec"),
        "tolerance_used_at_solve": float(solver_config.tol),
        "max_iter_used_at_solve": int(solver_config.max_iter),
        "delta_bequest_used": float(delta),
        "residual_metric": "relative" if args.use_relative else "absolute",
        "n_ages_evaluated": len(per_age_stats),
        "global_max_residual": float(global_max),
        "global_nan_count": int(global_nan),
        "per_age": per_age_stats,
    }

    print("\n" + "=" * 70, flush=True)
    print("EE residuals summary", flush=True)
    print("=" * 70, flush=True)
    print(f"  Bundle           : {bundle_path}", flush=True)
    print(f"  Spec             : {summary['wealth_dynamics_spec']}", flush=True)
    print(f"  Solver tol       : {summary['tolerance_used_at_solve']:.1e}", flush=True)
    print(f"  Metric           : {summary['residual_metric']}", flush=True)
    print(f"  Ages evaluated   : {summary['n_ages_evaluated']}", flush=True)
    print(f"  Global max resid : {summary['global_max_residual']:.3e}", flush=True)
    print(f"  Global NaN cells : {summary['global_nan_count']}", flush=True)

    # Pass / fail tag (per the handoff's section 7 thresholds).
    fail = any(
        (s["frac_above_1e-2"] is not None and s["frac_above_1e-2"] > 0.0)
        for s in per_age_stats
    ) or summary["global_nan_count"] > 0
    concerning = (
        not fail
        and (
            summary["global_max_residual"] > 1e-3
            or any(
                (s["frac_above_1e-5"] is not None and s["frac_above_1e-5"] > 0.01)
                for s in per_age_stats
            )
        )
    )
    if fail:
        verdict = "FAIL  (some age has frac_above_1e-2 > 0 or NaN cells)"
    elif concerning:
        verdict = "CONCERNING  (max > 1e-3 or some age has >1% above 1e-5)"
    else:
        verdict = "PASS  (all ages: max < 1e-2; <=1% above 1e-5; no NaN)"
    summary["verdict"] = verdict.split()[0]
    print(f"  Verdict          : {verdict}", flush=True)

    print("=" * 70, flush=True)

    if not args.no_save:
        out_path = bundle_path / "ee_residuals.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path}", flush=True)
    else:
        print("\n(--no-save: skipping JSON write)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
