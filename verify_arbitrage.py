"""verify_arbitrage.py -- Spurious-arbitrage check for saved policy bundles.

Loads a bundle, evaluates per-cell arbitrage gap at the solved (alpha_s, alpha_b)
using the same CCV log-return arithmetic the solver used. Per cell, the gap is

    gap = min_{k_v, k_r} R_p(alpha)  -  E_quad[R_bill]

where R_p uses ``_ccv_log_return_and_grad`` and R_bill = exp(log_R_bill_kv) is
read from the next-period state vector at ``rtb_index_in_state``. If any cell's
gap > 0 the discrete quadrature certifies a "free lunch" (a feasible portfolio
that the discrete integrator says cannot lose) that the continuous lognormal
distribution does not actually admit.

This is the cheap pre-launch test for whether Lobatto-style tail nodes are
needed: if the bundle's existing (Gauss-Hermite) quadrature passes here at the
solved policy, downstream analysis is safe; if it fails, re-solve with
``ret_lobatto_Z`` / ``state_lobatto_Z`` configured.

Usage
-----
    python verify_arbitrage.py <bundle-name-or-path>

Examples
--------
    python verify_arbitrage.py system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark
    python verify_arbitrage.py saved_runs/system_iv_full_var_..._jax_benchmark

Output
------
    ./<bundle>/arbitrage.json plus per-age summary on stdout.

Pass criteria
-------------
- PASS:        max gap < 1e-6 globally, no NaN cells, no cell with gap > 0
- CONCERNING:  max in [1e-6, 1e-4]   OR  fraction-above-1e-6 > 1%
- FAIL:        max > 1e-4            OR  fraction-above-1e-6 > 5%

Scope
-----
Initial port: arbitrage gap at the SOLVED policy only. The alpha-sweep
variant from main's _diag_arbitrage_quadsweep is deferred (TODO at end of
file). Adapted for the JAX rewrite: 4-D state, CCV log returns, rtb-as-state
(R_bill comes from s_next[k_v, rtb_idx], not a return draw). Memory is
bounded by the same chunk-outside-JIT pattern used in
``verify_ee_residuals.py``.
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
)
from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import (
    _ccv_log_return_and_grad,
    _pc_to_jnp,
    _precompute_per_is_tensors,
)
from lifecycle.var import build_nominal_system1_var_config_hardcoded


# =============================================================================
# Bundle loading + config rehydration  (mirrors verify_ee_residuals.py)
# =============================================================================

def _resolve_bundle_path(bundle_arg: str) -> Path:
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
        if k in tuple_fields:
            kwargs[k] = _list_to_tuple_recursive(v)
        else:
            kwargs[k] = v
    return DiscretizationConfig(**kwargs)


def _rehydrate_solver_config(d: dict | None) -> SolverConfig:
    if d is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    kwargs = {k: v for k, v in d.items() if k in valid}
    return SolverConfig(**kwargs)


def _rebuild_model_and_pc(metadata: dict, verbose: bool = False):
    run_config = metadata.get("run_config")
    if run_config is None:
        raise ValueError(
            "Bundle metadata has no 'run_config'. Cannot rebuild the precompute. "
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


def _solved_mask_from_metadata(metadata: dict, n_age: int) -> np.ndarray | None:
    diag_summary = metadata.get("diagnostics_summary", {})
    if not isinstance(diag_summary, dict):
        return None
    sma = diag_summary.get("solved_age_mask")
    if isinstance(sma, dict) and "values" in sma:
        m = np.asarray(sma["values"], dtype=bool)
    elif isinstance(sma, list):
        m = np.asarray(sma, dtype=bool)
    else:
        return None
    if m.size != n_age:
        return None
    return m


def _infer_solved_mask_from_C(C: np.ndarray) -> np.ndarray:
    return np.array(
        [not np.isnan(C[t]).any() for t in range(C.shape[0])],
        dtype=bool,
    )


# =============================================================================
# Cell-axis chunking helpers (mirrors verify_ee_residuals.py)
# =============================================================================

def _build_padded_cell_indices(n_z: int, N_state: int, n_chunks: int):
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
    """Return a runner that calls jit_chunk_fn n_chunks times, concatenates,
    and reshapes back to (n_z, N_state, n_w). Output is a single array
    (the per-cell arbitrage gap)."""
    def runner(*kernel_args):
        parts = []
        for i in range(n_chunks):
            start = i * chunk_size
            z_chunk = jax.device_put(z_idx_padded[start:start + chunk_size])
            is_chunk = jax.device_put(is_idx_padded[start:start + chunk_size])
            piece = jit_chunk_fn(*kernel_args, z_chunk, is_chunk)
            piece.block_until_ready()
            parts.append(piece)
        full = jnp.concatenate(parts, axis=0)[:n_cells]
        n_w = full.shape[-1]
        return full.reshape(n_z, N_state, n_w)
    return runner


# =============================================================================
# Per-age arbitrage-gap kernel
# =============================================================================

def _build_arbitrage_kernel(pcj, sc: SolverConfig, per_is_tensors,
                             n_z: int, N_state: int, n_chunks: int):
    """JIT'd per-age kernel returning per-cell arbitrage gap of shape
    ``(n_z, N_state, n_w)``.

    For each cell ``(z_idx, i_s, w_idx)``:
      1. Look up the per-i_s log-return scenario tensors (shape
         ``(n_state_quad, n_ret_quad)``).
      2. Compute ``R_p[k_v, k_r]`` at the solved (alpha_s, alpha_b) via
         ``_ccv_log_return_and_grad`` (we drop the grad outputs).
      3. ``R_bill[k_v, k_r] = exp(log_R_bill[k_v, k_r])`` — broadcast across
         k_r since rtb is a state-axis realisation, not a return draw.
      4. ``gap = min_{k_v, k_r} R_p  -  sum(weight_kv_kr * R_bill)``.

    Cells where c is NaN/Inf or where savings <= sc.tiny_savings are NaN'd
    in the output (the policy was not actually solved at that cell).
    """
    log_R_bill_all, log_x_s_all, log_x_b_all, _, _ = per_is_tensors
    weight_kv_kr = pcj.weight_kv_kr

    sigma2_xr = pcj.sigma2_xr
    sigma2_xb = pcj.sigma2_xb
    sigma_xrxb = pcj.sigma_xrxb

    tiny_savings = jnp.float64(sc.tiny_savings)

    z_idx_padded, is_idx_padded, n_cells, chunk_size = _build_padded_cell_indices(
        n_z, N_state, n_chunks,
    )

    def per_cell(z_idx, i_s, C_t, S_t, B_t):
        log_R_bill_i = log_R_bill_all[i_s]              # (n_state_quad, n_ret_quad)
        log_x_s_i = log_x_s_all[i_s]
        log_x_b_i = log_x_b_all[i_s]

        # E[R_bill] under the joint quadrature. R_bill is constant across
        # k_r so this also equals sum_kv v_weights[kv] * exp(log_R_bill_kv);
        # writing it as the joint sum keeps the diagnostic symmetric with the
        # joint-cloud arbitrage test on main and means we don't need to
        # separately materialise v_weights / ret_weights here.
        R_bill = jnp.exp(log_R_bill_i)
        R_bill_mean = jnp.sum(weight_kv_kr * R_bill)

        c_arr = C_t[z_idx, i_s, :]                       # (n_w,)
        s_arr = S_t[z_idx, i_s, :]
        b_arr = B_t[z_idx, i_s, :]
        wealth_arr = pcj.wealth_grid                     # (n_w,)

        def at_w(c, alpha_s, alpha_b, wealth):
            R_p, _, _ = _ccv_log_return_and_grad(
                alpha_s, alpha_b,
                log_R_bill_i, log_x_s_i, log_x_b_i,
                sigma2_xr, sigma2_xb, sigma_xrxb,
            )                                             # (n_state_quad, n_ret_quad)
            R_p_min = jnp.min(R_p)
            gap = R_p_min - R_bill_mean

            savings = wealth - c
            invalid = jnp.logical_or(
                jnp.logical_not(jnp.isfinite(c)),
                savings <= tiny_savings,
            )
            invalid = jnp.logical_or(
                invalid,
                jnp.logical_not(jnp.isfinite(alpha_s) & jnp.isfinite(alpha_b)),
            )
            return jnp.where(invalid, jnp.float64(jnp.nan), gap)

        return vmap(at_w)(c_arr, s_arr, b_arr, wealth_arr)  # (n_w,)

    @jit
    def per_chunk(C_t, S_t, B_t, z_chunk, is_chunk):
        return vmap(per_cell, in_axes=(0, 0, None, None, None))(
            z_chunk, is_chunk, C_t, S_t, B_t,
        )

    runner = _make_chunk_runner(
        per_chunk, z_idx_padded, is_idx_padded,
        n_cells, chunk_size, n_chunks, n_z, N_state,
    )

    def per_age(C_t, S_t, B_t):
        return runner(C_t, S_t, B_t)

    return per_age


# =============================================================================
# Aggregation + reporting
# =============================================================================

def _per_age_stats(gap_arr: np.ndarray, age: int) -> dict:
    finite = gap_arr[np.isfinite(gap_arr)]
    n_cells = int(gap_arr.size)
    n_finite = int(finite.size)
    n_nan = n_cells - n_finite
    if n_finite == 0:
        return {
            "age": int(age),
            "n_cells": n_cells,
            "n_nan": n_nan,
            "min_gap": None,
            "median_gap": None,
            "p99_gap": None,
            "max_gap": None,
            "frac_gap_pos": None,
            "frac_above_1e-6": None,
            "frac_above_1e-4": None,
        }
    return {
        "age": int(age),
        "n_cells": n_cells,
        "n_nan": n_nan,
        "min_gap": float(np.min(finite)),
        "median_gap": float(np.median(finite)),
        "p99_gap": float(np.percentile(finite, 99.0)),
        "max_gap": float(np.max(finite)),
        "frac_gap_pos": float(np.sum(finite > 0.0) / n_finite),
        "frac_above_1e-6": float(np.sum(finite > 1e-6) / n_finite),
        "frac_above_1e-4": float(np.sum(finite > 1e-4) / n_finite),
    }


def _format_age_line(s: dict) -> str:
    if s["max_gap"] is None:
        return f"Age {s['age']:3d}: all-NaN ({s['n_nan']}/{s['n_cells']})"
    return (
        f"Age {s['age']:3d}: "
        f"max={s['max_gap']:+.2e}  "
        f"p99={s['p99_gap']:+.2e}  "
        f"median={s['median_gap']:+.2e}  "
        f"min={s['min_gap']:+.2e}  "
        f"gap>0: {s['frac_gap_pos']*100:5.2f}%  "
        f">1e-6: {s['frac_above_1e-6']*100:5.2f}%  "
        f">1e-4: {s['frac_above_1e-4']*100:5.2f}%  "
        f"NaN: {s['n_nan']}/{s['n_cells']}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "bundle",
        help="Bundle directory or bare bundle name (looked up under ./saved_runs/<name>/).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing arbitrage.json.",
    )
    parser.add_argument(
        "--cell-chunks",
        type=int,
        default=None,
        help="Number of fixed-size chunks the per-age vmap is split into. "
             "Defaults to a heuristic that keeps per-chunk cells <= 256 — but the "
             "arbitrage kernel's per-cell working set is much smaller than EE's "
             "(no c_corners gather), so 1 is fine for most configs.",
    )
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle_path}", flush=True)

    print("Loading bundle...", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    print(f"  Policy shape: C={C.shape}, dtype={C.dtype}", flush=True)

    spec = metadata.get("wealth_dynamics_spec", "?")
    if spec != "ccv_log":
        print(
            f"  WARNING: bundle wealth_dynamics_spec={spec!r}, but the arbitrage "
            f"kernel evaluates the CCV log specification. Gaps will not be "
            f"meaningful for non-ccv_log bundles.",
            flush=True,
        )

    print("\nRebuilding model + precompute from bundle metadata...", flush=True)
    t0 = time.time()
    model, pc, solver_config, _run_config = _rebuild_model_and_pc(metadata, verbose=False)
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
            f"rebuilt precompute expects {expected_shape}."
        )

    pcj = _pc_to_jnp(pc, delta)
    per_is_tensors = _precompute_per_is_tensors(pcj)

    # Cell-axis chunking. Per-cell working set here is just the (n_state_quad
    # x n_ret_quad) R_p tensor — no c_corners gather — so memory is minor.
    # Default: at most 1024 cells per chunk (more relaxed than EE's 256).
    n_cells_total = pc.n_z * pc.N_state
    if args.cell_chunks is not None:
        n_chunks = max(1, int(args.cell_chunks))
    else:
        n_chunks = max(1, (n_cells_total + 1023) // 1024)
    chunk_size_dbg = (n_cells_total + n_chunks - 1) // n_chunks
    print(
        f"  Cell-chunk split: {n_chunks} chunk(s) over {n_cells_total} cells "
        f"(~{chunk_size_dbg} cells/chunk)",
        flush=True,
    )

    arb_kernel = _build_arbitrage_kernel(
        pcj, solver_config, per_is_tensors,
        pc.n_z, pc.N_state, n_chunks,
    )

    ages = np.asarray(pc.ages)
    n_age = pc.n_age

    solved_mask = _solved_mask_from_metadata(metadata, n_age)
    if solved_mask is None:
        print("  (no solved_age_mask in metadata; inferring from C NaN slabs)", flush=True)
        solved_mask = _infer_solved_mask_from_C(np.asarray(C))
    n_solved = int(solved_mask.sum())
    print(f"  Solved ages: {n_solved}/{n_age}", flush=True)

    eligible_t = [t for t in range(n_age) if solved_mask[t]]
    if not eligible_t:
        print("\nNo solved ages to evaluate. Exiting.", flush=True)
        return 1

    print(
        f"\nEvaluating arbitrage gap at {len(eligible_t)} solved age(s): "
        f"ages {int(ages[eligible_t[0]])}..{int(ages[eligible_t[-1]])}",
        flush=True,
    )

    per_age_stats: list[dict] = []
    global_max = -np.inf
    global_min = np.inf
    global_nan = 0
    global_pos_cells = 0
    global_finite_cells = 0
    t_sweep = time.time()

    for t in eligible_t:
        age = int(ages[t])
        C_jnp = jnp.asarray(C[t])
        S_jnp = jnp.asarray(S[t])
        B_jnp = jnp.asarray(B[t])

        gap = arb_kernel(C_jnp, S_jnp, B_jnp)
        gap_np = np.asarray(jax.device_get(gap))
        stats = _per_age_stats(gap_np, age)
        per_age_stats.append(stats)

        if stats["max_gap"] is not None:
            global_max = max(global_max, stats["max_gap"])
            global_min = min(global_min, stats["min_gap"])
            n_finite_age = stats["n_cells"] - stats["n_nan"]
            global_finite_cells += n_finite_age
            global_pos_cells += int(round(stats["frac_gap_pos"] * n_finite_age))
        global_nan += stats["n_nan"]

        elapsed = time.time() - t_sweep
        print(
            f"  t={t:3d} age={age:3d}  ({elapsed:6.1f}s)  "
            + _format_age_line(stats),
            flush=True,
        )

    total_sweep = time.time() - t_sweep
    print(f"\nTotal arbitrage-sweep wall: {total_sweep:.1f}s", flush=True)

    per_age_stats.sort(key=lambda s: s["age"])

    if not np.isfinite(global_max):
        global_max = None
    if not np.isfinite(global_min):
        global_min = None

    summary = {
        "bundle_path": str(bundle_path),
        "wealth_dynamics_spec": metadata.get("wealth_dynamics_spec"),
        "ret_lobatto_Z": _list_to_tuple_recursive(
            (metadata.get("run_config", {}) or {}).get("discretization_config", {}).get("ret_lobatto_Z")
        ),
        "state_lobatto_Z": _list_to_tuple_recursive(
            (metadata.get("run_config", {}) or {}).get("discretization_config", {}).get("state_lobatto_Z")
        ),
        "n_ages_evaluated": len(per_age_stats),
        "global_max_gap": global_max,
        "global_min_gap": global_min,
        "global_nan_count": int(global_nan),
        "global_finite_cells": int(global_finite_cells),
        "global_cells_with_gap_pos": int(global_pos_cells),
        "per_age": per_age_stats,
    }

    print("\n" + "=" * 70, flush=True)
    print("Arbitrage-gap summary", flush=True)
    print("=" * 70, flush=True)
    print(f"  Bundle           : {bundle_path}", flush=True)
    print(f"  Spec             : {summary['wealth_dynamics_spec']}", flush=True)
    print(f"  ret_lobatto_Z    : {summary['ret_lobatto_Z']}", flush=True)
    print(f"  state_lobatto_Z  : {summary['state_lobatto_Z']}", flush=True)
    print(f"  Ages evaluated   : {summary['n_ages_evaluated']}", flush=True)
    print(f"  Global max gap   : "
          f"{global_max:+.3e}" if global_max is not None else "  Global max gap   : N/A", flush=True)
    print(f"  Global min gap   : "
          f"{global_min:+.3e}" if global_min is not None else "  Global min gap   : N/A", flush=True)
    print(f"  Cells with gap>0 : {global_pos_cells}/{global_finite_cells}",
          flush=True)
    print(f"  Global NaN cells : {summary['global_nan_count']}", flush=True)

    # Pass / concerning / fail (per the handoff §3 thresholds).
    max_for_verdict = global_max if global_max is not None else 0.0
    frac_above_1e6_global = (
        max(
            (s["frac_above_1e-6"] or 0.0) for s in per_age_stats
        ) if per_age_stats else 0.0
    )
    fail = (
        max_for_verdict > 1e-4
        or frac_above_1e6_global > 0.05
    )
    concerning = (
        not fail
        and (
            max_for_verdict > 1e-6
            or frac_above_1e6_global > 0.01
        )
    )
    if fail:
        verdict = "FAIL  (max gap > 1e-4 or fraction-above-1e-6 > 5%)"
    elif concerning:
        verdict = "CONCERNING  (max gap in [1e-6, 1e-4] or some age has >1% above 1e-6)"
    else:
        verdict = "PASS  (max gap < 1e-6 globally; no spurious arbitrage)"
    summary["verdict"] = verdict.split()[0]
    print(f"  Verdict          : {verdict}", flush=True)
    print("=" * 70, flush=True)

    if not args.no_save:
        out_path = bundle_path / "arbitrage.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path}", flush=True)
    else:
        print("\n(--no-save: skipping JSON write)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# TODO (deferred — see HANDOFF_PORT_ARBITRAGE_DIAGNOSTIC.md §8.1):
# Add an alpha-sweep variant. The stronger test from main's
# _diag_arbitrage_quadsweep evaluates:
#     gap_sweep(cell) = max_alpha (min_{k_v,k_r} R_p(alpha)) - E[R_bill]
# over a coarse alpha grid (e.g. alpha_s in {-2, -1, 0, 1, 2} x alpha_b
# similar) at each cell. If gap_sweep > 0 the *quadrature* admits arbitrage
# even where the solver did not exploit it. Cheap to add (vmap over
# alpha-grid points + take max), but expand the per-cell working set by the
# alpha-grid factor; bump n_chunks accordingly.
