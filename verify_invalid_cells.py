"""verify_invalid_cells.py -- Cheap NumPy-level health check on a saved policy bundle.

Loads a bundle and flags cells that are NaN/Inf, have extreme alphas, or whose
savings collapsed to the documented ``tiny_savings`` design-fallback. Reports
per-age counts and saves ``<bundle>/invalid_cells.json``.

Usage
-----
    python verify_invalid_cells.py <bundle-name-or-path>

Examples
--------
    python verify_invalid_cells.py system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark
    python verify_invalid_cells.py saved_runs/system_iv_full_var_..._jax_benchmark

Pass criterion
--------------
Zero NaN/Inf cells in solved ages and zero extreme-alpha cells.
``tiny_savings`` cells are reported separately (not a failure — a documented
design fallback in the solver's per-cell EGM).

Scope
-----
Pure NumPy on the loaded ``(C, S, B)`` arrays. No precompute, no quadrature,
no JAX. ``solved_age_mask`` (or NaN-slab fallback) is honoured so that ages
the solver intentionally skipped (e.g. ``youngest_age_to_solve=67``) don't
masquerade as failed cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import numpy as np

from lifecycle.model import SolverConfig
from lifecycle.policy_io import load_policy_bundle


# Defaults match _diag_invalid_cells on main: |alpha_s| > 20, |alpha_b| > 50.
# Tunable per-bundle via CLI; the defaults are deliberately loose so we only
# flag clearly pathological corners, not aggressive-but-sensible Merton tilts.
DEFAULT_EXTREME_ALPHA_S = 20.0
DEFAULT_EXTREME_ALPHA_B = 50.0


# =============================================================================
# Bundle path resolution (mirrors verify_ee_residuals.py)
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


# =============================================================================
# Solved-age mask (mirrors verify_ee_residuals.py)
# =============================================================================

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
    """Fallback: an age is solved iff its slab has no NaNs (the solver fills
    unsolved ages with NaN)."""
    return np.array(
        [not np.isnan(C[t]).any() for t in range(C.shape[0])],
        dtype=bool,
    )


# =============================================================================
# Per-age scan
# =============================================================================

def _per_age_check(
    C_t: np.ndarray, S_t: np.ndarray, B_t: np.ndarray,
    wealth_grid: np.ndarray | None,
    sc: SolverConfig,
    extreme_alpha_s: float,
    extreme_alpha_b: float,
    age: int,
    is_solved: bool,
) -> dict:
    """Scan one age slab. Each input is shape ``(n_z, N_state, n_w)``.

    Counts:
      nan/inf in c, alpha_s, alpha_b
      extreme |alpha_s| > extreme_alpha_s
      extreme |alpha_b| > extreme_alpha_b
      tiny savings (w - c <= sc.tiny_savings) — design fallback, reported
        separately so it doesn't trip pass/fail
      sub-floor consumption (c <= sc.min_consumption) excluding NaN cells
    """
    n_cells = int(C_t.size)

    nan_c = ~np.isfinite(C_t)
    nan_s = ~np.isfinite(S_t)
    nan_b = ~np.isfinite(B_t)
    nan_any = nan_c | nan_s | nan_b

    if not is_solved:
        # By-design unsolved age: NaN slabs are expected. Report only the
        # totals so the user can confirm the slab looks like the solver's
        # canonical "not solved" filling.
        return {
            "age": int(age),
            "is_solved": False,
            "n_cells": n_cells,
            "n_nan_any": int(nan_any.sum()),
            "n_nan_c": int(nan_c.sum()),
            "n_nan_alpha_s": int(nan_s.sum()),
            "n_nan_alpha_b": int(nan_b.sum()),
            "n_extreme_alpha_s": 0,
            "n_extreme_alpha_b": 0,
            "n_tiny_savings": 0,
            "n_subfloor_c": 0,
            "alpha_s_max_abs": None,
            "alpha_b_max_abs": None,
            "c_min_finite": None,
        }

    extreme_s = (np.abs(S_t) > extreme_alpha_s) & ~nan_s
    extreme_b = (np.abs(B_t) > extreme_alpha_b) & ~nan_b

    if wealth_grid is not None:
        # wealth_grid shape (n_w,) broadcasts across (n_z, N_state, n_w)
        savings = wealth_grid[None, None, :] - C_t
        tiny_sav = (~nan_c) & (savings <= sc.tiny_savings)
    else:
        tiny_sav = np.zeros_like(nan_c)

    subfloor_c = (~nan_c) & (C_t <= sc.min_consumption)

    # Summaries on the finite parts only.
    finite_S = S_t[~nan_s]
    finite_B = B_t[~nan_b]
    finite_C = C_t[~nan_c]

    return {
        "age": int(age),
        "is_solved": True,
        "n_cells": n_cells,
        "n_nan_any": int(nan_any.sum()),
        "n_nan_c": int(nan_c.sum()),
        "n_nan_alpha_s": int(nan_s.sum()),
        "n_nan_alpha_b": int(nan_b.sum()),
        "n_extreme_alpha_s": int(extreme_s.sum()),
        "n_extreme_alpha_b": int(extreme_b.sum()),
        "n_tiny_savings": int(tiny_sav.sum()),
        "n_subfloor_c": int(subfloor_c.sum()),
        "alpha_s_max_abs": float(np.max(np.abs(finite_S))) if finite_S.size else None,
        "alpha_b_max_abs": float(np.max(np.abs(finite_B))) if finite_B.size else None,
        "c_min_finite": float(np.min(finite_C)) if finite_C.size else None,
    }


def _format_age_line(s: dict) -> str:
    if not s["is_solved"]:
        return (
            f"Age {s['age']:3d}: UNSOLVED  "
            f"NaN={s['n_nan_any']}/{s['n_cells']} (expected = full slab)"
        )
    bad = (
        s["n_nan_any"] + s["n_extreme_alpha_s"] + s["n_extreme_alpha_b"]
    )
    return (
        f"Age {s['age']:3d}: "
        f"NaN={s['n_nan_any']:>4d}  "
        f"|a_s|max={s['alpha_s_max_abs']:.2f}  "
        f"|a_b|max={s['alpha_b_max_abs']:.2f}  "
        f"extreme_s={s['n_extreme_alpha_s']:>3d}  "
        f"extreme_b={s['n_extreme_alpha_b']:>3d}  "
        f"tiny_sav={s['n_tiny_savings']:>5d}  "
        f"subfloor_c={s['n_subfloor_c']:>3d}  "
        f"{'BAD' if bad > 0 else 'ok'}"
    )


# =============================================================================
# Solver-config rehydration (only need tiny_savings + min_consumption)
# =============================================================================

def _rehydrate_solver_config(d: dict | None) -> SolverConfig:
    if d is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    kwargs = {k: v for k, v in d.items() if k in valid}
    return SolverConfig(**kwargs)


def _wealth_grid_from_metadata(metadata: dict) -> np.ndarray | None:
    """Pull the bundle's wealth grid out of metadata if available so we can
    compute savings = wealth - c without rebuilding the precompute."""
    run_config = metadata.get("run_config", {}) or {}
    disc = run_config.get("discretization_config")
    if not isinstance(disc, dict):
        return None
    n_w = disc.get("n_wealth")
    w_min = disc.get("wealth_min")
    w_max = disc.get("wealth_max")
    if n_w is None or w_min is None or w_max is None:
        return None
    # Solver builds wealth_grid as linspace(wealth_min, wealth_max, n_wealth);
    # this matches lifecycle/precompute's construction (same as in
    # verify_ee_residuals.py's pcj.wealth_grid).
    return np.linspace(float(w_min), float(w_max), int(n_w), dtype=np.float64)


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
        help="Skip writing invalid_cells.json.",
    )
    parser.add_argument(
        "--extreme-alpha-s",
        type=float,
        default=DEFAULT_EXTREME_ALPHA_S,
        help=f"|alpha_s| above this counts as extreme (default {DEFAULT_EXTREME_ALPHA_S}).",
    )
    parser.add_argument(
        "--extreme-alpha-b",
        type=float,
        default=DEFAULT_EXTREME_ALPHA_B,
        help=f"|alpha_b| above this counts as extreme (default {DEFAULT_EXTREME_ALPHA_B}).",
    )
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle_path}", flush=True)

    print("Loading bundle...", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    print(f"  Policy shape: C={C.shape}, dtype={C.dtype}", flush=True)

    # SolverConfig (for tiny_savings / min_consumption thresholds).
    run_config = metadata.get("run_config", {}) or {}
    solver_config = _rehydrate_solver_config(run_config.get("solver_config"))

    # Wealth grid for savings = w - c (optional: skipped if not in metadata,
    # in which case n_tiny_savings is reported as 0 with a note).
    wealth_grid = _wealth_grid_from_metadata(metadata)
    if wealth_grid is None:
        print(
            "  NOTE: wealth grid not in metadata; tiny_savings count will be 0. "
            "Re-save bundle with current verify_benchmark_bundle.py to populate.",
            flush=True,
        )

    # Solved-age mask: prefer metadata, fall back to NaN-slab inference.
    n_age = C.shape[0]
    solved_mask = _solved_mask_from_metadata(metadata, n_age)
    if solved_mask is None:
        print("  (no solved_age_mask in metadata; inferring from C NaN slabs)", flush=True)
        solved_mask = _infer_solved_mask_from_C(np.asarray(C))
    n_solved = int(solved_mask.sum())
    print(f"  Solved ages: {n_solved}/{n_age}", flush=True)

    # Age axis is conventionally indexed by t into pc.ages. We don't have pc
    # without rebuilding precompute, but absolute age can be recovered from
    # base_config.start_age if present; otherwise we just report the t index.
    base = run_config.get("base_config")
    start_age = None
    if isinstance(base, dict):
        start_age = base.get("start_age")
    if start_age is None:
        # No model rehydration — report by t (slab index).
        ages = np.arange(n_age, dtype=int)
    else:
        ages = np.arange(int(start_age), int(start_age) + n_age, dtype=int)

    print("\nScanning per-age slabs...", flush=True)
    per_age_stats: list[dict] = []
    for t in range(n_age):
        stats = _per_age_check(
            np.asarray(C[t]), np.asarray(S[t]), np.asarray(B[t]),
            wealth_grid, solver_config,
            args.extreme_alpha_s, args.extreme_alpha_b,
            int(ages[t]), bool(solved_mask[t]),
        )
        per_age_stats.append(stats)
        print(" " + _format_age_line(stats), flush=True)

    # Globals over solved ages only.
    solved_stats = [s for s in per_age_stats if s["is_solved"]]
    total_cells_solved = sum(s["n_cells"] for s in solved_stats)
    total_nan = sum(s["n_nan_any"] for s in solved_stats)
    total_extreme_s = sum(s["n_extreme_alpha_s"] for s in solved_stats)
    total_extreme_b = sum(s["n_extreme_alpha_b"] for s in solved_stats)
    total_tiny_sav = sum(s["n_tiny_savings"] for s in solved_stats)
    total_subfloor = sum(s["n_subfloor_c"] for s in solved_stats)

    summary = {
        "bundle_path": str(bundle_path),
        "wealth_dynamics_spec": metadata.get("wealth_dynamics_spec"),
        "tiny_savings_threshold": float(solver_config.tiny_savings),
        "min_consumption_threshold": float(solver_config.min_consumption),
        "extreme_alpha_s_threshold": float(args.extreme_alpha_s),
        "extreme_alpha_b_threshold": float(args.extreme_alpha_b),
        "n_age": int(n_age),
        "n_ages_solved": int(n_solved),
        "total_cells_solved": int(total_cells_solved),
        "total_nan": int(total_nan),
        "total_extreme_alpha_s": int(total_extreme_s),
        "total_extreme_alpha_b": int(total_extreme_b),
        "total_tiny_savings": int(total_tiny_sav),
        "total_subfloor_c": int(total_subfloor),
        "per_age": per_age_stats,
    }

    print("\n" + "=" * 70, flush=True)
    print("Invalid-cells summary", flush=True)
    print("=" * 70, flush=True)
    print(f"  Bundle           : {bundle_path}", flush=True)
    print(f"  Solved ages      : {n_solved}/{n_age}", flush=True)
    print(f"  Cells (solved)   : {total_cells_solved}", flush=True)
    print(f"  NaN/Inf cells    : {total_nan}", flush=True)
    print(f"  Extreme |a_s|    : {total_extreme_s}", flush=True)
    print(f"  Extreme |a_b|    : {total_extreme_b}", flush=True)
    print(f"  Tiny savings     : {total_tiny_sav}  "
          f"(design fallback; not a failure)", flush=True)
    print(f"  Sub-floor c      : {total_subfloor}", flush=True)

    fail = (total_nan > 0) or (total_extreme_s > 0) or (total_extreme_b > 0)
    verdict = "FAIL" if fail else "PASS"
    summary["verdict"] = verdict
    detail = (
        "(zero NaN, no extreme alphas)"
        if not fail
        else "(some solved age has NaN or extreme alpha)"
    )
    print(f"  Verdict          : {verdict}  {detail}", flush=True)
    print("=" * 70, flush=True)

    if not args.no_save:
        out_path = bundle_path / "invalid_cells.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path}", flush=True)
    else:
        print("\n(--no-save: skipping JSON write)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
