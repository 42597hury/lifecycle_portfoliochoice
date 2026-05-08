"""
System II × state-grid density convergence analysis.

Reads the three System II ablation bundles
  saved_runs/ablations/system_ii_grid{4x4,5x5,6x6}_nz11_sq4x5_rq4x4_calib1
all at the production-aligned canonical (n_z=11, sq=(4,5), rq=(4,4),
n_w=n_s=180, log1p wealth grid, ccv_log dynamics, gather_precision=f32,
state_n_stds=(2.0, 2.25), state_grid_mode="cholesky"), and quantifies
how 2-axis state-grid density on (rtb, y_1) affects the consumption /
risky-share / bond-share policies.

Approach mirrors `inf_horizon_grid_quad_convergence.py`:
  * The Cholesky-mode state grid is built as a tensor product of bracket
    grids `linspace(-n_stds[d], +n_stds[d], N_d)` in u-coords (Cholesky
    basis), with the physical state given by a fixed affine transform
    `mu_s + L @ u`. Since `state_n_stds=(2.0, 2.25)` matches across the
    three bundles, the bracket-coordinate grids are nested by density
    and the comparison is clean.
  * Reshape `(78, 11, N_state, 180)` -> `(78, 11, N_rtb, N_y1, 180)` in
    the lex (C-order) layout that `np.ndindex(*N_vec)` produces inside
    `_independence_rouwenhorst_pi`.
  * Pick (6, 6) as the reference. Interpolate the (4,4) and (5,5) onto
    the (6,6) bracket grid for the bulk metrics tensor (small-memory).
  * Render per-(rtb, y_1) heatmaps via a separate 50x50 interpolation
    on a single (age, z, wealth) probe slice (tiny, no memory issue).

Output: machine-readable metrics JSON + a small companion `.npz` for
the plot script (`plot_system_ii_grid_convergence.py`).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lifecycle.policy_io import load_policy_bundle  # noqa: E402

BUNDLE_NAMES = (
    "system_ii_grid4x4_nz11_sq4x5_rq4x4_calib1",
    "system_ii_grid5x5_nz11_sq4x5_rq4x4_calib1",
    "system_ii_grid6x6_nz11_sq4x5_rq4x4_calib1",
)
REFERENCE_NAME = BUNDLE_NAMES[-1]   # (6, 6)
HEATMAP_GRID_SIZE = 50              # for the (rtb, y_1) probe heatmap
STATE_AXIS_NAMES = ("rtb", "y_1")


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def load_bundle(bundles_root: Path, name: str) -> dict[str, Any]:
    bundle = bundles_root / name
    if not bundle.exists():
        raise FileNotFoundError(bundle)
    C, S, B, _diag, metadata = load_policy_bundle(bundle)
    diag_path = bundle / "diagnostics.pkl"
    with diag_path.open("rb") as f:
        diag = pickle.load(f)
    rc = metadata.get("run_config") or {}
    disc = rc.get("discretization_config") or diag.get("disc_config", {})
    state_grid_sizes = tuple(int(v) for v in disc["state_grid_sizes"])
    state_n_stds = tuple(float(v) for v in disc["state_n_stds"])
    if len(state_grid_sizes) != 2:
        raise ValueError(
            f"{name}: expected 2-axis state grid, got "
            f"state_grid_sizes={state_grid_sizes}"
        )
    if state_n_stds != (2.0, 2.25):
        raise ValueError(
            f"{name}: state_n_stds={state_n_stds} != (2.0, 2.25); "
            "bundles are not on the same Cholesky bracket interval, "
            "interpolation is not comparable across them."
        )
    C = np.asarray(C)
    S = np.asarray(S)
    B = np.asarray(B)
    if C.ndim != 4:
        raise ValueError(f"{name}: expected 4-D policy, got shape {C.shape}")
    n_age, n_z, n_state, n_w = C.shape
    if n_state != int(np.prod(state_grid_sizes)):
        raise ValueError(
            f"{name}: state axis size {n_state} != prod(state_grid_sizes) = "
            f"{int(np.prod(state_grid_sizes))}"
        )
    return {
        "name": name,
        "bundle": bundle,
        "C": C, "S": S, "B": B,
        "metadata": metadata,
        "disc": disc,
        "diag": diag,
        "n_age": int(n_age),
        "n_z": int(n_z),
        "n_w": int(n_w),
        "state_grid_sizes": state_grid_sizes,
        "state_n_stds": state_n_stds,
        "wall_seconds": float(rc.get("wall_time_seconds", float("nan"))),
        "total_newton_failures": int(diag.get("total_newton_failures", -1)),
        "newton_iter_histogram": diag.get("newton_iter_histogram", {}),
        "age_newton_fail": diag.get("age_newton_fail"),
    }


# ---------------------------------------------------------------------------
# Reshape + per-axis bracket coords
# ---------------------------------------------------------------------------

def reshape_state_axis(arr: np.ndarray, sizes: tuple[int, int]) -> np.ndarray:
    """Split the flat state axis (axis=2) into (N_rtb, N_y1).

    The state cells are stored in row-major (C-order) lex order over
    state_grid_sizes — produced by `np.ndindex(N_rtb, N_y1)` in
    `_independence_rouwenhorst_pi`. So a `(..., N_state, ...)` array
    reshapes to `(..., N_rtb, N_y1, ...)` with C-order semantics.
    """
    n_age, n_z, n_state, n_w = arr.shape
    N_rtb, N_y1 = sizes
    if N_rtb * N_y1 != n_state:
        raise ValueError(f"sizes {sizes} don't multiply to {n_state}")
    return arr.reshape(n_age, n_z, N_rtb, N_y1, n_w)


def bracket_axis(N: int, sd: float) -> np.ndarray:
    if N == 1:
        return np.array([0.0])
    return np.linspace(-sd, sd, N)


# ---------------------------------------------------------------------------
# 2D interpolation onto a target (M_rtb, M_y1) grid in u-coords
# ---------------------------------------------------------------------------

def interp_to_grid_5d(
    arr_5d: np.ndarray,
    src_rtb: np.ndarray,
    src_y1: np.ndarray,
    dst_rtb: np.ndarray,
    dst_y1: np.ndarray,
) -> np.ndarray:
    """Interpolate `arr_5d` (A, Z, N_rtb, N_y1, W) along axes 2 and 3
    from `(src_rtb, src_y1)` to `(dst_rtb, dst_y1)`."""
    if arr_5d.shape[2] != src_rtb.size or arr_5d.shape[3] != src_y1.size:
        raise ValueError(
            f"shape {arr_5d.shape} incompatible with src grids "
            f"({src_rtb.size}, {src_y1.size})"
        )
    out = np.apply_along_axis(
        lambda v: np.interp(dst_rtb, src_rtb, v), axis=2, arr=arr_5d,
    )
    out = np.apply_along_axis(
        lambda v: np.interp(dst_y1, src_y1, v), axis=3, arr=out,
    )
    return out


def interp_to_grid_2d(
    arr_2d: np.ndarray,
    src_rtb: np.ndarray,
    src_y1: np.ndarray,
    dst_rtb: np.ndarray,
    dst_y1: np.ndarray,
) -> np.ndarray:
    """Interpolate (N_rtb, N_y1) -> (M_rtb, M_y1) by sequential np.interp."""
    interim = np.empty((dst_rtb.size, src_y1.size), dtype=arr_2d.dtype)
    for j in range(src_y1.size):
        interim[:, j] = np.interp(dst_rtb, src_rtb, arr_2d[:, j])
    out = np.empty((dst_rtb.size, dst_y1.size), dtype=arr_2d.dtype)
    for i in range(dst_rtb.size):
        out[i, :] = np.interp(dst_y1, src_y1, interim[i, :])
    return out


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

def _ref_threshold(ref: np.ndarray, sample_size: int = 200_000) -> float:
    """Estimate `1e-3 * median(|ref|[|ref|>0])` via subsample (memory-safe)."""
    flat = ref.ravel()
    if flat.size > sample_size:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(flat.size, size=sample_size, replace=False)
        s = flat[idx]
    else:
        s = flat
    s_abs = np.abs(s)
    pos = s_abs[s_abs > 0]
    if pos.size == 0:
        return 1e-8
    return max(1e-8, 1e-3 * float(np.median(pos)))


def scalar_metrics(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    sup = float(delta.max())
    p99 = float(np.percentile(delta, 99))
    p95 = float(np.percentile(delta, 95))
    rms = float(np.sqrt((delta ** 2).mean()))
    mean_abs = float(delta.mean())
    threshold = _ref_threshold(ref)
    ref_abs = np.abs(ref)
    mask = ref_abs > threshold
    if mask.any():
        rel = delta[mask] / ref_abs[mask]
        sup_rel = float(rel.max())
        p99_rel = float(np.percentile(rel, 99))
    else:
        sup_rel = float("nan")
        p99_rel = float("nan")
    return {
        "sup": sup, "p99": p99, "p95": p95, "rms": rms,
        "mean_abs": mean_abs,
        "sup_rel": sup_rel, "p99_rel": p99_rel,
        "rel_threshold": float(threshold),
    }


def per_axis_max(delta: np.ndarray) -> dict[str, list[float]]:
    """delta has shape (A, Z, N_rtb, N_y1, W) — reduce to per-axis profile."""
    return {
        "per_age":    [float(x) for x in delta.max(axis=(1, 2, 3, 4))],
        "per_z":      [float(x) for x in delta.max(axis=(0, 2, 3, 4))],
        "per_rtb":    [float(x) for x in delta.max(axis=(0, 1, 3, 4))],
        "per_y1":     [float(x) for x in delta.max(axis=(0, 1, 2, 4))],
        "per_wealth": [float(x) for x in delta.max(axis=(0, 1, 2, 3))],
    }


# ---------------------------------------------------------------------------
# Heatmap probe (cheap — single (age, z, wealth) slice)
# ---------------------------------------------------------------------------

def heatmap_probe(
    interim_5d: dict[str, np.ndarray],
    src_rtb: np.ndarray,
    src_y1: np.ndarray,
    age_idx: int, z_idx: int, w_idx: int,
    dst_rtb: np.ndarray,
    dst_y1: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, arr in interim_5d.items():
        slice_2d = arr[age_idx, z_idx, :, :, w_idx]
        out[k] = interp_to_grid_2d(slice_2d, src_rtb, src_y1, dst_rtb, dst_y1)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles-root", type=Path,
                    default=REPO / "saved_runs" / "ablations")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "docs" / "scans")
    ap.add_argument("--metrics-name", type=str,
                    default="system_ii_grid_convergence_metrics.json")
    ap.add_argument("--save-plot-bundle", type=Path, default=None,
                    help="Optional .npz with heatmap + per-axis arrays for plotting.")
    ap.add_argument("--heatmap-age", type=int, default=22,
                    help="Age index for the heatmap probe (0=22, default mid-working).")
    ap.add_argument("--heatmap-z-idx", type=int, default=5,
                    help="z index for the heatmap probe (5 = mean for n_z=11).")
    ap.add_argument("--heatmap-w-idx", type=int, default=89,
                    help="Wealth index for the heatmap probe (~SCF median area).")
    ap.add_argument("--heatmap-grid-size", type=int, default=HEATMAP_GRID_SIZE)
    args = ap.parse_args(argv)

    print(f"Loading bundles from: {args.bundles_root}")
    bundles: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_NAMES:
        b = load_bundle(args.bundles_root, name)
        bundles[name] = b
        print(f"  {name}: shape={b['C'].shape}, "
              f"state_grid_sizes={b['state_grid_sizes']}, "
              f"state_n_stds={b['state_n_stds']}, "
              f"wall={b['wall_seconds']:.0f}s, "
              f"total_newton_fails={b['total_newton_failures']}")

    # Validation gates
    n_age = bundles[REFERENCE_NAME]["n_age"]
    n_z = bundles[REFERENCE_NAME]["n_z"]
    n_w = bundles[REFERENCE_NAME]["n_w"]
    n_stds = bundles[REFERENCE_NAME]["state_n_stds"]
    print("\nValidation gates:")
    for name, b in bundles.items():
        if (b["n_age"], b["n_z"], b["n_w"]) != (n_age, n_z, n_w):
            print(f"  WARN {name}: shape disagree on (n_age, n_z, n_w)")
        if b["state_n_stds"] != n_stds:
            print(f"  WARN {name}: state_n_stds={b['state_n_stds']} != ref {n_stds}")
        for arr_name in ("C", "S", "B"):
            if not np.all(np.isfinite(b[arr_name])):
                n_bad = int(np.sum(~np.isfinite(b[arr_name])))
                print(f"  WARN {name}: {arr_name} has {n_bad} non-finite cells")
    ref = bundles[REFERENCE_NAME]
    print(f"  ref={REFERENCE_NAME} | C in [{ref['C'].min():.3e}, {ref['C'].max():.3e}], "
          f"S in [{ref['S'].min():.3f}, {ref['S'].max():.3f}], "
          f"B in [{ref['B'].min():.3f}, {ref['B'].max():.3f}]")

    # Reshape
    print("\nReshaping (A, Z, N_state, W) -> (A, Z, N_rtb, N_y1, W)...")
    for name, b in bundles.items():
        sizes = b["state_grid_sizes"]
        for k in ("C", "S", "B"):
            b[k + "_5d"] = reshape_state_axis(b[k], sizes)
        b["src_rtb"] = bracket_axis(sizes[0], n_stds[0])
        b["src_y1"] = bracket_axis(sizes[1], n_stds[1])
        print(f"  {name}: 5d shape {b['C_5d'].shape}, "
              f"rtb-bracket={np.array2string(b['src_rtb'], precision=4)}, "
              f"y1-bracket={np.array2string(b['src_y1'], precision=4)}")

    # Bulk-metrics eval grid: the (6,6) reference's bracket grid
    ref_rtb = ref["src_rtb"]
    ref_y1 = ref["src_y1"]
    print(f"\nBulk-metrics eval grid: ref bracket grid "
          f"({ref_rtb.size}, {ref_y1.size}) on u-coords "
          f"rtb in [{ref_rtb[0]:.2f}, {ref_rtb[-1]:.2f}], "
          f"y1 in [{ref_y1[0]:.2f}, {ref_y1[-1]:.2f}]")

    # Interpolate coarse bundles onto ref bracket grid (small memory)
    print("\nInterpolating coarse bundles onto ref bracket grid...")
    interp_on_ref: dict[str, dict[str, np.ndarray]] = {REFERENCE_NAME: {}}
    for k in ("C", "S", "B"):
        interp_on_ref[REFERENCE_NAME][k] = ref[k + "_5d"]   # already on ref grid
    for name in BUNDLE_NAMES:
        if name == REFERENCE_NAME:
            continue
        b = bundles[name]
        d: dict[str, np.ndarray] = {}
        for k in ("C", "S", "B"):
            d[k] = interp_to_grid_5d(
                b[k + "_5d"], b["src_rtb"], b["src_y1"], ref_rtb, ref_y1,
            )
        interp_on_ref[name] = d
        print(f"  {name}: interp shape {d['C'].shape}")

    # Self-compare sanity: ref reshaped == ref
    print("\nSelf-compare sanity (ref vs ref on ref grid):")
    for k in ("C", "S", "B"):
        d = float(np.max(np.abs(ref[k + "_5d"] - interp_on_ref[REFERENCE_NAME][k])))
        print(f"  sup|delta_{k}| = {d:.3e}")

    # Pairwise divergence on the ref bracket grid
    print("\nPairwise divergence vs (6, 6) on ref bracket grid:")
    print(f"  {'pair':<28} {'arr':>3} {'sup':>10} {'p99':>10} "
          f"{'rms':>10} {'sup_rel':>10}")
    pairs: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_NAMES:
        if name == REFERENCE_NAME:
            continue
        per_arr: dict[str, Any] = {}
        for k in ("C", "S", "B"):
            d = np.abs(interp_on_ref[name][k] - interp_on_ref[REFERENCE_NAME][k])
            scalar = scalar_metrics(d, interp_on_ref[REFERENCE_NAME][k])
            per_axis = per_axis_max(d)
            per_arr[k] = {**scalar, **per_axis}
            print(f"  {name:<28} {k:>3} {scalar['sup']:>10.3e} "
                  f"{scalar['p99']:>10.3e} {scalar['rms']:>10.3e} "
                  f"{scalar['sup_rel']:>10.3e}")
        pairs[name] = {
            "state_grid_sizes": list(bundles[name]["state_grid_sizes"]),
            "wall_seconds": bundles[name]["wall_seconds"],
            "total_newton_failures": bundles[name]["total_newton_failures"],
            "C": per_arr["C"], "S": per_arr["S"], "B": per_arr["B"],
        }

    # Heatmap probe — 50x50 (rtb, y_1) on a single (age, z, wealth) slice
    M = int(args.heatmap_grid_size)
    dst_rtb = np.linspace(-n_stds[0], n_stds[0], M)
    dst_y1 = np.linspace(-n_stds[1], n_stds[1], M)
    a_idx, z_idx, w_idx = (
        int(args.heatmap_age), int(args.heatmap_z_idx), int(args.heatmap_w_idx),
    )
    print(f"\nHeatmap probe at (age_idx={a_idx} -> age={22+a_idx}, "
          f"z_idx={z_idx}/{n_z-1}, w_idx={w_idx}/{n_w-1}) on {M}x{M} eval grid:")
    heatmap: dict[str, dict[str, np.ndarray]] = {}
    for name, b in bundles.items():
        slice_in_5d = {k: b[k + "_5d"] for k in ("C", "S", "B")}
        heatmap[name] = heatmap_probe(
            slice_in_5d, b["src_rtb"], b["src_y1"],
            a_idx, z_idx, w_idx, dst_rtb, dst_y1,
        )
    # Heatmap deltas vs ref on shared (50, 50) grid
    heatmap_delta: dict[str, dict[str, np.ndarray]] = {}
    for name in BUNDLE_NAMES:
        if name == REFERENCE_NAME:
            continue
        d: dict[str, np.ndarray] = {}
        for k in ("C", "S", "B"):
            d[k] = np.abs(heatmap[name][k] - heatmap[REFERENCE_NAME][k])
        heatmap_delta[name] = d
        print(f"  {name}: heatmap sup|deltaS|={d['S'].max():.3e}, "
              f"sup|deltaB|={d['B'].max():.3e}, sup|deltaC|={d['C'].max():.3e}")

    # Probe lines for the convergence-curve plot — at the same (age, z, wealth)
    # but VARYING age over working years, fixed z=mean, fixed w=median, at a
    # fixed (rtb, y_1) cell — pick the centre of the eval grid (rtb=0, y_1=0).
    # Sample at the eval-grid centre via interp_to_grid_2d on a 1x1 dst grid.
    dst_centre_rtb = np.array([0.0])
    dst_centre_y1 = np.array([0.0])
    probe_lines: dict[str, dict[str, np.ndarray]] = {}
    for name, b in bundles.items():
        per_arr: dict[str, np.ndarray] = {}
        for k in ("C", "S", "B"):
            arr = b[k + "_5d"][:, z_idx, :, :, w_idx]   # (A, N_rtb, N_y1)
            out = np.empty(arr.shape[0])
            for a in range(arr.shape[0]):
                v = interp_to_grid_2d(
                    arr[a], b["src_rtb"], b["src_y1"],
                    dst_centre_rtb, dst_centre_y1,
                )
                out[a] = float(v[0, 0])
            per_arr[k] = out
        probe_lines[name] = per_arr

    # Distribution snapshots
    distribution = {}
    for name, b in bundles.items():
        distribution[name] = {
            "C_quantiles": [float(q) for q in np.quantile(
                b["C"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "S_quantiles": [float(q) for q in np.quantile(
                b["S"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "B_quantiles": [float(q) for q in np.quantile(
                b["B"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "S_min": float(b["S"].min()), "S_max": float(b["S"].max()),
            "B_min": float(b["B"].min()), "B_max": float(b["B"].max()),
            "C_min": float(b["C"].min()), "C_max": float(b["C"].max()),
        }

    # Newton-failure cross-reference
    newton_xref = {}
    for name, b in bundles.items():
        afl = b["age_newton_fail"]
        if isinstance(afl, np.ndarray):
            arr_list = [int(x) for x in afl]
        else:
            arr_list = []
        hist = b["newton_iter_histogram"] or {}
        newton_xref[name] = {
            "total_newton_failures": b["total_newton_failures"],
            "age_newton_fail": arr_list,
            "iter_p50": float(hist.get("p50", float("nan"))),
            "iter_p95": float(hist.get("p95", float("nan"))),
            "iter_p99": float(hist.get("p99", float("nan"))),
            "iter_max": float(hist.get("max", float("nan"))),
            "n_cells": int(hist.get("n_cells", -1)),
        }

    # Convergence-curve summary: N_state -> sup-norm vs ref
    convergence_curve = {
        "labels": list(BUNDLE_NAMES),
        "N_state": [int(np.prod(bundles[n]["state_grid_sizes"])) for n in BUNDLE_NAMES],
        "state_grid_sizes": [list(bundles[n]["state_grid_sizes"]) for n in BUNDLE_NAMES],
        "wall_seconds": [bundles[n]["wall_seconds"] for n in BUNDLE_NAMES],
        "sup_C": [], "sup_S": [], "sup_B": [],
        "rms_C": [], "rms_S": [], "rms_B": [],
    }
    for name in BUNDLE_NAMES:
        if name == REFERENCE_NAME:
            for k in ("C", "S", "B"):
                convergence_curve[f"sup_{k}"].append(0.0)
                convergence_curve[f"rms_{k}"].append(0.0)
        else:
            for k in ("C", "S", "B"):
                convergence_curve[f"sup_{k}"].append(pairs[name][k]["sup"])
                convergence_curve[f"rms_{k}"].append(pairs[name][k]["rms"])

    payload = {
        "bundles_root": str(args.bundles_root),
        "bundle_names": list(BUNDLE_NAMES),
        "reference": REFERENCE_NAME,
        "reference_state_grid_sizes": list(bundles[REFERENCE_NAME]["state_grid_sizes"]),
        "reference_state_n_stds": list(bundles[REFERENCE_NAME]["state_n_stds"]),
        "reference_shape": list(bundles[REFERENCE_NAME]["C"].shape),
        "bulk_eval_grid": {
            "kind": "ref_bracket",
            "ref_rtb": ref_rtb.tolist(),
            "ref_y1": ref_y1.tolist(),
        },
        "heatmap_eval_grid": {
            "M": M,
            "dst_rtb": dst_rtb.tolist(),
            "dst_y1": dst_y1.tolist(),
            "probe_age_idx": a_idx,
            "probe_age": 22 + a_idx,
            "probe_z_idx": z_idx,
            "probe_w_idx": w_idx,
        },
        "pairs": pairs,
        "convergence_curve": convergence_curve,
        "distributions": distribution,
        "newton_xref": newton_xref,
        "state_axis_names": list(STATE_AXIS_NAMES),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.metrics_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote metrics to {out_path}")

    if args.save_plot_bundle is not None:
        save_kwargs: dict[str, np.ndarray] = {
            "dst_rtb": dst_rtb, "dst_y1": dst_y1,
            "ref_rtb": ref_rtb, "ref_y1": ref_y1,
            "ages": np.arange(22, 22 + n_age),
        }
        for name in BUNDLE_NAMES:
            tag = name.replace("system_ii_grid", "g").split("_nz")[0]
            for k in ("C", "S", "B"):
                save_kwargs[f"{tag}_heatmap_{k}"] = heatmap[name][k]
                save_kwargs[f"{tag}_probe_{k}"] = probe_lines[name][k]
            if name != REFERENCE_NAME:
                for k in ("C", "S", "B"):
                    save_kwargs[f"{tag}_heatdelta_{k}"] = heatmap_delta[name][k]
        # Also save the per-axis decomposition arrays for the pairs
        for name in BUNDLE_NAMES:
            if name == REFERENCE_NAME:
                continue
            tag = name.replace("system_ii_grid", "g").split("_nz")[0]
            for k in ("C", "S", "B"):
                m = pairs[name][k]
                save_kwargs[f"{tag}_per_age_{k}"] = np.asarray(m["per_age"])
                save_kwargs[f"{tag}_per_z_{k}"] = np.asarray(m["per_z"])
                save_kwargs[f"{tag}_per_rtb_{k}"] = np.asarray(m["per_rtb"])
                save_kwargs[f"{tag}_per_y1_{k}"] = np.asarray(m["per_y1"])
                save_kwargs[f"{tag}_per_wealth_{k}"] = np.asarray(m["per_wealth"])
        args.save_plot_bundle.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_plot_bundle, **save_kwargs)
        print(f"Wrote plot bundle to {args.save_plot_bundle}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
