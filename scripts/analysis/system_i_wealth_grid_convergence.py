"""
LEGACY: System I (pre-pivot) wealth-grid convergence: log1p vs Bakhvalov.

This script is read-only on the eight paired pre-pivot bundles
    saved_runs/ablations/system_i_grid7_nz25_w{60,90,120,180}_{log,bakh}_calib1
and does not import any VAR builder, so it still runs on the post-pivot
branch. The analogous post-pivot study would re-solve the same paired
sweep on System 1 (real bill yield only) and analyse via a renamed
script `system_1_wealth_grid_convergence.py`; until then, this file
documents the pre-pivot finding for retrospective use.

Takes log_180 as the reference (the pre-pivot production canonical),
interpolates each non-reference bundle's policies onto a shared
log1p-uniform eval grid, and computes pairwise sup/p99/RMS divergences
for C/S/B. Also builds the load-bearing side-by-side n_w table, per-axis
decomposition, Newton failure rates, and the equidistribution check
(per-cell h^2 * |V''| should be ~constant on Bakhvalov, varying on log1p).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lifecycle.policy_io import load_policy_bundle  # noqa: E402
from lifecycle.wealth_grid import load_wealth_grid_from_bundle, wealth_grid_hash  # noqa: E402

ABL = REPO / "saved_runs" / "ablations"
N_W_VALUES: tuple[int, ...] = (60, 90, 120, 180)
GRID_KINDS: tuple[str, ...] = ("log", "bakh")
DEFAULT_REFERENCE_KEY: tuple[str, int] = ("log", 180)
BUNDLE_TEMPLATE = "system_i_grid7_nz25_w{nw}_{kind}_calib1"

# Shared evaluation grid: log1p-uniform, ~500 nodes covering the canonical range.
WMIN: float = 0.05
WMAX: float = 750.0
N_EVAL: int = 500


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _bundle_path(kind: str, n_w: int) -> Path:
    return ABL / BUNDLE_TEMPLATE.format(nw=n_w, kind=kind)


def load_one(kind: str, n_w: int) -> dict[str, Any]:
    bundle = _bundle_path(kind, n_w)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")
    C, S, B, diag, metadata = load_policy_bundle(bundle)
    wg = load_wealth_grid_from_bundle(bundle, metadata)
    if wg.size != C.shape[-1]:
        raise ValueError(
            f"{bundle.name}: wealth_grid size {wg.size} != policy axis {C.shape[-1]}"
        )
    # Hash sanity: file hash must match metadata hash (load_wealth_grid_from_bundle
    # already enforces this; we re-record for the report).
    h = wealth_grid_hash(wg)
    return {
        "key": (kind, n_w),
        "label": f"{kind}_{n_w}",
        "kind": kind,
        "n_w": n_w,
        "bundle": bundle,
        "C": np.asarray(C, dtype=np.float64),
        "S": np.asarray(S, dtype=np.float64),
        "B": np.asarray(B, dtype=np.float64),
        "wealth_grid": wg,
        "wealth_grid_hash": h,
        "metadata": metadata,
        "diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Interp
# ---------------------------------------------------------------------------

def build_eval_grid(wmin: float = WMIN, wmax: float = WMAX, n_eval: int = N_EVAL) -> np.ndarray:
    return np.expm1(np.linspace(np.log1p(wmin), np.log1p(wmax), n_eval))


def interp_to_eval(wealth_grid: np.ndarray, policy: np.ndarray, eval_grid: np.ndarray) -> np.ndarray:
    """Vectorized linear interp along the wealth axis (axis=-1).

    Manually computes (idx, frac) once on the eval grid, then gathers across
    the leading (age, z, state) axes. Equivalent to per-cell np.interp but
    ~50x faster on the (78, 25, 7, n_w) -> (78, 25, 7, 500) reshape.
    """
    wg = np.ascontiguousarray(wealth_grid, dtype=np.float64)
    n_w = wg.size
    # For each eval point, find left bracket index in [0, n_w-2].
    idx = np.searchsorted(wg, eval_grid, side="right") - 1
    idx = np.clip(idx, 0, n_w - 2)
    w_left = wg[idx]
    w_right = wg[idx + 1]
    # Clip frac to [0, 1] -> linear extrapolation becomes endpoint clamp,
    # which matches np.interp's default behaviour.
    frac = np.clip((eval_grid - w_left) / (w_right - w_left), 0.0, 1.0)
    left = policy[..., idx]
    right = policy[..., idx + 1]
    return left + (right - left) * frac


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _scalar_metrics(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    sup = float(np.max(delta))
    p99 = float(np.percentile(delta, 99))
    p95 = float(np.percentile(delta, 95))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    mean_abs = float(np.mean(delta))
    ref_abs = np.abs(ref)
    pos = ref_abs[ref_abs > 0]
    threshold = max(1e-8, 1e-3 * float(np.median(pos)) if pos.size else 1e-8)
    mask = ref_abs > threshold
    if mask.any():
        rel = delta[mask] / ref_abs[mask]
        sup_rel = float(np.max(rel))
        p99_rel = float(np.percentile(rel, 99))
    else:
        sup_rel = float("nan")
        p99_rel = float("nan")
    return {
        "sup": sup, "p99": p99, "p95": p95, "rms": rms, "mean_abs": mean_abs,
        "sup_rel": sup_rel, "p99_rel": p99_rel, "rel_threshold": float(threshold),
    }


def _per_axis_max(delta: np.ndarray) -> dict[str, list[float]]:
    return {
        "per_age":    [float(x) for x in np.max(delta, axis=(1, 2, 3))],
        "per_z":      [float(x) for x in np.max(delta, axis=(0, 2, 3))],
        "per_state":  [float(x) for x in np.max(delta, axis=(0, 1, 3))],
        "per_wealth": [float(x) for x in np.max(delta, axis=(0, 1, 2))],
    }


def newton_fail_rate(bundle: dict[str, Any]) -> float:
    """Fraction of (age, z, state, w) cells where Newton hit max_iter without convergence.

    n_cells_per_age = n_z * n_state * n_w; total cells across solved ages.
    """
    C = bundle["C"]
    n_ages, n_z, n_state, n_w = C.shape
    diag = bundle["diagnostics"] or {}
    total_fail = int(diag.get("total_newton_failures", 0))
    n_solved = int(diag.get("n_ages_solved", n_ages))
    cells = n_solved * n_z * n_state * n_w
    return float(total_fail) / max(1, cells)


# ---------------------------------------------------------------------------
# Wealth-axis curvature (equidistribution check, in-process)
# ---------------------------------------------------------------------------

def second_deriv_nonuniform(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """3-point FD 2nd derivative on a non-uniform 1-D grid (returns interior length-(N-2))."""
    h_left = x[1:-1] - x[:-2]
    h_right = x[2:] - x[1:-1]
    return 2.0 * ((y[2:] - y[1:-1]) / h_right - (y[1:-1] - y[:-2]) / h_left) / (h_left + h_right)


def cellwise_h2_d2(policy_slab: np.ndarray, wealth_grid: np.ndarray) -> np.ndarray:
    """Per-cell h^2 * |V''| on a 1-D policy slab. Length n_w-2 (interior)."""
    d2 = np.abs(second_deriv_nonuniform(policy_slab, wealth_grid))
    h = 0.5 * (wealth_grid[2:] - wealth_grid[:-2])
    return h * h * d2


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_grid_density(bundles: dict, fig_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    cmap_log = plt.get_cmap("Blues")
    cmap_bak = plt.get_cmap("Oranges")
    n_levels = len(N_W_VALUES)
    for i, n_w in enumerate(N_W_VALUES):
        wg_log = bundles[("log", n_w)]["wealth_grid"]
        wg_bak = bundles[("bakh", n_w)]["wealth_grid"]
        # color intensity grows with n_w
        c_log = cmap_log(0.4 + 0.6 * i / max(1, n_levels - 1))
        c_bak = cmap_bak(0.4 + 0.6 * i / max(1, n_levels - 1))
        y_log = 1.0 + 0.1 * i
        y_bak = 0.0 - 0.1 * i
        ax.scatter(wg_log, np.full_like(wg_log, y_log), s=8, color=c_log,
                   marker="|", label=f"log_{n_w}")
        ax.scatter(wg_bak, np.full_like(wg_bak, y_bak), s=8, color=c_bak,
                   marker="|", label=f"bakh_{n_w}")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("wealth W (AWI units)")
    ax.set_ylabel("(top: log1p-uniform / bottom: Bakhvalov)")
    ax.set_yticks([])
    ax.set_title("Wealth-grid node placement: log1p-uniform vs Bakhvalov density-weighted")
    ax.axvspan(0.1, 1.0, alpha=0.10, color="gray", label="kink band [0.1, 1.0]")
    ax.legend(ncol=4, fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def plot_convergence_curves(results: dict, fig_path: Path,
                            reference_key: tuple[str, int]) -> None:
    ref_label = f"{reference_key[0]}_{reference_key[1]}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, ("C", "S", "B")):
        for kind, color in (("log", "#1f77b4"), ("bakh", "#ff7f0e")):
            xs, ys = [], []
            for n_w in N_W_VALUES:
                key = (kind, n_w)
                if key == reference_key:
                    continue
                xs.append(n_w)
                ys.append(results[key][name]["sup"])
            ax.plot(xs, ys, marker="o", lw=1.6, color=color, label=kind)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_w")
        ax.set_ylabel(f"sup |Δ{name}| vs {ref_label}")
        ax.set_title(f"{name} convergence")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"System I wealth-grid convergence: log1p-uniform vs Bakhvalov vs {ref_label} reference"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def plot_per_cell_probe(bundles: dict, eval_grid: np.ndarray,
                        interp_cache: dict, fig_path: Path,
                        reference_key: tuple[str, int]) -> None:
    """Overlay all 8 grid policies at a single (age, z, state) probe cell.

    Uses age=45 (working age, in the kink band), z=mid, state=mid (rtb=middle node).
    """
    ref = bundles[reference_key]
    n_ages, n_z, n_state, _ = ref["C"].shape
    age_probe = 45 - 22  # start_age=22 for full lifecycle System I
    age_probe = max(0, min(age_probe, n_ages - 1))
    z_mid = n_z // 2
    s_mid = n_state // 2
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    arrays = ("C", "S", "B")
    for ax, name in zip(axes, arrays):
        for kind, ls in (("log", "-"), ("bakh", "--")):
            for n_w in N_W_VALUES:
                key = (kind, n_w)
                arr = interp_cache[key][name][age_probe, z_mid, s_mid, :]
                base_alpha = 0.35 + 0.65 * (N_W_VALUES.index(n_w) / (len(N_W_VALUES) - 1))
                color = "#1f77b4" if kind == "log" else "#ff7f0e"
                lw = 2.0 if key == reference_key else 1.0
                ax.plot(eval_grid, arr, ls=ls, color=color, alpha=base_alpha, lw=lw,
                        label=f"{kind}_{n_w}")
        ax.set_xscale("log")
        ax.set_xlabel("wealth W (AWI units)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} at age 45, z=mid, state=mid")
        ax.grid(True, which="both", alpha=0.3)
        ax.axvspan(0.1, 1.0, alpha=0.07, color="gray")
        if name == "C":
            ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Per-cell probe: log_60..180 vs bakh_60..180 (gray band = kink band)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def plot_per_wealth_bin(results: dict, eval_grid: np.ndarray, fig_path: Path,
                        reference_key: tuple[str, int]) -> None:
    """sup |Δ| reduced over (age, z, state) per wealth bin -- one line per bundle."""
    ref_label = f"{reference_key[0]}_{reference_key[1]}"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, ("C", "S", "B")):
        for kind, color in (("log", "#1f77b4"), ("bakh", "#ff7f0e")):
            for n_w in N_W_VALUES:
                key = (kind, n_w)
                if key == reference_key:
                    continue
                per_w = np.asarray(results[key][name]["per_wealth"])
                base_alpha = 0.35 + 0.65 * (N_W_VALUES.index(n_w) / (len(N_W_VALUES) - 1))
                ax.plot(eval_grid, per_w, color=color, alpha=base_alpha,
                        label=f"{kind}_{n_w}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wealth W (AWI units)")
        ax.set_ylabel(f"sup |Δ{name}| over (age, z, state)")
        ax.set_title(f"Per-wealth-bin divergence vs {ref_label}, {name}")
        ax.axvspan(0.1, 1.0, alpha=0.07, color="gray")
        ax.grid(True, which="both", alpha=0.3)
        if name == "C":
            ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Per-wealth-bin sup-divergence (gray = kink band [0.1, 1.0] AWI)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def plot_equidistribution(bundles: dict, fig_path: Path,
                          reference_key: tuple[str, int]) -> None:
    """Per-cell h^2 |V''| at age 45, z=mid, state=mid for log_180 and bakh_180.

    Bakhvalov's invariant: this should be ~flat. log1p will be sharply peaked
    in the kink band.
    """
    ref = bundles[reference_key]
    n_ages, n_z, n_state, _ = ref["C"].shape
    age_probe = 45 - 22
    z_mid = n_z // 2
    s_mid = n_state // 2
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, ("C", "S", "B")):
        for kind, color in (("log", "#1f77b4"), ("bakh", "#ff7f0e")):
            b = bundles[(kind, 180)]
            slab = b[name][age_probe, z_mid, s_mid, :]
            wg = b["wealth_grid"]
            err = cellwise_h2_d2(slab, wg)
            interior_w = wg[1:-1]
            ax.plot(interior_w, err + 1e-30, marker="o", ms=2.5, lw=1.0,
                    color=color, label=f"{kind}_180")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wealth W (AWI units)")
        ax.set_ylabel(f"h^2 * |{name}''(W)|")
        ax.set_title(f"Equidistribution check, {name}, age 45, z=mid, state=mid")
        ax.axvspan(0.1, 1.0, alpha=0.07, color="gray")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Per-cell h^2*|V''| -- Bakhvalov should be flatter than log1p")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", type=Path, default=ABL)
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs" / "scans")
    parser.add_argument("--metrics-name", type=str,
                        default="system_i_wealth_grid_convergence_metrics.json")
    parser.add_argument("--fig-dir", type=Path,
                        default=REPO / "docs" / "scans" / "figures")
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--reference-kind", type=str, default=DEFAULT_REFERENCE_KEY[0],
                        choices=list(GRID_KINDS),
                        help="Grid kind to use as reference (log or bakh).")
    parser.add_argument("--reference-nw", type=int, default=DEFAULT_REFERENCE_KEY[1],
                        choices=list(N_W_VALUES),
                        help="n_w of reference bundle.")
    parser.add_argument("--fig-suffix", type=str, default="",
                        help="Suffix appended to figure filenames (e.g. '_bakh_ref').")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    reference_key: tuple[str, int] = (args.reference_kind, args.reference_nw)
    ref_label = f"{reference_key[0]}_{reference_key[1]}"
    print(f"Reference bundle: {ref_label}")

    eval_grid = build_eval_grid(WMIN, WMAX, args.n_eval)

    print(f"Loading 8 bundles from {args.bundles_root}...")
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    for kind in GRID_KINDS:
        for n_w in N_W_VALUES:
            b = load_one(kind, n_w)
            bundles[(kind, n_w)] = b
            wg = b["wealth_grid"]
            print(f"  {b['label']:>9}: shape={b['C'].shape}, "
                  f"wg=[{wg[0]:.4g}, {wg[-1]:.4g}], n={wg.size}, "
                  f"hash={b['wealth_grid_hash'][:12]}")

    # --- Validation
    print("\nValidation gates:")
    ref = bundles[reference_key]
    for key, b in bundles.items():
        if not np.all(np.isfinite(b["C"])) or not np.all(np.isfinite(b["S"])) \
           or not np.all(np.isfinite(b["B"])):
            print(f"  FAIL: {b['label']} has non-finite policy entries")
            return 1
        if b["C"].shape[1:3] != ref["C"].shape[1:3] or b["C"].shape[0] != ref["C"].shape[0]:
            print(f"  FAIL: {b['label']} shape {b['C'].shape} mismatch with ref {ref['C'].shape}")
            return 1
    print(f"  all 8 bundles loaded; shapes consistent: (n_ages, n_z, n_state) = "
          f"{tuple(ref['C'].shape[:3])}")
    # Check Bakhvalov grids really are denser at the kink band than log_180
    log180_wg = bundles[("log", 180)]["wealth_grid"]
    bak180_wg = bundles[("bakh", 180)]["wealth_grid"]
    band_lo, band_hi = 0.05, 1.0
    n_log_band = int(((log180_wg >= band_lo) & (log180_wg <= band_hi)).sum())
    n_bak_band = int(((bak180_wg >= band_lo) & (bak180_wg <= band_hi)).sum())
    print(f"  kink-band [0.05, 1.0] node count: log_180={n_log_band}, bakh_180={n_bak_band}")

    # --- Interpolate everything to shared eval grid
    print(f"\nInterpolating all bundles to {args.n_eval}-pt shared eval grid "
          f"[{WMIN}, {WMAX}]...")
    interp: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, b in bundles.items():
        interp[key] = {
            name: interp_to_eval(b["wealth_grid"], b[name], eval_grid)
            for name in ("C", "S", "B")
        }
        print(f"  {b['label']:>9}: interp shapes "
              f"{tuple(interp[key]['C'].shape)}")

    # Sanity: ref vs itself = exactly 0 (modulo float roundoff)
    ref_interp = interp[reference_key]
    self_diff = float(np.max(np.abs(
        interp_to_eval(ref["wealth_grid"], ref["C"], eval_grid) - ref_interp["C"]
    )))
    print(f"\nSelf-compare ({ref_label} vs itself) sup_C diff: {self_diff:.3e}")
    if self_diff > 1e-12:
        print(f"  WARN: self-compare > 1e-12 (interp framework reproducibility)")

    # --- Compute pairwise metrics
    print(f"\nComputing pairwise divergences vs {ref_label}...")
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for key in bundles:
        if key == reference_key:
            continue
        kind, n_w = key
        bundle_metrics: dict[str, Any] = {"label": f"{kind}_{n_w}", "kind": kind, "n_w": n_w}
        for name in ("C", "S", "B"):
            ref_arr = ref_interp[name]
            arr = interp[key][name]
            delta = np.abs(arr - ref_arr)
            scalar = _scalar_metrics(delta, ref_arr)
            per_axis = _per_axis_max(delta)
            bundle_metrics[name] = {**scalar, **per_axis}
        # Newton failure rate at this bundle
        bundle_metrics["newton_fail_rate"] = newton_fail_rate(bundles[key])
        results[key] = bundle_metrics

    # Reference Newton fail rate
    ref_nfr = newton_fail_rate(ref)
    print(f"\nNewton failure rates (max_iter=30 hits / total cells):")
    print(f"  {ref_label} (ref): {ref_nfr*100:.2f}%")
    for n_w in N_W_VALUES:
        for kind in GRID_KINDS:
            key = (kind, n_w)
            if key == reference_key:
                continue
            r = results[key]["newton_fail_rate"]
            print(f"  {kind}_{n_w}: {r*100:.2f}%")

    # --- Headline tables
    print(f"\n=== Headline divergence (sup, p99, RMS) vs {ref_label} ===")
    print(f"{'bundle':>10} | {'sup_C':>10} {'p99_C':>10} {'rms_C':>10} | "
          f"{'sup_S':>10} {'sup_B':>10}")
    for n_w in N_W_VALUES:
        for kind in GRID_KINDS:
            key = (kind, n_w)
            if key == reference_key:
                continue
            r = results[key]
            print(f"  {r['label']:>8} | {r['C']['sup']:.4e} {r['C']['p99']:.4e} {r['C']['rms']:.4e} | "
                  f"{r['S']['sup']:.4e} {r['B']['sup']:.4e}")

    print(f"\n=== Side-by-side: log_N vs {ref_label}  vs  bakh_N vs {ref_label} ===")
    print(f"{'n_w':>4} | {'sup|dC| log':>14} {'sup|dC| bakh':>14} {'gain (log/bakh)':>18}")
    side_by_side = {}
    for n_w in N_W_VALUES:
        log_key = ("log", n_w)
        bakh_key = ("bakh", n_w)
        log_sup_C = 0.0 if log_key == reference_key else results[log_key]["C"]["sup"]
        bakh_sup_C = 0.0 if bakh_key == reference_key else results[bakh_key]["C"]["sup"]
        if log_sup_C > 0 and bakh_sup_C > 0:
            gain_C = log_sup_C / bakh_sup_C
        else:
            gain_C = float("nan")
        side_by_side[n_w] = {
            "log_sup_C": log_sup_C,
            "bakh_sup_C": bakh_sup_C,
            "gain_C": gain_C,
        }
        for name in ("S", "B"):
            log_sup = 0.0 if log_key == reference_key else results[log_key][name]["sup"]
            bakh_sup = 0.0 if bakh_key == reference_key else results[bakh_key][name]["sup"]
            side_by_side[n_w][f"log_sup_{name}"] = log_sup
            side_by_side[n_w][f"bakh_sup_{name}"] = bakh_sup
            if log_sup > 0 and bakh_sup > 0:
                side_by_side[n_w][f"gain_{name}"] = log_sup / bakh_sup
            else:
                side_by_side[n_w][f"gain_{name}"] = float("nan")
        print(f"  {n_w:>3} | {log_sup_C:14.4e} {bakh_sup_C:14.4e} {gain_C:18.2f}x")

    # --- Plots
    sfx = args.fig_suffix
    print("\nWriting figures...")
    plot_grid_density(bundles, args.fig_dir / f"wealth_grid_density{sfx}.png")
    plot_convergence_curves(results, args.fig_dir / f"wealth_grid_convergence_curves{sfx}.png",
                            reference_key)
    plot_per_cell_probe(bundles, eval_grid, interp,
                        args.fig_dir / f"wealth_grid_per_cell_probe{sfx}.png",
                        reference_key)
    plot_per_wealth_bin(results, eval_grid,
                        args.fig_dir / f"wealth_grid_per_wealth_bin{sfx}.png",
                        reference_key)
    plot_equidistribution(bundles, args.fig_dir / f"wealth_grid_equidistribution{sfx}.png",
                          reference_key)

    # --- Equidistribution numerical summary
    eq_summary: dict[str, Any] = {}
    n_ages, n_z, n_state, _ = ref["C"].shape
    age_probe = 45 - 22
    z_mid, s_mid = n_z // 2, n_state // 2
    for kind in GRID_KINDS:
        b = bundles[(kind, 180)]
        wg = b["wealth_grid"]
        slab_C = b["C"][age_probe, z_mid, s_mid, :]
        err = cellwise_h2_d2(slab_C, wg)
        eq_summary[f"{kind}_180_C"] = {
            "h2_d2_max": float(err.max()),
            "h2_d2_min": float(err.min() + 1e-30),
            "h2_d2_median": float(np.median(err)),
            "h2_d2_mean": float(err.mean()),
            "h2_d2_max_over_min": float((err.max() + 1e-30) / (err.min() + 1e-30)),
        }
    print("\n=== Equidistribution: per-cell h^2 |C''| at age 45, z=mid, s=mid ===")
    for k, v in eq_summary.items():
        print(f"  {k}: max/min ratio = {v['h2_d2_max_over_min']:.2e}, "
              f"max={v['h2_d2_max']:.3e}, median={v['h2_d2_median']:.3e}")

    # --- Distribution snapshots
    distributions = {}
    for key, b in bundles.items():
        distributions[b["label"]] = {
            "C_quantiles": [float(q) for q in np.quantile(b["C"], [0.01, 0.5, 0.99])],
            "S_quantiles": [float(q) for q in np.quantile(b["S"], [0.01, 0.5, 0.99])],
            "B_quantiles": [float(q) for q in np.quantile(b["B"], [0.01, 0.5, 0.99])],
            "S_min": float(b["S"].min()), "S_max": float(b["S"].max()),
            "B_min": float(b["B"].min()), "B_max": float(b["B"].max()),
            "C_min": float(b["C"].min()), "C_max": float(b["C"].max()),
        }

    # --- Serialize
    payload = {
        "bundles_root": str(args.bundles_root),
        "n_w_values": list(N_W_VALUES),
        "grid_kinds": list(GRID_KINDS),
        "reference": list(reference_key),
        "reference_shape": list(ref["C"].shape),
        "eval_grid_min": float(eval_grid[0]),
        "eval_grid_max": float(eval_grid[-1]),
        "eval_grid_n": int(eval_grid.size),
        "self_compare_sup_C": self_diff,
        "kink_band_node_counts": {
            "log_180": n_log_band,
            "bakh_180": n_bak_band,
            "band_range_AWI": [band_lo, band_hi],
        },
        "newton_fail_rate": {
            **{f"{kind}_{n_w}": results[(kind, n_w)]["newton_fail_rate"]
               for kind in GRID_KINDS for n_w in N_W_VALUES if (kind, n_w) != reference_key},
            "log_180": ref_nfr,
        },
        "wealth_grid_hash": {
            f"{kind}_{n_w}": bundles[(kind, n_w)]["wealth_grid_hash"]
            for kind in GRID_KINDS for n_w in N_W_VALUES
        },
        "side_by_side": side_by_side,
        "results": {
            f"{r['label']}": {k: v for k, v in r.items() if k != "label"}
            for r in results.values()
        },
        "equidistribution_summary_age45_zmid_smid_C": eq_summary,
        "distributions": distributions,
    }
    metrics_path = args.output_dir / args.metrics_name
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote metrics to {metrics_path}")
    print(f"Wrote 5 figures to {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
