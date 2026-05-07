"""
Visualize the System I × n_z convergence study.

Reads:
  - {input-dir}/system_i_nz_convergence_metrics.json (from
    `system_i_nz_convergence.py`)
  - The same four bundles for probe-cell line plots and S/B histograms.

Writes (to {fig-dir}):
  - convergence_curves.png   (sup-norm, p99, p99-relative, RMS vs n_z)
  - per_age_divergence.png   (sup_<X>(N) per age, one panel per array)
  - per_z_divergence.png     (sup_<X>(N) per reference-z, one panel per array)
  - probe_C_vs_age.png       (C at z=0, mid-state, three wealth probes; one curve per n_z)
  - probe_S_vs_age.png       (similar)
  - probe_B_vs_age.png       (similar)
  - alpha_distribution.png   (histograms of S and B across all cells, per n_z)

Usage:
    python scripts/analysis/plot_nz_convergence.py
        [--bundles-root saved_runs/ablations]
        [--input-dir docs/scans]
        [--fig-dir docs/scans/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.system_i_nz_convergence import (  # noqa: E402
    NZ_VALUES,
    REFERENCE_NZ,
    load_one,
)


def _ages_from_shape(shape) -> np.ndarray:
    """Return an age axis matching the bundle's first dim.

    Bundles canonically span ages 22..99 (78 cells). If the shape disagrees,
    fall back to integer indices.
    """
    n_age = int(shape[0])
    if n_age == 78:
        return np.arange(22, 100)
    return np.arange(n_age)


def _ref_z_index_at_zero(z_ref: np.ndarray) -> int:
    return int(np.argmin(np.abs(z_ref)))


def plot_convergence_curves(metrics: dict, out_path: Path) -> None:
    nzs = sorted(int(nz) for nz in metrics["pairs"].keys())
    arrays = ("C", "S", "B")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("sup", "Sup-norm |coarse − ref|"),
        ("rms", "RMS |coarse − ref|"),
        ("p99", "99th-pctile |coarse − ref|"),
        ("sup_rel", "Sup of |coarse − ref| / |ref| (relative)"),
    ]
    colors = {"C": "C0", "S": "C1", "B": "C2"}
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name in arrays:
            vals = [metrics["pairs"][str(nz)][name][key] for nz in nzs]
            ax.plot(nzs, vals, marker="o", label=name, color=colors[name])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_z (log scale)")
        ax.set_ylabel(title)
        ax.set_title(title + f" vs n_z={REFERENCE_NZ}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(
        f"System I × n_z convergence — coarse vs reference n_z={REFERENCE_NZ}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_axis(metrics: dict, axis_key: str, axis_label: str, out_path: Path) -> None:
    nzs = sorted(int(nz) for nz in metrics["pairs"].keys())
    arrays = ("C", "S", "B")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes, arrays):
        for i, nz in enumerate(nzs):
            vals = metrics["pairs"][str(nz)][name][axis_key]
            x = np.arange(len(vals))
            ax.plot(x, vals, color=cmap(i / max(1, len(nzs) - 1)),
                    label=f"n_z={nz}", lw=1.6)
        ax.set_yscale("log")
        ax.set_xlabel(axis_label)
        ax.set_ylabel(f"sup |{name}_coarse − {name}_ref|")
        ax.set_title(f"{name}: divergence per {axis_label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Per-{axis_label} divergence vs n_z={REFERENCE_NZ}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_probe(
    bundles: dict[int, dict],
    array_name: str,
    z_target: float,
    state_idx: int,
    wealth_idxs: list[int],
    out_path: Path,
) -> None:
    """One subpanel per wealth probe; overlay one curve per n_z."""
    fig, axes = plt.subplots(1, len(wealth_idxs), figsize=(5 * len(wealth_idxs), 4),
                             sharey=False)
    if len(wealth_idxs) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    nzs_sorted = sorted(bundles.keys())
    for ax, w_idx in zip(axes, wealth_idxs):
        for i, nz in enumerate(nzs_sorted):
            b = bundles[nz]
            arr = b[array_name]
            z_grid = b["z_grid"]
            ages = _ages_from_shape(arr.shape)
            # Slice over (age, ., state, w_idx); interp along z to z_target.
            slab = arr[:, :, state_idx, w_idx]  # shape (n_age, n_z_bundle)
            curve = np.array([np.interp(z_target, z_grid, slab[a]) for a in range(slab.shape[0])])
            color = cmap(i / max(1, len(nzs_sorted) - 1))
            ls = "-" if nz == REFERENCE_NZ else "--"
            ax.plot(ages, curve, label=f"n_z={nz}", color=color, ls=ls, lw=1.4)
        ax.set_xlabel("age")
        ax.set_ylabel(f"{array_name}(age)")
        ax.set_title(
            f"{array_name} at z={z_target:.2f}, state idx={state_idx}, w idx={w_idx}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{array_name}: probe-cell trajectories across n_z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_distribution(bundles: dict[int, dict], out_path: Path) -> None:
    nzs_sorted = sorted(bundles.keys())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = 80
    for ax, name in zip(axes, ("S", "B")):
        for i, nz in enumerate(nzs_sorted):
            arr = bundles[nz][name].ravel()
            color = cmap(i / max(1, len(nzs_sorted) - 1))
            ax.hist(arr, bins=bins, density=True, histtype="step",
                    color=color, label=f"n_z={nz}", lw=1.4)
        ax.set_xlabel(f"{name} (share)")
        ax.set_ylabel("density")
        ax.set_title(f"{name} distribution across all (age, z, state, wealth) cells")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", type=Path,
                        default=REPO / "saved_runs" / "ablations")
    parser.add_argument("--input-dir", type=Path,
                        default=REPO / "docs" / "scans")
    parser.add_argument("--metrics-name", type=str,
                        default="system_i_nz_convergence_metrics.json")
    parser.add_argument("--fig-dir", type=Path,
                        default=REPO / "docs" / "scans" / "figures")
    args = parser.parse_args(argv)

    metrics_path = args.input_dir / args.metrics_name
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Run system_i_nz_convergence.py first; missing {metrics_path}"
        )
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    fig_dir: Path = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Convergence-curve plots ...")
    plot_convergence_curves(metrics, fig_dir / "convergence_curves.png")
    plot_per_axis(metrics, "per_age", "age (index from start_age)", fig_dir / "per_age_divergence.png")
    plot_per_axis(metrics, "per_z", "z index (reference grid)", fig_dir / "per_z_divergence.png")
    plot_per_axis(metrics, "per_wealth", "wealth index", fig_dir / "per_wealth_divergence.png")

    print("Reloading bundles for probe + distribution plots ...")
    bundles: dict[int, dict] = {}
    for nz in NZ_VALUES:
        bundles[nz] = load_one(args.bundles_root, nz)

    ref_shape = bundles[REFERENCE_NZ]["C"].shape
    n_state = int(ref_shape[2])
    n_wealth = int(ref_shape[3])
    state_mid = n_state // 2
    wealth_low = max(1, int(0.10 * n_wealth))
    wealth_med = n_wealth // 2
    wealth_high = min(n_wealth - 1, int(0.90 * n_wealth))
    wealth_idxs = [wealth_low, wealth_med, wealth_high]

    print(f"  state_mid_idx={state_mid}/{n_state}; "
          f"wealth_idxs={wealth_idxs}/{n_wealth}")

    for name in ("C", "S", "B"):
        plot_probe(
            bundles, name,
            z_target=0.0, state_idx=state_mid, wealth_idxs=wealth_idxs,
            out_path=fig_dir / f"probe_{name}_vs_age.png",
        )

    plot_distribution(bundles, fig_dir / "alpha_distribution.png")

    print(f"\nAll figures written under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
