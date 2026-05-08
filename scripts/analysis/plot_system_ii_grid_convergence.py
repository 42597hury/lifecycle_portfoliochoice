"""Render figures for the System II x state-grid density convergence scan.

Reads the metrics JSON + plot bundle from
  docs/scans/system_ii_grid_convergence_metrics.json
  docs/scans/system_ii_grid_plot.npz
and writes:
  docs/scans/figures/system_ii_grid_convergence_curve.png
  docs/scans/figures/system_ii_grid_per_axis.png
  docs/scans/figures/system_ii_grid_heatmap.png
  docs/scans/figures/system_ii_grid_probe_age.png
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


# ---------------------------------------------------------------------------
# Convergence curve
# ---------------------------------------------------------------------------

def plot_convergence_curve(metrics: dict, out_path: Path) -> None:
    cc = metrics["convergence_curve"]
    Ns = np.asarray(cc["N_state"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("sup", "rms")):
        for k, lbl, color in (
            ("C", "consumption", "C0"),
            ("S", "alpha_s (risky share)", "C1"),
            ("B", "alpha_b (bond share)", "C2"),
        ):
            ax.plot(Ns, cc[f"{kind}_{k}"], "o-", color=color, label=lbl)
        ax.set_xlabel("N_state = N_rtb x N_y_1")
        ax.set_xticks(Ns)
        ax.set_xticklabels([f"({s[0]},{s[1]})\nN={n}"
                            for s, n in zip(cc["state_grid_sizes"], Ns)])
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel(f"{kind}|delta| vs (6, 6)")
        ax.set_title(f"{kind.upper()}-norm divergence vs (6, 6)")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("System II grid-density convergence: divergence vs (6, 6) reference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Per-axis decomposition
# ---------------------------------------------------------------------------

def plot_per_axis(metrics: dict, out_path: Path) -> None:
    pairs = metrics["pairs"]
    pair_keys = list(pairs.keys())
    n_age = len(metrics["reference_shape"])  # placeholder
    n_age = 78
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex="col")
    arrays = ("C", "S", "B")
    for r, k in enumerate(arrays):
        ax_age = axes[r, 0]
        ax_w = axes[r, 1]
        for pk in pair_keys:
            label = pk.split("_")[2]   # grid4x4 / grid5x5
            d = pairs[pk][k]
            ages = np.arange(22, 22 + len(d["per_age"]))
            ax_age.plot(ages, d["per_age"], label=label)
            ws = np.arange(len(d["per_wealth"]))
            ax_w.plot(ws, d["per_wealth"], label=label)
        ax_age.axvline(67, color="grey", ls=":", alpha=0.6)   # retirement
        ax_age.set_ylabel(f"sup |delta {k}|")
        ax_age.set_title(f"{k} divergence vs (6,6) by age (working|retirement)")
        ax_age.grid(True, alpha=0.3)
        ax_age.legend(fontsize=9)
        ax_w.set_yscale("log")
        ax_w.set_title(f"{k} divergence vs (6,6) by wealth bin")
        ax_w.grid(True, alpha=0.3)
        ax_w.legend(fontsize=9)
    axes[-1, 0].set_xlabel("age")
    axes[-1, 1].set_xlabel("wealth idx (0=lowest, 179=highest)")
    fig.suptitle("Per-axis decomposition of |coarse - (6,6)|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# (rtb, y_1) heatmap at probe (age, z, wealth)
# ---------------------------------------------------------------------------

def plot_heatmap(metrics: dict, plot_bundle: dict, out_path: Path) -> None:
    h = metrics["heatmap_eval_grid"]
    dst_rtb = np.asarray(h["dst_rtb"])
    dst_y1 = np.asarray(h["dst_y1"])
    age = h["probe_age"]
    z_idx = h["probe_z_idx"]
    w_idx = h["probe_w_idx"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ks = ("C", "S", "B")
    titles = ("|delta C|", "|delta alpha_s|", "|delta alpha_b|")
    extent = [dst_y1[0], dst_y1[-1], dst_rtb[0], dst_rtb[-1]]
    for col, (k, title) in enumerate(zip(ks, titles)):
        for row, src_label in enumerate(("g4x4", "g5x5")):
            ax = axes[row, col]
            data = plot_bundle[f"{src_label}_heatdelta_{k}"]
            im = ax.imshow(data, origin="lower", aspect="auto",
                           extent=extent, cmap="viridis")
            ax.set_xlabel("y_1 (u-coord)")
            ax.set_ylabel("rtb (u-coord)")
            ax.set_title(f"{src_label} vs (6,6) — {title}\n"
                         f"sup={data.max():.3e}")
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"Heatmap of policy divergence at age={age}, z_idx={z_idx}, "
                 f"w_idx={w_idx}\n(rtb, y_1 in Cholesky u-coords; corners = +/-n_stds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Probe-line plot at (rtb=0, y_1=0) over the working life
# ---------------------------------------------------------------------------

def plot_probe_lines(metrics: dict, plot_bundle: dict, out_path: Path) -> None:
    ages = np.asarray(plot_bundle["ages"])
    bundle_names = list(metrics["bundle_names"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ("c (consumption)", "alpha_s (risky share)", "alpha_b (bond share)")
    colors = ("C0", "C1", "C2")
    for ax, k, title in zip(axes, ("C", "S", "B"), titles):
        for i, name in enumerate(bundle_names):
            tag = name.replace("system_ii_grid", "g").split("_nz")[0]
            line = plot_bundle[f"{tag}_probe_{k}"]
            label = name.split("_")[2]
            ax.plot(ages, line, label=label, color=colors[i], lw=1.6)
        ax.axvline(67, color="grey", ls=":", alpha=0.6)
        ax.set_xlabel("age")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Policy at (rtb=0, y_1=0), z=mean, w idx 89 — overlay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path,
                    default=REPO / "docs" / "scans" / "system_ii_grid_convergence_metrics.json")
    ap.add_argument("--plot-bundle", type=Path,
                    default=REPO / "docs" / "scans" / "system_ii_grid_plot.npz")
    ap.add_argument("--figures-dir", type=Path,
                    default=REPO / "docs" / "scans" / "figures")
    args = ap.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    plot_bundle = dict(np.load(args.plot_bundle))

    print("Rendering figures:")
    plot_convergence_curve(metrics,
        args.figures_dir / "system_ii_grid_convergence_curve.png")
    plot_per_axis(metrics,
        args.figures_dir / "system_ii_grid_per_axis.png")
    plot_heatmap(metrics, plot_bundle,
        args.figures_dir / "system_ii_grid_heatmap.png")
    plot_probe_lines(metrics, plot_bundle,
        args.figures_dir / "system_ii_grid_probe_age.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
