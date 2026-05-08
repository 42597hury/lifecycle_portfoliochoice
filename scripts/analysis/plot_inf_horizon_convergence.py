"""
LEGACY plots: pre-pivot System IV inf-horizon resolution sweeps. Reads the
metrics JSON produced by `inf_horizon_grid_quad_convergence.py` (which is
itself flagged legacy) and the legacy bundles for raw policy curves.

Outputs (under docs/scans/figures/):
  axisbump_summary.png       — bar chart of sup-norm divergence per axis
  gridsize_summary.png       — sup/RMS/wRMS vs grid size + wall time
  gridsize_per_wealth.png    — per-wealth profile of g3 vs g5, g4 vs g5 in C/S/B
  gridsize_per_state_heatmap.png — heat map showing where divergence lives
  policy_vs_wealth_inf.png   — C/S/B(wealth) at the median state, one line per bundle
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

from scripts.analysis.inf_horizon_grid_quad_convergence import (  # noqa: E402
    AXIS_BUMP_BUNDLES, GRID_BUNDLES, GRID_REFERENCE, INF_HORIZON_ROOT, STATE_NAMES,
    load_bundle, interp_policy_to_finer_grid,
)


def plot_axisbump_summary(metrics: dict, out_path: Path) -> None:
    pairs = metrics["axis_bump_sweep"]["pairs"]
    labels: list[str] = []
    sup_C: list[float] = []
    sup_S: list[float] = []
    sup_B: list[float] = []
    walls: list[float] = []
    for name, p in pairs.items():
        sq = "".join(str(v) for v in p["sq"])
        rq = "".join(str(v) for v in p["rq"])
        # axis label from which entry differs from baseline (3,3,3,3)/(3,3)
        if p["sq"] != [3, 3, 3, 3]:
            d = next(i for i, v in enumerate(p["sq"]) if v != 3)
            label = f"sq[{STATE_NAMES[d]}]: 3->{p['sq'][d]}"
        elif p["rq"] != [3, 3]:
            d = next(i for i, v in enumerate(p["rq"]) if v != 3)
            ret_names = ("xr", "xb")
            label = f"rq[{ret_names[d]}]: 3->{p['rq'][d]}"
        else:
            label = name[-30:]
        labels.append(label)
        sup_C.append(p["C"]["sup"])
        sup_S.append(p["S"]["sup"])
        sup_B.append(p["B"]["sup"])
        walls.append(p["wall_seconds"] or float("nan"))

    order = np.argsort(sup_C)[::-1]
    labels = [labels[i] for i in order]
    sup_C = [sup_C[i] for i in order]
    sup_S = [sup_S[i] for i in order]
    sup_B = [sup_B[i] for i in order]
    walls = [walls[i] for i in order]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(len(labels))

    axes[0].barh(x, sup_C, color="C0")
    axes[0].set_yticks(x); axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("sup |ΔC|")
    axes[0].set_title("Consumption sup-norm divergence vs run1 baseline")
    axes[0].set_xscale("log")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(x, sup_S, color="C1")
    axes[1].set_yticks(x); axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("sup |Δα_s|")
    axes[1].set_title("Stock share sup-norm")
    axes[1].set_xscale("log")
    axes[1].grid(True, axis="x", alpha=0.3)

    axes[2].barh(x, sup_B, color="C2")
    axes[2].set_yticks(x); axes[2].set_yticklabels(labels)
    axes[2].set_xlabel("sup |Δα_b|")
    axes[2].set_title("Bond share sup-norm")
    axes[2].set_xscale("log")
    axes[2].grid(True, axis="x", alpha=0.3)

    axes[3].barh(x, walls, color="C3")
    axes[3].set_yticks(x); axes[3].set_yticklabels(labels)
    axes[3].set_xlabel("wall (seconds)")
    axes[3].set_title("Solve wall time")
    axes[3].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Axis-bump sweep: changing one quadrature axis from 3→5 "
                 "(System IV inf-horizon, baseline sq=(3,3,3,3) rq=(3,3), grid 5⁴=625)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_gridsize_summary(metrics: dict, out_path: Path) -> None:
    pairs = metrics["grid_size_sweep"]["pairs"]
    grids: list[int] = []
    sup_C: list[float] = []
    rms_C: list[float] = []
    wRMS_C: list[float] = []
    sup_S: list[float] = []
    sup_B: list[float] = []
    walls: list[float] = []
    for name, p in pairs.items():
        grids.append(int(p["state_grid_sizes"][0]))
        sup_C.append(p["C"]["sup"])
        rms_C.append(p["C"]["rms"])
        wRMS_C.append(p["C"].get("stat_weighted_rms", float("nan")))
        sup_S.append(p["S"]["sup"])
        sup_B.append(p["B"]["sup"])
        walls.append(p["wall_seconds"] or float("nan"))
    grids.append(int(metrics["grid_size_sweep"]["reference_state_grid_sizes"][0]))
    sup_C.append(0.0); rms_C.append(0.0); wRMS_C.append(0.0); sup_S.append(0.0); sup_B.append(0.0)
    walls.append(metrics["grid_size_sweep"]["reference_wall_seconds"] or float("nan"))

    order = np.argsort(grids)
    grids = [grids[i] for i in order]
    sup_C = [sup_C[i] for i in order]
    rms_C = [rms_C[i] for i in order]
    wRMS_C = [wRMS_C[i] for i in order]
    sup_S = [sup_S[i] for i in order]
    sup_B = [sup_B[i] for i in order]
    walls = [walls[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(grids, sup_C, "o-", label="sup |ΔC|", color="C0")
    axes[0].plot(grids, rms_C, "s-", label="RMS |ΔC|", color="C0", alpha=0.6)
    axes[0].plot(grids, wRMS_C, "^-", label="stat-weighted RMS |ΔC|", color="C0", alpha=0.4)
    axes[0].plot(grids, sup_S, "o-", label="sup |Δα_s|", color="C1")
    axes[0].plot(grids, sup_B, "o-", label="sup |Δα_b|", color="C2")
    axes[0].set_xlabel("state grid size per axis (g)")
    axes[0].set_ylabel("divergence vs g5")
    axes[0].set_yscale("log")
    axes[0].set_xticks(grids)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title("Policy divergence (after multilinear projection)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(grids, walls, "o-", color="C3")
    axes[1].set_xlabel("state grid size per axis (g)")
    axes[1].set_ylabel("wall time (seconds)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(grids)
    axes[1].set_title("Wall time vs g (g^4 cells)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(walls[:-1], sup_C[:-1], "o-", label="sup |ΔC|", color="C0")
    axes[2].plot(walls[:-1], wRMS_C[:-1], "^-",
                 label="stat-weighted RMS |ΔC|", color="C0", alpha=0.4)
    axes[2].set_xlabel("wall time (seconds)")
    axes[2].set_ylabel("divergence vs g5")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].set_title("Accuracy vs compute (Pareto)")
    axes[2].grid(True, alpha=0.3)
    for i, g in enumerate(grids[:-1]):
        axes[2].annotate(f"g{g}", (walls[i], sup_C[i]),
                         xytext=(4, 4), textcoords="offset points")

    fig.suptitle("State-grid-size sweep: System IV inf-horizon "
                 "(reference g5; quad=(3,3,3,4) rq=(4,4))", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_per_wealth_grid(metrics: dict, out_path: Path) -> None:
    w_grid = load_bundle(GRID_REFERENCE)["wealth_grid"]
    pairs = metrics["grid_size_sweep"]["pairs"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    arrays = ("C", "S", "B")
    titles = {"C": "consumption", "S": "stock share α_s", "B": "bond share α_b"}
    for ax, arr in zip(axes, arrays):
        for name, p in pairs.items():
            g = int(p["state_grid_sizes"][0])
            per_w = np.asarray(p[arr]["per_wealth_max"])
            ax.plot(w_grid, per_w, "o-", label=f"g{g} vs g5", lw=1.4, ms=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wealth W (AWI units)")
        ax.set_ylabel(f"sup |Δ{arr}| at this wealth (across all 625 states)")
        ax.set_title(f"Per-wealth {titles[arr]} divergence")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Where the grid-size divergence lives (per-wealth profile)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_per_state_heatmap(metrics: dict, out_path: Path) -> None:
    pairs = metrics["grid_size_sweep"]["pairs"]
    bundles_to_plot = list(pairs.keys())[:2]  # g3 vs g5, g4 vs g5

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axis_names = STATE_NAMES   # ("dp", "spr", "rtb", "y_1")
    for r, name in enumerate(bundles_to_plot):
        p = pairs[name]
        per_state = np.asarray(p["C"]["per_state_max"])    # length 625
        # Reshape into (5, 5, 5, 5) — order matches state index ordering used
        # in inf_horizon_grid_quad_convergence: i = i1 + 5*i2 + 25*i3 + 125*i4
        # where i1=dp, i2=spr, i3=rtb, i4=y_1.
        per_4d = per_state.reshape(5, 5, 5, 5, order="F")  # axes: (dp, spr, rtb, y_1)
        # For each pair (axis_a, axis_b) NOT in (axis_a, axis_b), max over the other two.
        pairs_to_show = [(0, 1), (0, 3), (2, 3), (1, 3)]
        for c, (a, b) in enumerate(pairs_to_show):
            other = tuple(i for i in range(4) if i not in (a, b))
            heat = per_4d.max(axis=other)   # shape (5, 5)
            ax = axes[r, c]
            im = ax.imshow(heat, origin="lower", cmap="viridis", aspect="equal")
            g = int(p["state_grid_sizes"][0])
            ax.set_title(f"g{g} vs g5: max |ΔC| over ({axis_names[a]}, {axis_names[b]})", fontsize=10)
            ax.set_xlabel(axis_names[b])
            ax.set_ylabel(axis_names[a])
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Where consumption divergence concentrates in state space "
                 "(maxed over the two off-axis dimensions)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_policy_vs_wealth_inf(out_path: Path) -> None:
    """C/S/B(wealth) at the modal state cell (mid index in each axis), one line per bundle."""
    bundles_grid = {name: load_bundle(name) for name in GRID_BUNDLES}
    bundles_axis = {name: load_bundle(name) for name in AXIS_BUMP_BUNDLES}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cmap = plt.get_cmap("viridis")

    # Row 0: grid-size sweep, slice at "central" state (each axis at midpoint).
    grids_sorted = sorted(bundles_grid.values(), key=lambda b: b["state_grid_sizes"][0])
    for c, arr_name in enumerate(("C", "S", "B")):
        ax = axes[0, c]
        for i, b in enumerate(grids_sorted):
            sizes = b["state_grid_sizes"]
            mid_idx = (sizes[0] // 2, sizes[1] // 2, sizes[2] // 2, sizes[3] // 2)
            flat_mid = mid_idx[0] + sizes[0] * (mid_idx[1] + sizes[1] *
                                                 (mid_idx[2] + sizes[2] * mid_idx[3]))
            color = cmap(i / max(1, len(grids_sorted) - 1))
            policy = b[arr_name][0, flat_mid, :]
            ls = "-" if b["name"] == GRID_REFERENCE else "--"
            ax.plot(b["wealth_grid"], policy, color=color,
                    label=f"g{sizes[0]}", lw=1.5, ls=ls)
        ax.set_xscale("log")
        ax.set_xlabel("wealth W")
        ax.set_ylabel({"C": "consumption", "S": "α_s", "B": "α_b"}[arr_name])
        ax.set_title(f"grid-size sweep — {arr_name} at midpoint state")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)

    # Row 1: axis-bump sweep, slice at midpoint of 5x5x5x5 (idx=(2,2,2,2)).
    base_sizes = (5, 5, 5, 5)
    flat_mid = 2 + 5 * (2 + 5 * (2 + 5 * 2))
    for c, arr_name in enumerate(("C", "S", "B")):
        ax = axes[1, c]
        for i, name in enumerate(AXIS_BUMP_BUNDLES):
            b = bundles_axis[name]
            sq = "".join(str(v) for v in b["n_state_quad_nodes"])
            rq = "".join(str(v) for v in b["n_ret_nodes_1d"])
            color = cmap(i / max(1, len(AXIS_BUMP_BUNDLES) - 1))
            policy = b[arr_name][0, flat_mid, :]
            ls = "-" if i == 0 else "--"
            ax.plot(b["wealth_grid"], policy, color=color,
                    label=f"sq={sq} rq={rq}", lw=1.5, ls=ls)
        ax.set_xscale("log")
        ax.set_xlabel("wealth W")
        ax.set_ylabel({"C": "consumption", "S": "α_s", "B": "α_b"}[arr_name])
        ax.set_title(f"axis-bump sweep — {arr_name} at midpoint state")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    fig.suptitle("Inf-horizon policy curves at modal state cell "
                 "(midpoint of every state axis)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=REPO / "docs" / "scans" / "inf_horizon_grid_quad_metrics.json")
    parser.add_argument("--fig-dir", type=Path,
                        default=REPO / "docs" / "scans" / "figures")
    args = parser.parse_args(argv)

    with args.metrics.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    plot_axisbump_summary(metrics, args.fig_dir / "axisbump_summary.png")
    plot_gridsize_summary(metrics, args.fig_dir / "gridsize_summary.png")
    plot_per_wealth_grid(metrics, args.fig_dir / "gridsize_per_wealth.png")
    plot_per_state_heatmap(metrics, args.fig_dir / "gridsize_per_state_heatmap.png")
    plot_policy_vs_wealth_inf(args.fig_dir / "policy_vs_wealth_inf.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
