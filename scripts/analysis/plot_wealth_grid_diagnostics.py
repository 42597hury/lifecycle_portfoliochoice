"""
Three-panel diagnostic: do log_180 and bakh_180 put nodes where households land?

Top panel:    Node positions on wealth axis (log_180 ticks vs bakh_180 ticks).
Middle panel: Simulated wealth density across the lifecycle, overlaid per age.
Bottom panel: Per-cell h^2 * |V''(W)| at multiple ages on log_180 and bakh_180.

All three panels share the wealth (log scale) axis so the reader can read off:
  - which grid puts more nodes in [0.1, 10] AWI (kink + median household)
  - which grid puts more nodes in [10, 750] AWI (where retirees live)
  - whether either grid's nodes line up with the curvature peaks

Reuses the simulation pipeline from wealth_axis_curvature.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG  # noqa: E402
from lifecycle.policy_io import load_policy_bundle  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.simulation import simulate_lifecycle  # noqa: E402
from verify._diag_helpers import build_bundle_var_config  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid  # noqa: E402

AWI_2019_USD = 54_099.99


def _list_to_tuple(v):
    if isinstance(v, list):
        return tuple(_list_to_tuple(x) for x in v)
    return v


def _rehydrate_disc(d):
    tuple_fields = {"state_grid_sizes", "n_state_quad_nodes", "n_ret_nodes_1d",
                    "state_n_stds", "ret_lobatto_Z", "state_lobatto_Z"}
    valid = set(DiscretizationConfig._fields)
    return DiscretizationConfig(
        **{k: (_list_to_tuple(v) if k in tuple_fields else v)
           for k, v in d.items() if k in valid}
    )


def second_deriv(y, x):
    h_l = x[1:-1] - x[:-2]
    h_r = x[2:] - x[1:-1]
    return 2.0 * ((y[2:] - y[1:-1]) / h_r - (y[1:-1] - y[:-2]) / h_l) / (h_l + h_r)


def simulate_for_bundle(bundle_path: Path, n_simulations: int, seed: int):
    C, S, B, _, metadata = load_policy_bundle(bundle_path)
    rc = metadata["run_config"]
    base_config = rc["base_config"]
    disc = _rehydrate_disc(rc["discretization_config"])
    disc = disc_config_with_bundle_wealth_grid(disc, bundle_path, metadata)
    var_config = build_bundle_var_config(metadata, bundle_path)
    model = build_model(base_config, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    sim = simulate_lifecycle(
        C, S, B, pc, model,
        n_simulations=n_simulations,
        initial_wealth=base_config.get("initial_wealth", 0.1),
        initial_z="stationary", initial_state="median",
        seed=seed, verbose=False,
    )
    return C, S, B, pc, model, sim, base_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-bundle", type=Path,
                        default=REPO / "saved_runs" / "ablations"
                                / "system_1_grid7_nz25_w180_log_calib1")
    parser.add_argument("--bakh-bundle", type=Path,
                        default=REPO / "saved_runs" / "ablations"
                                / "system_1_grid7_nz25_w180_bakh_calib1")
    parser.add_argument("--n-simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ages", type=int, nargs="+",
                        default=[25, 35, 45, 55, 67, 80])
    parser.add_argument("--curv-ages", type=int, nargs="+",
                        default=[35, 55, 67])
    parser.add_argument("--out",
                        default=REPO / "docs" / "scans" / "figures"
                                / "wealth_grid_node_density_curvature.png",
                        type=Path)
    args = parser.parse_args(argv)

    print(f"Simulating {args.n_simulations} households on log_180 (production canonical)...",
          flush=True)
    C_log, _, _, pc_log, _, sim, base_config = simulate_for_bundle(
        args.log_bundle, args.n_simulations, args.seed
    )
    print(f"Simulating same on bakh_180 (for the curvature-on-its-own-grid panel)...",
          flush=True)
    C_bak, _, _, pc_bak, _, _, _ = simulate_for_bundle(
        args.bakh_bundle, args.n_simulations, args.seed
    )

    wg_log = np.asarray(pc_log.wealth_grid)
    wg_bak = np.asarray(pc_bak.wealth_grid)

    sim_x = np.asarray(sim["x"])
    sim_alive = np.asarray(sim["alive"])
    start_age = int(base_config["start_age"])
    n_ages = sim_x.shape[1]

    # Wealth-axis: log scale spanning the full grid range.
    w_min = max(min(wg_log.min(), wg_bak.min()), 1e-2)
    w_max = max(wg_log.max(), wg_bak.max())

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 2.2, 2.2]})

    # ------------------------------------------------------------------
    # Top panel: node positions (log_180 above, bakh_180 below).
    # ------------------------------------------------------------------
    ax_nodes = axes[0]
    ax_nodes.scatter(wg_log, np.full_like(wg_log, 1.0),
                     marker="|", s=200, color="#1f77b4", linewidth=1.4,
                     label=f"log_180 ({wg_log.size} nodes)")
    ax_nodes.scatter(wg_bak, np.full_like(wg_bak, 0.0),
                     marker="|", s=200, color="#ff7f0e", linewidth=1.4,
                     label=f"bakh_180 ({wg_bak.size} nodes)")
    ax_nodes.axvspan(0.1, 1.0, alpha=0.10, color="gray")
    ax_nodes.set_ylim(-0.6, 1.6)
    ax_nodes.set_yticks([0.0, 1.0])
    ax_nodes.set_yticklabels(["bakh_180", "log_180"])
    ax_nodes.set_title("Top: node placement (gray band = kink [0.1, 1.0] AWI)")
    ax_nodes.grid(True, axis="x", which="both", alpha=0.3)
    ax_nodes.legend(loc="upper right", fontsize=9)

    # Annotate per-band node counts
    bands = [(0.05, 1.0, "kink"), (1.0, 10.0, "low-mid"),
             (10.0, 100.0, "p25-p90"), (100.0, 750.0, "tail")]
    for lo, hi, name in bands:
        n_log = int(((wg_log >= lo) & (wg_log < hi)).sum())
        n_bak = int(((wg_bak >= lo) & (wg_bak < hi)).sum())
        x_label = np.sqrt(lo * hi)
        ax_nodes.text(x_label, 1.45, f"{n_log}", color="#1f77b4",
                      ha="center", fontsize=9, fontweight="bold")
        ax_nodes.text(x_label, -0.45, f"{n_bak}", color="#ff7f0e",
                      ha="center", fontsize=9, fontweight="bold")
        ax_nodes.text(x_label, 0.5, name, color="dimgray",
                      ha="center", fontsize=8, alpha=0.7)

    # ------------------------------------------------------------------
    # Middle panel: simulated wealth density per age.
    # ------------------------------------------------------------------
    ax_dens = axes[1]
    eval_w = np.geomspace(max(w_min, 1e-2), w_max, 600)
    log_eval_w = np.log(eval_w)
    cmap = plt.get_cmap("viridis")
    for i, age in enumerate(args.ages):
        t = age - start_age
        if t < 0 or t >= n_ages:
            continue
        m = sim_alive[:, t]
        if not m.any():
            continue
        w_alive = sim_x[m, t]
        w_alive = w_alive[w_alive > 0]
        if w_alive.size < 5:
            continue
        # KDE on log-wealth (avoids zero-mass at small W)
        log_w = np.log(np.clip(w_alive, 1e-6, None))
        # silverman bandwidth
        bw = 1.06 * log_w.std(ddof=1) * log_w.size ** (-1 / 5)
        bw = max(bw, 1e-2)
        # vectorized kernel sum
        diff = (log_eval_w[:, None] - log_w[None, :]) / bw
        kvals = np.exp(-0.5 * diff * diff) / (np.sqrt(2 * np.pi) * bw)
        density = kvals.mean(axis=1)
        color = cmap(i / max(1, len(args.ages) - 1))
        # plot quantiles as thick markers
        q = np.quantile(w_alive, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        ax_dens.plot(eval_w, density, color=color, lw=1.6, label=f"age {age}",
                     alpha=0.85)
        ax_dens.scatter(q, np.full_like(q, density.max() * 0.05 * (1 + 0.4 * i)),
                        color=color, s=12, marker="v", alpha=0.7)
    ax_dens.set_yscale("linear")
    ax_dens.set_ylabel("Simulated wealth density (KDE on log W)")
    ax_dens.set_title(
        "Middle: simulated wealth density per age (10 000 households)\n"
        "  triangles = quantiles {p5, p25, p50, p75, p95, p99} (slight vertical jitter per age)"
    )
    ax_dens.axvspan(0.1, 1.0, alpha=0.07, color="gray")
    ax_dens.grid(True, which="both", alpha=0.3)
    ax_dens.legend(loc="upper right", fontsize=8, ncol=2)
    # Show node positions as faint vertical ticks at the bottom of this panel
    for w in wg_log:
        ax_dens.axvline(w, ymin=0.0, ymax=0.03, color="#1f77b4",
                        alpha=0.5, lw=0.5)
    for w in wg_bak:
        ax_dens.axvline(w, ymin=0.97, ymax=1.0, color="#ff7f0e",
                        alpha=0.5, lw=0.5)

    # ------------------------------------------------------------------
    # Bottom panel: per-cell h^2 * |V''| at multiple working/retirement ages.
    # ------------------------------------------------------------------
    ax_curv = axes[2]
    n_z = int(pc_log.n_z)
    n_state = int(np.prod(pc_log.state_grid_sizes))
    z_mid = n_z // 2
    s_mid = n_state // 2
    cmap_c = plt.get_cmap("plasma")
    for i, age in enumerate(args.curv_ages):
        t = age - start_age
        if t < 0 or t >= C_log.shape[0]:
            continue
        # log_180
        slab = C_log[t, z_mid, s_mid, :]
        d2 = np.abs(second_deriv(slab, wg_log))
        h = 0.5 * (wg_log[2:] - wg_log[:-2])
        err = h * h * d2
        color = cmap_c(0.15 + 0.55 * (i / max(1, len(args.curv_ages) - 1)))
        ax_curv.plot(wg_log[1:-1], err + 1e-30, marker="o", ms=3, lw=0.8,
                     color="#1f77b4", alpha=0.4 + 0.5 * (i / max(1, len(args.curv_ages) - 1)),
                     label=f"log_180 age {age}")
        # bakh_180
        slab_b = C_bak[t, z_mid, s_mid, :]
        d2_b = np.abs(second_deriv(slab_b, wg_bak))
        h_b = 0.5 * (wg_bak[2:] - wg_bak[:-2])
        err_b = h_b * h_b * d2_b
        ax_curv.plot(wg_bak[1:-1], err_b + 1e-30, marker="o", ms=3, lw=0.8,
                     color="#ff7f0e", alpha=0.4 + 0.5 * (i / max(1, len(args.curv_ages) - 1)),
                     label=f"bakh_180 age {age}")
    ax_curv.set_xscale("log")
    ax_curv.set_yscale("log")
    ax_curv.set_xlabel("wealth W (AWI units)  -- 1 AWI = $54.1k")
    ax_curv.set_ylabel("h^2 * |C''(W)|  (per-cell interp-error budget)")
    ax_curv.set_title(
        "Bottom: per-cell interp-error budget for consumption policy\n"
        "  (Bakhvalov design target: this should be flatter than log1p; cells with low budget = redundant resolution)"
    )
    ax_curv.axvspan(0.1, 1.0, alpha=0.07, color="gray")
    ax_curv.grid(True, which="both", alpha=0.3)
    ax_curv.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(
        "System I wealth-grid diagnostics: log_180 vs bakh_180\n"
        "(do nodes track the simulated wealth distribution + curvature?)",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
