"""
Locate the wealth ranges where policy curvature is highest.

For the n_z=70 System I lifecycle bundle, computes |V''(W)| (second derivative
along the wealth axis) for C, alpha_s, alpha_b on the existing log-spaced
180-node wealth grid. Weighted by the simulated wealth distribution to get
"interp-error contribution per wealth bin" — so we can identify where extra
nodes would buy the most accuracy.

Three views:
  1. Raw |V''(W)| vs W per policy.
  2. Interpolation-error proxy: h^2 * |V''| where h is the local wealth-grid
     spacing in physical W (already log-spaced, so h grows with W).
  3. Same but multiplied by the simulated wealth-density at each W (=
     "expected interp error a randomly-drawn household will experience").
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


def second_derivative_nonuniform(y, x):
    """3-point finite-difference 2nd derivative on a non-uniform grid.
    y, x: 1-D arrays (length N). Returns length N-2 (interior).
    """
    h_left = x[1:-1] - x[:-2]
    h_right = x[2:] - x[1:-1]
    return 2 * ((y[2:] - y[1:-1]) / h_right - (y[1:-1] - y[:-2]) / h_left) / (h_left + h_right)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=REPO / "saved_runs" / "ablations"
                                / "system_1_grid7_nz70_calib1")
    parser.add_argument("--n-simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig-dir", type=Path,
                        default=REPO / "docs" / "scans" / "figures")
    parser.add_argument("--ages", type=int, nargs="+", default=[25, 40, 55, 67, 80])
    args = parser.parse_args(argv)

    print(f"Loading bundle {args.bundle.name}...", flush=True)
    C, S, B, _, metadata = load_policy_bundle(args.bundle)
    rc = metadata["run_config"]
    base_config = rc["base_config"]
    disc = _rehydrate_disc(rc["discretization_config"])
    disc = disc_config_with_bundle_wealth_grid(disc, args.bundle, metadata)
    var_config = build_bundle_var_config(metadata, args.bundle)
    model = build_model(base_config, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)

    w_grid = np.asarray(pc.wealth_grid)
    n_w = w_grid.size
    n_z = int(pc.n_z)
    N_state = int(np.prod(pc.state_grid_sizes))

    z_mid = n_z // 2
    state_mid = N_state // 2

    print(f"\nSimulating {args.n_simulations} households for wealth distribution...",
          flush=True)
    sim = simulate_lifecycle(
        C, S, B, pc, model,
        n_simulations=args.n_simulations,
        initial_wealth=0.1,
        initial_z="stationary",
        initial_state="median",
        seed=args.seed, verbose=False,
    )
    sim_x = np.asarray(sim["x"])
    sim_alive = np.asarray(sim["alive"])

    # Empirical wealth density per (age, w_bin) — count of alive households whose
    # wealth at age t falls into the wealth-grid cell j.
    print("Building empirical wealth density per age x wealth bin...", flush=True)
    age_w_density = np.zeros((C.shape[0], n_w))
    for t in range(C.shape[0]):
        m = sim_alive[:, t]
        if not m.any(): continue
        w = sim_x[m, t]
        # Bracket each w into a wealth-grid bin (interp-style)
        idx = np.clip(np.searchsorted(w_grid, w, side="right") - 1, 0, n_w - 2)
        # frac to nearest right node
        frac = np.clip((w - w_grid[idx]) / (w_grid[idx+1] - w_grid[idx]), 0, 1)
        # Distribute mass between idx and idx+1
        np.add.at(age_w_density[t], idx, 1 - frac)
        np.add.at(age_w_density[t], idx+1, frac)
        age_w_density[t] /= max(1, m.sum())  # normalize to a probability per bin

    # Compute |V''(W)| at each (age, state_mid, z_mid, w) cell, for each policy.
    print("Computing wealth-axis curvature for C, alpha_s, alpha_b...", flush=True)
    rows = []
    for arr_name, arr in (("C", C), ("S", S), ("B", B)):
        for age in args.ages:
            t = age - base_config["start_age"]
            if t < 0 or t >= arr.shape[0]: continue
            slab = arr[t, z_mid, state_mid, :]   # (n_w,)
            d2 = second_derivative_nonuniform(slab, w_grid)  # (n_w - 2,)
            d2_abs = np.abs(d2)
            interior_w = w_grid[1:-1]
            local_h = 0.5 * (w_grid[2:] - w_grid[:-2])
            interp_err = local_h ** 2 * d2_abs / 8.0
            # Wealth-density at interior bins
            density = age_w_density[t, 1:-1]
            weighted_err = interp_err * density
            for j in range(d2_abs.size):
                rows.append({
                    "array": arr_name, "age": age, "w_idx": j+1,
                    "wealth": float(interior_w[j]),
                    "wealth_USD": float(interior_w[j] * AWI_2019_USD),
                    "second_deriv_abs": float(d2_abs[j]),
                    "local_h": float(local_h[j]),
                    "interp_err_proxy": float(interp_err[j]),
                    "wealth_density": float(density[j]),
                    "weighted_interp_err": float(weighted_err[j]),
                })

    # Top-curvature wealth bands per array, AGGREGATED across ages
    print("\n=== Top wealth bands by RAW |V''| (max across selected ages) ===")
    for arr_name in ("C", "S", "B"):
        d2_by_w = np.zeros(n_w - 2)
        for age in args.ages:
            t = age - base_config["start_age"]
            if t < 0 or t >= C.shape[0]: continue
            arr = {"C": C, "S": S, "B": B}[arr_name]
            slab = arr[t, z_mid, state_mid, :]
            d2 = np.abs(second_derivative_nonuniform(slab, w_grid))
            d2_by_w = np.maximum(d2_by_w, d2)
        # Top 5 wealth bins
        top_idx = np.argsort(d2_by_w)[::-1][:8]
        print(f"\n  Policy {arr_name}: top 8 wealth bins by max |V''(W)|")
        print(f"    {'w_idx':>6} {'W (AWI)':>10} {'W ($)':>10} {'|V_ww|':>11}")
        for j in top_idx:
            w = w_grid[1 + j]
            print(f"    {1+j:>6} {w:>10.3f} ${w*AWI_2019_USD/1000:>8.0f}k {d2_by_w[j]:>11.3e}")

    # Same but for INTERPOLATION ERROR (h^2 * |V''| / 8)
    print("\n=== Top wealth bands by RAW interp-error proxy h^2 |V''|/8 (max across ages) ===")
    for arr_name in ("C", "S", "B"):
        err_by_w = np.zeros(n_w - 2)
        for age in args.ages:
            t = age - base_config["start_age"]
            if t < 0 or t >= C.shape[0]: continue
            arr = {"C": C, "S": S, "B": B}[arr_name]
            slab = arr[t, z_mid, state_mid, :]
            d2 = np.abs(second_derivative_nonuniform(slab, w_grid))
            local_h = 0.5 * (w_grid[2:] - w_grid[:-2])
            err = local_h ** 2 * d2 / 8.0
            err_by_w = np.maximum(err_by_w, err)
        top_idx = np.argsort(err_by_w)[::-1][:8]
        print(f"\n  Policy {arr_name}: top 8 wealth bins by max interp-error proxy")
        print(f"    {'w_idx':>6} {'W (AWI)':>10} {'W ($)':>10} {'h^2|V''|/8':>14}")
        for j in top_idx:
            w = w_grid[1 + j]
            print(f"    {1+j:>6} {w:>10.3f} ${w*AWI_2019_USD/1000:>8.0f}k {err_by_w[j]:>14.3e}")

    # Same but DENSITY-WEIGHTED (what the typical household experiences)
    print("\n=== Top wealth bands by DENSITY-WEIGHTED interp-error proxy ===")
    print("    (= probability that a random household has wealth in this bin x interp error)")
    for arr_name in ("C", "S", "B"):
        werr_by_w = np.zeros(n_w - 2)
        for age in args.ages:
            t = age - base_config["start_age"]
            if t < 0 or t >= C.shape[0]: continue
            arr = {"C": C, "S": S, "B": B}[arr_name]
            slab = arr[t, z_mid, state_mid, :]
            d2 = np.abs(second_derivative_nonuniform(slab, w_grid))
            local_h = 0.5 * (w_grid[2:] - w_grid[:-2])
            density = age_w_density[t, 1:-1]
            err = local_h ** 2 * d2 / 8.0 * density
            werr_by_w = np.maximum(werr_by_w, err)
        top_idx = np.argsort(werr_by_w)[::-1][:8]
        print(f"\n  Policy {arr_name}: top 8 by density-weighted error")
        print(f"    {'w_idx':>6} {'W (AWI)':>10} {'W ($)':>10} {'mass*err':>14}")
        for j in top_idx:
            w = w_grid[1 + j]
            print(f"    {1+j:>6} {w:>10.3f} ${w*AWI_2019_USD/1000:>8.0f}k {werr_by_w[j]:>14.3e}")

    # ---- Plot ----
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    arrays = ("C", "S", "B")
    labels = {"C": "consumption c", "S": r"$\alpha_s$", "B": r"$\alpha_b$"}
    cmap = plt.get_cmap("plasma")
    for r, arr_name in enumerate(arrays):
        arr = {"C": C, "S": S, "B": B}[arr_name]
        ax_curv = axes[r, 0]
        ax_err = axes[r, 1]
        ax_werr = axes[r, 2]
        for i, age in enumerate(args.ages):
            t = age - base_config["start_age"]
            if t < 0 or t >= arr.shape[0]: continue
            slab = arr[t, z_mid, state_mid, :]
            d2 = np.abs(second_derivative_nonuniform(slab, w_grid))
            interior_w = w_grid[1:-1]
            local_h = 0.5 * (w_grid[2:] - w_grid[:-2])
            err = local_h ** 2 * d2 / 8.0
            density = age_w_density[t, 1:-1]
            werr = err * density
            color = cmap(i / max(1, len(args.ages) - 1))
            ax_curv.plot(interior_w, d2 + 1e-30, color=color, label=f"age {age}", lw=1.4)
            ax_err.plot(interior_w, err + 1e-30, color=color, label=f"age {age}", lw=1.4)
            ax_werr.plot(interior_w, werr + 1e-30, color=color, label=f"age {age}", lw=1.4)
        ax_curv.set_xscale("log"); ax_curv.set_yscale("log")
        ax_err.set_xscale("log"); ax_err.set_yscale("log")
        ax_werr.set_xscale("log"); ax_werr.set_yscale("log")
        ax_curv.set_xlabel("wealth W (AWI units)")
        ax_err.set_xlabel("wealth W (AWI units)")
        ax_werr.set_xlabel("wealth W (AWI units)")
        ax_curv.set_ylabel(f"|d²{labels[arr_name]}/dW²|")
        ax_err.set_ylabel(f"interp-err proxy h²|V''|/8 ({labels[arr_name]})")
        ax_werr.set_ylabel(f"density-weighted interp err ({labels[arr_name]})")
        ax_curv.set_title(f"{labels[arr_name]}: raw curvature")
        ax_err.set_title(f"{labels[arr_name]}: interp-error proxy")
        ax_werr.set_title(f"{labels[arr_name]}: density-weighted error\n(= per-household experience)")
        ax_curv.grid(True, which="both", alpha=0.3)
        ax_err.grid(True, which="both", alpha=0.3)
        ax_werr.grid(True, which="both", alpha=0.3)
        ax_curv.legend(fontsize=8)
    fig.suptitle("Wealth-axis policy curvature, n_z=70 System I, z=mid, state=mid")
    fig.tight_layout()
    out = args.fig_dir / "wealth_axis_curvature.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
