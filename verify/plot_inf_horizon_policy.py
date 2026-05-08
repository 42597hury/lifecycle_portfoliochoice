"""verify/plot_inf_horizon_policy.py -- Plot inf-horizon stationary policy.

Visualises how equity (alpha_s) and bond (alpha_b) shares vary across the
state variables of whichever real-yields system the bundle was solved on
(Full = (cape, spr, y_1); System 2 = (spr, y_1); System 1 = (y_1,)) at
several wealth levels, using the saved inf-horizon bundle (single
stationary retirement-phase policy).

Coordinates: the state grid is Cholesky-decorrelated, so a sweep of axis k
holding the other axes at their median index dominantly varies original-coord
state variable k. We plot vs the decorrelated axis index but annotate with the
original-coord range of the dominant variable.

Usage
-----
    python verify/plot_inf_horizon_policy.py <bundle> [--wealth-fracs 0.05,0.2,0.5,0.9]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid
from verify._diag_helpers import build_bundle_var_config
from verify.ee_simpath_inf_horizon import _rehydrate_disc_config


def _resolve_bundle_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    p2 = Path("saved_runs") / "inf_horizon" / arg
    if p2.is_dir():
        return p2
    p3 = Path("saved_runs") / arg
    if p3.is_dir():
        return p3
    raise FileNotFoundError(f"Bundle not found: tried {p}, {p2}, {p3}")


def _wealth_indices(wealth_grid: np.ndarray, fracs: list[float]) -> list[int]:
    n_w = len(wealth_grid)
    return [int(round(f * (n_w - 1))) for f in fracs]


def _axis_median_indices(axis_lengths: list[int]) -> list[int]:
    return [n // 2 for n in axis_lengths]


def _flat_index(per_axis_idx: tuple[int, ...], axis_lengths: list[int]) -> int:
    s = 1
    flat = 0
    for d in range(len(axis_lengths) - 1, -1, -1):
        flat += per_axis_idx[d] * s
        s *= axis_lengths[d]
    return flat


def _gather_axis_sweep(arr3d: np.ndarray, axis_k: int, axis_lengths: list[int],
                       fixed_idx: list[int], wealth_idx: int) -> np.ndarray:
    """For each value i along decorrelated axis k, hold other axes at fixed_idx
    and read arr3d[0, flat_idx, wealth_idx]. Returns a (len_k,) vector."""
    out = np.empty(axis_lengths[axis_k], dtype=arr3d.dtype)
    for i in range(axis_lengths[axis_k]):
        idx = list(fixed_idx)
        idx[axis_k] = i
        flat = _flat_index(tuple(idx), axis_lengths)
        out[i] = arr3d[0, flat, wealth_idx]
    return out


def _gather_axis_wealth_grid(arr3d: np.ndarray, axis_k: int,
                              axis_lengths: list[int], fixed_idx: list[int]
                              ) -> np.ndarray:
    """Returns shape (len_k, n_w): arr3d[0, flat(i_k=i), :] for i sweeping axis k."""
    n_w = arr3d.shape[-1]
    out = np.empty((axis_lengths[axis_k], n_w), dtype=arr3d.dtype)
    for i in range(axis_lengths[axis_k]):
        idx = list(fixed_idx)
        idx[axis_k] = i
        flat = _flat_index(tuple(idx), axis_lengths)
        out[i, :] = arr3d[0, flat, :]
    return out


def _original_coords_along_axis(state_grid: np.ndarray, axis_k: int,
                                 axis_lengths: list[int],
                                 fixed_idx: list[int]) -> np.ndarray:
    """Read original-coord state vector for each value along axis k holding
    others fixed. Returns (len_k, n_state) of state variables in the bundle's
    actual order (Full = (cape, spr, y_1); System 2 = (spr, y_1); System 1 =
    (y_1,))."""
    out = np.empty((axis_lengths[axis_k], state_grid.shape[1]), dtype=float)
    for i in range(axis_lengths[axis_k]):
        idx = list(fixed_idx)
        idx[axis_k] = i
        flat = _flat_index(tuple(idx), axis_lengths)
        out[i, :] = state_grid[flat, :]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("bundle", nargs="?",
                        default="full_system_inf_grid_g5_quad334_ret44_calib1")
    parser.add_argument("--wealth-fracs", default="0.05,0.2,0.5,0.8",
                        help="Comma-separated fractions of n_w for line-plot wealth slices.")
    parser.add_argument("--out-dir", default="docs/scans/figures",
                        help="Directory to save PNG files.")
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle_path}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading bundle + rebuilding precompute...", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    disc = _rehydrate_disc_config(metadata["run_config"]["discretization_config"])
    disc = disc_config_with_bundle_wealth_grid(disc, bundle_path, metadata)
    var_cfg = build_bundle_var_config(metadata, bundle_path)
    model = build_model(metadata["run_config"]["base_config"], var_cfg, verbose=False)
    pc = build_precompute(model, disc, verbose=False)

    state_names = list(model.state_names)
    n_state = len(state_names)
    axis_lengths = [g.shape[0] for g in pc.state_bracket_grids]
    state_grid = np.asarray(pc.state_grid)
    wealth_grid = np.asarray(pc.wealth_grid)
    n_w = len(wealth_grid)

    fracs = [float(x) for x in args.wealth_fracs.split(",")]
    w_idx_list = _wealth_indices(wealth_grid, fracs)
    fixed_idx = _axis_median_indices(axis_lengths)
    bundle_label = bundle_path.name

    print(f"  state names: {state_names}", flush=True)
    print(f"  axis lengths: {axis_lengths}", flush=True)
    print(f"  wealth slices (frac, idx, value):", flush=True)
    for f, i in zip(fracs, w_idx_list):
        print(f"    {f:.2f}  idx={i:3d}  W={wealth_grid[i]:.3f}", flush=True)
    print(f"  fixed (median) per-axis idx: {fixed_idx}", flush=True)

    decor_grids = [np.asarray(g) for g in pc.state_bracket_grids]

    # =========================================================================
    # Plot 1: alpha_s + alpha_b vs each state axis at multiple wealth levels
    #   2 rows (alpha_s, alpha_b) x 4 cols (one per state axis). Lines = wealth.
    # =========================================================================
    print("\nPlot 1: alpha vs state axis @ multiple wealth levels...", flush=True)
    fig, axes = plt.subplots(2, n_state, figsize=(4.3 * n_state, 7), sharey="row")
    cmap = plt.get_cmap("viridis")
    for k in range(n_state):
        coords = _original_coords_along_axis(state_grid, k, axis_lengths, fixed_idx)
        x_decor = decor_grids[k]
        x_orig_k = coords[:, k]
        for j, (frac, w_idx) in enumerate(zip(fracs, w_idx_list)):
            color = cmap(j / max(1, len(fracs) - 1))
            alpha_s_line = _gather_axis_sweep(S, k, axis_lengths, fixed_idx, w_idx)
            alpha_b_line = _gather_axis_sweep(B, k, axis_lengths, fixed_idx, w_idx)
            axes[0, k].plot(x_orig_k, alpha_s_line, "o-", color=color,
                            label=f"W={wealth_grid[w_idx]:.2f}")
            axes[1, k].plot(x_orig_k, alpha_b_line, "o-", color=color,
                            label=f"W={wealth_grid[w_idx]:.2f}")
        axes[0, k].set_title(f"axis {k}: dominant {state_names[k]}\n"
                             f"(others fixed at decor idx={fixed_idx[k]}={x_decor[fixed_idx[k]]:+.2f})",
                             fontsize=10)
        axes[1, k].set_xlabel(f"{state_names[k]} (original coord)")
        axes[0, k].grid(True, alpha=0.3)
        axes[1, k].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel(r"$\alpha_s$ (equity share)")
    axes[1, 0].set_ylabel(r"$\alpha_b$ (bond share)")
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Inf-horizon stationary policy: portfolio shares vs state\n"
                 f"Bundle: {bundle_label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"inf_horizon_policy_lines_{bundle_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}", flush=True)

    # =========================================================================
    # Plot 2: alpha_s heatmap in (state-axis, wealth) for each axis (4 panels)
    # =========================================================================
    print("\nPlot 2: alpha_s heatmap (state-axis x wealth)...", flush=True)
    fig, axes = plt.subplots(1, n_state, figsize=(4.3 * n_state, 4.5), sharey=True)
    for k in range(n_state):
        S_kw = _gather_axis_wealth_grid(S, k, axis_lengths, fixed_idx)
        coords = _original_coords_along_axis(state_grid, k, axis_lengths, fixed_idx)
        x_orig_k = coords[:, k]
        # log-spaced y-axis (wealth) for readability
        im = axes[k].pcolormesh(
            x_orig_k, wealth_grid, S_kw.T,
            shading="auto", cmap="viridis",
        )
        axes[k].set_yscale("log")
        axes[k].set_xlabel(state_names[k])
        axes[k].set_title(f"axis {k}: {state_names[k]}")
        plt.colorbar(im, ax=axes[k], label=r"$\alpha_s$")
    axes[0].set_ylabel("Wealth (log scale)")
    fig.suptitle(rf"Equity share $\alpha_s$ across (state, wealth) -- {bundle_label}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / f"inf_horizon_alpha_s_heatmap_{bundle_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}", flush=True)

    # =========================================================================
    # Plot 3: alpha_s vs wealth for several state-axis values, one panel per axis
    # =========================================================================
    print("\nPlot 3: alpha_s vs wealth for several state values...", flush=True)
    fig, axes = plt.subplots(1, n_state, figsize=(4.3 * n_state, 4.5), sharey=True)
    cmap = plt.get_cmap("plasma")
    for k in range(n_state):
        S_kw = _gather_axis_wealth_grid(S, k, axis_lengths, fixed_idx)
        coords = _original_coords_along_axis(state_grid, k, axis_lengths, fixed_idx)
        x_orig_k = coords[:, k]
        for i in range(axis_lengths[k]):
            color = cmap(i / max(1, axis_lengths[k] - 1))
            axes[k].plot(wealth_grid, S_kw[i, :], "-", color=color,
                         label=f"{state_names[k]}={x_orig_k[i]:+.3f}")
        axes[k].set_xscale("log")
        axes[k].set_xlabel("Wealth (log scale)")
        axes[k].set_title(f"varying {state_names[k]}")
        axes[k].grid(True, alpha=0.3)
        axes[k].legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"$\alpha_s$ (equity share)")
    fig.suptitle(f"Equity share vs wealth, varying one state at a time -- {bundle_label}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / f"inf_horizon_alpha_s_vs_wealth_{bundle_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}", flush=True)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
