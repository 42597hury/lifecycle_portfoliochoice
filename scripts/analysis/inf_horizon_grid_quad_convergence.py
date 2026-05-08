"""
LEGACY analysis: pre-pivot System IV infinite-horizon resolution sweeps.

These bundles (system_iv_inf_*) were solved on the 4-axis nominal model and
are still on disk under saved_runs/inf_horizon/. This script is read-only
on those bundles' policy arrays + diagnostics — it does not rebuild a
precompute or call into lifecycle/var, so it still runs on the post-pivot
branch. The analogous post-pivot script will read full_system_inf_* bundles
once the new sweeps complete; they are not yet available.

Two sweeps under `saved_runs/inf_horizon/`:

  axis-bump (6 runs) — varies state-quadrature and return-quadrature node
  counts around a baseline of sq=(3,3,3,3) rq=(3,3) on a 5x5x5x5 state grid:

    run1: sq=(3,3,3,3) rq=(3,3)            <- baseline
    run2: sq=(3,3,3,5) rq=(3,3)            (bump y_1 state-quad)
    run3: sq=(5,3,3,3) rq=(3,3)            (bump dp state-quad)
    run4: sq=(3,5,3,3) rq=(3,3)            (bump spr state-quad)
    run5: sq=(3,3,3,3) rq=(3,5)            (bump xb return-quad)
    run6: sq=(3,3,3,3) rq=(5,3)            (bump xr return-quad)

  state-grid-size (3 runs) — varies isotropic state grid size with
  quad=(3,3,3,4) rq=(4,4):

    g3: state_grid_sizes=(3,3,3,3) -> 81 states
    g4: state_grid_sizes=(4,4,4,4) -> 256 states
    g5: state_grid_sizes=(5,5,5,5) -> 625 states

All bundles share n_wealth=180 (identical wealth grid), n_z=1 (z is inert in
inf-horizon under no-pension, no-survival), state_n_stds=(2.0, 2.25, 2.0, 2.25).

This script computes:
  * axis-bump sweep: per-axis sup-norm / RMS / p99 divergence vs run1.
  * grid-size sweep: g3 and g4 multilinearly interpolated onto g5's state
    grid, then per-array divergence vs g5 reference.
  * stationary-mass-weighted error proxy using the run-specific
    stationary_probs from build_state_grid (computed via Pi_state).
  * solver-side diagnostics flags (n_iter cap, final_stopping_supnorm).

Writes a JSON metrics blob to docs/scans/.
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
from lifecycle.wealth_grid import load_wealth_grid_from_bundle  # noqa: E402

INF_HORIZON_ROOT = REPO / "saved_runs" / "inf_horizon"

AXIS_BUMP_BUNDLES = [
    "system_iv_inf_axisbump_run1_sq3333_rq33_calib1",
    "system_iv_inf_axisbump_run2_sq3335_rq33_calib1",
    "system_iv_inf_axisbump_run3_sq5333_rq33_calib1",
    "system_iv_inf_axisbump_run4_sq3533_rq33_calib1",
    "system_iv_inf_axisbump_run5_sq3333_rq35_calib1",
    "system_iv_inf_axisbump_run6_sq3333_rq53_calib1",
]
AXIS_BUMP_BASELINE = AXIS_BUMP_BUNDLES[0]

GRID_BUNDLES = [
    "system_iv_inf_grid_g3_quad3334_ret44_calib1",
    "system_iv_inf_grid_g4_quad3334_ret44_calib1",
    "system_iv_inf_grid_g5_quad3334_ret44_calib1",
]
GRID_REFERENCE = GRID_BUNDLES[-1]   # g5

STATE_NAMES = ("dp", "spr", "rtb", "y_1")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_bundle(name: str) -> dict[str, Any]:
    bundle = INF_HORIZON_ROOT / name
    if not bundle.exists():
        raise FileNotFoundError(bundle)
    C, S, B, _diag, metadata = load_policy_bundle(bundle)
    diag_path = bundle / "diagnostics.pkl"
    with diag_path.open("rb") as f:
        diag = pickle.load(f)
    rc = metadata["run_config"]
    disc = rc["discretization_config"]
    return {
        "name": name,
        "bundle": bundle,
        "C": np.asarray(C),
        "S": np.asarray(S),
        "B": np.asarray(B),
        "wealth_grid": load_wealth_grid_from_bundle(bundle, metadata),
        "metadata": metadata,
        "disc": disc,
        "solver_config": rc["solver_config"],
        "wall": rc.get("wall_time_seconds"),
        "diag": diag,
        "state_grid_sizes": tuple(int(v) for v in disc["state_grid_sizes"]),
        "state_n_stds": tuple(float(v) for v in disc["state_n_stds"]),
        "n_state_quad_nodes": tuple(int(v) for v in disc["n_state_quad_nodes"]),
        "n_ret_nodes_1d": tuple(int(v) for v in disc["n_ret_nodes_1d"]),
    }


# ---------------------------------------------------------------------------
# Stationary probabilities (under Cholesky bracket grid, build_state_grid)
# ---------------------------------------------------------------------------
# build_state_grid sets stationary_probs as the product of per-axis bin
# probabilities under N(0, I) for the bracket grids. Here we re-derive them.

from lifecycle.numerics import _normal_bin_probs  # noqa: E402


def stationary_state_probs(state_grid_sizes: tuple[int, ...],
                           state_n_stds: tuple[float, ...]) -> np.ndarray:
    """Per-state-cell stationary probability under the Cholesky construction.

    The state grid is built as a tensor product of axis bracket grids
    np.linspace(-n_stds[d], +n_stds[d], state_grid_sizes[d]); under
    `_normal_bin_probs` each axis is a probability vector summing to 1.
    Returns a flat (N_state,) array.
    """
    probs_1d = []
    for N, sd in zip(state_grid_sizes, state_n_stds):
        if N == 1:
            probs_1d.append(np.array([1.0]))
        else:
            grid = np.linspace(-sd, sd, N)
            probs_1d.append(_normal_bin_probs(grid))
    out = np.array([1.0])
    for p in probs_1d:
        out = np.outer(out, p).ravel()
    out /= out.sum()
    return out


# ---------------------------------------------------------------------------
# Cross-grid multilinear interpolation
# ---------------------------------------------------------------------------

def axis_grid(N: int, sd: float) -> np.ndarray:
    return np.linspace(-sd, sd, N)


def interp_policy_to_finer_grid(
    coarse_policy: np.ndarray,        # shape (1, N_coarse, n_w)
    coarse_sizes: tuple[int, ...],
    coarse_n_stds: tuple[float, ...],
    fine_sizes: tuple[int, ...],
    fine_n_stds: tuple[float, ...],
) -> np.ndarray:
    """Project a coarse-grid policy onto the fine grid by multilinear interp
    along each of the four state axes.

    The state grid is a Cholesky tensor product so the per-axis bracket grids
    are np.linspace(-n_stds[d], +n_stds[d], N[d]). We interpolate in those
    bracket coordinates.

    Returns shape (1, N_fine_total, n_w).
    """
    assert coarse_policy.ndim == 3 and coarse_policy.shape[0] == 1
    n_w = coarse_policy.shape[-1]
    n_axes = len(coarse_sizes)

    coarse_axes = [axis_grid(coarse_sizes[d], coarse_n_stds[d]) for d in range(n_axes)]
    fine_axes = [axis_grid(fine_sizes[d], fine_n_stds[d]) for d in range(n_axes)]

    # Reshape coarse to (n_w, *coarse_sizes), interpolate axis-by-axis.
    coarse = coarse_policy[0].reshape(*coarse_sizes, n_w)        # (*coarse_sizes, n_w)

    out = coarse
    for d in range(n_axes):
        # Interp axis d from coarse_axes[d] to fine_axes[d].
        fine_grid_d = fine_axes[d]
        coarse_grid_d = coarse_axes[d]
        # Move axis d to position 0 for vectorised np.interp via broadcasting.
        out = np.moveaxis(out, d, 0)
        original_shape = out.shape
        flat = out.reshape(coarse_grid_d.size, -1)
        new_flat = np.empty((fine_grid_d.size, flat.shape[1]), dtype=flat.dtype)
        for k in range(flat.shape[1]):
            new_flat[:, k] = np.interp(fine_grid_d, coarse_grid_d, flat[:, k])
        out = new_flat.reshape(fine_grid_d.size, *original_shape[1:])
        out = np.moveaxis(out, 0, d)

    fine_total = int(np.prod(fine_sizes))
    return out.reshape(1, fine_total, n_w)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_pair(coarse: np.ndarray, ref: np.ndarray,
                 stationary_probs: np.ndarray | None = None,
                 array_label: str = "?") -> dict[str, Any]:
    """Compute scalar + per-axis divergence metrics for one (coarse, ref) pair.

    Both arrays have shape (1, N_state, n_w). If stationary_probs is provided,
    also reports stationary-mass-weighted RMS along the state axis.
    """
    assert coarse.shape == ref.shape, f"shape mismatch: {coarse.shape} vs {ref.shape}"
    delta = np.abs(coarse - ref)
    sup = float(delta.max())
    rms = float(np.sqrt(np.mean(delta ** 2)))
    mean_abs = float(delta.mean())
    p99 = float(np.percentile(delta, 99))
    p95 = float(np.percentile(delta, 95))

    ref_abs = np.abs(ref)
    threshold = 1e-3 * float(np.median(ref_abs[ref_abs > 0])) if (ref_abs > 0).any() else 1e-8
    mask = ref_abs > threshold
    if mask.any():
        rel = delta[mask] / ref_abs[mask]
        sup_rel = float(rel.max())
        p99_rel = float(np.percentile(rel, 99))
    else:
        sup_rel = float("nan")
        p99_rel = float("nan")

    out = {
        "array": array_label,
        "sup": sup,
        "rms": rms,
        "p99": p99,
        "p95": p95,
        "mean_abs": mean_abs,
        "sup_rel": sup_rel,
        "p99_rel": p99_rel,
        "per_state_max": [float(x) for x in delta[0].max(axis=1)],     # one per state cell
        "per_wealth_max": [float(x) for x in delta[0].max(axis=0)],    # one per wealth cell
    }
    if stationary_probs is not None:
        # State-axis stationary-weighted RMS, uniform over wealth.
        w = stationary_probs / stationary_probs.sum()
        out["stat_weighted_rms"] = float(
            np.sqrt(((delta[0] ** 2) * w[:, None]).sum() / coarse.shape[-1])
        )
        out["stat_weighted_mean_abs"] = float(
            ((delta[0]) * w[:, None]).sum() / coarse.shape[-1]
        )
    return out


# ---------------------------------------------------------------------------
# Sweep analyses
# ---------------------------------------------------------------------------

def analyse_axis_bump_sweep() -> dict[str, Any]:
    print("Loading axis-bump sweep bundles...", flush=True)
    bundles = {name: load_bundle(name) for name in AXIS_BUMP_BUNDLES}
    base = bundles[AXIS_BUMP_BASELINE]
    print(f"  Baseline: {base['name']}", flush=True)
    print(f"  Shape: {base['C'].shape}, state_grid_sizes={base['state_grid_sizes']}",
          flush=True)
    print(f"  Solver: tol={base['solver_config']['tol']}, "
          f"max_iter={base['solver_config']['max_iter']}", flush=True)

    sup_baseline = base["diag"].get("final_stopping_supnorm", float("nan"))
    print(f"  Baseline solver final_stopping_supnorm = {sup_baseline:.3e} "
          f"(must be ~tol for the comparison to be trustworthy)", flush=True)

    state_probs = stationary_state_probs(
        base["state_grid_sizes"], base["state_n_stds"])

    pairs = {}
    print("\nDivergence vs run1 baseline (sq=(3,3,3,3), rq=(3,3)):", flush=True)
    print(f"  {'run':<60} {'sq':>14} {'rq':>10} {'wall':>7} {'sup_C':>9} "
          f"{'sup_S':>9} {'sup_B':>9} {'rms_C':>9}", flush=True)
    for name, b in bundles.items():
        if name == AXIS_BUMP_BASELINE:
            continue
        m_c = metrics_pair(b["C"], base["C"], state_probs, "C")
        m_s = metrics_pair(b["S"], base["S"], state_probs, "S")
        m_b = metrics_pair(b["B"], base["B"], state_probs, "B")
        pairs[name] = {
            "sq": list(b["n_state_quad_nodes"]),
            "rq": list(b["n_ret_nodes_1d"]),
            "wall_seconds": b["wall"],
            "solver_diag": {
                "converged": bool(b["diag"].get("converged", False)),
                "n_iter": int(b["diag"].get("n_iter", -1)),
                "final_stopping_supnorm": float(b["diag"].get(
                    "final_stopping_supnorm", float("nan"))),
            },
            "C": m_c, "S": m_s, "B": m_b,
        }
        print(f"  {name:<60} {str(list(b['n_state_quad_nodes'])):>14} "
              f"{str(list(b['n_ret_nodes_1d'])):>10} "
              f"{b['wall']:>6.0f}s {m_c['sup']:>9.3e} {m_s['sup']:>9.3e} "
              f"{m_b['sup']:>9.3e} {m_c['rms']:>9.3e}", flush=True)

    return {
        "baseline": AXIS_BUMP_BASELINE,
        "baseline_shape": list(base["C"].shape),
        "baseline_state_grid_sizes": list(base["state_grid_sizes"]),
        "baseline_state_n_stds": list(base["state_n_stds"]),
        "baseline_wall_seconds": base["wall"],
        "baseline_diag": {
            "converged": bool(base["diag"].get("converged", False)),
            "n_iter": int(base["diag"].get("n_iter", -1)),
            "final_stopping_supnorm": float(base["diag"].get(
                "final_stopping_supnorm", float("nan"))),
        },
        "pairs": pairs,
    }


def analyse_grid_size_sweep() -> dict[str, Any]:
    print("\n\nLoading grid-size sweep bundles...", flush=True)
    bundles = {name: load_bundle(name) for name in GRID_BUNDLES}
    ref = bundles[GRID_REFERENCE]
    print(f"  Reference: {ref['name']} (state_grid_sizes={ref['state_grid_sizes']})",
          flush=True)
    for name, b in bundles.items():
        sup = b["diag"].get("final_stopping_supnorm", float("nan"))
        conv = b["diag"].get("converged", "?")
        n_iter = b["diag"].get("n_iter", "?")
        print(f"  {name}: sizes={b['state_grid_sizes']} wall={b['wall']:.0f}s "
              f"converged={conv} n_iter={n_iter} stop={sup:.3e}", flush=True)

    ref_state_probs = stationary_state_probs(
        ref["state_grid_sizes"], ref["state_n_stds"])

    pairs = {}
    print("\nDivergence vs g5 reference (after multilinear projection onto g5 grid):",
          flush=True)
    print(f"  {'run':<55} {'wall':>7} {'sup_C':>9} {'sup_S':>9} {'sup_B':>9} "
          f"{'rms_C':>9} {'wRMS_C':>9}", flush=True)
    for name, b in bundles.items():
        if name == GRID_REFERENCE:
            continue
        # Project coarse onto g5
        C_proj = interp_policy_to_finer_grid(
            b["C"], b["state_grid_sizes"], b["state_n_stds"],
            ref["state_grid_sizes"], ref["state_n_stds"])
        S_proj = interp_policy_to_finer_grid(
            b["S"], b["state_grid_sizes"], b["state_n_stds"],
            ref["state_grid_sizes"], ref["state_n_stds"])
        B_proj = interp_policy_to_finer_grid(
            b["B"], b["state_grid_sizes"], b["state_n_stds"],
            ref["state_grid_sizes"], ref["state_n_stds"])
        m_c = metrics_pair(C_proj, ref["C"], ref_state_probs, "C")
        m_s = metrics_pair(S_proj, ref["S"], ref_state_probs, "S")
        m_b = metrics_pair(B_proj, ref["B"], ref_state_probs, "B")
        pairs[name] = {
            "state_grid_sizes": list(b["state_grid_sizes"]),
            "wall_seconds": b["wall"],
            "solver_diag": {
                "converged": bool(b["diag"].get("converged", False)),
                "n_iter": int(b["diag"].get("n_iter", -1)),
                "final_stopping_supnorm": float(b["diag"].get(
                    "final_stopping_supnorm", float("nan"))),
            },
            "C": m_c, "S": m_s, "B": m_b,
        }
        print(f"  {name:<55} {b['wall']:>6.0f}s {m_c['sup']:>9.3e} "
              f"{m_s['sup']:>9.3e} {m_b['sup']:>9.3e} {m_c['rms']:>9.3e} "
              f"{m_c.get('stat_weighted_rms', float('nan')):>9.3e}", flush=True)

    return {
        "reference": GRID_REFERENCE,
        "reference_state_grid_sizes": list(ref["state_grid_sizes"]),
        "reference_state_n_stds": list(ref["state_n_stds"]),
        "reference_wall_seconds": ref["wall"],
        "reference_diag": {
            "converged": bool(ref["diag"].get("converged", False)),
            "n_iter": int(ref["diag"].get("n_iter", -1)),
            "final_stopping_supnorm": float(ref["diag"].get(
                "final_stopping_supnorm", float("nan"))),
        },
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs" / "scans" / "inf_horizon_grid_quad_metrics.json")
    args = parser.parse_args(argv)

    axis = analyse_axis_bump_sweep()
    grid = analyse_grid_size_sweep()

    payload = {
        "axis_bump_sweep": axis,
        "grid_size_sweep": grid,
        "state_axis_names": list(STATE_NAMES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote metrics to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
