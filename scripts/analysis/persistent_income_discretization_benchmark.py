"""Benchmark quadrature-vs-Pi_z persistent-income integration on System I.

The script can reuse existing quadrature bundles and solve missing bundles for
either method.  Outputs a metrics JSON and an accuracy-vs-wall plot under
``docs/scans`` by default.

Examples
--------
Run only the Pi_z variants against the existing n_z=70 quadrature reference:

    python scripts/analysis/persistent_income_discretization_benchmark.py \
        --methods pi_z --n-z-values 10 15 20 30

Run both methods, solving any missing bundles:

    python scripts/analysis/persistent_income_discretization_benchmark.py \
        --methods quadrature pi_z --n-z-values 10 15 20 30 50
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER  # noqa: E402
from lifecycle.discretization import discretize_income_ar1_mixture  # noqa: E402
from lifecycle.model import SolveControl  # noqa: E402
from lifecycle.policy_io import load_policy_bundle, save_policy_bundle  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.predictability_ablation import prepare_predictability_system  # noqa: E402
from lifecycle.solver import run_lifecycle_solver  # noqa: E402
from lifecycle.solver_pi_z_variant import run_lifecycle_solver_pi_z  # noqa: E402


CSV_PATH = REPO / "data" / "var_dataset.csv"
DEFAULT_NZ = (10, 15, 20, 30, 50)
REFERENCE_NZ = 70


def _template_disc(n_z: int):
    # 3-axis (cape, spr, y_1) template post real-yields pivot. System 1
    # projection keeps only the y_1 entry (last axis).
    return CANONICAL_DISC._replace(
        wealth_min=0.05,
        state_grid_sizes=(7, 7, 7),
        state_n_stds=(2.0, 2.25, 2.25),
        n_stds=2.25,
        n_z=int(n_z),
        n_eta_nodes=3,
        n_eps_nodes=4,
        n_state_quad_nodes=(3, 3, 5),
        state_lobatto_Z=None,
        n_ret_nodes_1d=(3, 3),
        ret_lobatto_Z=None,
    )


def _solver_config():
    return CANONICAL_SOLVER._replace(
        wealth_dynamics_spec="ccv_log",
        max_iter=100,
        max_iter_unconstrained=100,
        delta_bequest=0.0,
        gather_precision="f32",
        cell_vmap_chunks=1,
    )


def _solve_control():
    return SolveControl(
        youngest_age_to_solve=22,
        checkpoint_every_n_ages=10,
        save_on_interrupt=True,
        return_partial_on_interrupt=True,
    )


def bundle_name(method: str, n_z: int) -> str:
    if method == "quadrature":
        return f"system_1_grid7_nz{n_z}_calib1"
    if method == "pi_z":
        return f"system_1_grid7_nz{n_z}_pi_z_calib1"
    raise ValueError(f"Unknown method {method!r}")


def _read_disc_config(bundle: Path, diagnostics: dict[str, Any] | None, metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata and "run_config" in metadata:
        disc = metadata["run_config"].get("discretization_config")
        if disc is not None:
            return dict(disc)
    if diagnostics and "disc_config" in diagnostics:
        return dict(diagnostics["disc_config"])
    diag_path = bundle / "diagnostics.pkl"
    if diag_path.exists():
        with diag_path.open("rb") as f:
            diag = pickle.load(f)
        if "disc_config" in diag:
            return dict(diag["disc_config"])
    raise KeyError(f"Cannot infer discretization config for {bundle}")


def reconstruct_z_grid(disc: dict[str, Any]) -> np.ndarray:
    z_grid, _ = discretize_income_ar1_mixture(
        rho=float(BASE_CONFIG["rho"]),
        p=float(BASE_CONFIG["pz"]),
        mu1=float(BASE_CONFIG["mu_eta1"]),
        sigma1=float(BASE_CONFIG["sigma_eta1"]),
        mu2=float(BASE_CONFIG["mu_eta2"]),
        sigma2=float(BASE_CONFIG["sigma_eta2"]),
        N=int(disc["n_z"]),
        n_stds=float(disc["n_stds"]),
    )
    return np.asarray(z_grid, dtype=float)


def load_bundle_payload(bundle: Path) -> dict[str, Any]:
    C, S, B, diag, metadata = load_policy_bundle(bundle)
    disc = _read_disc_config(bundle, diag, metadata)
    wall = None
    if diag:
        wall = diag.get("wall_time_sec")
    if wall is None and metadata:
        wall = metadata.get("run_config", {}).get("wall_time_seconds")
    return {
        "bundle": bundle,
        "C": np.asarray(C),
        "S": np.asarray(S),
        "B": np.asarray(B),
        "diagnostics": diag or {},
        "metadata": metadata,
        "disc_config": disc,
        "z_grid": reconstruct_z_grid(disc),
        "wall_sec": float(wall) if wall is not None else float("nan"),
    }


def interp_along_z(values: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda v: np.interp(z_dst, z_src, v), axis=1, arr=values)


def _abs_delta_metrics(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    sup = float(np.max(delta))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    p99 = float(np.percentile(delta, 99))
    ref_abs = np.abs(ref)
    positive = ref_abs[ref_abs > 0]
    threshold = max(1e-8, 1e-3 * float(np.median(positive) if positive.size else 0.0))
    mask = ref_abs > threshold
    rel_sup = float(np.max(delta[mask] / ref_abs[mask])) if mask.any() else float("nan")
    return {"sup": sup, "rms": rms, "p99": p99, "sup_rel": rel_sup}


def policy_metrics(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("C", "S", "B"):
        projected = interp_along_z(candidate[name], candidate["z_grid"], reference["z_grid"])
        delta = np.abs(projected - reference[name])
        out[name] = _abs_delta_metrics(delta, reference[name])
    return out


def pi_z_transition_diagnostics(n_z_values: list[int], n_stds: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for n_z in n_z_values:
        z, Pi = discretize_income_ar1_mixture(
            rho=float(BASE_CONFIG["rho"]),
            p=float(BASE_CONFIG["pz"]),
            mu1=float(BASE_CONFIG["mu_eta1"]),
            sigma1=float(BASE_CONFIG["sigma_eta1"]),
            mu2=float(BASE_CONFIG["mu_eta2"]),
            sigma2=float(BASE_CONFIG["sigma_eta2"]),
            N=int(n_z),
            n_stds=float(n_stds),
        )
        mean_disc = Pi @ z
        mean_true = float(BASE_CONFIG["rho"]) * z
        bias = mean_disc - mean_true
        rows[str(n_z)] = {
            "z_min": float(z[0]),
            "z_max": float(z[-1]),
            "max_abs_row_sum_error": float(np.max(np.abs(Pi.sum(axis=1) - 1.0))),
            "max_abs_conditional_mean_bias": float(np.max(np.abs(bias))),
            "median_abs_conditional_mean_bias": float(np.median(np.abs(bias))),
            "left_edge_mean_bias": float(bias[0]),
            "right_edge_mean_bias": float(bias[-1]),
        }
    return rows


def solve_one(method: str, n_z: int, bundle_dir: Path, overwrite: bool) -> dict[str, Any]:
    template_disc = _template_disc(n_z)
    meta = prepare_predictability_system(
        "1",
        csv_path=str(CSV_PATH),
        disc_config_template=template_disc,
    )
    var_config = meta["var_config"]
    disc_config = meta["disc_config"]
    solver_config = _solver_config()
    solve_control = _solve_control()

    t0 = time.time()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc_config, verbose=False)
    setup_wall = time.time() - t0

    solve_fn = run_lifecycle_solver if method == "quadrature" else run_lifecycle_solver_pi_z
    t0 = time.time()
    C, S, B, diag = solve_fn(model, pc, solver_config, verbose=1, solve_control=solve_control)
    solve_wall = time.time() - t0

    if bundle_dir.exists() and overwrite:
        shutil.rmtree(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)

    run_config_snapshot = {
        "base_config": dict(BASE_CONFIG),
        "discretization_config": disc_config._asdict(),
        "solver_config": solver_config._asdict(),
        "solve_control": solve_control._asdict(),
        "predictability_ablation": {
            "system_code": meta["system_code"],
            "system_label": meta["system_label"],
            "system_title": meta["system_title"],
            "system_description": meta.get("system_description", ""),
            "state_names": list(meta["state_names"]),
        },
        "bundle_name": bundle_dir.name,
        "wall_time_seconds": float(solve_wall),
        "setup_wall_time_seconds": float(setup_wall),
        "solver_kind": "lifecycle_full",
        "income_discretization_method": method,
        "sweep_dimension": "n_z",
        "n_z_value": int(n_z),
    }
    save_policy_bundle(
        bundle_dir,
        C,
        S,
        B,
        diagnostics=diag,
        run_config=run_config_snapshot,
        overwrite=True,
        wealth_grid=pc.wealth_grid,
    )
    return load_bundle_payload(bundle_dir)


def plot_tradeoff(runs: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"quadrature": "#2f6f9f", "pi_z": "#b2453a"}
    labels = {"quadrature": "eta quadrature + linear z", "pi_z": "Pi_z discrete chain"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    for ax, key, title in zip(axes, ("C", "S", "B"), ("Consumption", "Stock Share", "Bond Share")):
        for method in ("quadrature", "pi_z"):
            sub = [r for r in runs if r["method"] == method and np.isfinite(r["wall_sec"])]
            sub.sort(key=lambda r: r["wall_sec"])
            if not sub:
                continue
            x = [r["wall_sec"] for r in sub]
            y = [r["metrics"][key]["sup"] for r in sub]
            ax.plot(x, y, marker="o", color=colors[method], label=labels[method])
            for r, xx, yy in zip(sub, x, y):
                ax.annotate(str(r["n_z"]), (xx, yy), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("solve wall seconds")
        ax.set_ylabel("sup divergence vs n_z=70 reference")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-z-values", nargs="+", type=int, default=list(DEFAULT_NZ))
    parser.add_argument("--methods", nargs="+", choices=("quadrature", "pi_z"), default=["quadrature", "pi_z"])
    parser.add_argument("--bundles-root", type=Path, default=REPO / "saved_runs" / "ablations")
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs" / "scans")
    parser.add_argument("--metrics-name", default="persistent_income_discretization_metrics.json")
    parser.add_argument("--plot-name", default="figures/persistent_income_discretization_tradeoff.png")
    parser.add_argument("--reference-bundle", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_bundle = args.reference_bundle or (
        args.bundles_root / bundle_name("quadrature", REFERENCE_NZ)
    )
    reference = load_bundle_payload(reference_bundle)

    runs: list[dict[str, Any]] = []
    for method in args.methods:
        for n_z in args.n_z_values:
            out_dir = args.bundles_root / bundle_name(method, n_z)
            if out_dir.exists() and not args.no_reuse_existing and not args.overwrite:
                print(f"REUSE {method} n_z={n_z}: {out_dir}", flush=True)
                payload = load_bundle_payload(out_dir)
            else:
                print(f"SOLVE {method} n_z={n_z}: {out_dir}", flush=True)
                payload = solve_one(method, n_z, out_dir, overwrite=args.overwrite)
            metrics = policy_metrics(payload, reference)
            run_row = {
                "method": method,
                "n_z": int(n_z),
                "bundle": str(out_dir),
                "wall_sec": payload["wall_sec"],
                "solve_status": payload["diagnostics"].get("solve_status"),
                "metrics": metrics,
            }
            runs.append(run_row)
            print(
                f"  wall={run_row['wall_sec']:.1f}s "
                f"sup_C={metrics['C']['sup']:.4e} "
                f"sup_S={metrics['S']['sup']:.4e} "
                f"sup_B={metrics['B']['sup']:.4e}",
                flush=True,
            )

    transition_diag = pi_z_transition_diagnostics(
        sorted(set(args.n_z_values) | {REFERENCE_NZ}),
        n_stds=2.25,
    )
    payload = {
        "reference_bundle": str(reference_bundle),
        "reference_n_z": REFERENCE_NZ,
        "reference_shape": list(reference["C"].shape),
        "runs": runs,
        "pi_z_transition_diagnostics": transition_diag,
    }
    metrics_path = args.output_dir / args.metrics_name
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {metrics_path}", flush=True)

    plot_path = args.output_dir / args.plot_name
    plot_tradeoff(runs, plot_path)
    print(f"Wrote {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
