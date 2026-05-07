"""
System I × n_z sensitivity convergence analysis.

Reads the four `system_i_grid7_nz<N>_calib1` bundles for N in (10, 15, 30, 70),
treats n_z=70 as the reference, projects each coarser policy onto the n_z=70
z-grid by linear interpolation along the z-axis, and reports a battery of
convergence metrics (sup-norm, L2, per-axis max, relative divergence) for the
consumption / risky-share / bond-share policies.

Usage:
    python scripts/analysis/system_i_nz_convergence.py
        [--bundles-root saved_runs/ablations]
        [--output-dir docs/scans]

Writes:
    {output-dir}/system_i_nz_convergence_metrics.json
        Machine-readable metrics for downstream plots / report.

The script is read-only with respect to `lifecycle/` — it only loads bundles
and reconstructs z-grids via `discretize_income_ar1_mixture`. It does not
re-solve or modify any solver code.
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
from lifecycle.discretization import discretize_income_ar1_mixture  # noqa: E402
from configs._canonical import BASE_CONFIG  # noqa: E402

NZ_VALUES: tuple[int, ...] = (10, 15, 30, 70)
REFERENCE_NZ: int = 70
BUNDLE_NAME_TEMPLATE = "system_i_grid7_nz{nz}_calib1"


# ---------------------------------------------------------------------------
# Bundle loading + z-grid reconstruction
# ---------------------------------------------------------------------------

def _bundle_path(bundles_root: Path, nz: int) -> Path:
    return bundles_root / BUNDLE_NAME_TEMPLATE.format(nz=nz)


def _read_disc_config_from_diagnostics(bundle: Path) -> dict[str, Any]:
    """Read disc_config from diagnostics.pkl when metadata.json is absent.

    The ablation bundles only ship policy_arrays.npz + diagnostics.pkl, so
    the run-time config has to come from diagnostics. `disc_config` and
    `solver_config` are stored verbatim there as plain dicts.
    """
    diag_path = bundle / "diagnostics.pkl"
    if not diag_path.exists():
        raise FileNotFoundError(f"No diagnostics.pkl in {bundle}; cannot infer disc_config")
    with diag_path.open("rb") as f:
        diag = pickle.load(f)
    if "disc_config" not in diag:
        raise KeyError(
            f"diagnostics.pkl in {bundle} has no 'disc_config' key — cannot rebuild z-grid"
        )
    return dict(diag["disc_config"])


def _reconstruct_z_grid(disc: dict[str, Any], nz_expected: int) -> np.ndarray:
    """Rebuild the z_grid the bundle was solved on.

    Mirrors `lifecycle.precompute.build_precompute`: calls
    `discretize_income_ar1_mixture` with the eta-mixture parameters from
    `BASE_CONFIG` (the canonical configs._canonical.BASE_CONFIG; identical
    across the System I sweep) and `n_z`/`n_stds` from `disc_config`.
    The handoff doc tentatively suggested the eps parameters; precompute.py
    line 336–345 confirms eta is the source of truth.
    """
    n_z_meta = int(disc["n_z"])
    if n_z_meta != int(nz_expected):
        raise ValueError(
            f"Bundle metadata reports n_z={n_z_meta}, expected {nz_expected}"
        )
    z_grid, _Pi_z = discretize_income_ar1_mixture(
        rho=float(BASE_CONFIG["rho"]),
        p=float(BASE_CONFIG["pz"]),
        mu1=float(BASE_CONFIG["mu_eta1"]),
        sigma1=float(BASE_CONFIG["sigma_eta1"]),
        mu2=float(BASE_CONFIG["mu_eta2"]),
        sigma2=float(BASE_CONFIG["sigma_eta2"]),
        N=n_z_meta,
        n_stds=float(disc["n_stds"]),
    )
    return np.asarray(z_grid, dtype=float)


def load_one(bundles_root: Path, nz: int) -> dict[str, Any]:
    bundle = _bundle_path(bundles_root, nz)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle}")
    C, S, B, _diag, metadata = load_policy_bundle(bundle)
    # Prefer metadata.json's run_config when present; otherwise fall back to
    # disc_config in diagnostics.pkl (these ablation bundles ship without
    # metadata.json).
    disc: dict[str, Any] | None = None
    if metadata and "run_config" in metadata:
        disc = metadata["run_config"].get("discretization_config")
    if disc is None:
        disc = _read_disc_config_from_diagnostics(bundle)
    z_grid = _reconstruct_z_grid(disc, nz_expected=nz)
    return {
        "nz": nz,
        "bundle": bundle,
        "C": np.asarray(C),
        "S": np.asarray(S),
        "B": np.asarray(B),
        "z_grid": z_grid,
        "metadata": metadata,
        "disc_config": disc,
    }


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _shape_summary(arr: np.ndarray) -> tuple[int, int, int, int]:
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D policy array, got shape {arr.shape}")
    return tuple(int(s) for s in arr.shape)  # type: ignore[return-value]


def validate_bundle(b: dict[str, Any]) -> list[str]:
    """Return a list of warning strings; empty list = clean."""
    warnings: list[str] = []
    nz = b["nz"]
    for name in ("C", "S", "B"):
        arr = b[name]
        if arr.shape[1] != nz:
            warnings.append(
                f"n_z={nz}: {name} axis-1 size {arr.shape[1]} != metadata n_z {nz}"
            )
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            warnings.append(
                f"n_z={nz}: {name} contains {n_bad} non-finite cells"
            )
    z = b["z_grid"]
    if z.size != nz:
        warnings.append(f"n_z={nz}: reconstructed z_grid has {z.size} pts (expected {nz})")
    elif nz >= 2:
        if not np.all(np.diff(z) > 0):
            warnings.append(f"n_z={nz}: z_grid is not strictly monotonic")
        if abs(z[0] + z[-1]) > 1e-8 * max(1.0, abs(z[-1])):
            warnings.append(f"n_z={nz}: z_grid not symmetric around 0 (z[0]+z[-1]={z[0]+z[-1]:.3e})")
    return warnings


# ---------------------------------------------------------------------------
# Interpolation + metrics
# ---------------------------------------------------------------------------

def interp_along_z(values: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate `values` along axis 1 from z_src to z_dst.

    `values` has shape (A, n_src, ..., ...); output has shape (A, n_dst, ..., ...).
    Cells at z_dst outside [z_src.min(), z_src.max()] are linearly extrapolated
    (np.interp clamps to endpoints — same behaviour the handoff implies). The
    actual z-grid endpoints across n_z bundles match modulo `std_z`, so the
    out-of-range case shouldn't fire for evenly-spaced grids with the same
    n_stds. This is verified by the validation gate (matching endpoints).
    """
    if values.shape[1] != z_src.size:
        raise ValueError(
            f"interp_along_z: axis-1 size {values.shape[1]} != z_src size {z_src.size}"
        )
    # np.interp handles 1D only -> use apply_along_axis. Per-cell call is fine
    # for the bundle sizes here (~78 * 7 * 180 = ~98k cells per array, 4 arrays).
    return np.apply_along_axis(
        lambda v: np.interp(z_dst, z_src, v), axis=1, arr=values
    )


def _abs_delta_metrics(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Common scalar summaries for an element-wise |coarse - ref| array."""
    sup = float(np.max(delta))
    p99 = float(np.percentile(delta, 99))
    p95 = float(np.percentile(delta, 95))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    mean_abs = float(np.mean(delta))
    # Relative versions where ref is meaningfully non-zero.
    ref_abs = np.abs(ref)
    threshold = max(1e-8, 1e-3 * float(np.median(ref_abs[ref_abs > 0]) if np.any(ref_abs > 0) else 0.0))
    mask = ref_abs > threshold
    if mask.any():
        rel = delta[mask] / ref_abs[mask]
        sup_rel = float(np.max(rel))
        p99_rel = float(np.percentile(rel, 99))
    else:
        sup_rel = float("nan")
        p99_rel = float("nan")
    return {
        "sup": sup,
        "p99": p99,
        "p95": p95,
        "rms": rms,
        "mean_abs": mean_abs,
        "sup_rel": sup_rel,
        "p99_rel": p99_rel,
        "rel_threshold": float(threshold),
    }


def _per_axis_max(delta: np.ndarray) -> dict[str, list[float]]:
    """Max |coarse - ref| reduced to one number per axis (age, z, state, wealth)."""
    return {
        "per_age": [float(x) for x in np.max(delta, axis=(1, 2, 3))],
        "per_z": [float(x) for x in np.max(delta, axis=(0, 2, 3))],
        "per_state": [float(x) for x in np.max(delta, axis=(0, 1, 3))],
        "per_wealth": [float(x) for x in np.max(delta, axis=(0, 1, 2))],
    }


def compute_pair_metrics(
    coarse: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    z_ref = reference["z_grid"]
    z_src = coarse["z_grid"]
    metrics: dict[str, Any] = {
        "nz_coarse": int(coarse["nz"]),
        "nz_reference": int(reference["nz"]),
        "shape": list(reference["C"].shape),
    }
    interp_arrays: dict[str, np.ndarray] = {}
    for name in ("C", "S", "B"):
        coarse_proj = interp_along_z(coarse[name], z_src, z_ref)
        ref_arr = reference[name]
        if coarse_proj.shape != ref_arr.shape:
            raise ValueError(
                f"Interpolated shape {coarse_proj.shape} != reference {ref_arr.shape} for {name}"
            )
        interp_arrays[name] = coarse_proj
        delta = np.abs(coarse_proj - ref_arr)
        scalar = _abs_delta_metrics(delta, ref_arr)
        per_axis = _per_axis_max(delta)
        metrics[name] = {**scalar, **per_axis}
    metrics["_interp"] = interp_arrays  # not serialized; kept for plot script
    return metrics


def self_compare_sanity(reference: dict[str, Any]) -> dict[str, float]:
    """sup-norm of reference-vs-itself after passing through interp_along_z.

    Should be exactly 0 modulo float-rounding; if not, the interpolation
    framework has a bug. Independent of bundle correctness.
    """
    z = reference["z_grid"]
    sups: dict[str, float] = {}
    for name in ("C", "S", "B"):
        arr = reference[name]
        proj = interp_along_z(arr, z, z)
        sups[name] = float(np.max(np.abs(proj - arr)))
    return sups


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles-root", type=Path,
        default=REPO / "saved_runs" / "ablations",
        help="Directory containing system_i_grid7_nz<N>_calib1 bundles",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "docs" / "scans",
        help="Where to write metrics JSON",
    )
    parser.add_argument(
        "--metrics-name", type=str,
        default="system_i_nz_convergence_metrics.json",
        help="Filename for the metrics JSON (under output-dir).",
    )
    parser.add_argument(
        "--save-interp-bundle", type=Path, default=None,
        help="Optional .npz path; if set, save the interpolated coarse policies "
             "and the reference for downstream plotting.",
    )
    args = parser.parse_args(argv)

    bundles_root: Path = args.bundles_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bundles from: {bundles_root}")
    bundles: dict[int, dict[str, Any]] = {}
    for nz in NZ_VALUES:
        bundles[nz] = load_one(bundles_root, nz)
        sh = _shape_summary(bundles[nz]["C"])
        print(f"  n_z={nz:>3}: shape={sh}, "
              f"z_range=[{bundles[nz]['z_grid'][0]:+.4f}, {bundles[nz]['z_grid'][-1]:+.4f}]")

    print("\nValidation gates:")
    any_warnings = False
    for nz, b in bundles.items():
        ws = validate_bundle(b)
        if ws:
            any_warnings = True
            for w in ws:
                print(f"  WARN {w}")
    if not any_warnings:
        print("  all clean.")

    # Sanity check the interpolation framework on the reference bundle.
    ref = bundles[REFERENCE_NZ]
    self_sups = self_compare_sanity(ref)
    print(f"\nSelf-compare (ref vs itself, must be ~0):")
    for k, v in self_sups.items():
        print(f"  sup_{k} = {v:.3e}")
        if v > 1e-12:
            print(f"  WARN: self-compare sup_{k} = {v:.3e} > 1e-12 — interp has bug")

    # Pairwise metrics: coarse n_z vs reference (n_z=70).
    print(f"\nDivergence vs n_z={REFERENCE_NZ}:")
    pairs: dict[int, dict[str, Any]] = {}
    interp_cache: dict[int, dict[str, np.ndarray]] = {}
    for nz in NZ_VALUES:
        if nz == REFERENCE_NZ:
            continue
        m = compute_pair_metrics(bundles[nz], ref)
        interp_cache[nz] = m.pop("_interp")
        pairs[nz] = m
        print(
            f"  n_z={nz:>3} | "
            f"sup_C={m['C']['sup']:.4e} sup_S={m['S']['sup']:.4e} sup_B={m['B']['sup']:.4e} | "
            f"rms_C={m['C']['rms']:.3e} | "
            f"rel_sup_C={m['C']['sup_rel']:.3e}"
        )

    # Distribution snapshots (light-weight; histograms saved at plot time).
    distribution = {}
    for nz, b in bundles.items():
        distribution[str(nz)] = {
            "S_quantiles": [float(q) for q in np.quantile(b["S"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "B_quantiles": [float(q) for q in np.quantile(b["B"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "C_quantiles": [float(q) for q in np.quantile(b["C"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "S_min": float(b["S"].min()),
            "S_max": float(b["S"].max()),
            "B_min": float(b["B"].min()),
            "B_max": float(b["B"].max()),
            "C_min": float(b["C"].min()),
            "C_max": float(b["C"].max()),
        }

    out_payload = {
        "bundles_root": str(bundles_root),
        "nz_values": list(NZ_VALUES),
        "reference_nz": REFERENCE_NZ,
        "reference_shape": list(ref["C"].shape),
        "self_sup": self_sups,
        "pairs": {str(nz): pairs[nz] for nz in pairs},
        "z_grids": {str(nz): bundles[nz]["z_grid"].tolist() for nz in NZ_VALUES},
        "distributions": distribution,
    }
    metrics_path = output_dir / args.metrics_name
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote metrics to {metrics_path}")

    if args.save_interp_bundle is not None:
        save_kwargs: dict[str, np.ndarray] = {}
        for name in ("C", "S", "B"):
            save_kwargs[f"ref_{name}"] = ref[name]
        save_kwargs["ref_z"] = ref["z_grid"]
        for nz in NZ_VALUES:
            if nz == REFERENCE_NZ:
                continue
            for name in ("C", "S", "B"):
                save_kwargs[f"nz{nz}_{name}_interp"] = interp_cache[nz][name]
                save_kwargs[f"nz{nz}_{name}_raw"] = bundles[nz][name]
            save_kwargs[f"nz{nz}_z"] = bundles[nz]["z_grid"]
        args.save_interp_bundle.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_interp_bundle, **save_kwargs)
        print(f"Wrote interp bundle to {args.save_interp_bundle}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
