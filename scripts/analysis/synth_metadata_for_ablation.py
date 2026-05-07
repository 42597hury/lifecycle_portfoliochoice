"""
Synthesize metadata.json for the System I ablation bundles.

The ablation sweep ships only `policy_arrays.npz` + `diagnostics.pkl` (no
metadata.json). `verify/ee_simpath.py` requires `metadata["run_config"]` with
base_config / discretization_config / solver_config to rebuild the
precompute, so we construct a minimal-but-faithful metadata.json from:

  - `disc_config` and `solver_config` saved verbatim in diagnostics.pkl
  - `BASE_CONFIG` from configs._canonical (the calibration these runs used;
    confirmed match against the sister sweep on saved_runs/system_iv...)
  - `wealth_dynamics_spec` from the solver_config inside diagnostics.pkl

Idempotent: skips bundles that already have metadata.json.

Usage:
    python scripts/analysis/synth_metadata_for_ablation.py
        [--bundles-root saved_runs/ablations]
        [--bundle-glob 'system_i_grid7_nz*_calib1']
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from configs._canonical import BASE_CONFIG  # noqa: E402


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def synthesize_for_bundle(bundle: Path) -> Path | None:
    meta_path = bundle / "metadata.json"
    if meta_path.exists():
        return None
    diag_path = bundle / "diagnostics.pkl"
    arrays_path = bundle / "policy_arrays.npz"
    if not diag_path.exists() or not arrays_path.exists():
        print(f"  SKIP {bundle.name}: missing diagnostics.pkl or policy_arrays.npz")
        return None

    with diag_path.open("rb") as f:
        diag = pickle.load(f)
    with np.load(arrays_path, allow_pickle=False) as data:
        shape = list(data["C_mat"].shape)
        dtype_C = str(data["C_mat"].dtype)
        dtype_S = str(data["S_mat"].dtype)
        dtype_B = str(data["B_mat"].dtype)

    disc = dict(diag.get("disc_config", {}))
    solver = dict(diag.get("solver_config", {}))
    spec = solver.get("wealth_dynamics_spec", "ccv_log")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "synthetic_note": (
            "metadata.json reconstructed post-hoc from diagnostics.pkl + "
            "configs._canonical.BASE_CONFIG (System I calib1 sweep)."
        ),
        "array_file": "policy_arrays.npz",
        "diagnostics_file": "diagnostics.pkl",
        "shape": shape,
        "dtype_C": dtype_C,
        "dtype_S": dtype_S,
        "dtype_B": dtype_B,
        "compressed": True,
        "wealth_dynamics_spec": str(spec),
        "run_config": {
            "base_config": _to_jsonable(BASE_CONFIG),
            "discretization_config": _to_jsonable(disc),
            "solver_config": _to_jsonable(solver),
            "solve_control": _to_jsonable(diag.get("solve_control", {})),
            "predictability_ablation": {
                "system_label": "system_i",
                "system_title": "System I (predictability ablation, calib1)",
            },
            "bundle_name": bundle.name,
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", type=Path,
                        default=REPO / "saved_runs" / "ablations")
    parser.add_argument("--bundle-glob", type=str,
                        default="system_i_grid7_nz*_calib1")
    args = parser.parse_args(argv)

    bundles = sorted(p for p in args.bundles_root.glob(args.bundle_glob) if p.is_dir())
    if not bundles:
        print(f"No bundles matching {args.bundles_root / args.bundle_glob}")
        return 1
    print(f"Found {len(bundles)} bundle(s) under {args.bundles_root}:")
    for b in bundles:
        path = synthesize_for_bundle(b)
        if path is None:
            print(f"  - {b.name}: already has metadata.json (skipping)")
        else:
            print(f"  - {b.name}: wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
