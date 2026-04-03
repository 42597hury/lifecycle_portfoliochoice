"""
policy_io.py -- Save and load lifecycle policy-function outputs.

This module is designed for outputs from:
    C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(...)

Files written to a bundle directory:
    policy_arrays.npz   (C_mat, S_mat, B_mat)
    diagnostics.pkl     (optional, lossless Python object)
    metadata.json       (human-readable summary + shapes/dtypes)
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Convert values to JSON-friendly representations."""
    if hasattr(value, "_asdict"):
        return {k: _to_jsonable(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        arr_meta = {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        # Keep small arrays fully reproducible in metadata while avoiding
        # huge JSON payloads for large solver diagnostics.
        if value.size <= 256:
            arr_meta["values"] = value.tolist()
        return arr_meta
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def save_policy_bundle(
    output_dir: str | Path,
    C_mat: np.ndarray,
    S_mat: np.ndarray,
    B_mat: np.ndarray,
    diagnostics: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
    compress: bool = True,
) -> Path:
    """
    Save policy outputs to disk for reuse.

    Parameters
    ----------
    output_dir
        Directory to write files into.
    C_mat, S_mat, B_mat
        Policy arrays from run_lifecycle_solver.
    diagnostics
        Diagnostics dict from run_lifecycle_solver (optional).
    run_config
        Run-time configuration snapshot (optional), e.g. calibration,
        discretization, solver settings, and VAR configuration.
    overwrite
        If False, raises FileExistsError when target files exist.
    compress
        If True, use np.savez_compressed. If False, use np.savez.

    Returns
    -------
    Path
        The bundle directory path.
    """
    C_mat = np.asarray(C_mat)
    S_mat = np.asarray(S_mat)
    B_mat = np.asarray(B_mat)

    if C_mat.shape != S_mat.shape or C_mat.shape != B_mat.shape:
        raise ValueError(
            "Policy arrays must have identical shapes. "
            f"Got C={C_mat.shape}, S={S_mat.shape}, B={B_mat.shape}"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arrays_path = out / "policy_arrays.npz"
    diag_path = out / "diagnostics.pkl"
    meta_path = out / "metadata.json"

    paths = [arrays_path, meta_path]
    if diagnostics is not None:
        paths.append(diag_path)
    if not overwrite:
        existing = [p for p in paths if p.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing files: "
                + ", ".join(str(p) for p in existing)
            )

    if compress:
        np.savez_compressed(arrays_path, C_mat=C_mat, S_mat=S_mat, B_mat=B_mat)
    else:
        np.savez(arrays_path, C_mat=C_mat, S_mat=S_mat, B_mat=B_mat)

    if diagnostics is not None:
        with diag_path.open("wb") as f:
            pickle.dump(diagnostics, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "array_file": arrays_path.name,
        "diagnostics_file": diag_path.name if diagnostics is not None else None,
        "shape": list(C_mat.shape),
        "dtype_C": str(C_mat.dtype),
        "dtype_S": str(S_mat.dtype),
        "dtype_B": str(B_mat.dtype),
        "compressed": bool(compress),
    }
    if diagnostics is not None:
        metadata["diagnostics_summary"] = _to_jsonable(diagnostics)
    if run_config is not None:
        metadata["run_config"] = _to_jsonable(run_config)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out


def load_policy_bundle(
    bundle_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None, dict[str, Any]]:
    """
    Load policy arrays and diagnostics from a saved bundle.

    Returns
    -------
    tuple
        (C_mat, S_mat, B_mat, diagnostics_or_none, metadata_dict)
    """
    bundle = Path(bundle_dir)
    arrays_path = bundle / "policy_arrays.npz"
    diag_path = bundle / "diagnostics.pkl"
    meta_path = bundle / "metadata.json"

    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing policy arrays file: {arrays_path}")

    with np.load(arrays_path, allow_pickle=False) as data:
        C_mat = data["C_mat"]
        S_mat = data["S_mat"]
        B_mat = data["B_mat"]

    diagnostics = None
    if diag_path.exists():
        with diag_path.open("rb") as f:
            diagnostics = pickle.load(f)

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return C_mat, S_mat, B_mat, diagnostics, metadata
