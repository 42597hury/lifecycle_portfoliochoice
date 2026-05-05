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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _dictify_namedtuples(value: Any) -> Any:
    """Recursively replace NamedTuple instances with plain dicts.

    Why: bundles are pickled, and a NamedTuple pickles its class identity
    (module + qualname). Storing the data as plain dicts removes any
    dependency on Python module structure, so bundles unpickle without
    needing the lifecycle.* (or legacy `model`) modules to be importable.
    """
    if hasattr(value, "_asdict") and hasattr(value, "_fields"):
        return {k: _dictify_namedtuples(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {k: _dictify_namedtuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dictify_namedtuples(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_dictify_namedtuples(v) for v in value)
    return value


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
            pickle.dump(
                _dictify_namedtuples(diagnostics), f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    # Wealth-dynamics spec — drawn from run_config.solver_config if present;
    # otherwise tagged "simple_clamp" (the legacy default). Storing this at
    # the top of the metadata makes it cheap to verify at load time that the
    # simulator/diagnostics are run under the same spec the solver used.
    spec = "simple_clamp"
    if run_config is not None:
        sc = run_config.get("solver_config") if isinstance(run_config, dict) else None
        if isinstance(sc, dict):
            spec = sc.get("wealth_dynamics_spec", spec)
        elif sc is not None and hasattr(sc, "wealth_dynamics_spec"):
            spec = getattr(sc, "wealth_dynamics_spec", spec)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "array_file": arrays_path.name,
        "diagnostics_file": diag_path.name if diagnostics is not None else None,
        "shape": list(C_mat.shape),
        "dtype_C": str(C_mat.dtype),
        "dtype_S": str(S_mat.dtype),
        "dtype_B": str(B_mat.dtype),
        "compressed": bool(compress),
        "wealth_dynamics_spec": str(spec),
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
        try:
            with diag_path.open("rb") as f:
                diagnostics = pickle.load(f)
        except Exception as exc:
            warnings.warn(
                f"Could not load diagnostics from {diag_path.name}: {exc}. "
                "Continuing with arrays + metadata only."
            )

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    # State-ordering guard: warn if the bundle was produced under a different
    # state_indices than the current lifecycle.var default.  Prevents silent
    # column mislabelling when an old (y_1, spr, cy) bundle is loaded after the
    # 2026-04-30 reorder default to (cy, spr, y_1).
    try:
        from lifecycle.var import build_nominal_system1_var_config as _bn
        import inspect as _inspect
        _default_state_indices = list(
            _inspect.signature(_bn).parameters["state_indices"].default
        )
        _bundle_var_cfg = (
            metadata.get("run_config", {}).get("var_config", {}) if metadata else {}
        )
        _bundle_state_indices = _bundle_var_cfg.get("state_indices")
        if (
            _bundle_state_indices is not None
            and list(_bundle_state_indices) != _default_state_indices
        ):
            warnings.warn(
                f"Bundle '{bundle.name}' was solved with state_indices="
                f"{list(_bundle_state_indices)}, but the current lifecycle.var default is "
                f"{_default_state_indices}. The bundle's state_grid columns are in "
                f"the OLD order; do not mix with arrays from a freshly-built "
                f"Precompute under the current default. Re-solve to migrate."
            )
    except Exception:
        # Guard is advisory; never block a load on an inspection failure.
        pass

    return C_mat, S_mat, B_mat, diagnostics, metadata


# ---------------------------------------------------------------------------
# Simulation data save / load
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    """Convert a human-readable label to a safe filename stem."""
    safe = label.strip().lower()
    for ch in (" ", "+", "/", "\\"):
        safe = safe.replace(ch, "_")
    # collapse repeated underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def save_sim_data(
    bundle_dir: str | Path,
    sim_data: dict[str, Any],
    label: str,
    sim_config: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Save simulation output arrays and metadata into a bundle's sims/ folder.

    Parameters
    ----------
    bundle_dir
        The solver bundle directory (must already exist with policy_arrays.npz).
    sim_data
        Dict returned by ``simulate_lifecycle()``.
    label
        Human-readable name, e.g. ``"baseline"`` or ``"high dp + high yield"``.
    sim_config
        Simulation configuration snapshot (seed, n_simulations, etc.).
    overwrite
        If False, raises FileExistsError when target files exist.

    Returns
    -------
    Path
        The .npz file that was written.
    """
    bundle = Path(bundle_dir)
    if not (bundle / "policy_arrays.npz").exists():
        raise FileNotFoundError(
            f"No policy_arrays.npz in {bundle}. Save solver output first."
        )

    sims_dir = bundle / "sims"
    sims_dir.mkdir(exist_ok=True)
    # Also create figs/ directory for future use
    (bundle / "figs").mkdir(exist_ok=True)

    safe = _sanitize_label(label)
    arrays_path = sims_dir / f"{safe}.npz"
    meta_path = sims_dir / f"{safe}_meta.json"

    if not overwrite:
        existing = [p for p in (arrays_path, meta_path) if p.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing sim files: "
                + ", ".join(str(p) for p in existing)
            )

    # Separate numpy arrays from non-array entries
    array_kw = {}
    non_array = {}
    for k, v in sim_data.items():
        if isinstance(v, np.ndarray):
            array_kw[k] = v
        else:
            non_array[k] = v

    np.savez_compressed(arrays_path, **array_kw)

    metadata = {
        "label": label,
        "safe_label": safe,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle),
        "array_keys": sorted(array_kw.keys()),
        "array_shapes": {k: list(v.shape) for k, v in array_kw.items()},
        "non_array_fields": _to_jsonable(non_array),
    }
    if sim_config is not None:
        metadata["sim_config"] = _to_jsonable(sim_config)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return arrays_path


def load_sim_data(
    bundle_dir: str | Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a saved simulation from a bundle's sims/ folder.

    Returns
    -------
    tuple
        (sim_data_dict, sim_metadata)
    """
    bundle = Path(bundle_dir)
    safe = _sanitize_label(label)
    arrays_path = bundle / "sims" / f"{safe}.npz"
    meta_path = bundle / "sims" / f"{safe}_meta.json"

    if not arrays_path.exists():
        available = list_sims(bundle)
        raise FileNotFoundError(
            f"No sim '{label}' (file: {safe}.npz) in {bundle / 'sims'}. "
            f"Available sims: {available}"
        )

    sim_data: dict[str, Any] = {}
    with np.load(arrays_path, allow_pickle=False) as data:
        for k in data.files:
            sim_data[k] = data[k]

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Restore non-array fields (e.g. 'label')
    if "non_array_fields" in metadata:
        for k, v in metadata["non_array_fields"].items():
            if k not in sim_data:
                sim_data[k] = v

    return sim_data, metadata


def list_sims(bundle_dir: str | Path) -> list[str]:
    """
    List available simulation labels in a bundle's sims/ folder.

    Returns
    -------
    list[str]
        Labels derived from filenames (e.g. ``["baseline", "high_dp_high_yield"]``).
    """
    sims_dir = Path(bundle_dir) / "sims"
    if not sims_dir.exists():
        return []
    return sorted(
        p.stem for p in sims_dir.glob("*.npz")
    )
