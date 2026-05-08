"""Wealth-grid helpers shared by solvers, bundles, and diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WEALTH_GRID_FILE = "wealth_grid.npy"


def legacy_log1p_wealth_grid(
    n_wealth: int,
    wealth_min: float,
    wealth_max: float,
) -> np.ndarray:
    """Return the historical log1p-uniform wealth grid exactly."""
    return np.expm1(
        np.linspace(
            np.log1p(float(wealth_min)),
            np.log1p(float(wealth_max)),
            int(n_wealth),
        )
    )


def wealth_grid_hash(wealth_grid: np.ndarray) -> str:
    """Hash the normalized numeric wealth grid, not file bytes or strings."""
    arr = np.ascontiguousarray(wealth_grid, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def wealth_grid_spacing_stats(wealth_grid: np.ndarray) -> dict[str, Any]:
    """Return float64/float32 spacing diagnostics for a wealth grid."""
    w64 = np.ascontiguousarray(wealth_grid, dtype=np.float64)
    w32 = w64.astype(np.float32)
    diff64 = np.diff(w64)
    diff32 = np.diff(w32).astype(np.float64)
    denom = np.maximum(np.abs(w64[:-1]), 1e-12)
    rel32 = diff32 / denom
    return {
        "min_diff64": float(np.min(diff64)) if diff64.size else float("nan"),
        "min_diff32": float(np.min(diff32)) if diff32.size else float("nan"),
        "min_rel_diff32": float(np.min(rel32)) if rel32.size else float("nan"),
        "n_nonpositive_diff32": int(np.sum(diff32 <= 0.0)),
    }


def load_wealth_grid_file(path: str | Path) -> np.ndarray:
    """Load a wealth grid from .npy or .npz and normalize to contiguous f64."""
    grid_path = Path(path)
    if not grid_path.exists():
        raise FileNotFoundError(f"Wealth grid file not found: {grid_path}")
    if grid_path.suffix.lower() == ".npy":
        arr = np.load(grid_path, allow_pickle=False)
    elif grid_path.suffix.lower() == ".npz":
        with np.load(grid_path, allow_pickle=False) as data:
            if "wealth_grid" in data.files:
                arr = data["wealth_grid"]
            elif len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                raise ValueError(
                    f"{grid_path} contains multiple arrays but no 'wealth_grid' key."
                )
    else:
        raise ValueError(
            f"Unsupported wealth grid suffix {grid_path.suffix!r}; use .npy or .npz."
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def validate_wealth_grid(
    wealth_grid: np.ndarray,
    *,
    n_wealth: int,
    wealth_min: float,
    wealth_max: float,
    source: str = "wealth grid",
    endpoint_atol: float | None = None,
    abs_spacing_floor: float = 1e-8,
    rel_spacing_floor32: float = 8.0 * np.finfo(np.float32).eps,
) -> dict[str, Any]:
    """Validate solver safety requirements for a predefined wealth grid.

    Returns the spacing diagnostics so callers can report them alongside the
    grid hash.
    """
    w = np.asarray(wealth_grid, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"{source} must be one-dimensional; got shape {w.shape}.")
    if w.size != int(n_wealth):
        raise ValueError(
            f"{source} size {w.size} does not match n_wealth={int(n_wealth)}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{source} contains non-finite values.")

    diff64 = np.diff(w)
    if np.any(diff64 <= 0.0):
        bad = int(np.where(diff64 <= 0.0)[0][0])
        raise ValueError(
            f"{source} must be strictly increasing in float64; "
            f"first non-increasing gap is at {bad}->{bad + 1}."
        )

    scale = max(1.0, abs(float(wealth_min)), abs(float(wealth_max)))
    atol = max(1e-10, 1e-12 * scale) if endpoint_atol is None else float(endpoint_atol)
    if not np.isclose(w[0], float(wealth_min), rtol=0.0, atol=atol):
        raise ValueError(
            f"{source} lower endpoint {w[0]:.17g} does not match "
            f"wealth_min={float(wealth_min):.17g} within atol={atol:.3g}."
        )
    if not np.isclose(w[-1], float(wealth_max), rtol=0.0, atol=atol):
        raise ValueError(
            f"{source} upper endpoint {w[-1]:.17g} does not match "
            f"wealth_max={float(wealth_max):.17g} within atol={atol:.3g}."
        )

    stats = wealth_grid_spacing_stats(w)
    if stats["n_nonpositive_diff32"]:
        raise ValueError(
            f"{source} is not strictly increasing after float32 casting "
            f"({stats['n_nonpositive_diff32']} non-positive gaps)."
        )
    if stats["min_diff64"] <= abs_spacing_floor:
        raise ValueError(
            f"{source} has min float64 spacing {stats['min_diff64']:.3e}, "
            f"below floor {abs_spacing_floor:.3e}."
        )
    if stats["min_rel_diff32"] <= rel_spacing_floor32:
        raise ValueError(
            f"{source} has min relative float32 spacing "
            f"{stats['min_rel_diff32']:.3e}, below floor "
            f"{rel_spacing_floor32:.3e}."
        )
    return stats


def _run_config_disc(metadata: dict[str, Any]) -> dict[str, Any] | None:
    run_config = metadata.get("run_config", {}) or {}
    disc = run_config.get("discretization_config")
    return disc if isinstance(disc, dict) else None


def _metadata_from_bundle(bundle_dir: Path) -> dict[str, Any]:
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bundle_wealth_grid_path(
    bundle_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Return the saved grid-file path if the bundle has one."""
    bundle = Path(bundle_dir)
    metadata = _metadata_from_bundle(bundle) if metadata is None else metadata
    file_name = metadata.get("wealth_grid_file")
    candidates: list[Path] = []
    if isinstance(file_name, str) and file_name:
        candidates.append(bundle / file_name)
    elif not metadata:
        candidates.append(bundle / DEFAULT_WEALTH_GRID_FILE)
    for path in candidates:
        if path.exists():
            return path
    return None


def metadata_implies_custom_grid(metadata: dict[str, Any]) -> bool:
    """Whether metadata says an exact saved grid is required."""
    if metadata.get("wealth_grid_kind") == "custom":
        return True
    disc = _run_config_disc(metadata)
    return bool(disc and disc.get("wealth_grid_path") is not None)


def load_wealth_grid_from_bundle(
    bundle_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    allow_legacy_log1p: bool = True,
) -> np.ndarray:
    """Load the authoritative wealth grid for a policy bundle.

    New bundles should contain ``wealth_grid.npy``. Legacy log1p bundles can be
    reconstructed exactly from their saved discretization config; custom
    bundles without a saved grid file are refused.
    """
    bundle = Path(bundle_dir)
    metadata = _metadata_from_bundle(bundle) if metadata is None else metadata

    grid_path = bundle_wealth_grid_path(bundle, metadata)
    if grid_path is not None:
        grid = load_wealth_grid_file(grid_path)
        meta_hash = metadata.get("wealth_grid_hash")
        if meta_hash is not None and str(meta_hash) != wealth_grid_hash(grid):
            raise ValueError(
                f"Wealth grid hash mismatch in {bundle}: metadata has "
                f"{meta_hash}, file hashes to {wealth_grid_hash(grid)}."
            )
        return grid

    if metadata_implies_custom_grid(metadata):
        raise FileNotFoundError(
            f"{bundle} uses a custom wealth grid but has no {DEFAULT_WEALTH_GRID_FILE}."
        )
    if not allow_legacy_log1p:
        raise FileNotFoundError(f"{bundle} has no saved {DEFAULT_WEALTH_GRID_FILE}.")

    disc = _run_config_disc(metadata)
    if disc is None:
        raise FileNotFoundError(
            f"{bundle} has no saved wealth grid and no run_config.discretization_config "
            "from which to reconstruct the legacy log1p grid."
        )
    grid = legacy_log1p_wealth_grid(
        int(disc["n_wealth"]), float(disc["wealth_min"]), float(disc["wealth_max"])
    )
    meta_hash = metadata.get("wealth_grid_hash")
    if meta_hash is not None and str(meta_hash) != wealth_grid_hash(grid):
        raise ValueError(
            f"Legacy reconstructed wealth grid hash mismatch in {bundle}: "
            f"metadata has {meta_hash}, reconstructed hashes to {wealth_grid_hash(grid)}."
        )
    return np.ascontiguousarray(grid, dtype=np.float64)


def disc_config_with_bundle_wealth_grid(
    disc_config: Any,
    bundle_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Point a DiscretizationConfig at a bundle's saved grid when needed."""
    bundle = Path(bundle_dir)
    metadata = _metadata_from_bundle(bundle) if metadata is None else metadata
    grid_path = bundle_wealth_grid_path(bundle, metadata)
    if grid_path is not None:
        return disc_config._replace(wealth_grid_path=str(grid_path))
    if metadata_implies_custom_grid(metadata):
        raise FileNotFoundError(
            f"{bundle} uses a custom wealth grid but has no {DEFAULT_WEALTH_GRID_FILE}."
        )
    return disc_config
