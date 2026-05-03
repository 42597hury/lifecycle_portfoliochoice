"""One-shot: re-pickle saved_runs/*/diagnostics.pkl with NamedTuples
expanded to plain dicts.

Why: legacy bundles store solver_config / disc_config as NamedTuple
instances, so the pickle qualnames are tied to specific Python modules
(`model.SolverConfig`, etc). After this migration the pickle contains
only built-in types + numpy arrays, so the root-level shim modules
(`model.py` etc.) can be deleted without breaking bundle loads.

Run from the repo root:
    python scripts/migrate_bundle_pickles.py

Idempotent: bundles already in dict form are skipped.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `model` (shim) and `lifecycle` resolve

from lifecycle.policy_io import _dictify_namedtuples


def needs_migration(diagnostics) -> bool:
    """True if any NamedTuple instance lives anywhere in the diagnostics tree."""
    if hasattr(diagnostics, "_asdict") and hasattr(diagnostics, "_fields"):
        return True
    if isinstance(diagnostics, dict):
        return any(needs_migration(v) for v in diagnostics.values())
    if isinstance(diagnostics, (list, tuple)):
        return any(needs_migration(v) for v in diagnostics)
    return False


def migrate_bundle(diag_path: Path) -> str:
    try:
        with diag_path.open("rb") as f:
            diagnostics = pickle.load(f)
    except Exception as exc:
        return f"SKIP (load failed: {exc})"

    if not needs_migration(diagnostics):
        return "skip (already dict)"

    new = _dictify_namedtuples(diagnostics)

    backup = diag_path.with_suffix(".pkl.pre_dictify_bak")
    diag_path.replace(backup)
    try:
        with diag_path.open("wb") as f:
            pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        backup.replace(diag_path)
        raise
    return "migrated"


def main() -> None:
    bundles = sorted((ROOT / "saved_runs").glob("*/diagnostics.pkl"))
    print(f"Scanning {len(bundles)} bundle(s) under saved_runs/")
    counts = {"migrated": 0, "skip (already dict)": 0, "skipped/error": 0}
    for diag_path in bundles:
        status = migrate_bundle(diag_path)
        print(f"  {diag_path.parent.name}: {status}")
        if status == "migrated":
            counts["migrated"] += 1
        elif status.startswith("skip "):
            counts["skip (already dict)"] += 1
        else:
            counts["skipped/error"] += 1
    print()
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if counts["migrated"]:
        print(
            "\nBackups written as <bundle>/diagnostics.pkl.pre_dictify_bak. "
            "Delete them once you've verified loads work."
        )


if __name__ == "__main__":
    main()
