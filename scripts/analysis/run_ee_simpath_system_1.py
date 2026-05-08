"""Thin wrapper around `verify/ee_simpath.py` for System 1 bundles.

After the 2026-05-08 real-yields pivot, `verify/ee_simpath.py` auto-detects the
right VAR builder from `metadata.run_config.predictability_ablation.system_code`
via `verify/_diag_helpers.build_bundle_var_config`. This wrapper exists for two
reasons:

  1. **Defensive guard**: assert the bundle's `system_code` is "1" before
     running. Catches a stale CLI invocation pointed at the wrong bundle (e.g.
     a Full System bundle handed to the System 1 dispatcher).
  2. **Convention**: keeps the per-system dispatch surface visible alongside
     the per-system benchmark wrappers (verify/benchmark_system_*).

System 1 is the most reduced real-yields ablation: state = (y_1,), where y_1 is
the real bill yield (Homer-Sylla expectation). Only y_1 predicts (no spread,
no CAPE).

Usage (positional + flag args identical to `verify/ee_simpath.py`):

    python scripts/analysis/run_ee_simpath_system_1.py <bundle> \\
        [--eval-mode {same,next_finer,double}] [--n-simulations 5000] ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


_EXPECTED_CODES = ("1", 1, "system_1", "system_1_real")


def _resolve_bundle(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    p2 = REPO / "saved_runs" / arg
    if p2.is_dir():
        return p2
    raise FileNotFoundError(f"Bundle not found: {arg}")


def _read_system_code(bundle_path: Path) -> str | None:
    meta_path = bundle_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    pa = (meta.get("run_config", {}) or {}).get("predictability_ablation", {}) or {}
    code = pa.get("system_code")
    return code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: run_ee_simpath_system_1.py <bundle> [ee_simpath flags]")
        return 2

    bundle_arg = argv[0]
    bundle_path = _resolve_bundle(bundle_arg)
    code = _read_system_code(bundle_path)
    if code not in _EXPECTED_CODES:
        raise ValueError(
            f"Bundle {bundle_path} has system_code={code!r}, expected one of "
            f"{_EXPECTED_CODES}. This wrapper is the System 1 dispatcher; pick "
            "scripts/analysis/run_ee_simpath_system_2.py for System 2 or "
            "scripts/analysis/run_ee_simpath_full_system.py for the Full System."
        )
    print(
        f"[wrapper] System 1 (real bill yield only) — bundle system_code={code!r}",
        flush=True,
    )

    # ee_simpath.main() reads sys.argv via argparse; route through.
    import verify.ee_simpath as ee  # imports JAX, slow first call
    sys.argv = ["verify/ee_simpath.py", *argv]
    return ee.main()


if __name__ == "__main__":
    raise SystemExit(main())
