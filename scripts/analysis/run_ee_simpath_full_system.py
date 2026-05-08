"""Thin wrapper around `verify/ee_simpath.py` for the Full System bundles.

The Full System is the three-axis real-yields headline: state = (cape, spr,
y_1). cape (= -log Shiller CAPE) drives equity predictability, the bond
predictor uses (y_1, spr), and the bill is state-determined (R_bill_{t+1} =
y_1_t).

`verify/ee_simpath.py` auto-detects the right VAR builder from the bundle's
saved metadata via `verify/_diag_helpers.build_bundle_var_config`. This
wrapper just guards against a misrouted bundle and forwards CLI args.

Usage (positional + flag args identical to `verify/ee_simpath.py`):

    python scripts/analysis/run_ee_simpath_full_system.py <bundle> \\
        [--eval-mode {same,next_finer,double}] [--n-simulations 5000] ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


_EXPECTED_CODES = ("full", "system_full", "full_system", "full_system_real")


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
        print("Usage: run_ee_simpath_full_system.py <bundle> [ee_simpath flags]")
        return 2

    bundle_arg = argv[0]
    bundle_path = _resolve_bundle(bundle_arg)
    code = _read_system_code(bundle_path)
    if code not in _EXPECTED_CODES:
        raise ValueError(
            f"Bundle {bundle_path} has system_code={code!r}, expected one of "
            f"{_EXPECTED_CODES}. This wrapper is the Full System dispatcher; "
            "pick scripts/analysis/run_ee_simpath_system_1.py for System 1 or "
            "scripts/analysis/run_ee_simpath_system_2.py for System 2."
        )
    print(
        f"[wrapper] Full System (cape + spr + y_1) — bundle system_code={code!r}",
        flush=True,
    )

    import verify.ee_simpath as ee  # imports JAX, slow first call
    sys.argv = ["verify/ee_simpath.py", *argv]
    return ee.main()


if __name__ == "__main__":
    raise SystemExit(main())
