"""
Thin wrapper around `verify/ee_simpath.py` that picks the right VAR builder
based on the bundle's `metadata.run_config.predictability_ablation.system_label`.

Why: `verify/ee_simpath.py` hardcodes `build_nominal_system1_var_config_hardcoded()`
(System IV — full 4-state VAR), which crashes when the bundle was solved with
an iid System I VAR (state vector = (rtb,), N_state=7). This wrapper inspects
the bundle's metadata, swaps in the matching builder, then delegates to
`ee_simpath.main`. Pure Python monkey-patch — no edit to ee_simpath.

Usage (positional args identical to ee_simpath):
    python scripts/analysis/run_ee_simpath_system_i.py <bundle> [--eval-mode same] ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lifecycle.var import (  # noqa: E402
    build_iid_var_config,
    build_no_cy_var_config,
    build_nominal_system1_var_config_hardcoded,
    build_rtb_y1_var_config,
)


_BUILDER_BY_LABEL = {
    "system_i_iid": build_iid_var_config,
    "system_ii_rtb_y1": build_rtb_y1_var_config,
    "system_iii_rtb_spr_y1": build_no_cy_var_config,
    "system_iv_full_var": build_nominal_system1_var_config_hardcoded,
}

_CSV_PATH = "data/var_dataset.csv"


def _resolve_bundle(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    p2 = REPO / "saved_runs" / arg
    if p2.is_dir():
        return p2
    raise FileNotFoundError(f"Bundle not found: {arg}")


def _read_system_label(bundle_path: Path) -> str:
    meta_path = bundle_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    pa = meta.get("run_config", {}).get("predictability_ablation", {}) or {}
    label = pa.get("system_label", "system_iv_full_var")
    return str(label)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: run_ee_simpath_system_i.py <bundle> [ee_simpath flags]")
        return 2

    bundle_arg = argv[0]
    bundle_path = _resolve_bundle(bundle_arg)
    label = _read_system_label(bundle_path)
    builder = _BUILDER_BY_LABEL.get(label)
    if builder is None:
        raise ValueError(f"Unknown system_label {label!r}")
    print(f"[wrapper] Bundle system_label = {label!r} -> {builder.__name__}", flush=True)

    if builder is build_nominal_system1_var_config_hardcoded:
        # Builder takes no args; nothing to patch.
        patched = builder
    else:
        # Wrap so ee_simpath can call with no args.
        def _patched():
            res = builder(csv_path=_CSV_PATH)
            # Builders other than the hardcoded one return (var_config, fit, data);
            # ee_simpath only consumes the var_config dict.
            return res[0] if isinstance(res, tuple) else res
        patched = _patched

    import verify.ee_simpath as ee  # imports JAX, sets up cache — slow first call
    ee.build_nominal_system1_var_config_hardcoded = patched

    # Re-route argv so ee_simpath's argparse sees the same args.
    sys.argv = ["verify/ee_simpath.py", *argv]
    return ee.main()


if __name__ == "__main__":
    raise SystemExit(main())
