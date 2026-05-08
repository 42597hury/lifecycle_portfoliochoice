"""Shared bundle-rehydration helpers for verify/* diagnostics.

After the 2026-05-08 real-yields pivot, the only VAR builders are the three
real-yields ones (System 1 / System 2 / Full). Legacy bundles solved against
the 4-axis nominal model (System I/II/III/IV with rtb-as-state) cannot be
reloaded on this code path: the precompute and FOC kernels are 3-axis now,
and the old hardcoded VAR builders have been removed from lifecycle.var.

These helpers centralize the reload-time policy:
  * If the bundle's metadata declares a NEW system code (1/2/full), pick the
    corresponding builder and rebuild the precompute.
  * Otherwise (legacy nominal codes I/II/III/IV with state names from
    {dp, cy, spr, rtb}), raise a clear error pointing the user at the
    pivot doc and at re-solving against the new model.
"""
from __future__ import annotations

from typing import Any


# Legacy state-name signals used by pre-pivot bundles. If we see any of these
# axis names in the bundle's predictability_ablation block we know we're
# looking at a 4-axis nominal bundle that the new code cannot rehydrate.
_LEGACY_STATE_NAMES = frozenset({"dp", "cy", "rtb"})

# Legacy system codes (pre-pivot Roman numerals). The current
# predictability_ablation module rejects these explicitly via
# _RETIRED_SYSTEM_CODES; we mirror that list here so we can produce a
# tailored message before the call ever reaches lifecycle.
_LEGACY_SYSTEM_CODES = frozenset({
    "I", "II", "III", "IV",
    "i", "ii", "iii", "iv",
    "system_i", "system_ii", "system_iii", "system_iv",
    "system_i_iid", "system_ii_no_dp", "system_iii_no_cy",
    "system_iv_full_var",
})


def _legacy_bundle_message(bundle_path: Any, marker: str) -> str:
    return (
        f"Bundle {bundle_path} was solved against the pre-pivot 4-axis nominal "
        f"model (state vector (dp/cy, spr, rtb, y_1)); marker = {marker!r}.\n"
        "The 2026-05-08 real-yields pivot replaced that model with a 3-axis "
        "real-yields model on (cape, spr, y_1). The old VAR builders and "
        "rtb-as-state precompute path have been removed from lifecycle/.\n"
        "Diagnostic scripts on this branch cannot rebuild the precompute that "
        "matches a legacy bundle's policy shape, so reload is rejected.\n"
        "Options:\n"
        "  * For retrospective analysis, use the pre-pivot revision of this "
        "    repo (commit 73dee67~ or earlier) to load the bundle.\n"
        "  * For forward analysis, re-solve under the new 3-axis canonical "
        "    via verify/benchmark_bundle.py and the corresponding new bundle.\n"
        "See docs/handoff/HANDOFF_REAL_YIELDS_PIVOT.md for details."
    )


def build_bundle_var_config(metadata: dict, bundle_path: Any = "<bundle>") -> dict:
    """Pick the correct real-yields VAR builder for a bundle's saved metadata.

    Reads metadata.run_config.predictability_ablation.system_code (preferred)
    or falls back to system_label / state_names heuristics. Raises a clear
    error for legacy nominal bundles (System I..IV) that cannot be rehydrated
    against the post-pivot 3-axis precompute.

    The returned dict is the same shape as build_real_full_var_config(...)[0]:
    it can be passed to lifecycle.precompute.build_model.
    """
    from lifecycle.var import (
        build_real_full_var_config_hardcoded,
        build_real_system1_var_config,
        build_real_system2_var_config,
    )

    run_config = (metadata or {}).get("run_config") or {}
    pa = run_config.get("predictability_ablation") or {}
    system_code = pa.get("system_code")
    system_label = pa.get("system_label", "")
    state_names = pa.get("state_names") or ()

    # Legacy detection — anything that looks like the old 4-axis nominal model.
    if (
        system_code in _LEGACY_SYSTEM_CODES
        or any(name in _LEGACY_STATE_NAMES for name in state_names)
        or any(s in (system_label or "") for s in ("system_i", "system_ii", "system_iii", "system_iv"))
    ):
        marker = system_code or system_label or repr(state_names)
        raise RuntimeError(_legacy_bundle_message(bundle_path, marker))

    # No predictability metadata at all: assume the headline Full real system.
    # This is what the pre-ablation legacy bundles look like at the bundle
    # root level; new bundles should always carry system_code so this path
    # is just a graceful fallback.
    if system_code is None and not state_names:
        return build_real_full_var_config_hardcoded()

    # Map system_code -> builder. State_names cross-checked when available.
    if system_code in ("1", 1, "system_1", "system_1_real"):
        var_config, _, _ = build_real_system1_var_config()
        return var_config
    if system_code in ("2", 2, "system_2", "system_2_real"):
        var_config, _, _ = build_real_system2_var_config()
        return var_config
    if system_code in ("full", "system_full", "full_system", "full_system_real"):
        return build_real_full_var_config_hardcoded()

    # Something we don't recognise. Surface it cleanly.
    raise RuntimeError(
        f"Bundle {bundle_path} has unrecognised predictability_ablation "
        f"metadata: system_code={system_code!r}, system_label={system_label!r}, "
        f"state_names={tuple(state_names)!r}. Update verify/_diag_helpers.py "
        "to dispatch this case."
    )
