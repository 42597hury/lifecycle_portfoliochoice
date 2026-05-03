"""Backward-compat shim — the implementation lives in lifecycle.predictability_ablation.

This shim aliases `sys.modules['predictability_ablation']` to the lifecycle.predictability_ablation module so that:
  * `from predictability_ablation import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `predictability_ablation.ClassName` continue to
    unpickle correctly (`predictability_ablation.ClassName` resolves to
    `lifecycle.predictability_ablation.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `predictability_ablation` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import predictability_ablation as _impl
sys.modules[__name__] = _impl