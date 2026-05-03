"""Backward-compat shim — the implementation lives in lifecycle.mortality.

This shim aliases `sys.modules['mortality']` to the lifecycle.mortality module so that:
  * `from mortality import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `mortality.ClassName` continue to
    unpickle correctly (`mortality.ClassName` resolves to
    `lifecycle.mortality.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `mortality` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import mortality as _impl
sys.modules[__name__] = _impl