"""Backward-compat shim — the implementation lives in lifecycle.diagnostics.

This shim aliases `sys.modules['diagnostics']` to the lifecycle.diagnostics module so that:
  * `from diagnostics import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `diagnostics.ClassName` continue to
    unpickle correctly (`diagnostics.ClassName` resolves to
    `lifecycle.diagnostics.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `diagnostics` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import diagnostics as _impl
sys.modules[__name__] = _impl