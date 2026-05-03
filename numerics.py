"""Backward-compat shim — the implementation lives in lifecycle.numerics.

This shim aliases `sys.modules['numerics']` to the lifecycle.numerics module so that:
  * `from numerics import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `numerics.ClassName` continue to
    unpickle correctly (`numerics.ClassName` resolves to
    `lifecycle.numerics.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `numerics` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import numerics as _impl
sys.modules[__name__] = _impl