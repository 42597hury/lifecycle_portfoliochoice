"""Backward-compat shim — the implementation lives in lifecycle.discretization.

This shim aliases `sys.modules['discretization']` to the lifecycle.discretization module so that:
  * `from discretization import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `discretization.ClassName` continue to
    unpickle correctly (`discretization.ClassName` resolves to
    `lifecycle.discretization.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `discretization` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import discretization as _impl
sys.modules[__name__] = _impl