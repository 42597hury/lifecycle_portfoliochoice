"""Backward-compat shim — the implementation lives in lifecycle.var.

This shim aliases `sys.modules['var']` to the lifecycle.var module so that:
  * `from var import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `var.ClassName` continue to
    unpickle correctly (`var.ClassName` resolves to
    `lifecycle.var.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `var` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import var as _impl
sys.modules[__name__] = _impl