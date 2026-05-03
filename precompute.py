"""Backward-compat shim — the implementation lives in lifecycle.precompute.

This shim aliases `sys.modules['precompute']` to the lifecycle.precompute module so that:
  * `from precompute import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `precompute.ClassName` continue to
    unpickle correctly (`precompute.ClassName` resolves to
    `lifecycle.precompute.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `precompute` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import precompute as _impl
sys.modules[__name__] = _impl