"""Backward-compat shim — the implementation lives in lifecycle.plots.

This shim aliases `sys.modules['plots']` to the lifecycle.plots module so that:
  * `from plots import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `plots.ClassName` continue to
    unpickle correctly (`plots.ClassName` resolves to
    `lifecycle.plots.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `plots` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import plots as _impl
sys.modules[__name__] = _impl