"""Backward-compat shim — the implementation lives in lifecycle.solver.

This shim aliases `sys.modules['solver']` to the lifecycle.solver module so that:
  * `from solver import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `solver.ClassName` continue to
    unpickle correctly (`solver.ClassName` resolves to
    `lifecycle.solver.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `solver` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import solver as _impl
sys.modules[__name__] = _impl