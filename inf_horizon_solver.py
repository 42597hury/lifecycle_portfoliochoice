"""Backward-compat shim — the implementation lives in lifecycle.inf_horizon_solver.

This shim aliases `sys.modules['inf_horizon_solver']` to the lifecycle.inf_horizon_solver module so that:
  * `from inf_horizon_solver import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `inf_horizon_solver.ClassName` continue to
    unpickle correctly (`inf_horizon_solver.ClassName` resolves to
    `lifecycle.inf_horizon_solver.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `inf_horizon_solver` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import inf_horizon_solver as _impl
sys.modules[__name__] = _impl