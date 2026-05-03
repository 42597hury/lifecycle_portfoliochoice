"""Backward-compat shim — the implementation lives in lifecycle.simulation.

This shim aliases `sys.modules['simulation']` to the lifecycle.simulation module so that:
  * `from simulation import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `simulation.ClassName` continue to
    unpickle correctly (`simulation.ClassName` resolves to
    `lifecycle.simulation.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `simulation` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import simulation as _impl
sys.modules[__name__] = _impl