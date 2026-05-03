"""Backward-compat shim — the implementation lives in lifecycle.model.

This shim aliases `sys.modules['model']` to the lifecycle.model module so that:
  * `from model import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `model.ClassName` continue to
    unpickle correctly (`model.ClassName` resolves to
    `lifecycle.model.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `model` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import model as _impl
sys.modules[__name__] = _impl