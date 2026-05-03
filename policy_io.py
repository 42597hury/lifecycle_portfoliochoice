"""Backward-compat shim — the implementation lives in lifecycle.policy_io.

This shim aliases `sys.modules['policy_io']` to the lifecycle.policy_io module so that:
  * `from policy_io import X` (used throughout tests/, scripts/, notebooks)
    keeps working without any import migration.
  * Pickled bundles whose qualnames embed `policy_io.ClassName` continue to
    unpickle correctly (`policy_io.ClassName` resolves to
    `lifecycle.policy_io.ClassName`).

When import migration is complete (see Phase E plan) and no consumer
imports `policy_io` at the top level any more, this shim can be removed.
Until then, leave it in place to preserve compatibility with existing
saved_runs/ bundles and external scripts.
"""
import sys
from lifecycle import policy_io as _impl
sys.modules[__name__] = _impl