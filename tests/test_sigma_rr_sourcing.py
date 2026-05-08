"""Regression test: precompute's eq.(10) constants must come from Sigma_rr.

Locked-in invariant (CCV_RETURN_IMPLEMENT.md §3.1.d):
  precompute.sigma2_xr  == model.Sigma_rr[xr, xr]
  precompute.sigma2_xb  == model.Sigma_rr[xb, xb]
  precompute.sigma_xrxb == model.Sigma_rr[xr, xb]

This is the May-2026 patch. Pre-patch, the constants were sourced from
Sigma_r_cond (the inner-quadrature Cholesky residual), which differs from
Sigma_rr by the M·Sigma_ss·M' projection — 10x to 90x in this calibration —
and yields wrong eq.(10) constants. This test guards against any future
"fix" that reverts the source.

The check is intentionally narrow: it greps the source for the literal
``model.Sigma_rr[`` pattern in the precompute eq.(10)-constant block, since
the constants are downstream private state we don't want to depend on at
import time.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PRECOMPUTE = REPO / "lifecycle" / "precompute.py"


def test_eq10_constants_sourced_from_sigma_rr():
    """The three scalars must be assigned from model.Sigma_rr, not Sigma_r_cond."""
    src = PRECOMPUTE.read_text(encoding="utf-8")
    # Match the assignment block; allow whitespace and float() wrapping.
    pat_xr = re.compile(r"sigma2_xr\s*=\s*float\(\s*model\.Sigma_rr\[")
    pat_xb = re.compile(r"sigma2_xb\s*=\s*float\(\s*model\.Sigma_rr\[")
    pat_xrxb = re.compile(r"sigma_xrxb\s*=\s*float\(\s*model\.Sigma_rr\[")
    assert pat_xr.search(src), "precompute.sigma2_xr must be sourced from model.Sigma_rr (not Sigma_r_cond)"
    assert pat_xb.search(src), "precompute.sigma2_xb must be sourced from model.Sigma_rr (not Sigma_r_cond)"
    assert pat_xrxb.search(src), "precompute.sigma_xrxb must be sourced from model.Sigma_rr (not Sigma_r_cond)"
    # Forbid the wrong source.
    forbidden_patterns = [
        re.compile(r"sigma2_xr\s*=\s*float\(\s*model\.Sigma_r_cond\["),
        re.compile(r"sigma2_xb\s*=\s*float\(\s*model\.Sigma_r_cond\["),
        re.compile(r"sigma_xrxb\s*=\s*float\(\s*model\.Sigma_r_cond\["),
    ]
    for pat in forbidden_patterns:
        assert not pat.search(src), (
            f"precompute must not source eq.(10) constants from Sigma_r_cond; "
            f"see CCV_RETURN_IMPLEMENT.md §3.1.d for why"
        )


def test_sigma_rr_differs_from_sigma_r_cond_at_runtime():
    """Numerical evidence the patch matters: the two matrices differ on production calibration.

    If they were equal, this test would still pass logically but the regression
    test above would have nothing to guard. This sanity check confirms the
    distinction is non-trivial under the current data.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from lifecycle.var import build_real_full_var_config_hardcoded
    from lifecycle.var import partition_var

    cfg = build_real_full_var_config_hardcoded()
    parts = partition_var(
        cfg["Phi"],
        cfg["Omega"],
        cfg["z_bar"],
        cfg["state_indices"],
        cfg["return_indices"],
        variable_names=cfg["variable_names"],
        verbose=False,
    )
    Sigma_rr = parts["Sigma_rr"]
    Sigma_r_cond = parts["Sigma_r_cond"]
    # They should differ materially. On the Option A (1920-2011) calibration,
    # the diagonal ratio is ~3-30x (xr) and ~13x (xb).
    diag_ratio_xr = Sigma_rr[0, 0] / Sigma_r_cond[0, 0]
    diag_ratio_xb = Sigma_rr[1, 1] / Sigma_r_cond[1, 1]
    assert diag_ratio_xr > 1.5, (
        f"Sigma_rr[xr, xr]/Sigma_r_cond[xr, xr] = {diag_ratio_xr:.3f} -- too close; "
        "regression test in test_eq10_constants_sourced_from_sigma_rr would be vacuous"
    )
    assert diag_ratio_xb > 1.5, (
        f"Sigma_rr[xb, xb]/Sigma_r_cond[xb, xb] = {diag_ratio_xb:.3f} -- too close"
    )
