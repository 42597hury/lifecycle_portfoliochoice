"""Regression tests for wealth-invariant FOC normalization in the EGM scan."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lifecycle.solver import EC_INTERIOR, _egm_scan_cell, terminal_foc_jac_ccv


def test_egm_scan_normalization_converges_at_high_savings():
    """CRRA terminal FOCs are homogeneous in savings.

    Before the normalization wrapper, the raw 2x2 Jacobian determinant crossed
    the absolute ``singular_det`` threshold around the high-savings points here,
    sending Newton into the fallback and leaving alphas at the cold init.
    """

    log_R_bill = jnp.array([[0.01, 0.01, 0.01, 0.01]], dtype=jnp.float64)
    log_x_s = jnp.array([[-0.12, -0.02, 0.08, 0.18]], dtype=jnp.float64)
    log_x_b = jnp.array([[-0.06, 0.03, -0.01, 0.09]], dtype=jnp.float64)
    weight = jnp.ones_like(log_x_s) / log_x_s.size
    s_grid = jnp.array([1.0, 10.0, 25.0, 75.0, 750.0], dtype=jnp.float64)

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return terminal_foc_jac_ccv(
                a_s,
                a_b,
                s_val,
                1.0,  # A_is
                log_R_bill,
                log_x_s,
                log_x_b,
                weight,
                0.04,   # sigma2_xr
                0.01,   # sigma2_xb
                0.005,  # sigma_xrxb
                5.0,    # gamma
                1.0,    # b_bar
                0.0,    # delta
            )

        return foc_fn

    (
        _x_egm,
        c_egm,
        a_s_egm,
        a_b_egm,
        _n_iters_egm,
        _n_backtrack_egm,
        exit_code_egm,
    ) = _egm_scan_cell(
        foc_factory,
        s_grid,
        jnp.full_like(s_grid, 0.85),
        jnp.full_like(s_grid, 0.44),
        gamma=5.0,
        beta=0.96,
        tol=1e-7,
        max_iter=50,
        max_backtrack_iter=10,
        line_search_max_step=2.0,
        singular_det=1e-15,
        grad_step_size=0.05,
        grad_denom_eps=1e-10,
        tiny_savings=1e-6,
        euler_inv_floor=1e-20,
        min_consumption=1e-10,
        egm_anchor=1e-10,
        use_fori=False,
        use_line_search=True,
    )

    exit_codes = np.asarray(exit_code_egm[1:])
    alpha_s = np.asarray(a_s_egm[1:])
    alpha_b = np.asarray(a_b_egm[1:])
    c_ratio = np.asarray(c_egm[1:] / s_grid)

    assert np.all(exit_codes == EC_INTERIOR)
    assert not np.allclose(alpha_s[-1], 0.85)
    assert not np.allclose(alpha_b[-1], 0.44)
    np.testing.assert_allclose(alpha_s, alpha_s[0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(alpha_b, alpha_b[0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(c_ratio, c_ratio[0], rtol=1e-11, atol=1e-11)
