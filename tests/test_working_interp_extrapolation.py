"""
Regression tests for working-age wealth extrapolation.

These tests pin down the contract between `find_bracket` and
`_interp_z_wealth`: once x moves off the wealth grid, the returned value and
the returned slope must still describe the same affine extrapolation.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lifecycle.solver import find_bracket, fast_interp_1d_with_slope, _interp_z_wealth


def test_find_bracket_extrapolates_both_boundaries():
    grid = np.array([1.0, 2.0, 4.0])

    iw_lo, frac_lo, inv_dw_lo = find_bracket(0.5, grid)
    assert iw_lo == 0
    assert abs(frac_lo + 0.5) < 1e-14
    assert abs(inv_dw_lo - 1.0) < 1e-14

    iw_hi, frac_hi, inv_dw_hi = find_bracket(4.5, grid)
    assert iw_hi == 1
    assert abs(frac_hi - 1.25) < 1e-14
    assert abs(inv_dw_hi - 0.5) < 1e-14


def test_interp_z_wealth_linear_branch_matches_1d_extrapolation():
    wealth_grid = np.array([1.0, 2.0, 4.0])
    x = 4.5
    iw, frac_w, inv_dw = find_bracket(x, wealth_grid)

    c_next_full = np.zeros((2, 1, 3))
    c_next_full[0, 0, :] = np.array([1.6, 2.0, 2.8])
    c_next_full[1, 0, :] = np.array([2.6, 3.0, 3.8])

    frac_z = 0.25
    c_val, mpc_val = _interp_z_wealth(
        c_next_full, 0, 0, frac_z, iw, frac_w, inv_dw, 2, False, 0.0
    )

    c_lo, mpc_lo = fast_interp_1d_with_slope(x, wealth_grid, c_next_full[0, 0, :])
    c_hi, mpc_hi = fast_interp_1d_with_slope(x, wealth_grid, c_next_full[1, 0, :])
    c_expected = (1.0 - frac_z) * c_lo + frac_z * c_hi
    mpc_expected = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi

    assert abs(c_val - c_expected) < 1e-14
    assert abs(mpc_val - mpc_expected) < 1e-14


def test_interp_z_wealth_cubic_branch_is_exact_for_linear_z_affine_wealth():
    """PCHIP preserves exactness on linear-in-z, affine-in-wealth data, even
    under wealth extrapolation (frac_w > 1). The cubic branch must reproduce
    the exact policy and exact wealth-derivative."""
    wealth_grid = np.array([1.0, 2.0, 4.0])
    x = 4.5
    iw, frac_w, inv_dw = find_bracket(x, wealth_grid)

    n_z = 5
    c_next_full = np.zeros((n_z, 1, wealth_grid.size))

    def a_of_z(z):
        return 1.0 + 0.2 * z

    def b_of_z(z):
        return 0.3 + 0.05 * z

    for iz in range(n_z):
        for iw_grid, w in enumerate(wealth_grid):
            c_next_full[iz, 0, iw_grid] = a_of_z(iz) + b_of_z(iz) * w

    iz_lo = 1
    frac_z = 0.35
    c_val, mpc_val = _interp_z_wealth(
        c_next_full, 0, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, True, 0.0
    )

    z_cont = iz_lo + frac_z
    c_expected = a_of_z(z_cont) + b_of_z(z_cont) * x
    mpc_expected = b_of_z(z_cont)

    assert abs(c_val - c_expected) < 1e-12
    assert abs(mpc_val - mpc_expected) < 1e-12


def test_interp_z_wealth_cubic_mpc_is_fd_consistent_under_extrapolation():
    """The FD-consistency invariant must hold even when wealth is being
    extrapolated (frac_w outside [0,1]). mpc returned by the cubic branch
    equals the analytical wealth derivative of c_val."""
    wealth_grid = np.array([1.0, 2.0, 4.0])
    n_z = 6
    rng = np.random.default_rng(11)
    c_next_full = rng.uniform(0.5, 5.0, size=(n_z, 1, wealth_grid.size))

    iz_lo = 2
    frac_z = 0.4
    for x in (0.5, 4.5):  # extrapolation below grid and above grid
        iw, frac_w, inv_dw = find_bracket(x, wealth_grid)
        c_val, mpc_val = _interp_z_wealth(
            c_next_full, 0, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, True, -1e30
        )
        c_at_iw, _ = _interp_z_wealth(
            c_next_full, 0, iz_lo, frac_z, iw, 0.0, inv_dw, n_z, True, -1e30
        )
        c_at_iw1, _ = _interp_z_wealth(
            c_next_full, 0, iz_lo, frac_z, iw, 1.0, inv_dw, n_z, True, -1e30
        )
        mpc_expected = (c_at_iw1 - c_at_iw) * inv_dw
        if 0.0 < mpc_expected < 1.0:
            assert abs(mpc_val - mpc_expected) < 1e-12
