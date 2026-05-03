"""
Tests for the non-uniform PCHIP interpolant `pchip_interp_1d` used by the
EGM secondary-interpolation step in `_solve_retirement_step_quad_jit` and
`_solve_working_age_step_quad_jit`.

These cover the surface contract:
  1. Affine reproduction on a non-uniform grid (no spurious curvature).
  2. Linear extrapolation outside grid bounds matches `fast_interp_1d`.
  3. Slope-zeroing at local extrema -> bounded-by-data interpolant.
  4. Monotonicity preservation on monotone-increasing inputs.
  5. Numerical conditioning at a tiny first-segment width.
"""

import numpy as np

from solver import fast_interp_1d, pchip_interp_1d


def test_pchip_interp_1d_affine_reproduction_nonuniform():
    """PCHIP must be exact on linear data, even on a non-uniform grid."""
    x = np.array([0.01, 0.05, 0.13, 0.34, 0.7, 1.5, 4.0, 12.0, 50.0, 200.0])
    y = 2.0 + 0.7 * x
    for x_query in [0.02, 0.1, 0.5, 1.0, 3.0, 8.0, 30.0, 100.0]:
        expected = 2.0 + 0.7 * x_query
        actual = pchip_interp_1d(x_query, x, y)
        assert abs(actual - expected) < 1e-12, (x_query, actual, expected)


def test_pchip_interp_1d_linear_extrapolation_matches_fast_interp_1d():
    """Outside the grid bounds, output must equal the linear extrapolant
    used by fast_interp_1d so call-site semantics are preserved."""
    x = np.array([0.01, 0.05, 0.13, 0.34, 0.7, 1.5, 4.0, 12.0, 50.0, 200.0])
    y = 2.0 + 0.7 * x
    for x_query in [-1.0, 1e-5, 250.0, 500.0]:
        lin = fast_interp_1d(x_query, x, y)
        pch = pchip_interp_1d(x_query, x, y)
        assert abs(lin - pch) < 1e-12, (x_query, lin, pch)


def test_pchip_interp_1d_no_overshoot_on_nonmonotone_data():
    """At local extrema, slopes get zeroed; the interpolant stays bounded
    by the surrounding data, no overshoot."""
    x = np.linspace(0.0, 1.0, 11)
    y = np.array([0.0, 0.5, 1.0, 0.7, 0.3, 0.0, 0.3, 0.7, 1.0, 0.5, 0.0])
    for x_query in np.linspace(0.05, 0.95, 91):
        v = pchip_interp_1d(x_query, x, y)
        assert -0.01 < v < 1.01, (x_query, v)


def test_pchip_interp_1d_monotonicity_preservation():
    """Monotone-increasing input must produce a monotone-increasing
    interpolant. This is the property linear interpolation preserves
    trivially and PCHIP preserves by design (Fritsch-Carlson)."""
    x = np.array([0.0, 0.1, 0.5, 1.0, 5.0, 25.0, 100.0])
    y = np.array([0.0, 0.05, 0.4, 0.85, 4.5, 23.0, 95.0])
    sample = np.linspace(x[0], x[-1], 1000)
    vals = np.array([pchip_interp_1d(xi, x, y) for xi in sample])
    assert np.all(np.diff(vals) >= -1e-12)


def test_pchip_interp_1d_tiny_first_segment_finite():
    """A tiny first interval must not produce NaN / inf in the boundary
    slope estimate."""
    x = np.array([1e-10, 1e-3, 0.05, 0.5, 5.0, 100.0])
    y = np.array([1e-10, 0.001, 0.05, 0.5, 5.0, 100.0])
    for x_query in [0.0, 1e-12, 1e-5, 1e-3, 0.5, 50.0]:
        v = pchip_interp_1d(x_query, x, y)
        assert np.isfinite(v), (x_query, v)
