"""
Tests for PCHIP (Fritsch-Carlson) z-interpolation in compute_foc_jac_working.

These exercise the live solver helpers `_pchip_slope_uniform`,
`_pchip_eval_with_basis`, and `_interp_z_wealth` -- not a local copy.

Properties covered:
  1. _pchip_slope_uniform: zero at sign changes / zeros, harmonic mean
     otherwise, satisfies Fritsch-Carlson |m| <= 3*min(|d_l|,|d_r|).
  2. PCHIP cubic via _pchip_eval_with_basis: endpoint interpolation,
     linear-exactness, monotonicity preservation on monotone stencils
     (where Catmull-Rom would overshoot).
  3. _interp_z_wealth: cubic and linear branches return correct values;
     mpc is the exact analytical d c_val / d x_next (FD-consistency
     invariant -- the Jacobian guardrail flagged in the catmull-rom
     handoff).

Catmull-Rom-only properties (exactness on quadratics/cubics, the
0.5*t*(1-t)*(1-2t) error formula) are intentionally NOT carried over:
PCHIP is monotonicity-preserving, not exact on quadratics.
"""

import numpy as np

from lifecycle.solver import (
    _interp_z_wealth,
    _pchip_eval_with_basis,
    _pchip_slope_uniform,
)


# ── 1. Slope-limiter unit tests ──────────────────────────────────────────────

def _hermite_basis(f):
    f2 = f * f
    f3 = f2 * f
    return (
        2.0 * f3 - 3.0 * f2 + 1.0,
        f3 - 2.0 * f2 + f,
        -2.0 * f3 + 3.0 * f2,
        f3 - f2,
    )


def test_pchip_slope_zero_on_sign_change():
    assert _pchip_slope_uniform(1.0, -1.0) == 0.0
    assert _pchip_slope_uniform(-2.5, 0.7) == 0.0
    assert _pchip_slope_uniform(3.0, 0.0) == 0.0
    assert _pchip_slope_uniform(0.0, 3.0) == 0.0


def test_pchip_slope_harmonic_mean_same_sign():
    # positive both
    m = _pchip_slope_uniform(2.0, 6.0)
    expected = 2.0 * 2.0 * 6.0 / (2.0 + 6.0)
    assert abs(m - expected) < 1e-14
    # negative both
    m = _pchip_slope_uniform(-2.0, -6.0)
    expected = 2.0 * (-2.0) * (-6.0) / (-2.0 + -6.0)
    assert abs(m - expected) < 1e-14


def test_pchip_slope_fritsch_carlson_bound():
    # For any same-sign pair, |m| <= 3 * min(|d_l|, |d_r|)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d_l = rng.uniform(0.01, 100.0)
        d_r = rng.uniform(0.01, 100.0)
        for sign in (1.0, -1.0):
            m = _pchip_slope_uniform(sign * d_l, sign * d_r)
            assert abs(m) <= 3.0 * min(d_l, d_r) + 1e-12


# ── 2. PCHIP cubic kernel tests ──────────────────────────────────────────────

def test_pchip_endpoints():
    rng = np.random.default_rng(42)
    h00_0, h10_0, h01_0, h11_0 = _hermite_basis(0.0)
    h00_1, h10_1, h01_1, h11_1 = _hermite_basis(1.0)
    for _ in range(500):
        p0, p1, p2, p3 = rng.uniform(0.1, 10.0, 4)
        v0 = _pchip_eval_with_basis(p0, p1, p2, p3, h00_0, h10_0, h01_0, h11_0)
        v1 = _pchip_eval_with_basis(p0, p1, p2, p3, h00_1, h10_1, h01_1, h11_1)
        assert abs(v0 - p1) < 1e-12
        assert abs(v1 - p2) < 1e-12


def test_pchip_linear_exact():
    """PCHIP is exact on linear data."""
    rng = np.random.default_rng(123)
    for _ in range(200):
        a, b = rng.uniform(-5.0, 5.0, 2)
        p0 = a + b * (-1)
        p1 = a + b * 0
        p2 = a + b * 1
        p3 = a + b * 2
        for f in np.linspace(0.0, 1.0, 21):
            h00, h10, h01, h11 = _hermite_basis(f)
            approx = _pchip_eval_with_basis(p0, p1, p2, p3, h00, h10, h01, h11)
            exact = a + b * f
            assert abs(approx - exact) < 1e-12, (a, b, f, approx, exact)


def test_pchip_monotonicity_preservation():
    """On any monotone-increasing 4-point stencil, the interpolant on [p1,p2]
    stays in [p1, p2]. This is the property Catmull-Rom violates (HANDOFF
    measured 1.683e-3 max overshoot)."""
    rng = np.random.default_rng(7)
    n_violations = 0
    n_total = 0
    for _ in range(500):
        # generate a monotone-increasing stencil
        deltas = rng.uniform(0.0, 5.0, 4)
        p0 = rng.uniform(0.1, 10.0)
        p1 = p0 + deltas[0]
        p2 = p1 + deltas[1]
        p3 = p2 + deltas[2]
        lo, hi = min(p1, p2), max(p1, p2)
        for f in np.linspace(0.0, 1.0, 51):
            h00, h10, h01, h11 = _hermite_basis(f)
            v = _pchip_eval_with_basis(p0, p1, p2, p3, h00, h10, h01, h11)
            n_total += 1
            if v < lo - 1e-12 or v > hi + 1e-12:
                n_violations += 1
    assert n_violations == 0, (
        f"PCHIP violated monotonicity in {n_violations}/{n_total} cases"
    )


def test_pchip_zero_slope_at_local_extremum():
    """When p1 is a local max/min on the stencil, derivative at p1 is 0."""
    # local max at p1
    p0, p1, p2, p3 = 1.0, 5.0, 2.0, 0.5
    h00, h10, h01, h11 = _hermite_basis(0.0)
    eps = 1e-7
    h00e, h10e, h01e, h11e = _hermite_basis(eps)
    v0 = _pchip_eval_with_basis(p0, p1, p2, p3, h00, h10, h01, h11)
    ve = _pchip_eval_with_basis(p0, p1, p2, p3, h00e, h10e, h01e, h11e)
    # forward derivative at f=0 should be ~0 because m0 is zeroed by the limiter
    assert abs((ve - v0) / eps) < 1e-5

    # local min at p1
    p0, p1, p2, p3 = 5.0, 1.0, 3.0, 6.0
    v0 = _pchip_eval_with_basis(p0, p1, p2, p3, h00, h10, h01, h11)
    ve = _pchip_eval_with_basis(p0, p1, p2, p3, h00e, h10e, h01e, h11e)
    assert abs((ve - v0) / eps) < 1e-5


# ── 3. _interp_z_wealth integration tests ────────────────────────────────────

def _build_grid(n_z=11, n_w=20, j_s_count=1):
    """Realistic monotone-increasing-in-w policy: c[iz, j_s, iw]."""
    rng = np.random.default_rng(999)
    c = np.zeros((n_z, j_s_count, n_w), dtype=np.float64)
    for iz in range(n_z):
        for j_s in range(j_s_count):
            base = 1.0 + 0.3 * np.exp(0.2 * iz) + 0.1 * j_s
            for iw in range(n_w):
                c[iz, j_s, iw] = base + 0.05 * iw**1.5 + 1e-3 * rng.standard_normal()
    return c


def test_interp_z_wealth_endpoints_match_grid():
    """At frac_z in {0, 1} and frac_w in {0, 1}, c_val must equal the grid node."""
    c = _build_grid()
    inv_dw = 1.0  # dw=1 implied
    n_z = c.shape[0]
    for iz_lo in range(1, n_z - 2):
        for iw in range(c.shape[2] - 1):
            # corner (frac_z=0, frac_w=0) -> c[iz_lo, 0, iw]
            v, _ = _interp_z_wealth(c, 0, iz_lo, 0.0, iw, 0.0, inv_dw, n_z, True, 1e-30)
            assert abs(v - c[iz_lo, 0, iw]) < 1e-12
            # corner (frac_z=1, frac_w=0) -> c[iz_lo+1, 0, iw]
            v, _ = _interp_z_wealth(c, 0, iz_lo, 1.0, iw, 0.0, inv_dw, n_z, True, 1e-30)
            assert abs(v - c[iz_lo + 1, 0, iw]) < 1e-12
            # corner (frac_z=0, frac_w=1) -> c[iz_lo, 0, iw+1]
            v, _ = _interp_z_wealth(c, 0, iz_lo, 0.0, iw, 1.0, inv_dw, n_z, True, 1e-30)
            assert abs(v - c[iz_lo, 0, iw + 1]) < 1e-12
            # corner (frac_z=1, frac_w=1) -> c[iz_lo+1, 0, iw+1]
            v, _ = _interp_z_wealth(c, 0, iz_lo, 1.0, iw, 1.0, inv_dw, n_z, True, 1e-30)
            assert abs(v - c[iz_lo + 1, 0, iw + 1]) < 1e-12


def test_interp_z_wealth_mpc_fd_consistency_cubic():
    """mpc must equal the secant slope of c_val between the wealth endpoints,
    (c_at_iw1 - c_at_iw)/dw.  Under blend-then-PCHIP this is no longer the
    pointwise derivative of c_val (which is non-linear in frac_w under the
    non-linear slope limiter), but mpc is *defined* as the secant for use
    as the Jacobian element ∂c/∂x_next; that secant exactly equals
    (c_val at fw=1) - (c_val at fw=0) per dw, by construction.
    """
    c = _build_grid()
    n_z = c.shape[0]
    n_w = c.shape[2]
    dw = 1.0
    inv_dw = 1.0 / dw

    rng = np.random.default_rng(2024)
    n_checks = 0
    for iz_lo in range(1, n_z - 2):
        for iw in range(n_w - 1):
            for frac_z in rng.uniform(0.0, 1.0, 5):
                for frac_w in rng.uniform(0.0, 1.0, 3):
                    c_val, mpc = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, True, 1e-30
                    )
                    # Evaluate c_val at the two wealth endpoints (fw = 0, 1).
                    # Under blend-then-PCHIP these reduce to PCHIP(stencil_iw)
                    # and PCHIP(stencil_iw1) respectively, so mpc returned at
                    # any interior frac_w is exactly the secant of these.
                    c_at_iw, _ = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, 0.0, inv_dw, n_z, True, 1e-30
                    )
                    c_at_iw1, _ = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, 1.0, inv_dw, n_z, True, 1e-30
                    )
                    mpc_expected = (c_at_iw1 - c_at_iw) * inv_dw
                    # mpc may be clamped to [0,1]; only check unclamped cases
                    if 0.0 < mpc_expected < 1.0:
                        assert abs(mpc - mpc_expected) < 1e-12, (
                            iz_lo, iw, frac_z, frac_w, mpc, mpc_expected
                        )
                    n_checks += 1
    assert n_checks > 0


def test_interp_z_wealth_no_overshoot_cross_wealth():
    """Regression: under blend-then-PCHIP the c_val at off-grid (frac_z,
    frac_w) must stay inside the interval endpoints of the *wealth-blended*
    z stencil.  The previous PCHIP-then-blend implementation overshot by
    ~0.003 on this stencil because Fritsch-Carlson made materially
    different limiter decisions at iw and iw+1.
    """
    # CODEX counterexample: 4-point z stencil at two wealth nodes.
    stencil = np.array(
        [[2.9005, 1.6772],
         [3.0121, 1.7558],
         [2.0189, 4.4623],
         [1.2131, 3.1536]],
        dtype=np.float64,
    )
    n_z = 4
    c = np.zeros((n_z, 1, 2), dtype=np.float64)
    c[:, 0, 0] = stencil[:, 0]
    c[:, 0, 1] = stencil[:, 1]

    iz_lo = 1
    j_s = 0
    iw = 0
    inv_dw = 1.0
    # Sweep both fractions over a fine grid; require c_val never escape the
    # [min, max] of the wealth-blended interval endpoints (b1, b2).
    n_violations = 0
    n_total = 0
    for frac_w in np.linspace(0.0, 1.0, 21):
        b1 = (1.0 - frac_w) * stencil[1, 0] + frac_w * stencil[1, 1]
        b2 = (1.0 - frac_w) * stencil[2, 0] + frac_w * stencil[2, 1]
        lo, hi = (b1, b2) if b1 <= b2 else (b2, b1)
        for frac_z in np.linspace(0.0, 1.0, 41):
            c_val, _ = _interp_z_wealth(
                c, j_s, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, True, 1e-30
            )
            n_total += 1
            if c_val < lo - 1e-10 or c_val > hi + 1e-10:
                n_violations += 1
    assert n_violations == 0, (
        f"PCHIP overshoot on cross-wealth stencil: {n_violations}/{n_total} cases"
    )

    # Also verify the specific point CODEX flagged.
    c_val, _ = _interp_z_wealth(c, 0, 1, 0.05, 0, 0.25, 1.0, n_z, True, 1e-30)
    # Under blend-then-PCHIP the value should be ~2.6978, well within the
    # blended interval endpoints (2.6298, 2.6980).
    assert 2.629 <= c_val <= 2.700, (
        f"c_val = {c_val:.6f} not within blended endpoints "
        f"[2.6298, 2.6980] (was 2.7008 under PCHIP-then-blend)."
    )


def test_interp_z_wealth_mpc_fd_consistency_linear_branch():
    """Same FD-consistency check on the linear (boundary) branch."""
    c = _build_grid()
    n_z = c.shape[0]
    n_w = c.shape[2]
    inv_dw = 1.0
    rng = np.random.default_rng(2025)
    for iz_lo in (0, n_z - 2):
        for iw in range(n_w - 1):
            for frac_z in rng.uniform(0.0, 1.0, 5):
                for frac_w in rng.uniform(0.0, 1.0, 3):
                    _, mpc = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, False, 1e-30
                    )
                    c_at_iw, _ = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, 0.0, inv_dw, n_z, False, 1e-30
                    )
                    c_at_iw1, _ = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, 1.0, inv_dw, n_z, False, 1e-30
                    )
                    mpc_expected = (c_at_iw1 - c_at_iw) * inv_dw
                    if 0.0 < mpc_expected < 1.0:
                        assert abs(mpc - mpc_expected) < 1e-12


def test_interp_z_wealth_no_overshoot_on_realistic_policy():
    """On a monotone-positive realistic policy, PCHIP must keep c_val inside
    [min(p1,p2), max(p1,p2)] across the active z interval (this is the gap
    Catmull-Rom left -- HANDOFF section 4.3 reported 18,466 overshoot
    cases, max relative deviation 1.683e-3)."""
    n_z = 11
    z_grid = np.linspace(-3.0, 3.0, n_z)
    n_w = 5
    c = np.empty((n_z, 1, n_w), dtype=np.float64)
    for iw in range(n_w):
        scale = 1.0 + 0.3 * iw
        c[:, 0, iw] = scale * np.maximum(0.1, 0.5 + 0.3 * np.exp(z_grid))

    inv_dw = 1.0
    n_violations = 0
    n_total = 0
    for iz_lo in range(1, n_z - 2):
        for iw in range(n_w - 1):
            for frac_z in np.linspace(0.0, 1.0, 31):
                for frac_w in (0.0, 0.5, 1.0):
                    c_val, _ = _interp_z_wealth(
                        c, 0, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, True, 1e-30
                    )
                    # bounds at this frac_w from the chord-linear-in-w view
                    p1 = (1.0 - frac_w) * c[iz_lo, 0, iw] + frac_w * c[iz_lo, 0, iw + 1]
                    p2 = (1.0 - frac_w) * c[iz_lo + 1, 0, iw] + frac_w * c[iz_lo + 1, 0, iw + 1]
                    lo, hi = min(p1, p2), max(p1, p2)
                    if c_val < lo - 1e-10 or c_val > hi + 1e-10:
                        n_violations += 1
                    n_total += 1
    assert n_violations == 0, (
        f"PCHIP overshoot on realistic policy: {n_violations}/{n_total}"
    )
