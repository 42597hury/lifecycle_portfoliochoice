"""
quadrature_with_tails.py — Gauss-Hermite quadrature with prescribed
tail nodes, for arbitrage-resistant integration on a chosen axis.

What this is
------------
A 1D quadrature rule for the standard normal measure with K nodes, two
of which are FIXED at +/- Z (in standardised z units). The remaining
K-2 nodes are placed optimally to maximise polynomial exactness on the
*full* (untruncated) Gaussian. This is the rigorous analog of
Gauss-Lobatto when the support has no finite boundary.

Polynomial exactness is 2K-3 (vs 2K-1 for pure Gauss-Hermite at the
same K). The trade-off you buy is guaranteed coverage of extreme tail
values that pure Hermite of practical K cannot reach -- e.g. Hermite
K=5 has max node ~2.86, while this rule with K=5 and Z=4 puts a node
at exactly 4.

Why this is useful
------------------
Spurious arbitrages in a finite quadrature happen when the discrete
node set fails to span the support of an asset's true return
distribution -- e.g. all sampled bond residuals dominate all sampled
stock residuals, even though in the continuous distribution they
don't. Adding explicit far-tail nodes is the targeted fix.

Top-level entry points used by the rest of the package:

  * `gauss_hermite_prescribed_tails(K, Z)` — the 1D rule. Imported by
    `lifecycle.discretization` to assemble per-axis tensor-product
    return / state quadratures with mixed Hermite + Lobatto axes.

  * `get_return_quadrature_with_axis_tails`, `get_state_quadrature_with_axis_tails`
    — single-axis convenience wrappers retained for the self-test
    suite below; production code uses the `lobatto_Z` argument on
    `get_return_quadrature` / `get_state_quadrature` instead.

Limitations
-----------
* K must be odd and in {3, 5, 7} (closed-form construction).
* Z must satisfy: Z > 1 (K=3); Z >= sqrt(5) (K=5); discrete valid
  windows for K=7 (see `gauss_hermite_prescribed_tails` docstring).
  Below threshold, the optimal interior node positions become
  imaginary and no valid rule exists.
* Polynomial exactness is 2K-3 vs 2K-1 for pure Hermite. For smooth
  bulk-dominated integrands pure Hermite converges faster.
"""

from __future__ import annotations

from math import prod

import numpy as np
from scipy.special import roots_hermite


# =============================================================================
# 1D rule: Gauss-Hermite with prescribed tail nodes
# =============================================================================

def _even_moment(k: int) -> float:
    """E[z^(2k)] for z ~ N(0,1):  1, 1, 3, 15, 105, 945, ..."""
    if k == 0:
        return 1.0
    return float(prod(range(1, 2 * k, 2)))


def gauss_hermite_prescribed_tails(K: int, Z: float) -> tuple[np.ndarray, np.ndarray]:
    """K-node 1D quadrature on N(0, 1) with prescribed nodes at +/- Z.

    Constructs a symmetric K-node rule {-Z, ..., 0, ..., +Z} (with K odd)
    that integrates polynomials of degree <= 2K-3 exactly on the full
    standard normal density.

    Parameters
    ----------
    K : int
        Number of nodes. Must be odd and in {3, 5, 7}.
    Z : float
        Magnitude of the prescribed tail nodes (in standardised units).
        For pure standard normal, Z=3 puts nodes at ~3-sigma; Z=4 at 4-sigma.

    Returns
    -------
    nodes : ndarray, shape (K,)
        Sorted ascending. nodes[0] = -Z, nodes[-1] = +Z, nodes[(K-1)/2] = 0.
    weights : ndarray, shape (K,)
        All positive, sum to 1.

    Raises
    ------
    ValueError
        If K is even, K is not in {3, 5, 7}, or Z is outside the valid
        window for the requested K.

    Validity windows in Z (closed-form regions where all weights are
    positive and interior nodes are real):
        K=3:  Z > 1
        K=5:  Z >= sqrt(5) ~ 2.236
        K=7:  Z < 1.36, or 1.81 < Z < 2.86, or Z >= 3.28
    """
    if K < 3:
        raise ValueError(f"K must be >= 3, got {K}")
    if K % 2 == 0:
        raise NotImplementedError(
            "Only odd K supported (symmetric construction with central node)."
        )
    if Z <= 1.0:
        raise ValueError(
            f"Z must be > 1 for K=3 (needed for positive central weight); "
            f"larger K imposes stricter lower bounds. Got Z = {Z}"
        )

    # m = (K - 1) / 2. By symmetry, the rule has:
    #   - Prescribed nodes:  +/- Z  (weight w_t each)
    #   - Free interior:     +/- z_1, ..., +/- z_{m-1}, 0  (weights w_1,...,w_{m-1}, w_0)
    # Total distinct unknowns: (m-1) free positions + (m+1) distinct weights = 2m.
    # Match 2m even moments E[z^0], E[z^2], ..., E[z^{4m-2}]; odd moments
    # auto-zero by symmetry. Polynomial exactness 4m - 1 = 2K - 3.
    m = (K - 1) // 2

    # ---- K = 3 closed form -----------------------------------------------
    if m == 1:
        w_t = 1.0 / (2.0 * Z * Z)
        w_0 = 1.0 - 1.0 / (Z * Z)
        if w_0 <= 0:
            raise ValueError(f"Z={Z} too small; w_0 = {w_0} <= 0")
        nodes   = np.array([-Z, 0.0, +Z])
        weights = np.array([w_t, w_0, w_t])
        return nodes, weights

    # ---- K = 5 closed form -----------------------------------------------
    if m == 2:
        # Optimal z_1^2 = c = 3 (Z^2 - 5) / (Z^2 - 3). Need c >= 0,
        # i.e. Z >= sqrt(5) ~ 2.236.
        denom = Z * Z - 3.0
        if denom == 0:
            raise ValueError(f"Z^2 = 3 is a degenerate case; pick another Z.")
        c = 3.0 * (Z * Z - 5.0) / denom
        if c < 0:
            raise ValueError(
                f"K=5 prescribed-tail rule requires Z >= sqrt(5) ~ 2.236 "
                f"for real interior nodes. Got Z = {Z} (=> z_1^2 = {c} < 0)."
            )
        z_1 = float(np.sqrt(c))
        # Linear system in (w_0, w_1, w_t) given z_1. By moment matching:
        # moment 0:  w_0 + 2 w_1 + 2 w_t = 1
        # moment 2:  2 z_1^2 w_1 + 2 Z^2 w_t = 1
        # moment 4:  2 z_1^4 w_1 + 2 Z^4 w_t = 3
        A = np.array(
            [[1.0,                      2.0,                          2.0           ],
             [0.0,                      2.0 * z_1 * z_1,              2.0 * Z * Z   ],
             [0.0,                      2.0 * z_1**4,                 2.0 * Z**4    ]]
        )
        b = np.array([1.0, 1.0, 3.0])
        w_0, w_1, w_t = np.linalg.solve(A, b)
        if min(w_0, w_1, w_t) <= 0:
            raise ValueError(
                f"K=5, Z={Z} gave non-positive weights: w_0={w_0}, w_1={w_1}, w_t={w_t}"
            )
        nodes   = np.array([-Z, -z_1, 0.0, +z_1, +Z])
        weights = np.array([w_t, w_1, w_0, w_1, w_t])
        return nodes, weights

    # ---- K = 7 closed form -----------------------------------------------
    # Free interior nodes (zeros of degree-5 odd polynomial p_5(x) = x q(x^2)
    # orthogonal to {1,x,x^2,x^3,x^4} w.r.t. (x^2 - Z^2) phi(x)) are
    # x = 0, +/- z_1, +/- z_2, with z_1^2, z_2^2 = roots of y^2 + a y + b
    # where (a, b) solve the linear system from <p_5, x>_rho = <p_5, x^3>_rho = 0:
    #
    #   3 a (5 - u) + b (3 - u) = 15 (u - 7)        u := Z^2
    #   5 a (7 - u) + b (5 - u) = 35 (u - 9)
    #
    # Solving by Cramer's rule:
    if m == 3:
        u = Z * Z
        denom = (u * u - 10.0 * u + 15.0)
        if abs(denom) < 1e-14:
            raise ValueError(f"K=7 degenerate at Z={Z} (Cramer denominator=0).")
        a = -10.0 * (u * u - 12.0 * u + 21.0) / denom
        b =  15.0 * (u * u - 14.0 * u + 35.0) / denom
        disc = a * a - 4.0 * b
        if a >= 0 or b <= 0 or disc <= 0:
            # The valid Z windows for K=7 are roughly Z<1.36, 1.81<Z<2.86,
            # or Z>3.28. Outside these the orthogonal polynomial fails to
            # produce two real positive squared roots.
            raise ValueError(
                f"K=7 prescribed-tails rule has no valid solution at Z={Z}.\n"
                f"   Got (a, b) = ({a:.4f}, {b:.4f}), discriminant = {disc:.4f}.\n"
                f"   Need a < 0, b > 0, a^2 - 4b > 0.\n"
                f"   Valid Z windows: Z < 1.36, 1.81 < Z < 2.86, or Z >= 3.28.\n"
                f"   For tail node at z=3 you must use K=3 or K=5 (or pure Hermite K>=7)."
            )
        sqrt_disc = float(np.sqrt(disc))
        y1 = (-a - sqrt_disc) / 2.0    # smaller positive root of y^2 + a y + b = 0
        y2 = (-a + sqrt_disc) / 2.0    # larger positive root
        z_1 = float(np.sqrt(y1))
        z_2 = float(np.sqrt(y2))
        # Solve for weights: 4 unknowns (w_0, w_1, w_2, w_t) from 4 even moments
        # (0, 2, 4, 6). w_0 from moment 0; the rest from moments 2, 4, 6.
        A = np.array([
            [1.0,            2.0,                  2.0,                  2.0      ],
            [0.0,            2.0 * z_1**2,         2.0 * z_2**2,         2.0 * Z**2 ],
            [0.0,            2.0 * z_1**4,         2.0 * z_2**4,         2.0 * Z**4 ],
            [0.0,            2.0 * z_1**6,         2.0 * z_2**6,         2.0 * Z**6 ],
        ])
        rhs = np.array([1.0, 1.0, 3.0, 15.0])
        w_0, w_1, w_2, w_t = np.linalg.solve(A, rhs)
        if min(w_0, w_1, w_2, w_t) <= 0:
            raise ValueError(
                f"K=7, Z={Z} gave non-positive weights: "
                f"w_0={w_0}, w_1={w_1}, w_2={w_2}, w_t={w_t}"
            )
        nodes   = np.array([-Z, -z_2, -z_1, 0.0, +z_1, +z_2, +Z])
        weights = np.array([w_t, w_2, w_1, w_0, w_1, w_2, w_t])
        return nodes, weights

    # ---- K >= 9 not implemented in closed form ---------------------------
    raise NotImplementedError(
        f"K={K} not implemented. Closed-form solutions are provided for "
        f"K in (3, 5, 7); for higher K the orthogonal-polynomial system "
        f"requires numerical solution and validity windows in Z become "
        f"complicated. If you need K >= 9 tail coverage, prefer pure "
        f"Gauss-Hermite at higher K (no Z to tune, no validity gaps)."
    )


# =============================================================================
# Tensor-product wrappers (legacy single-axis interface, kept for self-test).
# Production code wires Lobatto into `get_return_quadrature` /
# `get_state_quadrature` directly via the `lobatto_Z` argument; these
# single-axis wrappers are retained because the self-test suite below
# exercises them and they are convenient one-off entry points.
# =============================================================================

def _tensor_product_quadrature(grids_z, grids_w, Sigma):
    """Stitch per-axis 1D rules into a tensor product, then apply Cholesky transform."""
    grid_1d   = np.meshgrid(*grids_z, indexing="ij")
    weight_1d = np.meshgrid(*grids_w, indexing="ij")
    z_nodes   = np.stack([g.ravel() for g in grid_1d], axis=1)
    weights   = np.prod(np.stack(weight_1d, axis=0), axis=0).ravel()

    Sigma_sym = 0.5 * (np.asarray(Sigma, dtype=float)
                       + np.asarray(Sigma, dtype=float).T)
    L = np.linalg.cholesky(Sigma_sym)
    nodes = z_nodes @ L.T
    return nodes, weights


def _build_per_axis_grids(K_per_dim, axis_with_tails, Z):
    """Build per-axis (z_nodes, weights) lists. Axis `axis_with_tails` uses the
    prescribed-tails rule with parameter Z; all other axes use pure Hermite."""
    grids_z, grids_w = [], []
    for d, K in enumerate(K_per_dim):
        if K < 1:
            raise ValueError(f"K[{d}] = {K} < 1")
        if d == axis_with_tails:
            if K < 3 or K % 2 == 0:
                raise ValueError(
                    f"axis {d} uses prescribed-tail rule which needs odd K >= 3; got {K}"
                )
            z, w = gauss_hermite_prescribed_tails(K, Z)
        elif K == 1:
            z = np.zeros(1, dtype=float)
            w = np.ones(1, dtype=float)
        else:
            z, w = roots_hermite(K)
            z = z * np.sqrt(2.0)
            w = w / np.sqrt(np.pi)
        grids_z.append(z)
        grids_w.append(w)
    return grids_z, grids_w


def get_return_quadrature_with_axis_tails(model, K_per_dim, axis: int, Z: float):
    """Return-residual quadrature with prescribed tails on the chosen axis.

    Drop-in replacement for `discretization.get_return_quadrature(model, K_per_dim)`
    that swaps the 1D rule on `axis` for a prescribed-tails rule with parameter Z.

    Axis convention (post rtb-as-state migration, ret_names = ('xr', 'xb')):
        axis = 0  ->  prescribed tails on xr residual (z_0)
        axis = 1  ->  prescribed tails on purified xb residual (z_1)  <-- bond axis
    """
    from lifecycle.discretization import _normalize_ret_nodes  # lazy: avoid cycle
    n_ret = int(model.n_ret)
    K = _normalize_ret_nodes(K_per_dim, n_ret)
    if axis < 0 or axis >= n_ret:
        raise ValueError(f"axis = {axis} out of range [0, {n_ret})")
    grids_z, grids_w = _build_per_axis_grids(K, axis, Z)
    return _tensor_product_quadrature(grids_z, grids_w, model.Sigma_r_cond)


def get_state_quadrature_with_axis_tails(model, K_per_dim, axis: int, Z: float):
    """State-innovation quadrature with prescribed tails on the chosen axis.

    Drop-in replacement for `discretization.get_state_quadrature(model, K_per_dim)`.
    Under the default state ordering (dp, spr, rtb, y_1) (post 2026-05-07 dp migration):
        axis = 0  ->  prescribed tails on dp axis
        axis = 1  ->  prescribed tails on spr-purified axis
        axis = 2  ->  prescribed tails on rtb-purified axis (inflation surprise)
        axis = 3  ->  prescribed tails on y_1-purified axis  <-- bond driver
    """
    from lifecycle.discretization import _normalize_state_nodes  # lazy: avoid cycle
    n_state = int(model.n_state)
    K = _normalize_state_nodes(K_per_dim, n_state)
    if axis < 0 or axis >= n_state:
        raise ValueError(f"axis = {axis} out of range [0, {n_state})")
    grids_z, grids_w = _build_per_axis_grids(K, axis, Z)
    return _tensor_product_quadrature(grids_z, grids_w, model.Sigma_ss)


# =============================================================================
# Validation suite (run as `python -m lifecycle.quadrature_with_tails`)
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from lifecycle.var import build_nominal_system1_var_config_hardcoded
    from lifecycle.precompute import build_model
    from lifecycle.discretization import get_return_quadrature, get_state_quadrature

    # mu_eta2 / mu_eps2 are derived inside `build_model` from
    # (pz, mu_eta1) and (pe, mu_eps1) per the zero-mean constraint
    # E[eta] = E[eps] = 0 (Fix A in
    # docs/scans/INCOME_PIPELINE_REVIEW_2026-05-09.md), so we don't pass
    # them here.
    BASE_CFG = {
        "beta": 0.96, "gamma": 5.0, "b_bar": 10,
        "start_age": 22, "retire_age": 67, "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991, "pz": 0.176,
        "mu_eta1": -0.524, "sigma_eta1": 0.113, "sigma_eta2": 0.046,
        "pe": 0.044, "mu_eps1": 0.134, "sigma_eps1": 0.762,
        "sigma_eps2": 0.055, "constrained": False,
    }
    model = build_model(BASE_CFG, build_nominal_system1_var_config_hardcoded(), verbose=False)
    Sigma_r_sym = 0.5 * (model.Sigma_r_cond + model.Sigma_r_cond.T)
    L_r = np.linalg.cholesky(Sigma_r_sym)
    Sigma_s_sym = 0.5 * (model.Sigma_ss + model.Sigma_ss.T)
    L_s = np.linalg.cholesky(Sigma_s_sym)

    # ---- 1: 1D rule moment recovery --------------------------------------
    print("=" * 72)
    print("1.  1D rule moment recovery (polynomial exactness)")
    print("=" * 72)
    for K in (3, 5, 7):
        Z_list = (3.0, 4.0)
        if K == 7:
            Z_list = (3.5, 4.0)   # K=7 has no valid solution at Z=3
        for Z in Z_list:
            try:
                z, w = gauss_hermite_prescribed_tails(K, Z)
            except ValueError as e:
                print(f"  K={K} Z={Z}: SKIP -- {e}")
                continue
            max_err_low = 0.0
            for d in range(2 * K - 2):
                truth = (_even_moment(d // 2) if d % 2 == 0 else 0.0)
                qval  = float(np.sum(w * z ** d))
                err   = abs(qval - truth) / max(1.0, abs(truth))
                max_err_low = max(max_err_low, err)
            print(f"  K={K} Z={Z}: sum(w)={w.sum():.16f}  max|err| over d<=2K-3 = {max_err_low:.2e}")
            assert max_err_low < 1e-10, (K, Z, max_err_low)
            assert (w > 0).all(), (K, Z, w)

    # ---- 2: tensor-product moment recovery -------------------------------
    print()
    print("=" * 72)
    print("2.  Tensor-product return quadrature: mean=0, cov=Sigma_r_cond")
    print("=" * 72)
    print(f"    {'config':<55}  {'|sumw-1|':>10}  {'|mean|':>10}  {'|cov-S|':>10}")
    # Post rtb-as-state: return block is (xr, xb), so axis 0 = xr and
    # axis 1 = xb (the bond residual). The legacy "tails on rtb" axis-0
    # case is dropped — rtb no longer lives in the return block.
    cases = [
        ("axis=1, Z=3, K=(5,3) [tails on xb]",   1, 3.0, (5, 3)),
        ("axis=1, Z=4, K=(5,3) [tails on xb]",   1, 4.0, (5, 3)),
        ("axis=1, Z=4, K=(5,5) [tails on xb]",   1, 4.0, (5, 5)),
        ("axis=0, Z=3, K=(5,3) [tails on xr]",   0, 3.0, (5, 3)),
    ]
    for label, axis, Z, K in cases:
        r, w = get_return_quadrature_with_axis_tails(model, K, axis, Z)
        e_mass = abs(w.sum() - 1.0)
        e_mean = float(np.max(np.abs(w @ r)))
        e_cov  = float(np.max(np.abs((r.T * w) @ r - Sigma_r_sym)))
        print(f"    {label:<55}  {e_mass:>10.2e}  {e_mean:>10.2e}  {e_cov:>10.2e}")
        assert e_mass < 1e-12 and e_mean < 1e-12 and e_cov < 1e-13, (label, e_mass, e_mean, e_cov)

    # ---- 3: tail-coverage comparison -------------------------------------
    print()
    print("=" * 72)
    print("3.  Bond-residual tail coverage: Hermite vs prescribed-tail rules")
    print("=" * 72)
    print(f"    {'rule':<55}  {'min(xb_resid)':>14}  {'max(xb_resid)':>14}")
    # Post rtb-as-state: return block is (xr, xb); xb is at column 1.
    for K in [(5, 3), (5, 5), (5, 7)]:
        r, _ = get_return_quadrature(model, n_nodes=K)
        print(f"    {'Hermite K=' + str(K):<55}  {r[:,1].min():>14.4f}  {r[:,1].max():>14.4f}")
    for Z in (3.0, 4.0):
        r, _ = get_return_quadrature_with_axis_tails(model, (5, 5), axis=1, Z=Z)
        print(f"    {f'Prescribed-tails axis=1 K=(5,5) Z={Z}':<55}  "
              f"{r[:,1].min():>14.4f}  {r[:,1].max():>14.4f}")

    # ---- 4: validity-window error message --------------------------------
    print()
    print("=" * 72)
    print("4.  Validity-window error messages")
    print("=" * 72)
    try:
        gauss_hermite_prescribed_tails(7, 3.0)
        raise AssertionError("(K=7, Z=3.0) should have raised ValueError")
    except ValueError as e:
        print(f"  (K=7, Z=3.0) correctly raised: {str(e).splitlines()[0]}")
    try:
        gauss_hermite_prescribed_tails(5, 2.0)
        raise AssertionError("(K=5, Z=2.0) should have raised ValueError")
    except ValueError as e:
        print(f"  (K=5, Z=2.0) correctly raised: {str(e).splitlines()[0]}")

    print()
    print("All self-tests passed.")
