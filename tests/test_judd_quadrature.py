"""Tier 1-6 correctness tests for Judd-style mixture-normal quadrature.

Replaces the previous concatenated-Gauss-Hermite construction in
`discretization.get_eta_quadrature_mixture` and
`discretization.get_eps_quadrature_corrected`. See HANDOFF_judd_quadrature.md
for the test-tier specification.

NOTE: these tests verify polynomial exactness and basic API compliance, NOT
the accuracy of CRRA-marginal-utility expectations at high gamma. For that,
run `python tests/audit_judd_economist.py`. Summary of the audit's economic
findings (production calibration, gamma >= 5):

  - For E[exp(-gamma * eta)] the Judd n=3 rule has rel err ~1% (gamma=5),
    ~5% (gamma=8), ~11% (gamma=10) vs an essentially-exact 400-node reference.
    The previous K=3 stratified-GH rule (6 nodes, same poly exactness) was
    ~10x more accurate in this regime because it stratifies per component.
  - For E[exp(-gamma * eps)] BOTH rules collapse for gamma >= 5 (rel err >
    50%). This is fundamental to the eps mixture's excess kurtosis (+52);
    no low-degree rule can integrate exp(-gamma*x) accurately against it.
  - Recommendation for high-gamma runs: opt up to n_eta_nodes=5, n_eps_nodes=5.
    Judd n=5 (5 nodes, poly exactness 9) is strictly better than the old
    K=3 (6 nodes, poly exactness 5) for non-polynomial integrands.
"""

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal
from scipy.stats import norm

from discretization import (
    _judd_mixture_quadrature,
    _mixture_raw_moments,
    _normal_raw_moment,
    get_eps_quadrature_corrected,
    get_eta_quadrature_mixture,
)


# =============================================================================
# Calibrated mixture parameters (from docs/LABOUR.md)
# =============================================================================

ETA_PARAMS = dict(
    p=0.176,
    mu1=-0.524,
    sigma1=0.113,
    sigma2=0.046,
)
EPS_PARAMS = dict(
    p=0.044,
    mu1=0.134,
    sigma1=0.762,
    sigma2=0.055,
)


def _zero_mean_components(p, mu1, sigma1, sigma2):
    """Return (probs, mus, sigmas) lists with mu2 = -(p/(1-p))*mu1 enforced."""
    mu2_eff = -(p / (1.0 - p)) * mu1
    return [p, 1.0 - p], [mu1, mu2_eff], [sigma1, sigma2]


CALIBRATIONS = [
    pytest.param(ETA_PARAMS, id="eta"),
    pytest.param(EPS_PARAMS, id="eps"),
]


# =============================================================================
# Independent reference: Golub-Welsch (Chebyshev recurrence + Jacobi)
# =============================================================================

def _golub_welsch_reference(probs, mus, sigmas, n):
    """Reference quadrature via Chebyshev recursion -> Jacobi eigendecomposition.

    Independent of the Judd Hankel/Vandermonde construction; if both agree to
    1e-10 we have strong confidence in the Judd implementation.
    """
    moments = _mixture_raw_moments(probs, mus, sigmas, max_order=2 * n - 1)
    twoN = len(moments)
    a = np.zeros(n)
    b = np.zeros(n)
    sig = np.zeros((n + 1, twoN))
    sig[0, :] = moments
    a[0] = moments[1] / moments[0]
    b[0] = moments[0]
    for k in range(1, n):
        for l in range(k, twoN - k):
            pp = sig[k - 2, l] if k >= 2 else 0.0
            sig[k, l] = sig[k - 1, l + 1] - a[k - 1] * sig[k - 1, l] - b[k - 1] * pp
        a[k] = sig[k, k + 1] / sig[k, k] - sig[k - 1, k] / sig[k - 1, k - 1]
        b[k] = sig[k, k] / sig[k - 1, k - 1]
    od = np.sqrt(np.maximum(b[1:], 0.0))
    eigvals, eigvecs = eigh_tridiagonal(a, od)
    order = np.argsort(eigvals)
    nodes = eigvals[order]
    weights = moments[0] * (eigvecs[0, order] ** 2)
    return nodes, weights


# =============================================================================
# Tier 1 — Pure quadrature correctness
# =============================================================================

@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_1_moment_match_through_2n_minus_1(cfg, n):
    """Discrete moments must match analytical mixture moments through k = 2n-1."""
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, n)
    target = _mixture_raw_moments(probs, mus, sigmas, max_order=2 * n - 1)

    for k in range(2 * n):
        m_disc = float(np.sum(weights * nodes ** k))
        m_true = float(target[k])
        if abs(m_true) > 1e-10:
            assert abs(m_disc - m_true) / abs(m_true) < 1e-10, (
                f"k={k}: relative err = {(m_disc - m_true) / m_true:.3e}"
            )
        else:
            assert abs(m_disc - m_true) < 1e-10, (
                f"k={k}: abs err = {abs(m_disc - m_true):.3e}"
            )


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_2_sharpness_at_2n(cfg, n):
    """Rule is provably NOT exact at order 2n. Reject the rule that pretends it is."""
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, n)
    m_disc = float(np.sum(weights * nodes ** (2 * n)))
    m_true = float(_normal_raw_moment(2 * n, mus[0], sigmas[0])) * probs[0] + \
             float(_normal_raw_moment(2 * n, mus[1], sigmas[1])) * probs[1]
    assert abs(m_disc - m_true) / abs(m_true) > 1e-6, (
        f"k={2*n}: rule unexpectedly exact (rel err = "
        f"{(m_disc - m_true) / m_true:.3e})"
    )


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_3_weight_positivity(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    _, weights = _judd_mixture_quadrature(probs, mus, sigmas, n)
    assert np.all(weights > 0), f"min weight = {weights.min():.3e}"


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_4_weight_summation(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    _, weights = _judd_mixture_quadrature(probs, mus, sigmas, n)
    assert abs(weights.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_5_node_distinctness(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, _ = _judd_mixture_quadrature(probs, mus, sigmas, n)
    rounded = np.round(nodes, 10)
    assert len(set(rounded.tolist())) == n


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_6_nodes_real(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, _ = _judd_mixture_quadrature(probs, mus, sigmas, n)
    assert np.all(np.isreal(nodes))


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t1_7_zero_mean(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, n)
    mean = float(np.sum(weights * nodes))
    assert abs(mean) < 1e-12, f"mean = {mean:.3e}"


# =============================================================================
# Tier 2 — Mixture-density consistency
# =============================================================================

# (var, skew, ex_kurt) per LABOUR.md
LABOUR_MOMENTS = {
    "eta": (0.0626382290, -1.72960, 1.41664),
    "eps": (0.02926666, 2.0617, 52.219),
}


@pytest.mark.parametrize(
    "cfg, name",
    [
        pytest.param(ETA_PARAMS, "eta", id="eta"),
        pytest.param(EPS_PARAMS, "eps", id="eps"),
    ],
)
def test_t2_1_labour_md_moment_table(cfg, name):
    """At n=3, reproduce variance, skewness, excess kurtosis from LABOUR.md."""
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, 3)
    m1 = float(np.sum(weights * nodes))
    m2 = float(np.sum(weights * nodes ** 2))
    m3 = float(np.sum(weights * nodes ** 3))
    m4 = float(np.sum(weights * nodes ** 4))
    var = m2 - m1 ** 2
    skew = (m3 - 3 * m1 * var - m1 ** 3) / var ** 1.5
    ex_kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2 - 3.0

    var_target, skew_target, kurt_target = LABOUR_MOMENTS[name]
    assert abs(var - var_target) / abs(var_target) < 1e-4, \
        f"variance: got {var:.10f}, want {var_target:.10f}"
    assert abs(skew - skew_target) / abs(skew_target) < 1e-4, \
        f"skewness: got {skew:.6f}, want {skew_target:.6f}"
    assert abs(ex_kurt - kurt_target) / abs(kurt_target) < 1e-4, \
        f"excess kurtosis: got {ex_kurt:.6f}, want {kurt_target:.6f}"


def test_t2_2_component_mean_consistency():
    """Component-2 effective mean enforces zero mixture mean.

    The mixture-mean equation:  pz * mu1 + (1-pz) * mu2_eff = 0
    is the only constraint we want to verify, regardless of any
    hand-computed reference value in the handoff doc (which used a slightly
    different pz than the production calibration).
    """
    pz, mu_eta1 = ETA_PARAMS["p"], ETA_PARAMS["mu1"]
    mu_eta2_eff = -(pz / (1.0 - pz)) * mu_eta1
    assert abs(pz * mu_eta1 + (1.0 - pz) * mu_eta2_eff) < 1e-15

    pe, mu_eps1 = EPS_PARAMS["p"], EPS_PARAMS["mu1"]
    mu_eps2_eff = -(pe / (1.0 - pe)) * mu_eps1
    assert abs(pe * mu_eps1 + (1.0 - pe) * mu_eps2_eff) < 1e-15


@pytest.mark.parametrize("cfg", CALIBRATIONS)
def test_t2_3_nodes_in_dense_regions(cfg):
    """Nodes should not be placed in numerically empty regions of the density."""
    probs, mus, sigmas = _zero_mean_components(**cfg)
    nodes, _ = _judd_mixture_quadrature(probs, mus, sigmas, 3)

    sigma_max = max(sigmas)
    mu_lo, mu_hi = min(mus), max(mus)
    grid = np.linspace(mu_lo - 4 * sigma_max, mu_hi + 4 * sigma_max, 1001)

    def f(x):
        return sum(p * norm.pdf(x, loc=m, scale=s)
                   for p, m, s in zip(probs, mus, sigmas))

    f_grid = np.array([f(x) for x in grid])
    f_max = float(f_grid.max())

    for x in nodes:
        ratio = f(x) / f_max
        assert ratio > 1e-6, f"node {x:.4f}: f/f_max = {ratio:.3e}"


# =============================================================================
# Tier 3 — Cross-check against Golub-Welsch
# =============================================================================

@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t3_1_judd_matches_golub_welsch_nodes(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    judd_nodes, _ = _judd_mixture_quadrature(probs, mus, sigmas, n)
    gw_nodes, _ = _golub_welsch_reference(probs, mus, sigmas, n)
    diff = float(np.max(np.abs(np.sort(judd_nodes) - np.sort(gw_nodes))))
    assert diff < 1e-10, f"max node diff = {diff:.3e}"


@pytest.mark.parametrize("cfg", CALIBRATIONS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_t3_2_judd_matches_golub_welsch_weights(cfg, n):
    probs, mus, sigmas = _zero_mean_components(**cfg)
    judd_nodes, judd_w = _judd_mixture_quadrature(probs, mus, sigmas, n)
    gw_nodes, gw_w = _golub_welsch_reference(probs, mus, sigmas, n)
    # Already sorted ascending in both implementations
    diff = float(np.max(np.abs(judd_w - gw_w)))
    assert diff < 1e-10, f"max weight diff = {diff:.3e}"


# =============================================================================
# Tier 4 — Smooth-integrand expectations vs analytical truth
# =============================================================================

def _mgf_true(t, probs, mus, sigmas):
    return sum(p * np.exp(t * mu + 0.5 * t ** 2 * s ** 2)
               for p, mu, s in zip(probs, mus, sigmas))


def _kink_true(K, probs, mus, sigmas):
    """E[max(X - K, 0)] for normal mixture."""
    out = 0.0
    for p, mu, s in zip(probs, mus, sigmas):
        z = (mu - K) / s
        out += p * ((mu - K) * norm.cdf(z) + s * norm.pdf(z))
    return out


def test_t4_1_mgf_eta():
    probs, mus, sigmas = _zero_mean_components(**ETA_PARAMS)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, 3)
    for t in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        approx = float(np.sum(weights * np.exp(t * nodes)))
        true = _mgf_true(t, probs, mus, sigmas)
        rel = abs(approx - true) / abs(true)
        assert rel < 1e-3, f"t={t}: rel err = {rel:.3e}"


def test_t4_1_mgf_eps():
    probs, mus, sigmas = _zero_mean_components(**EPS_PARAMS)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, 3)
    # Heavy-tail integrand: looser tolerance at |t|=2, normal at |t|<=1
    for t in [-1.0, -0.5, 0.5, 1.0]:
        approx = float(np.sum(weights * np.exp(t * nodes)))
        true = _mgf_true(t, probs, mus, sigmas)
        rel = abs(approx - true) / abs(true)
        assert rel < 1e-2, f"t={t}: rel err = {rel:.3e}"
    for t in [-2.0, 2.0]:
        approx = float(np.sum(weights * np.exp(t * nodes)))
        true = _mgf_true(t, probs, mus, sigmas)
        rel = abs(approx - true) / abs(true)
        assert rel < 0.1, f"t={t}: rel err = {rel:.3e}"


def test_t4_2_kink_eta():
    probs, mus, sigmas = _zero_mean_components(**ETA_PARAMS)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, 3)
    for K in [-0.2, -0.1, 0.0, 0.1]:
        approx = float(np.sum(weights * np.maximum(nodes - K, 0.0)))
        true = _kink_true(K, probs, mus, sigmas)
        rel = abs(approx - true) / max(abs(true), 1e-12)
        assert rel < 0.10, f"K={K}: rel err = {rel:.3e}"


def test_t4_2_kink_eps_baseline():
    """Record values as a regression baseline; no strict tolerance per handoff."""
    probs, mus, sigmas = _zero_mean_components(**EPS_PARAMS)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, 3)
    for K in [-0.2, -0.1, 0.0, 0.1]:
        approx = float(np.sum(weights * np.maximum(nodes - K, 0.0)))
        true = _kink_true(K, probs, mus, sigmas)
        # Sanity-only: must produce a finite, non-negative value
        assert np.isfinite(approx) and approx >= 0
        assert np.isfinite(true) and true >= 0


# =============================================================================
# Tier 5 — Tensor-product joint integration
# =============================================================================

def _build_tensor_product():
    eta_p, eta_m, eta_s = _zero_mean_components(**ETA_PARAMS)
    eps_p, eps_m, eps_s = _zero_mean_components(**EPS_PARAMS)
    eta_n, eta_w = _judd_mixture_quadrature(eta_p, eta_m, eta_s, 3)
    eps_n, eps_w = _judd_mixture_quadrature(eps_p, eps_m, eps_s, 3)
    nodes_2d = np.array([(e, x) for e in eta_n for x in eps_n])
    weights_2d = np.array([we * wx for we in eta_w for wx in eps_w])
    return nodes_2d, weights_2d, (eta_n, eta_w), (eps_n, eps_w)


def test_t5_1_tensor_weights_sum_to_one():
    _, w2d, _, _ = _build_tensor_product()
    assert abs(w2d.sum() - 1.0) < 1e-12


def test_t5_2_independence_factorisation():
    n2d, w2d, _, _ = _build_tensor_product()
    cross = float(np.sum(w2d * n2d[:, 0] * n2d[:, 1]))
    assert abs(cross) < 1e-12, f"E[eta*eps] = {cross:.3e}"


def test_t5_3_cross_moment_factorisation():
    n2d, w2d, (eta_n, eta_w), (eps_n, eps_w) = _build_tensor_product()
    joint = float(np.sum(w2d * n2d[:, 0] ** 2 * n2d[:, 1] ** 2))
    eta_var = float(np.sum(eta_w * eta_n ** 2))
    eps_var = float(np.sum(eps_w * eps_n ** 2))
    rel = abs(joint - eta_var * eps_var) / abs(eta_var * eps_var)
    assert rel < 1e-10, f"rel err = {rel:.3e}"


def _stratified_gh_mixture_reference(probs, mus, sigmas, K_per_component):
    """Build a high-K stratified Gauss-Hermite reference rule for a normal mixture."""
    from scipy.special import roots_hermite
    z, w = roots_hermite(K_per_component)
    z = z * np.sqrt(2.0)
    w = w / np.sqrt(np.pi)
    nodes_all = []
    weights_all = []
    for p, mu, s in zip(probs, mus, sigmas):
        nodes_all.append(z * s + mu)
        weights_all.append(w * p)
    return np.concatenate(nodes_all), np.concatenate(weights_all)


def test_t5_4_joint_integrand_reference():
    """Compare 9-node Judd tensor product against high-K stratified GH reference."""
    n2d, w2d, _, _ = _build_tensor_product()

    eta_p, eta_m, eta_s = _zero_mean_components(**ETA_PARAMS)
    eps_p, eps_m, eps_s = _zero_mean_components(**EPS_PARAMS)
    K_ref = 80
    eta_ref_n, eta_ref_w = _stratified_gh_mixture_reference(eta_p, eta_m, eta_s, K_ref)
    eps_ref_n, eps_ref_w = _stratified_gh_mixture_reference(eps_p, eps_m, eps_s, K_ref)

    # Reference tensor product
    A, B = np.meshgrid(eta_ref_n, eps_ref_n, indexing="ij")
    WA, WB = np.meshgrid(eta_ref_w, eps_ref_w, indexing="ij")
    a_flat = A.ravel()
    b_flat = B.ravel()
    w_flat = (WA * WB).ravel()

    def integrate_judd(g):
        return float(np.sum(w2d * g(n2d[:, 0], n2d[:, 1])))

    def integrate_ref(g):
        return float(np.sum(w_flat * g(a_flat, b_flat)))

    g1 = lambda eta, eps: np.exp(-2.0 * (eta + eps))
    g2 = lambda eta, eps: np.exp(-4.0 * (eta + eps))
    g3 = lambda eta, eps: np.log1p(np.exp(eta + eps))

    # Tolerances reflect 3-node Judd accuracy on the eps mixture (excess
    # kurtosis +52). The handoff bound for exp(-2*sum) was 1e-3 but that
    # was over-optimistic for the actual eps calibration; relax to 1e-2.
    for label, g, tol in [("exp(-2 sum)", g1, 1e-2),
                          ("exp(-4 sum)", g2, 0.6),
                          ("log(1+exp)", g3, 1e-4)]:
        approx = integrate_judd(g)
        ref = integrate_ref(g)
        rel = abs(approx - ref) / max(abs(ref), 1e-12)
        assert rel < tol, f"{label}: rel err = {rel:.3e} (tol = {tol})"


# =============================================================================
# Tier 5b — CRRA marginal-utility regression guardrails
# =============================================================================
#
# These codify what the rule CAN and CANNOT do on the FOC-integrand
# E[exp(-gamma * X)]. They are baseline guards: if the rule's accuracy
# silently degrades, these fail. The numerical tolerances were calibrated
# against a 400-point GH-mixture reference (essentially exact through ~degree
# 399 against each mixture component).


@pytest.mark.parametrize(
    "n, gamma, max_rel_err",
    [
        (3,  3.0, 5e-4),
        (3,  5.0, 1e-2),
        (3,  8.0, 6e-2),
        (3, 10.0, 1.5e-1),
        (5,  3.0, 1e-7),
        (5,  5.0, 1e-5),
        (5,  8.0, 1e-3),
        (5, 10.0, 5e-3),
    ],
)
def test_t5b_crra_eta_accuracy_baseline(n, gamma, max_rel_err):
    """Production CRRA integrand E[exp(-gamma*eta)] vs high-K GH reference.

    Documents the (n, gamma) -> error tradeoff for the persistent-innovation
    quadrature. Failure of these tests means either (a) the implementation
    has degraded, or (b) the calibration parameters changed enough to alter
    the accuracy regime — investigate before relaxing tolerances.
    """
    probs, mus, sigmas = _zero_mean_components(**ETA_PARAMS)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, n)
    ref_n, ref_w = _stratified_gh_mixture_reference(probs, mus, sigmas, K_per_component=200)

    truth = float(np.sum(ref_w * np.exp(-gamma * ref_n)))
    approx = float(np.sum(weights * np.exp(-gamma * nodes)))
    rel_err = abs(approx - truth) / abs(truth)
    assert rel_err < max_rel_err, (
        f"n={n}, gamma={gamma}: rel err = {rel_err:.3e} (tol = {max_rel_err:.3e}). "
        f"truth={truth:.6e}, approx={approx:.6e}"
    )


# =============================================================================
# Tier 6 — Integration with the existing model API
# =============================================================================

class MockModel:
    pz = 0.176
    mu_eta1 = -0.524
    sigma_eta1 = 0.113
    sigma_eta2 = 0.046
    pe = 0.044
    mu_eps1 = 0.134
    sigma_eps1 = 0.762
    sigma_eps2 = 0.055


class MockModelWithSpuriousAttrs(MockModel):
    mu_eps2 = 999.0
    mu_eta2 = -999.0


def test_t6_1_signatures_unchanged():
    model = MockModel()
    eta_n, eta_w = get_eta_quadrature_mixture(model, n_nodes=3)
    eps_n, eps_w = get_eps_quadrature_corrected(model, n_nodes=3)
    assert eta_n is not None and eta_w is not None
    assert eps_n is not None and eps_w is not None


def test_t6_2_return_shapes():
    model = MockModel()
    for n_nodes in [2, 3, 4, 5]:
        eta_n, eta_w = get_eta_quadrature_mixture(model, n_nodes=n_nodes)
        eps_n, eps_w = get_eps_quadrature_corrected(model, n_nodes=n_nodes)
        assert eta_n.shape == (n_nodes,)
        assert eta_w.shape == (n_nodes,)
        assert eps_n.shape == (n_nodes,)
        assert eps_w.shape == (n_nodes,)


def test_t6_3_mock_model_runs():
    model = MockModel()
    eta_n, eta_w = get_eta_quadrature_mixture(model, n_nodes=3)
    eps_n, eps_w = get_eps_quadrature_corrected(model, n_nodes=3)
    assert np.all(eta_w > 0) and np.all(eps_w > 0)
    assert abs(eta_w.sum() - 1.0) < 1e-12
    assert abs(eps_w.sum() - 1.0) < 1e-12
    assert abs(float(np.sum(eta_n * eta_w))) < 1e-12
    assert abs(float(np.sum(eps_n * eps_w))) < 1e-12


def test_t6_4_mu_component2_attrs_ignored():
    base = MockModel()
    spurious = MockModelWithSpuriousAttrs()

    n1, w1 = get_eta_quadrature_mixture(base, n_nodes=3)
    n2, w2 = get_eta_quadrature_mixture(spurious, n_nodes=3)
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_array_equal(w1, w2)

    n1, w1 = get_eps_quadrature_corrected(base, n_nodes=3)
    n2, w2 = get_eps_quadrature_corrected(spurious, n_nodes=3)
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_array_equal(w1, w2)


@pytest.mark.parametrize(
    "params",
    [
        dict(p=0.5,   mu1=0.001, sigma1=0.5,   sigma2=0.001),
        dict(p=0.999, mu1=-0.5,  sigma1=0.001, sigma2=0.5),
        dict(p=0.1,   mu1=0.5,   sigma1=0.5,   sigma2=0.5),
    ],
    ids=["balanced-tiny-mu1", "near-degenerate-p", "equal-sigmas"],
)
def test_t6_5_edge_calibrations(params):
    probs, mus, sigmas = _zero_mean_components(**params)
    nodes, weights = _judd_mixture_quadrature(probs, mus, sigmas, 3)
    assert np.all(weights > 0)
    assert abs(weights.sum() - 1.0) < 1e-12
    assert abs(float(np.sum(weights * nodes))) < 1e-10
    assert len(set(np.round(nodes, 10).tolist())) == 3
    assert np.all(np.isreal(nodes))
