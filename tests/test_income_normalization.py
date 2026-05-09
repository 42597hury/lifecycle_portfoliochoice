"""Income mixture normalization regression tests.

Locks in the zero-mean invariants surfaced by Fix A in
docs/scans/INCOME_PIPELINE_REVIEW_2026-05-09.md:

  E[eta] = E[eps] = 0   <=>   mu_eta2 = -(pz/(1-pz)) * mu_eta1
                              mu_eps2 = -(pe/(1-pe)) * mu_eps1

Three test classes:
  1. Property-based: across a parameter grid, the eta and eps Judd quadratures
     have empirical mean within 1e-12 of zero. Garbage `mu_eta2`/`mu_eps2`
     fields on the model NamedTuple WERE silently overridden by the
     pre-Fix-A quadrature; after Fix A those stored values are authoritative
     and must come from `build_model`.
  2. build_model derivation: after `build_model(base_config, var_config)`,
     `model.mu_eta2 == -(pz/(1-pz)) * mu_eta1` and the analogous identity
     for `mu_eps2` hold to machine epsilon.
  3. Regression on the canonical: `model.mu_eta2` and `model.mu_eps2`
     match the values implied by `configs._canonical.{pz, mu_eta1, pe, mu_eps1}`.
"""
from typing import NamedTuple

import numpy as np
import pytest

from lifecycle.discretization import (
    get_eps_quadrature_corrected,
    get_eta_quadrature_mixture,
)
from lifecycle.precompute import build_model


# Stand-in NamedTuple matching the fields the quadratures read on the
# model. Mirrors LifecyclePortfolioModel's labour-income block (the
# quadratures only access pe/pz and mu_*/sigma_*); other fields are not
# touched, so we don't include them.
class _IncomeModel(NamedTuple):
    pe: float
    pz: float
    mu_eta1: float
    sigma_eta1: float
    mu_eta2: float
    sigma_eta2: float
    mu_eps1: float
    sigma_eps1: float
    mu_eps2: float
    sigma_eps2: float


def _zero_mean_mu2(p: float, mu1: float) -> float:
    """Component-2 mean implied by E[mixture] = 0."""
    return -(p / (1.0 - p)) * mu1


# -----------------------------------------------------------------------------
# 1. Property-based zero-mean invariants on the Judd quadrature.
# -----------------------------------------------------------------------------
# (p, mu1, sigma1, sigma2) parameter grid:
#   - canonical eps  (pe, mu_eps1, sigma_eps1, sigma_eps2)
#   - canonical eta  (pz, mu_eta1, sigma_eta1, sigma_eta2)
#   - arbitrary positive-shift mixture
#   - symmetric extreme: 50/50 weights
_QUAD_GRID = [
    (0.044, 0.134, 0.762, 0.055),
    (0.176, -0.524, 0.113, 0.046),
    (0.10, 0.5, 0.3, 0.1),
    (0.5, -0.2, 0.8, 0.4),
]


@pytest.mark.parametrize("p,mu1,s1,s2", _QUAD_GRID)
def test_eps_quadrature_zero_mean(p, mu1, s1, s2):
    """E[eps] from the eps Judd quadrature is zero to machine epsilon."""
    mu2 = _zero_mean_mu2(p, mu1)
    m = _IncomeModel(
        pe=p, pz=0.5,
        mu_eta1=0.0, sigma_eta1=1.0, mu_eta2=0.0, sigma_eta2=1.0,
        mu_eps1=mu1, sigma_eps1=s1, mu_eps2=mu2, sigma_eps2=s2,
    )
    nodes, weights = get_eps_quadrature_corrected(m, n_nodes=4)
    mean = float(np.sum(nodes * weights))
    assert abs(mean) < 1e-12, f"E[eps] = {mean:.3e} (parameters p={p}, mu1={mu1})"

    # Variance must match the closed-form mixture variance under zero-mean.
    var_theory = p * (s1 ** 2 + mu1 ** 2) + (1.0 - p) * (s2 ** 2 + mu2 ** 2)
    var_quad = float(np.sum(nodes ** 2 * weights)) - mean ** 2
    assert abs(var_quad - var_theory) < 1e-10, (var_quad, var_theory)


@pytest.mark.parametrize("p,mu1,s1,s2", _QUAD_GRID)
def test_eta_quadrature_zero_mean(p, mu1, s1, s2):
    """E[eta] from the eta Judd quadrature is zero to machine epsilon."""
    mu2 = _zero_mean_mu2(p, mu1)
    m = _IncomeModel(
        pe=0.5, pz=p,
        mu_eta1=mu1, sigma_eta1=s1, mu_eta2=mu2, sigma_eta2=s2,
        mu_eps1=0.0, sigma_eps1=1.0, mu_eps2=0.0, sigma_eps2=1.0,
    )
    nodes, weights = get_eta_quadrature_mixture(m, n_nodes=3)
    mean = float(np.sum(nodes * weights))
    assert abs(mean) < 1e-12, f"E[eta] = {mean:.3e} (parameters p={p}, mu1={mu1})"

    var_theory = p * (s1 ** 2 + mu1 ** 2) + (1.0 - p) * (s2 ** 2 + mu2 ** 2)
    var_quad = float(np.sum(nodes ** 2 * weights)) - mean ** 2
    assert abs(var_quad - var_theory) < 1e-10, (var_quad, var_theory)


# -----------------------------------------------------------------------------
# 2. build_model derives mu_eta2 and mu_eps2 from the zero-mean constraint.
# -----------------------------------------------------------------------------
# Sweep through several (pz, mu_eta1) and (pe, mu_eps1) combos. Each call
# constructs a real LifecyclePortfolioModel via the canonical hardcoded
# real-yields VAR config (no CSV dependency).
_BUILD_MODEL_GRID = [
    # (pz, mu_eta1, pe, mu_eps1)  -- canonical first
    (0.176, -0.524, 0.044, 0.134),
    (0.30, -0.30, 0.10, 0.05),
    (0.50, 0.20, 0.20, -0.40),
]


@pytest.mark.parametrize("pz,mu_eta1,pe,mu_eps1", _BUILD_MODEL_GRID)
def test_build_model_derives_mu2_from_zero_mean(pz, mu_eta1, pe, mu_eps1):
    """After build_model, model.mu_eta2 and model.mu_eps2 equal the
    zero-mean-implied values to machine epsilon. The model field is the
    single source of truth (Fix A in INCOME_PIPELINE_REVIEW_2026-05-09)."""
    from lifecycle.var import build_real_full_var_config_hardcoded

    base_config = {
        "beta": 0.96, "gamma": 5.0, "b_bar": 10,
        "start_age": 22, "retire_age": 67, "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991,
        "pz": pz, "mu_eta1": mu_eta1,
        "sigma_eta1": 0.113, "sigma_eta2": 0.046,
        "pe": pe, "mu_eps1": mu_eps1,
        "sigma_eps1": 0.762, "sigma_eps2": 0.055,
    }
    var_config = build_real_full_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=False)

    expected_mu_eta2 = _zero_mean_mu2(pz, mu_eta1)
    expected_mu_eps2 = _zero_mean_mu2(pe, mu_eps1)
    assert np.isclose(model.mu_eta2, expected_mu_eta2, atol=1e-12), (
        model.mu_eta2, expected_mu_eta2,
    )
    assert np.isclose(model.mu_eps2, expected_mu_eps2, atol=1e-12), (
        model.mu_eps2, expected_mu_eps2,
    )

    # E[eta] and E[eps] of the underlying mixture must be exactly zero
    # at the values the model carries.
    e_eta = pz * mu_eta1 + (1.0 - pz) * model.mu_eta2
    e_eps = pe * mu_eps1 + (1.0 - pe) * model.mu_eps2
    assert abs(e_eta) < 1e-12, e_eta
    assert abs(e_eps) < 1e-12, e_eps


def test_build_model_ignores_legacy_mu2_keys():
    """build_model must silently ignore explicit mu_eta2 / mu_eps2 keys in
    base_config (legacy-bundle compatibility) and ALWAYS use the derived
    values. Regression for the HIGH-1 silent-override hazard documented in
    INCOME_PIPELINE_REVIEW_2026-05-09."""
    from lifecycle.var import build_real_full_var_config_hardcoded

    pz, mu_eta1 = 0.176, -0.524
    pe, mu_eps1 = 0.044, 0.134
    base_config = {
        "beta": 0.96, "gamma": 5.0, "b_bar": 10,
        "start_age": 22, "retire_age": 67, "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991,
        "pz": pz, "mu_eta1": mu_eta1,
        "sigma_eta1": 0.113, "sigma_eta2": 0.046,
        "pe": pe, "mu_eps1": mu_eps1,
        "sigma_eps1": 0.762, "sigma_eps2": 0.055,
        # Legacy garbage values that MUST NOT propagate:
        "mu_eta2": 999.0,
        "mu_eps2": -999.0,
    }
    var_config = build_real_full_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=False)

    expected_mu_eta2 = _zero_mean_mu2(pz, mu_eta1)
    expected_mu_eps2 = _zero_mean_mu2(pe, mu_eps1)
    assert np.isclose(model.mu_eta2, expected_mu_eta2, atol=1e-12), model.mu_eta2
    assert np.isclose(model.mu_eps2, expected_mu_eps2, atol=1e-12), model.mu_eps2


# -----------------------------------------------------------------------------
# 3. Regression: canonical model has the expected mu_*2 values.
# -----------------------------------------------------------------------------
def test_canonical_model_zero_mean():
    """Regression on the canonical. Locks in that BASE_CONFIG + build_model
    yields the income process Fix A intends."""
    from configs._canonical import BASE_CONFIG
    from lifecycle.var import build_real_full_var_config_hardcoded

    var_config = build_real_full_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)

    # Recall canonical: pz=0.176, mu_eta1=-0.524, pe=0.044, mu_eps1=0.134.
    expected_mu_eta2 = -(0.176 / (1.0 - 0.176)) * (-0.524)
    expected_mu_eps2 = -(0.044 / (1.0 - 0.044)) * 0.134
    assert np.isclose(model.mu_eta2, expected_mu_eta2, atol=1e-12), model.mu_eta2
    assert np.isclose(model.mu_eps2, expected_mu_eps2, atol=1e-12), model.mu_eps2

    # Also verify the canonical mixture has E=0 to machine epsilon.
    e_eta = model.pz * model.mu_eta1 + (1.0 - model.pz) * model.mu_eta2
    e_eps = model.pe * model.mu_eps1 + (1.0 - model.pe) * model.mu_eps2
    assert abs(e_eta) < 1e-12, e_eta
    assert abs(e_eps) < 1e-12, e_eps
