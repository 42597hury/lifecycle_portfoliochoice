"""Diagnostic-vs-solver R_p consistency, plus bundle refuse-to-mix invariants.

Covers PR2 §1 of the handoff:
    - At fixed (state, alpha, shock), the diagnostic's R_p must match the
      solver kernel's R_p to 1e-10 in BOTH branches (simple+clamp, ccv_log).
      Otherwise the EE diagnostic is producing residuals against a different
      wealth dynamics than the bundle was solved under.

Plus the bundle-tagging invariant from §5.9:
    - A bundle saved with wealth_dynamics_spec="ccv_log" must round-trip with
      that tag and the diagnostic must pick it up via ctx.use_ccv=True.
    - Legacy bundles (no tag) default to "simple_clamp"/use_ccv=False.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------
# §1: diag-vs-solver R_p consistency (1e-10)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("a_s, a_b, log_R_bill, log_x_s, log_x_b", [
    (0.0, 0.0, 0.02, 0.05, 0.01),
    (0.5, 0.3, 0.02, 0.07, -0.02),
    (1.0, 0.0, 0.03, 0.10, 0.05),
    (0.6, 0.4, 0.01, -0.04, 0.06),
    (1.5, -0.5, 0.025, 0.05, 0.02),    # leveraged
])
def test_diagnostic_ccv_Rp_matches_solver(a_s, a_b, log_R_bill, log_x_s, log_x_b):
    """The diagnostic's CVC R_p formula must match the solver kernel's CVC R_p
    formula bit-for-bit. We don't call the diagnostic's quadrature loop here
    (it's wrapped in @njit and would need full kernel inputs); instead we
    verify the formula by hand-computing it from the same inputs both call
    sites use, and check against the solver kernel via V_dot extraction.
    """
    sigma2_xr, sigma2_xb, sigma_xrxb = 0.025, 0.005, 0.001

    # Diagnostic-side formula (copied from _compute_euler_sum_retirement_continuous).
    r_p_diag = (log_R_bill
                + a_s * log_x_s + a_b * log_x_b
                + 0.5 * (a_s * sigma2_xr + a_b * sigma2_xb)
                - 0.5 * (a_s * a_s * sigma2_xr
                         + 2.0 * a_s * a_b * sigma_xrxb
                         + a_b * a_b * sigma2_xb))
    R_p_diag = math.exp(r_p_diag)

    # Solver-side: extract R_p from compute_terminal_foc_jac_shifted's V_dot
    # output (V_dot = mu_b * R_p when there is one quadrature node).
    from lifecycle.solver import compute_terminal_foc_jac_shifted

    log_R_bill_arr = np.array([[log_R_bill]])
    log_x_s_arr = np.array([[log_x_s]])
    log_x_b_arr = np.array([[log_x_b]])
    Rx_bill = np.exp(log_R_bill_arr)
    Rx_stock_mult = np.exp(log_x_s_arr)
    Rx_bond_mult = np.exp(log_x_b_arr)
    state_weights = np.array([1.0])
    ret_weights = np.array([1.0])
    s_val = 1.0; A_is = 1.0; gamma = 5.0; b_bar = 10; delta = 0.005
    _, _, _, _, _, V_dot = compute_terminal_foc_jac_shifted(
        a_s, a_b, s_val, A_is,
        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
        log_R_bill_arr, log_x_s_arr, log_x_b_arr,
        sigma2_xr, sigma2_xb, sigma_xrxb, True,
        gamma, b_bar, delta,
    )
    mu_b_diag = b_bar * (R_p_diag + delta) ** (-gamma)
    V_dot_diag = mu_b_diag * R_p_diag
    assert abs(V_dot - V_dot_diag) < 1e-10, (
        f"Diagnostic and solver R_p disagree at alpha=({a_s},{a_b}): "
        f"V_dot_diff={abs(V_dot - V_dot_diag):.2e}"
    )


def test_diagnostic_simple_Rp_matches_solver():
    """Same check for simple+clamp branch. Reusing the existing helper."""
    a_s, a_b = 0.5, 0.3
    log_R_bill, log_x_s, log_x_b = 0.02, 0.07, -0.02

    R_bill = math.exp(log_R_bill)
    R_s = math.exp(log_R_bill + log_x_s)
    R_b = math.exp(log_R_bill + log_x_b)
    R_p_diag = a_s * R_s + a_b * R_b + (1 - a_s - a_b) * R_bill

    from lifecycle.solver import compute_terminal_foc_jac_shifted
    log_R_bill_arr = np.array([[log_R_bill]])
    log_x_s_arr = np.array([[log_x_s]])
    log_x_b_arr = np.array([[log_x_b]])
    Rx_bill = np.exp(log_R_bill_arr)
    Rx_stock_mult = np.exp(log_x_s_arr)
    Rx_bond_mult = np.exp(log_x_b_arr)
    state_weights = np.array([1.0])
    ret_weights = np.array([1.0])
    s_val = 1.0; A_is = 1.0; gamma = 5.0; b_bar = 10; delta = 0.005
    _, _, _, _, _, V_dot = compute_terminal_foc_jac_shifted(
        a_s, a_b, s_val, A_is,
        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
        log_R_bill_arr, log_x_s_arr, log_x_b_arr,
        0.0, 0.0, 0.0, False,
        gamma, b_bar, delta,
    )
    if R_p_diag > 0:
        mu_b = b_bar * (R_p_diag + delta) ** (-gamma)
        V_dot_diag = mu_b * R_p_diag
    else:
        V_dot_diag = 0.0
    assert abs(V_dot - V_dot_diag) < 1e-10


# ----------------------------------------------------------------------------
# Bundle refuse-to-mix invariant: metadata round-trip + diagnostic flag.
# ----------------------------------------------------------------------------


def test_bundle_metadata_carries_wealth_dynamics_spec(tmp_path):
    """Saving with run_config.solver_config.wealth_dynamics_spec='ccv_log'
    propagates to the bundle's top-level metadata.wealth_dynamics_spec.
    """
    from lifecycle.policy_io import save_policy_bundle, load_policy_bundle

    C = np.zeros((2, 1, 1, 3))
    S = np.zeros_like(C)
    B = np.zeros_like(C)

    bundle_dir = tmp_path / "ccv_bundle"
    save_policy_bundle(
        bundle_dir, C, S, B,
        run_config={"solver_config": {"wealth_dynamics_spec": "ccv_log",
                                       "delta_bequest": -1.0}},
    )

    _, _, _, _, meta = load_policy_bundle(bundle_dir)
    assert meta["wealth_dynamics_spec"] == "ccv_log"


def test_bundle_metadata_defaults_to_simple_clamp(tmp_path):
    """Legacy bundle (no run_config) tags as 'simple_clamp'."""
    from lifecycle.policy_io import save_policy_bundle, load_policy_bundle

    C = np.zeros((2, 1, 1, 3))
    S = np.zeros_like(C)
    B = np.zeros_like(C)

    bundle_dir = tmp_path / "legacy_bundle"
    save_policy_bundle(bundle_dir, C, S, B)

    _, _, _, _, meta = load_policy_bundle(bundle_dir)
    assert meta["wealth_dynamics_spec"] == "simple_clamp"


def test_simulator_default_is_ccv_log():
    """simulate_lifecycle's wealth_dynamics_spec default is 'ccv_log' since
    CCV became the production spec. Callers working with legacy simple+clamp
    bundles MUST pass 'simple_clamp' explicitly. There is no auto-detection;
    diagnostics enforce parity by reading the bundle's metadata tag.
    """
    import inspect
    from lifecycle.simulation import simulate_lifecycle

    sig = inspect.signature(simulate_lifecycle)
    assert "wealth_dynamics_spec" in sig.parameters
    assert sig.parameters["wealth_dynamics_spec"].default == "ccv_log"
