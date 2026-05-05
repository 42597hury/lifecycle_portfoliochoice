"""Math-only unit tests for the Campbell-Viceira (CCV w8566) log-wealth
specification of the portfolio FOC and Jacobian.

These tests do NOT require a solved policy bundle. They probe the FOC kernel
directly with hand-constructed quadrature inputs and verify:

    1. r_p^CVC reduces correctly at corner alphas (e_s, e_b, 0).
    2. dr_p/dalpha matches the corrected formula (1/2 Jensen, not 1).
    3. Hessian-of-V Jacobian agrees with finite differences (FD).
    4. Jacobian is symmetric (Schwarz: J_sb == J_bs).
    5. Simple+clamp branch unchanged when use_ccv=False.

Also does a §4.1-§4.3 spot check of the r_p formula at corner alphas.
"""

import math
import numpy as np
import pytest

from lifecycle.solver import compute_terminal_foc_jac_shifted


# ---- Fixed inputs for reproducibility ----------------------------------------

GAMMA = 5.0
B_BAR = 10
DELTA = 0.005
A_IS = 4.0
S_VAL = 1.0
SIGMA2_XR = 0.025
SIGMA2_XB = 0.005
SIGMA_XRXB = 0.001


def _scalar_kernel(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b, use_ccv):
    """Helper: call compute_terminal_foc_jac_shifted with single (state, ret) node."""
    log_R_bill_arr = np.array([[log_R_bill]])
    log_x_s_arr = np.array([[log_x_s]])
    log_x_b_arr = np.array([[log_x_b]])
    Rx_bill = np.exp(log_R_bill_arr)
    Rx_stock_mult = np.exp(log_x_s_arr)
    Rx_bond_mult = np.exp(log_x_b_arr)
    state_weights = np.array([1.0])
    ret_weights = np.array([1.0])
    return compute_terminal_foc_jac_shifted(
        alpha_s, alpha_b, S_VAL, A_IS,
        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
        log_R_bill_arr, log_x_s_arr, log_x_b_arr,
        SIGMA2_XR, SIGMA2_XB, SIGMA_XRXB, use_ccv,
        GAMMA, B_BAR, DELTA,
    )


# ---- §4.1-§4.3: r_p formula at corner alphas ---------------------------------

def test_rp_at_full_stock():
    """alpha=(1, 0): r_p = r_bill + log_x_s (Jensen and Ito cancel)."""
    log_R_bill, log_x_s = 0.02, 0.05
    _, _, _, _, _, V_dot = _scalar_kernel(1.0, 0.0, log_R_bill, log_x_s, 0.01, True)
    expected_R_p = math.exp(log_R_bill + log_x_s)
    expected_mu = B_BAR * (S_VAL * expected_R_p / A_IS + DELTA) ** (-GAMMA) / A_IS
    expected_V_dot = expected_mu * expected_R_p
    assert abs(V_dot - expected_V_dot) < 1e-10


def test_rp_at_full_bond():
    """alpha=(0, 1): r_p = r_bill + log_x_b."""
    log_R_bill, log_x_b = 0.02, 0.01
    _, _, _, _, _, V_dot = _scalar_kernel(0.0, 1.0, log_R_bill, 0.05, log_x_b, True)
    expected_R_p = math.exp(log_R_bill + log_x_b)
    expected_mu = B_BAR * (S_VAL * expected_R_p / A_IS + DELTA) ** (-GAMMA) / A_IS
    assert abs(V_dot - expected_mu * expected_R_p) < 1e-10


def test_rp_at_all_bills():
    """alpha=(0, 0): r_p = r_bill (no Jensen/Ito at corner)."""
    log_R_bill = 0.02
    _, _, _, _, _, V_dot = _scalar_kernel(0.0, 0.0, log_R_bill, 0.05, 0.01, True)
    expected_R_p = math.exp(log_R_bill)
    expected_mu = B_BAR * (S_VAL * expected_R_p / A_IS + DELTA) ** (-GAMMA) / A_IS
    assert abs(V_dot - expected_mu * expected_R_p) < 1e-10


# ---- Gradient at corner alphas (corrected 1/2 Jensen) ------------------------

def test_dr_da_at_zero_alpha():
    """alpha=(0, 0): dr_p/dalpha_s = log_x_s + sigma2_xr/2."""
    log_R_bill, log_x_s, log_x_b = 0.02, 0.05, 0.01
    foc_s, foc_b, _, _, _, V_dot = _scalar_kernel(0.0, 0.0, log_R_bill, log_x_s, log_x_b, True)
    R_p = math.exp(log_R_bill)
    mu = B_BAR * (S_VAL * R_p / A_IS + DELTA) ** (-GAMMA) / A_IS
    expected_dr_da_s = log_x_s + SIGMA2_XR * 0.5
    expected_dr_da_b = log_x_b + SIGMA2_XB * 0.5
    assert abs(foc_s - mu * R_p * expected_dr_da_s) < 1e-10
    assert abs(foc_b - mu * R_p * expected_dr_da_b) < 1e-10


def test_dr_da_at_full_stock():
    """alpha=(1, 0): dr_p/dalpha_s = log_x_s + sigma2_xr*(1/2 - 1) = log_x_s - sigma2_xr/2.

    This is the corner-of-handoff-bug check: the original handoff had
    sigma2_xr*(1 - alpha_s), giving log_x_s - sigma2_xr; the corrected formula
    halves the Jensen term to sigma2_xr*(1/2 - alpha_s).
    """
    log_R_bill, log_x_s = 0.02, 0.05
    foc_s, _, _, _, _, _ = _scalar_kernel(1.0, 0.0, log_R_bill, log_x_s, 0.01, True)
    R_p = math.exp(log_R_bill + log_x_s)
    mu = B_BAR * (S_VAL * R_p / A_IS + DELTA) ** (-GAMMA) / A_IS
    expected_dr_da_s = log_x_s + SIGMA2_XR * (0.5 - 1.0)  # = log_x_s - SIGMA2_XR/2
    assert abs(foc_s - mu * R_p * expected_dr_da_s) < 1e-10


# ---- Hessian-of-V Jacobian: FD agreement and symmetry ------------------------

def _multinode_kernel(alpha_s, alpha_b):
    """4x4 quadrature with random nodes for FD checks."""
    np.random.seed(0)
    n_state, n_ret = 4, 4
    state_weights = np.random.dirichlet(np.ones(n_state)).reshape(-1)
    ret_weights = np.random.dirichlet(np.ones(n_ret)).reshape(-1)
    log_R_bill_arr = np.random.normal(0.02, 0.03, (n_state, n_ret))
    log_x_s_arr = np.random.normal(0.05, 0.15, (n_state, n_ret))
    log_x_b_arr = np.random.normal(0.01, 0.05, (n_state, n_ret))
    Rx_bill = np.exp(log_R_bill_arr)
    Rx_stock_mult = np.exp(log_x_s_arr)
    Rx_bond_mult = np.exp(log_x_b_arr)
    return compute_terminal_foc_jac_shifted(
        alpha_s, alpha_b, S_VAL, A_IS,
        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
        log_R_bill_arr, log_x_s_arr, log_x_b_arr,
        SIGMA2_XR, SIGMA2_XB, SIGMA_XRXB, True,
        GAMMA, B_BAR, DELTA,
    )


@pytest.mark.parametrize("a_s, a_b", [
    (0.0, 0.0), (0.5, 0.3), (1.0, 0.0), (0.8, -0.2), (1.5, 0.5),
])
def test_jacobian_fd_agreement(a_s, a_b):
    """Analytical Jacobian agrees with finite-difference to 1e-6."""
    fs, fb, Jss, Jbb, Jsb, _ = _multinode_kernel(a_s, a_b)
    h = 1e-6
    fs_p, fb_p, _, _, _, _ = _multinode_kernel(a_s + h, a_b)
    fs_m, fb_m, _, _, _, _ = _multinode_kernel(a_s - h, a_b)
    Jss_fd = (fs_p - fs_m) / (2 * h)
    Jbs_fd = (fb_p - fb_m) / (2 * h)

    fs_p, fb_p, _, _, _, _ = _multinode_kernel(a_s, a_b + h)
    fs_m, fb_m, _, _, _, _ = _multinode_kernel(a_s, a_b - h)
    Jsb_fd = (fs_p - fs_m) / (2 * h)
    Jbb_fd = (fb_p - fb_m) / (2 * h)

    rel = lambda a, fd: abs(a - fd) / max(abs(fd), 1e-12)
    assert rel(Jss, Jss_fd) < 1e-6
    assert rel(Jbb, Jbb_fd) < 1e-6
    assert rel(Jsb, Jsb_fd) < 1e-6
    # Jacobian symmetry (Schwarz): both off-diagonal FDs should match analytical Jsb.
    assert rel(Jsb, Jbs_fd) < 1e-6


# ---- Backward-compat: simple+clamp branch unchanged --------------------------

def test_simple_clamp_branch_at_full_stock():
    """use_ccv=False reproduces the simple+clamp result at alpha=(1, 0)."""
    log_R_bill, log_x_s = 0.02, 0.05
    foc_s, _, _, _, _, V_dot = _scalar_kernel(1.0, 0.0, log_R_bill, log_x_s, 0.01, False)
    R_p = math.exp(log_R_bill + log_x_s)  # alpha=(1,0): R_p = R_s
    R_bill = math.exp(log_R_bill)
    mu = B_BAR * (S_VAL * R_p / A_IS + DELTA) ** (-GAMMA) / A_IS
    Rex_s = R_p - R_bill
    expected_V_dot = mu * R_p
    expected_foc_s = mu * Rex_s   # E[mu * (R_s - R_bill)]
    assert abs(V_dot - expected_V_dot) < 1e-10
    assert abs(foc_s - expected_foc_s) < 1e-10
