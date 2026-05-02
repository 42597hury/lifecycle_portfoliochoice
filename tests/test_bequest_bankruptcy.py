"""Tests for the bequest=0-on-bankruptcy primitive in the FOC kernels.

Two scenarios exercised against `compute_foc_jac_retirement_quad`:
  1. Solvent equivalence — when every quadrature node has s*R_p > 0, the
     patched kernel must agree with a hand-coded reference integrand.
  2. Bankruptcy branch — when at least one node has s*R_p <= 0, the kernel
     must (a) return finite, non-explosive numbers, (b) suppress the bequest
     contribution at the bankrupt node, and (c) still pick up the alive
     contribution at that node from x_next = pension_next_scalar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver import compute_foc_jac_retirement_quad


# ─── Reference Python integrand ──────────────────────────────────────────
def _reference_retirement_foc(
    *,
    alpha_s,
    alpha_b,
    s_val,
    pension_next_scalar,
    annuity_factor_is,
    c_grid_y,
    wealth_grid,
    exp_ret_bill,
    exp_ret_stock,
    exp_ret_bond,
    ret_weights,
    base_mu_r_i,
    v_weights,
    M_v_nodes,
    gamma,
    psi,
    b_bar,
):
    """Plain Python reimplementation of the retirement integrand for a
    single state-quadrature node and a 1-point state grid (so trilinear
    interpolation collapses to a single c_next_full[0, :] lookup).
    """
    a_bill = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi
    foc_s = foc_b = 0.0
    J_ss = J_bb = J_sb = 0.0
    euler_sum = 0.0

    n_state_quad = len(v_weights)
    n_ret_quad = len(ret_weights)
    for k_v in range(n_state_quad):
        w_v = v_weights[k_v]
        exp_mu_bill = np.exp(base_mu_r_i[0] + M_v_nodes[k_v, 0])
        exp_mu_s = np.exp(base_mu_r_i[1] + M_v_nodes[k_v, 1])
        exp_mu_b = np.exp(base_mu_r_i[2] + M_v_nodes[k_v, 2])
        for k_r in range(n_ret_quad):
            weight = w_v * ret_weights[k_r]
            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            sR_p = s_val * R_p
            if sR_p > 0.0:
                x_next = sR_p + pension_next_scalar
            else:
                x_next = pension_next_scalar

            # Linear interpolation (matches fast_interp_1d_with_slope on a
            # uniform 2-point grid; mpc = constant slope).
            c_next, mpc = _linear_interp_with_slope(x_next, wealth_grid, c_grid_y)

            mu_alive = c_next ** (-gamma)
            mup_alive = -gamma * mu_alive / c_next * mpc
            if sR_p > 0.0:
                w_A = sR_p / annuity_factor_is
                mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
                mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
            else:
                mu_bequest = 0.0
                mup_bequest = 0.0
            mu_comb = psi * mu_alive + prob_death * mu_bequest
            mup_comb = psi * mup_alive + prob_death * mup_bequest

            wmu = weight * mu_comb
            wmup = weight * mup_comb
            euler_sum += wmu * R_p
            foc_s += wmu * Rex_s
            foc_b += wmu * Rex_b
            jac = wmup * s_val
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum


def _linear_interp_with_slope(x, grid, y):
    n = len(grid)
    if x <= grid[0]:
        slope = (y[1] - y[0]) / (grid[1] - grid[0])
        return y[0] + slope * (x - grid[0]), slope
    if x >= grid[-1]:
        slope = (y[-1] - y[-2]) / (grid[-1] - grid[-2])
        return y[-1] + slope * (x - grid[-1]), slope
    for i in range(n - 1):
        if grid[i + 1] >= x:
            slope = (y[i + 1] - y[i]) / (grid[i + 1] - grid[i])
            return y[i] + slope * (x - grid[i]), slope
    raise RuntimeError("interp failed")


# ─── Common minimal inputs ───────────────────────────────────────────────
def _build_minimal_inputs(
    *,
    alpha_s,
    alpha_b,
    s_val,
    exp_ret_bill,
    exp_ret_stock,
    exp_ret_bond,
    ret_weights,
):
    """Build the kernel-arg dict for a 1-state-quad / 1-state-grid setup."""
    wealth_grid = np.linspace(0.0, 100.0, 11, dtype=np.float64)
    # Linear continuation policy: c(w) = 0.05*w + 0.5 (constant MPC 0.05).
    c_grid_y = 0.05 * wealth_grid + 0.5
    # 1-point state grid → trilinear interp collapses to a single value.
    c_next_full = c_grid_y.reshape(1, -1).astype(np.float64)

    grids_0 = np.array([0.0], dtype=np.float64)
    grids_1 = np.array([0.0], dtype=np.float64)
    grids_2 = np.array([0.0], dtype=np.float64)
    state_bracket_shift = np.zeros(3, dtype=np.float64)
    state_bracket_L_inv = np.eye(3, dtype=np.float64)
    Phi_0_state = np.zeros(3, dtype=np.float64)
    Phi_11 = np.zeros((3, 3), dtype=np.float64)
    state_grid_i = np.zeros(3, dtype=np.float64)

    v_nodes = np.zeros((1, 3), dtype=np.float64)
    v_weights = np.array([1.0], dtype=np.float64)
    M_v_nodes = np.zeros((1, 3), dtype=np.float64)
    base_mu_r_i = np.zeros(3, dtype=np.float64)

    return dict(
        alpha_s=alpha_s,
        alpha_b=alpha_b,
        s_val=s_val,
        z_idx=0,
        i_s=0,
        wealth_grid=wealth_grid,
        c_next_full=c_next_full,
        pension_next_scalar=2.0,
        annuity_factor_is=8.0,
        v_nodes=v_nodes,
        v_weights=v_weights,
        M_v_nodes=M_v_nodes,
        base_mu_r_i=base_mu_r_i,
        Phi_0_state=Phi_0_state,
        Phi_11=Phi_11,
        state_grid_i=state_grid_i,
        state_bracket_shift=state_bracket_shift,
        state_bracket_L_inv=state_bracket_L_inv,
        grids_0=grids_0,
        grids_1=grids_1,
        grids_2=grids_2,
        N1=1,
        N2=1,
        exp_ret_bill=np.asarray(exp_ret_bill, dtype=np.float64),
        exp_ret_stock=np.asarray(exp_ret_stock, dtype=np.float64),
        exp_ret_bond=np.asarray(exp_ret_bond, dtype=np.float64),
        ret_weights=np.asarray(ret_weights, dtype=np.float64),
        gamma=5.0,
        psi=0.97,
        beta=0.96,
        b_bar=10,
    )


# ─── Test 1: solvent equivalence ─────────────────────────────────────────
def test_solvent_equivalence():
    """When every node has s*R_p > 0, the kernel matches the hand-coded
    integrand to ~machine precision."""
    inputs = _build_minimal_inputs(
        alpha_s=0.5,
        alpha_b=0.3,
        s_val=10.0,
        # Two return nodes, both keep R_p > 0.
        exp_ret_bill=[1.0, 1.05],
        exp_ret_stock=[1.10, 0.95],
        exp_ret_bond=[1.05, 1.00],
        ret_weights=[0.5, 0.5],
    )

    # Sanity: every node solvent under these params.
    a_bill = 1.0 - inputs["alpha_s"] - inputs["alpha_b"]
    for k in range(2):
        R_bill = inputs["exp_ret_bill"][k]
        R_s = R_bill * inputs["exp_ret_stock"][k]
        R_b = R_bill * inputs["exp_ret_bond"][k]
        R_p = inputs["alpha_s"] * R_s + inputs["alpha_b"] * R_b + a_bill * R_bill
        assert inputs["s_val"] * R_p > 0.0, f"node {k} unexpectedly bankrupt"

    out_kernel = compute_foc_jac_retirement_quad(**inputs)

    ref_inputs = dict(inputs)
    ref_inputs.pop("z_idx"); ref_inputs.pop("i_s")
    ref_inputs.pop("Phi_0_state"); ref_inputs.pop("Phi_11"); ref_inputs.pop("state_grid_i")
    ref_inputs.pop("state_bracket_shift"); ref_inputs.pop("state_bracket_L_inv")
    ref_inputs.pop("grids_0"); ref_inputs.pop("grids_1"); ref_inputs.pop("grids_2")
    ref_inputs.pop("N1"); ref_inputs.pop("N2")
    ref_inputs.pop("v_nodes"); ref_inputs.pop("beta")
    ref_inputs["c_grid_y"] = ref_inputs["c_next_full"][0, :]
    ref_inputs.pop("c_next_full")

    out_ref = _reference_retirement_foc(**ref_inputs)

    for name, a, b in zip(
        ("foc_s", "foc_b", "J_ss", "J_bb", "J_sb", "euler_sum"),
        out_kernel, out_ref,
    ):
        assert np.isclose(a, b, rtol=1e-12, atol=1e-14), (
            f"{name}: kernel={a!r} ref={b!r}"
        )


# ─── Test 2: bankruptcy branch ───────────────────────────────────────────
def test_bankruptcy_branch_finite_and_zero_bequest():
    """At a node where s*R_p <= 0, the kernel must (a) return finite values,
    (b) zero out the bequest contribution, and (c) still pick up the alive
    contribution evaluated at x_next = pension_next_scalar."""
    # Leveraged: alpha_s = 5 (long stock 5x), alpha_b = 0, a_bill = -4.
    # Node 0: stock up,   solvent.
    # Node 1: stock down to half, bankrupt: R_p = 5*0.5 - 4*1.0 = -1.5.
    inputs = _build_minimal_inputs(
        alpha_s=5.0,
        alpha_b=0.0,
        s_val=10.0,
        exp_ret_bill=[1.0, 1.0],
        exp_ret_stock=[1.10, 0.50],
        exp_ret_bond=[1.0, 1.0],
        ret_weights=[0.5, 0.5],
    )

    a_bill = 1.0 - inputs["alpha_s"] - inputs["alpha_b"]
    R_p_node1 = (
        inputs["alpha_s"] * inputs["exp_ret_bill"][1] * inputs["exp_ret_stock"][1]
        + inputs["alpha_b"] * inputs["exp_ret_bill"][1] * inputs["exp_ret_bond"][1]
        + a_bill * inputs["exp_ret_bill"][1]
    )
    assert R_p_node1 < 0.0, f"node 1 should be bankrupt, R_p={R_p_node1}"

    out_kernel = compute_foc_jac_retirement_quad(**inputs)

    # All outputs finite (no 1e51-scale explosions).
    for name, val in zip(("foc_s", "foc_b", "J_ss", "J_bb", "J_sb", "euler_sum"), out_kernel):
        assert np.isfinite(val), f"{name} not finite: {val!r}"
        assert abs(val) < 1e8, f"{name} suspiciously large: {val!r}"

    # Independent reference computation matches.
    ref_inputs = dict(inputs)
    ref_inputs.pop("z_idx"); ref_inputs.pop("i_s")
    ref_inputs.pop("Phi_0_state"); ref_inputs.pop("Phi_11"); ref_inputs.pop("state_grid_i")
    ref_inputs.pop("state_bracket_shift"); ref_inputs.pop("state_bracket_L_inv")
    ref_inputs.pop("grids_0"); ref_inputs.pop("grids_1"); ref_inputs.pop("grids_2")
    ref_inputs.pop("N1"); ref_inputs.pop("N2")
    ref_inputs.pop("v_nodes"); ref_inputs.pop("beta")
    ref_inputs["c_grid_y"] = ref_inputs["c_next_full"][0, :]
    ref_inputs.pop("c_next_full")
    out_ref = _reference_retirement_foc(**ref_inputs)
    for name, a, b in zip(
        ("foc_s", "foc_b", "J_ss", "J_bb", "J_sb", "euler_sum"),
        out_kernel, out_ref,
    ):
        assert np.isclose(a, b, rtol=1e-12, atol=1e-14), (
            f"{name}: kernel={a!r} ref={b!r}"
        )

    # Sanity: the bankrupt-node alive contribution evaluated at
    # x_next = pension_next_scalar yields a finite c_next.
    pension = inputs["pension_next_scalar"]
    c_next_at_pension = 0.05 * pension + 0.5
    assert np.isfinite(c_next_at_pension ** (-inputs["gamma"]))


if __name__ == "__main__":
    test_solvent_equivalence()
    test_bankruptcy_branch_finite_and_zero_bequest()
    print("all tests passed")
