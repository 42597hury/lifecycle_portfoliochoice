"""FD-Jacobian tests for retirement and working-age FOC kernels under CVC.

These tests use a small Precompute to construct realistic kernel inputs (state
grid, quadrature, conditional return means), then probe the kernel's gradient
against finite differences. Tolerance: 1e-6.

Mirrors the math-only tests in test_cvc_kernels.py but exercises the full
retirement/working kernels with their state-quadrature + return-quadrature
loops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import Precompute, build_model
from lifecycle.solver import (
    compute_foc_jac_retirement_quad,
    compute_foc_jac_working_quad,
)
from lifecycle.var import build_nominal_system1_var_config


def _reference_base_config() -> dict:
    return {
        "beta": 0.96,
        "gamma": 5.0,
        "b_bar": 10,
        "start_age": 22,
        "retire_age": 67,
        "terminal_age": 99,
        "b0": -6.142,
        "b1": 0.3040,
        "b2": -0.051,
        "b3": 0.002586,
        "rho": 0.991,
        "pz": 0.176,
        "mu_eta1": -0.524,
        "sigma_eta1": 0.113,
        "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524),
        "sigma_eta2": 0.046,
        "pe": 0.044,
        "mu_eps1": 0.134,
        "sigma_eps1": 0.762,
        "mu_eps2": 0.0,
        "sigma_eps2": 0.055,
        "constrained": True,
    }


@pytest.fixture(scope="module")
def small_pc():
    var_config, _, _ = build_nominal_system1_var_config(
        csv_path=str(ROOT / "data" / "var_dataset.csv")
    )
    model = build_model(_reference_base_config(), var_config, verbose=False)
    disc = DiscretizationConfig(
        n_wealth=10,
        n_savings=10,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=2.0,
        n_z=3,
        n_stds=2.0,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=2,
        n_state_quad_nodes=2,
    )
    pc = Precompute(model, disc, verbose=False)
    return model, pc


def _retirement_kernel_inputs(model, pc):
    """Build the positional args to compute_foc_jac_retirement_quad for a fixed
    representative cell. Returns a dict so tests can swap (alpha_s, alpha_b)
    cleanly while reusing everything else.
    """
    i_s = pc.N_state // 2
    z_idx = pc.n_z // 2
    s_val = 5.0
    pension_next = pc.pension_after_tax[-1, z_idx]

    # Mock c_next_full as an arbitrary smooth array of shape (N_state, n_w)
    rng = np.random.default_rng(0)
    c_next_full = 0.5 * pc.wealth_grid[None, :].repeat(pc.N_state, axis=0) + 0.05 * rng.standard_normal(
        (pc.N_state, pc.n_w)
    )
    c_next_full = np.maximum(c_next_full, 0.05)

    base_mu_r_i = pc.const_r + pc.A_r @ pc.state_grid[i_s]

    n_state_grid = len(pc.state_grid_sizes)
    grids = list(pc.state_bracket_grids) + [np.array([0.0])] * (3 - n_state_grid)
    grids_0 = grids[0]; grids_1 = grids[1]; grids_2 = grids[2]
    N1 = len(grids_1); N2 = len(grids_2)

    # Pad state_grid_i and bracketing matrices to 3D as solver does
    s_i = np.zeros(3); s_i[: pc.state_grid.shape[1]] = pc.state_grid[i_s]
    Phi_0_state = np.zeros(3); Phi_0_state[: model.Phi_0_state.shape[0]] = model.Phi_0_state
    Phi_11 = np.eye(3)
    Phi_11[: model.Phi_11.shape[0], : model.Phi_11.shape[1]] = model.Phi_11
    state_bracket_shift = np.zeros(3)
    state_bracket_shift[: pc.state_bracket_shift.shape[0]] = pc.state_bracket_shift
    state_bracket_L_inv = np.eye(3)
    state_bracket_L_inv[: pc.state_bracket_L_inv.shape[0], : pc.state_bracket_L_inv.shape[1]] = pc.state_bracket_L_inv
    v_nodes_3d = np.zeros((pc.v_nodes.shape[0], 3))
    v_nodes_3d[:, : pc.v_nodes.shape[1]] = pc.v_nodes
    base_mu_r_i_3d = base_mu_r_i

    return dict(
        s_val=s_val, z_idx=z_idx, i_s=i_s,
        wealth_grid=pc.wealth_grid, c_next_full=c_next_full,
        pension_next_scalar=float(pension_next),
        annuity_factor_is=float(pc.annuity_factors[i_s]),
        v_nodes=v_nodes_3d, v_weights=pc.v_weights, M_v_nodes=pc.M_v_nodes,
        base_mu_r_i=base_mu_r_i_3d,
        Phi_0_state=Phi_0_state, Phi_11=Phi_11, state_grid_i=s_i,
        state_bracket_shift=state_bracket_shift,
        state_bracket_L_inv=state_bracket_L_inv,
        grids_0=grids_0, grids_1=grids_1, grids_2=grids_2, N1=N1, N2=N2,
        exp_ret_bill=pc.exp_ret_bill, exp_ret_stock=pc.exp_ret_stock,
        exp_ret_bond=pc.exp_ret_bond, ret_weights=pc.ret_weights,
        ret_nodes=pc.ret_nodes,
        sigma2_xr=pc.sigma2_xr, sigma2_xb=pc.sigma2_xb, sigma_xrxb=pc.sigma_xrxb,
        gamma=model.gamma, psi=0.95, beta=model.beta, b_bar=model.b_bar,
    )


def _call_retire(alpha_s, alpha_b, kw, use_ccv):
    return compute_foc_jac_retirement_quad(
        alpha_s, alpha_b,
        kw["s_val"], kw["z_idx"], kw["i_s"],
        kw["wealth_grid"], kw["c_next_full"], kw["pension_next_scalar"],
        kw["annuity_factor_is"],
        kw["v_nodes"], kw["v_weights"], kw["M_v_nodes"],
        kw["base_mu_r_i"],
        kw["Phi_0_state"], kw["Phi_11"], kw["state_grid_i"],
        kw["state_bracket_shift"], kw["state_bracket_L_inv"],
        kw["grids_0"], kw["grids_1"], kw["grids_2"], kw["N1"], kw["N2"],
        kw["exp_ret_bill"], kw["exp_ret_stock"], kw["exp_ret_bond"], kw["ret_weights"],
        kw["ret_nodes"],
        kw["sigma2_xr"], kw["sigma2_xb"], kw["sigma_xrxb"], use_ccv,
        kw["gamma"], kw["psi"], kw["beta"], kw["b_bar"],
    )


@pytest.mark.parametrize("a_s, a_b", [
    (0.0, 0.0), (0.3, 0.2), (0.5, 0.4), (0.8, 0.1), (1.0, 0.0),
])
def test_retirement_jacobian_fd_ccv(small_pc, a_s, a_b):
    model, pc = small_pc
    kw = _retirement_kernel_inputs(model, pc)

    fs, fb, Jss, Jbb, Jsb, _ = _call_retire(a_s, a_b, kw, True)
    h = 1e-6
    fs_p, fb_p, *_ = _call_retire(a_s + h, a_b, kw, True)
    fs_m, fb_m, *_ = _call_retire(a_s - h, a_b, kw, True)
    Jss_fd = (fs_p - fs_m) / (2 * h)
    Jbs_fd = (fb_p - fb_m) / (2 * h)

    fs_p, fb_p, *_ = _call_retire(a_s, a_b + h, kw, True)
    fs_m, fb_m, *_ = _call_retire(a_s, a_b - h, kw, True)
    Jsb_fd = (fs_p - fs_m) / (2 * h)
    Jbb_fd = (fb_p - fb_m) / (2 * h)

    rel = lambda a, fd: abs(a - fd) / max(abs(fd), 1e-8)
    assert rel(Jss, Jss_fd) < 1e-5, f"Jss err {rel(Jss, Jss_fd):.2e}"
    assert rel(Jbb, Jbb_fd) < 1e-5
    assert rel(Jsb, Jsb_fd) < 1e-5
    assert rel(Jsb, Jbs_fd) < 1e-5  # symmetry


def test_retirement_simple_clamp_unchanged(small_pc):
    """Default use_ccv=False reproduces simple+clamp result; sanity check."""
    model, pc = small_pc
    kw = _retirement_kernel_inputs(model, pc)
    fs, fb, _, _, _, e = _call_retire(0.5, 0.3, kw, False)
    # Just verify the call returns finite values; the regression test on the
    # Newton solver convergence is in test_partial_solve.py.
    assert np.isfinite(fs) and np.isfinite(fb) and np.isfinite(e)


# ---- Working-age kernel ------------------------------------------------------

def _working_kernel_inputs(model, pc):
    i_s = pc.N_state // 2
    z_idx = pc.n_z // 2
    s_val = 5.0
    rng = np.random.default_rng(1)
    c_next_full = 0.5 * pc.wealth_grid[None, None, :].repeat(pc.n_z, axis=0).repeat(
        pc.N_state, axis=1
    ) + 0.05 * rng.standard_normal((pc.n_z, pc.N_state, pc.n_w))
    c_next_full = np.maximum(c_next_full, 0.05)
    log_det_next = float(pc.log_det_profile[10])
    base_mu_r_i = pc.const_r + pc.A_r @ pc.state_grid[i_s]

    grids = list(pc.state_bracket_grids) + [np.array([0.0])] * (3 - len(pc.state_grid_sizes))
    grids_0 = grids[0]; grids_1 = grids[1]; grids_2 = grids[2]
    N1 = len(grids_1); N2 = len(grids_2)

    s_i = np.zeros(3); s_i[: pc.state_grid.shape[1]] = pc.state_grid[i_s]
    Phi_0_state = np.zeros(3); Phi_0_state[: model.Phi_0_state.shape[0]] = model.Phi_0_state
    Phi_11 = np.eye(3); Phi_11[: model.Phi_11.shape[0], : model.Phi_11.shape[1]] = model.Phi_11
    state_bracket_shift = np.zeros(3)
    state_bracket_shift[: pc.state_bracket_shift.shape[0]] = pc.state_bracket_shift
    state_bracket_L_inv = np.eye(3)
    state_bracket_L_inv[: pc.state_bracket_L_inv.shape[0], : pc.state_bracket_L_inv.shape[1]] = pc.state_bracket_L_inv
    v_nodes_3d = np.zeros((pc.v_nodes.shape[0], 3))
    v_nodes_3d[:, : pc.v_nodes.shape[1]] = pc.v_nodes

    return dict(
        s_val=s_val, z_idx=z_idx, i_s=i_s,
        wealth_grid=pc.wealth_grid, c_next_full=c_next_full,
        log_det_next=log_det_next,
        annuity_factor_is=float(pc.annuity_factors[i_s]),
        z_grid=pc.z_grid, rho=model.rho,
        eta_nodes=pc.eta_nodes, eta_weights=pc.eta_weights, dz=pc.dz,
        v_nodes=v_nodes_3d, v_weights=pc.v_weights, M_v_nodes=pc.M_v_nodes,
        base_mu_r_i=base_mu_r_i,
        Phi_0_state=Phi_0_state, Phi_11=Phi_11, state_grid_i=s_i,
        state_bracket_shift=state_bracket_shift,
        state_bracket_L_inv=state_bracket_L_inv,
        grids_0=grids_0, grids_1=grids_1, grids_2=grids_2, N1=N1, N2=N2,
        exp_ret_bill=pc.exp_ret_bill, exp_ret_stock=pc.exp_ret_stock,
        exp_ret_bond=pc.exp_ret_bond, ret_weights=pc.ret_weights,
        ret_nodes=pc.ret_nodes,
        sigma2_xr=pc.sigma2_xr, sigma2_xb=pc.sigma2_xb, sigma_xrxb=pc.sigma_xrxb,
        eps_nodes=pc.eps_nodes, eps_weights=pc.eps_weights,
        gamma=model.gamma, psi=0.99, beta=model.beta, b_bar=model.b_bar,
        use_pension_next=False,
        pension_next_by_z=pc.pension_after_tax[-1, :].copy(),
    )


def _call_work(alpha_s, alpha_b, kw, use_ccv):
    return compute_foc_jac_working_quad(
        alpha_s, alpha_b,
        kw["s_val"], kw["z_idx"], kw["i_s"],
        kw["wealth_grid"], kw["c_next_full"], kw["log_det_next"],
        kw["annuity_factor_is"],
        kw["z_grid"], kw["rho"], kw["eta_nodes"], kw["eta_weights"], kw["dz"],
        kw["v_nodes"], kw["v_weights"], kw["M_v_nodes"],
        kw["base_mu_r_i"],
        kw["Phi_0_state"], kw["Phi_11"], kw["state_grid_i"],
        kw["state_bracket_shift"], kw["state_bracket_L_inv"],
        kw["grids_0"], kw["grids_1"], kw["grids_2"], kw["N1"], kw["N2"],
        kw["exp_ret_bill"], kw["exp_ret_stock"], kw["exp_ret_bond"], kw["ret_weights"],
        kw["ret_nodes"],
        kw["sigma2_xr"], kw["sigma2_xb"], kw["sigma_xrxb"], use_ccv,
        kw["eps_nodes"], kw["eps_weights"],
        kw["gamma"], kw["psi"], kw["beta"], kw["b_bar"],
        kw["use_pension_next"], kw["pension_next_by_z"],
    )


@pytest.mark.parametrize("a_s, a_b", [
    (0.0, 0.0), (0.3, 0.2), (0.5, 0.4), (0.8, 0.1), (1.0, 0.0),
])
def test_working_jacobian_fd_ccv(small_pc, a_s, a_b):
    model, pc = small_pc
    kw = _working_kernel_inputs(model, pc)

    fs, fb, Jss, Jbb, Jsb, _ = _call_work(a_s, a_b, kw, True)
    h = 1e-6
    fs_p, fb_p, *_ = _call_work(a_s + h, a_b, kw, True)
    fs_m, fb_m, *_ = _call_work(a_s - h, a_b, kw, True)
    Jss_fd = (fs_p - fs_m) / (2 * h)
    Jbs_fd = (fb_p - fb_m) / (2 * h)

    fs_p, fb_p, *_ = _call_work(a_s, a_b + h, kw, True)
    fs_m, fb_m, *_ = _call_work(a_s, a_b - h, kw, True)
    Jsb_fd = (fs_p - fs_m) / (2 * h)
    Jbb_fd = (fb_p - fb_m) / (2 * h)

    rel = lambda a, fd: abs(a - fd) / max(abs(fd), 1e-8)
    assert rel(Jss, Jss_fd) < 1e-5, f"Jss err {rel(Jss, Jss_fd):.2e}"
    assert rel(Jbb, Jbb_fd) < 1e-5
    assert rel(Jsb, Jsb_fd) < 1e-5
    assert rel(Jsb, Jbs_fd) < 1e-5  # symmetry


def test_working_simple_clamp_unchanged(small_pc):
    model, pc = small_pc
    kw = _working_kernel_inputs(model, pc)
    fs, fb, _, _, _, e = _call_work(0.5, 0.3, kw, False)
    assert np.isfinite(fs) and np.isfinite(fb) and np.isfinite(e)
