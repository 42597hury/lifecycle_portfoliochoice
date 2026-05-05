from __future__ import annotations

import sys

import numpy as np
import pytest

from lifecycle.inf_horizon_solver import (
    _markowitz_cold_start,
    compile_inner_kernel_smoke_test,
    extract_policy_at_point,
    run_infinite_horizon_solver,
)
from lifecycle.model import DiscretizationConfig, SolveControl, SolverConfig
from lifecycle.precompute import Precompute, build_model
from lifecycle.solver import run_lifecycle_solver
from lifecycle.var import build_nominal_system1_var_config_hardcoded


def _reference_base_config() -> dict:
    return {
        "beta": 0.96,
        "gamma": 3.0,
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


def _build_small_problem():
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(_reference_base_config(), var_config, verbose=False)
    disc = DiscretizationConfig(
        n_wealth=10,
        n_savings=10,
        wealth_max=60.0,
        savings_max=60.0,
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


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Dedicated nested-JIT smoke helper is unstable under pytest on Windows; "
    "the end-to-end solver test below still exercises the real outer-core path.",
)
def test_compile_inner_kernel_smoke():
    model, pc = _build_small_problem()
    result = compile_inner_kernel_smoke_test(
        model,
        pc,
        solver_config=SolverConfig(),
        verbose=False,
    )

    assert result["compiled"] is True
    assert result["shape"] == (pc.n_z, pc.N_state, pc.n_w)


def test_markowitz_cold_start_properties():
    model, pc = _build_small_problem()
    C_init, S_init, B_init = _markowitz_cold_start(model, pc)

    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    assert C_init.shape == expected_shape
    assert S_init.shape == expected_shape
    assert B_init.shape == expected_shape
    assert C_init.dtype == np.float64
    assert S_init.dtype == np.float64
    assert B_init.dtype == np.float64

    assert C_init.flags["C_CONTIGUOUS"]
    assert S_init.flags["C_CONTIGUOUS"]
    assert B_init.flags["C_CONTIGUOUS"]

    assert np.all(np.isfinite(C_init))
    assert np.all(np.isfinite(S_init))
    assert np.all(np.isfinite(B_init))

    assert np.all(S_init >= 0.0)
    assert np.all(B_init >= 0.0)
    assert np.all(S_init + B_init <= 1.0 + 1e-12)

    xi_expected = 1.0 - float(model.beta)
    xi_actual = C_init / pc.wealth_grid[None, None, :]
    assert np.allclose(xi_actual, xi_expected, atol=1e-12)

    for i_s in range(pc.N_state):
        s_slice_S = S_init[:, i_s, :]
        s_slice_B = B_init[:, i_s, :]
        assert np.ptp(s_slice_S) < 1e-15
        assert np.ptp(s_slice_B) < 1e-15

    s_per_state = S_init[0, :, 0]
    b_per_state = B_init[0, :, 0]
    assert np.ptp(s_per_state) > 1e-6, "Stock share should vary across states"
    assert np.ptp(b_per_state) > 1e-6, "Bond share should vary across states"


def test_infinite_horizon_solver_converges_from_markowitz_cold_start():
    model, pc = _build_small_problem()
    solver_config = SolverConfig()

    solve_control = SolveControl(youngest_age_to_solve=model.retire_age)
    C_mat, S_mat, B_mat, _ = run_lifecycle_solver(
        model,
        pc,
        solver_config=solver_config,
        solve_control=solve_control,
        verbose=0,
    )

    C_warm, S_warm, B_warm, _ = run_infinite_horizon_solver(
        model,
        pc,
        solver_config=solver_config,
        warm_start_c=C_mat,
        warm_start_s=S_mat,
        warm_start_b=B_mat,
        tol=5e-7,
        max_iter=500,
        damping=1.0,
        trim_wealth_points=2,
        verbose=False,
    )

    C_cold, S_cold, B_cold, diag_cold = run_infinite_horizon_solver(
        model,
        pc,
        solver_config=solver_config,
        tol=5e-7,
        max_iter=500,
        damping=1.0,
        trim_wealth_points=2,
        verbose=False,
    )

    assert diag_cold["converged"] is True
    assert diag_cold["n_iter"] >= 1
    assert np.all(np.isfinite(C_cold))
    assert np.all(np.isfinite(S_cold))
    assert np.all(np.isfinite(B_cold))
    assert diag_cold["final_stopping_supnorm"] < 5e-7
    assert diag_cold["max_z_slice_diff_c"] < 1e-5
    assert diag_cold["max_z_slice_diff_s"] < 5e-5
    assert diag_cold["max_z_slice_diff_b"] < 5e-5
    assert diag_cold["max_xi_spread_across_w"] < 5e-2
    assert diag_cold["max_share_spread_across_w"] < 5e-2
    assert np.isfinite(diag_cold["stability_proxy"])
    assert diag_cold["stability_proxy"] < 1.0

    cross_tol = 5e-4
    trim = 2
    diff_C = np.max(
        np.abs(C_warm[:, :, trim:] - C_cold[:, :, trim:])
        / np.maximum(np.abs(C_warm[:, :, trim:]), 1.0)
    )
    diff_S = np.max(np.abs(S_warm[:, :, trim:] - S_cold[:, :, trim:]))
    diff_B = np.max(np.abs(B_warm[:, :, trim:] - B_cold[:, :, trim:]))

    assert diff_C < cross_tol, f"C diverges between cold and warm: {diff_C}"
    assert diff_S < cross_tol, f"S diverges between cold and warm: {diff_S}"
    assert diff_B < cross_tol, f"B diverges between cold and warm: {diff_B}"

    point = extract_policy_at_point(
        C_cold,
        S_cold,
        B_cold,
        i_z=pc.n_z // 2,
        i_s=pc.N_state // 2,
        i_w=pc.n_w // 2,
    )
    assert np.isfinite(point["consumption"])
    assert np.isfinite(point["alpha_stock"])
    assert np.isfinite(point["alpha_bond"])
    assert np.isfinite(point["alpha_bill"])
