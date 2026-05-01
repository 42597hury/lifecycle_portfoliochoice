from __future__ import annotations

import sys

import numpy as np
import pytest

from inf_horizon_solver import (
    compile_inner_kernel_smoke_test,
    extract_policy_at_point,
    run_infinite_horizon_solver,
)
from model import DiscretizationConfig, SolveControl, SolverConfig
from precompute import Precompute, build_model
from solver import run_lifecycle_solver
from var import build_nominal_system1_var_config_hardcoded


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
        state_grid_mode="principal",
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


def test_infinite_horizon_solver_converges_from_lifecycle_warm_start():
    model, pc = _build_small_problem()
    solver_config = SolverConfig()

    solve_control = SolveControl(youngest_age_to_solve=model.retire_age)
    C_mat, S_mat, B_mat, lifecycle_diag = run_lifecycle_solver(
        model,
        pc,
        solver_config=solver_config,
        solve_control=solve_control,
        verbose=0,
    )

    assert lifecycle_diag["youngest_solved_age"] == model.retire_age

    C_inf, S_inf, B_inf, diag = run_infinite_horizon_solver(
        model,
        pc,
        solver_config=solver_config,
        warm_start_c=C_mat,
        warm_start_s=S_mat,
        warm_start_b=B_mat,
        tol=1e-5,
        max_iter=300,
        damping=1.0,
        trim_wealth_points=2,
        verbose=False,
    )

    assert diag["converged"] is True
    assert diag["n_iter"] >= 1
    assert np.all(np.isfinite(C_inf))
    assert np.all(np.isfinite(S_inf))
    assert np.all(np.isfinite(B_inf))
    assert diag["final_stopping_supnorm"] < 1e-5
    assert diag["max_z_slice_diff_c"] < 1e-5
    assert diag["max_z_slice_diff_s"] < 5e-5
    assert diag["max_z_slice_diff_b"] < 5e-5
    assert diag["max_xi_spread_across_w"] < 1e-4
    assert diag["max_share_spread_across_w"] < 1e-3
    assert np.isfinite(diag["stability_proxy"])
    assert diag["stability_proxy"] < 1.0

    point = extract_policy_at_point(
        C_inf,
        S_inf,
        B_inf,
        i_z=pc.n_z // 2,
        i_s=pc.N_state // 2,
        i_w=pc.n_w // 2,
    )
    assert np.isfinite(point["consumption"])
    assert np.isfinite(point["alpha_stock"])
    assert np.isfinite(point["alpha_bond"])
    assert np.isfinite(point["alpha_bill"])
