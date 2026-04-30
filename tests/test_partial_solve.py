from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import DiscretizationConfig, SolveControl, SolverConfig
from policy_io import load_policy_bundle
from precompute import Precompute, build_model
from solver import run_lifecycle_solver
from var import build_nominal_system1_var_config


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


def _build_small_problem():
    var_config, _, _ = build_nominal_system1_var_config(
        csv_path=str(ROOT / "data" / "var_dataset.csv")
    )
    model = build_model(_reference_base_config(), var_config, verbose=False)
    disc = DiscretizationConfig(
        n_wealth=12,
        n_savings=12,
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


def test_partial_solve_checkpoint_bundle(tmp_path):
    model, pc = _build_small_problem()
    checkpoint_dir = tmp_path / "debug_retirement_tail"
    solve_control = SolveControl(
        youngest_age_to_solve=95,
        checkpoint_path=str(checkpoint_dir),
        checkpoint_every_n_ages=1,
        save_on_interrupt=True,
        return_partial_on_interrupt=True,
    )

    C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(
        model,
        pc,
        solver_config=SolverConfig(),
        solve_control=solve_control,
        verbose=0,
    )

    solved_mask = diagnostics["solved_age_mask"]
    expected_mask = pc.ages >= 95

    assert diagnostics["solve_status"] == "stopped_early"
    assert diagnostics["is_partial"] is True
    assert diagnostics["youngest_solved_age"] == 95
    assert diagnostics["oldest_solved_age"] == int(pc.ages[-1])
    assert diagnostics["n_ages_solved"] == int(expected_mask.sum())
    assert diagnostics["checkpoint_save_count"] >= 1
    assert diagnostics["last_saved_bundle_path"] == str(checkpoint_dir)
    assert np.array_equal(solved_mask, expected_mask)

    assert np.all(np.isfinite(C_mat[solved_mask]))
    assert np.all(np.isfinite(S_mat[solved_mask]))
    assert np.all(np.isfinite(B_mat[solved_mask]))
    assert np.all(np.isnan(C_mat[~solved_mask]))
    assert np.all(np.isnan(S_mat[~solved_mask]))
    assert np.all(np.isnan(B_mat[~solved_mask]))

    C_loaded, S_loaded, B_loaded, diag_loaded, meta = load_policy_bundle(checkpoint_dir)
    assert meta["shape"] == list(C_mat.shape)
    assert diag_loaded["solve_status"] == "stopped_early"
    assert np.array_equal(diag_loaded["solved_age_mask"], solved_mask)
    assert np.all(np.isnan(C_loaded[~solved_mask]))
    assert np.all(np.isnan(S_loaded[~solved_mask]))
    assert np.all(np.isnan(B_loaded[~solved_mask]))
