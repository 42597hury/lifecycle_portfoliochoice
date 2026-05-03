from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.model import DiscretizationConfig, SolveControl, SolverConfig
from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import Precompute, build_model
from lifecycle.solver import _build_progress_wealth_schedule, _normalize_solve_control, run_lifecycle_solver
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


def test_progress_wealth_schedule_uses_scf_median_with_linear_age_interpolation():
    ages = np.array([22, 30, 35, 40, 50, 80, 99], dtype=np.int64)
    w_grid = np.geomspace(1e-4, 200.0, 12)

    schedule, label = _build_progress_wealth_schedule(ages, w_grid, "scf_median")
    awi_kusd = 54.09999

    expected_age30 = 39.0 / awi_kusd
    expected_age40 = 135.6 / awi_kusd
    expected_age35 = 0.5 * (expected_age30 + expected_age40)
    expected_age50 = 247.2 / awi_kusd
    expected_age80 = 335.6 / awi_kusd

    assert "SCF median wealth" in label
    assert schedule[0] == pytest.approx(expected_age30)
    assert schedule[1] == pytest.approx(expected_age30)
    assert schedule[2] == pytest.approx(expected_age35)
    assert schedule[3] == pytest.approx(expected_age40)
    assert schedule[4] == pytest.approx(expected_age50)
    assert schedule[5] == pytest.approx(expected_age80)
    assert schedule[6] == pytest.approx(expected_age80)


def test_progress_wealth_schedule_grid_midpoint_matches_legacy_constant_probe():
    ages = np.array([22, 45, 80], dtype=np.int64)
    w_grid = np.geomspace(1e-4, 200.0, 12)

    schedule, label = _build_progress_wealth_schedule(ages, w_grid, "grid_midpoint")
    expected = float(w_grid[len(w_grid) // 2])

    assert "legacy constant probe" in label
    assert np.allclose(schedule, expected)


def test_normalize_solve_control_accepts_legacy_shape_and_defaults_progress_source():
    model, pc = _build_small_problem()

    class LegacySolveControl:
        youngest_age_to_solve = 95
        checkpoint_path = None
        checkpoint_every_n_ages = 1
        save_on_interrupt = True
        return_partial_on_interrupt = False

    solve_control, control_active = _normalize_solve_control(model, pc, LegacySolveControl())

    assert control_active is True
    assert solve_control.youngest_age_to_solve == 95
    assert solve_control.checkpoint_every_n_ages == 1
    assert solve_control.save_on_interrupt is True
    assert solve_control.progress_wealth_source == "scf_median"
