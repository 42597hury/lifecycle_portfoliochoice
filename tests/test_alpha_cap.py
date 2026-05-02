"""Tests for the numerical leverage cap (alpha_min, alpha_max) in
unconstrained Newton portfolio solvers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import DiscretizationConfig, SolverConfig
from precompute import Precompute, build_model
from solver import run_lifecycle_solver
from var import build_nominal_system1_var_config_hardcoded


def _reference_base_config(constrained: bool, gamma: float = 3.0,
                           start_age: int = 22, retire_age: int = 67,
                           terminal_age: int = 99) -> dict:
    return {
        "beta": 0.96,
        "gamma": gamma,
        "b_bar": 10,
        "start_age": start_age,
        "retire_age": retire_age,
        "terminal_age": terminal_age,
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
        "constrained": constrained,
    }


def _build_small_problem(constrained: bool, gamma: float = 3.0,
                         start_age: int = 22, retire_age: int = 67,
                         terminal_age: int = 99):
    var_config = build_nominal_system1_var_config_hardcoded()
    base_cfg = _reference_base_config(
        constrained=constrained, gamma=gamma,
        start_age=start_age, retire_age=retire_age, terminal_age=terminal_age,
    )
    model = build_model(base_cfg, var_config, verbose=False)
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


def test_cap_noop_on_nonbinding_bounds():
    """A wide-enough cap should produce bit-identical output to the
    sentinel-disabled cap (-1e30, +1e30) when no iterate ever reaches the box.

    On this 10x10 small grid the unconstrained optimum can leverage to
    several hundred at corners, so the comparison is run against a cap
    sized two orders of magnitude above the largest visited iterate.
    """
    model, pc = _build_small_problem(constrained=False, gamma=3.0)

    sc_open = SolverConfig(alpha_min=-1e30, alpha_max=+1e30)
    C1, S1, B1, _ = run_lifecycle_solver(model, pc, solver_config=sc_open, verbose=0)

    finite_mask = np.isfinite(S1) & np.isfinite(B1)
    max_alpha = float(max(np.abs(S1[finite_mask]).max(), np.abs(B1[finite_mask]).max()))
    wide_cap = max(1.0, max_alpha) * 100.0

    sc_wide = SolverConfig(alpha_min=-wide_cap, alpha_max=+wide_cap)
    C2, S2, B2, _ = run_lifecycle_solver(model, pc, solver_config=sc_wide, verbose=0)

    assert np.array_equal(S1, S2)
    assert np.array_equal(B1, B2)
    assert np.array_equal(C1, C2)


def test_cap_clips_and_reports_failure():
    """A tight cap should clip max |alpha| and increase EC_NEWTON_FAIL count
    relative to the open-cap baseline."""
    model, pc = _build_small_problem(constrained=False, gamma=3.0)

    sc_open = SolverConfig(alpha_min=-1e30, alpha_max=+1e30)
    sc_tight = SolverConfig(alpha_min=-0.5, alpha_max=+0.5)

    _, S_open, B_open, diag_open = run_lifecycle_solver(
        model, pc, solver_config=sc_open, verbose=0
    )
    _, S_tight, B_tight, diag_tight = run_lifecycle_solver(
        model, pc, solver_config=sc_tight, verbose=0
    )

    eps = 1e-12
    assert np.nanmax(np.abs(S_tight)) <= 0.5 + eps
    assert np.nanmax(np.abs(B_tight)) <= 0.5 + eps
    assert np.nanmin(S_tight) >= -0.5 - eps
    assert np.nanmin(B_tight) >= -0.5 - eps

    # The tight cap must produce at least as many Newton failures as the
    # open cap (typically strictly more, because cells whose interior
    # optimum lies outside [-0.5, +0.5] now fail).
    assert diag_tight["total_newton_failures"] >= diag_open["total_newton_failures"]
    # Sanity: with a tight 0.5 cap on a well-conditioned gamma=3 problem,
    # at least one cell should fail (the optimum is well above 0.5).
    assert diag_tight["total_newton_failures"] > 0


def test_constrained_branch_unaffected_by_cap():
    """With constrained=True the cap fields must be ignored: the SolverConfig
    default (-10, +10) and the sentinel (-1e30, +1e30) must produce
    bit-identical output."""
    model, pc = _build_small_problem(constrained=True, gamma=3.0)

    sc_default = SolverConfig()  # alpha_min=-10, alpha_max=+10
    sc_open = SolverConfig(alpha_min=-1e30, alpha_max=+1e30)

    C1, S1, B1, _ = run_lifecycle_solver(model, pc, solver_config=sc_default, verbose=0)
    C2, S2, B2, _ = run_lifecycle_solver(model, pc, solver_config=sc_open, verbose=0)

    assert np.array_equal(S1, S2)
    assert np.array_equal(B1, B2)
    assert np.array_equal(C1, C2)


def test_cap_exercises_both_retirement_and_working_ages():
    """With a tiny age window covering both work and retirement, the cap
    must be a no-op when wide and clip when tight on both age-paths."""
    model, pc = _build_small_problem(
        constrained=False, gamma=3.0,
        start_age=64, retire_age=66, terminal_age=68,
    )

    sc_open = SolverConfig(alpha_min=-1e30, alpha_max=+1e30)
    sc_tight = SolverConfig(alpha_min=-0.5, alpha_max=+0.5)

    C_open, S_open, B_open, _ = run_lifecycle_solver(
        model, pc, solver_config=sc_open, verbose=0
    )

    # Size the wide cap relative to the actual iterates visited by the
    # sentinel-open solve so that the clamp is the identity everywhere.
    finite_mask = np.isfinite(S_open) & np.isfinite(B_open)
    max_alpha = float(max(
        np.abs(S_open[finite_mask]).max(),
        np.abs(B_open[finite_mask]).max(),
    ))
    wide_cap = max(1.0, max_alpha) * 100.0
    sc_wide = SolverConfig(alpha_min=-wide_cap, alpha_max=+wide_cap)

    C_wide, S_wide, B_wide, _ = run_lifecycle_solver(
        model, pc, solver_config=sc_wide, verbose=0
    )
    _, S_tight, B_tight, _ = run_lifecycle_solver(
        model, pc, solver_config=sc_tight, verbose=0
    )

    # 5.1 on this small lifecycle: wide cap is a no-op
    assert np.array_equal(S_open, S_wide)
    assert np.array_equal(B_open, B_wide)
    assert np.array_equal(C_open, C_wide)

    # 5.2 on this small lifecycle: tight cap clips |S|, |B|
    eps = 1e-12
    assert np.nanmax(np.abs(S_tight)) <= 0.5 + eps
    assert np.nanmax(np.abs(B_tight)) <= 0.5 + eps
    assert np.nanmin(S_tight) >= -0.5 - eps
    assert np.nanmin(B_tight) >= -0.5 - eps
