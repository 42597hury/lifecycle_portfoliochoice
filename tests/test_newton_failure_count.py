"""Regression tests for the Newton-failure-count diagnostic wiring.

Three gates:
  1. Bit-identity: enabling the diagnostic plumbing must not move policies.
     The exit-code propagation is purely informational; ``c_w``, ``a_s_w``,
     ``a_b_w`` are computed from the Newton optimum independently of whether
     the exit code was discarded or propagated. We can't compare against a
     pre-fix reference inside this repo (the source files have already been
     updated), so the bit-identity check here is structural: solving the
     same tiny config twice from a clean Python session produces identical
     policies.
  2. Failure-count smoke (lifecycle): a deliberately under-budgeted Newton
     (``max_iter=3``) must register non-zero ``total_newton_failures``;
     the same config with a generous budget (``max_iter=200``) must
     register zero (or near-zero) failures.
  3. Failure-count smoke (inf-horizon): same idea, against the fixed-point
     loop in ``inf_horizon_solver``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import EC_INTERIOR, run_lifecycle_solver
from lifecycle import inf_horizon_solver as ihs


# ---------------------------------------------------------------------------
# Tiny configs (shared across gates; same shape as the histogram-fix bench).
# Real-yields pivot: 3-axis state vector (cape, spr, y_1).
# ---------------------------------------------------------------------------

def _tiny_lifecycle_setup():
    disc = DiscretizationConfig(
        n_wealth=15, wealth_min=0.13, wealth_max=200.0,
        n_savings=15,
        state_grid_sizes=(2, 2, 2),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.25),
        n_z=4,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=(2, 2),
        n_state_quad_nodes=(2, 2, 2),
    )
    base = dict(BASE_CONFIG)
    base.update(start_age=60, retire_age=63, terminal_age=65)
    var_config = build_real_full_var_config_hardcoded()
    model = build_model(base, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    return model, pc


def _tiny_inf_horizon_setup():
    disc = DiscretizationConfig(
        n_wealth=20, wealth_min=0.13, wealth_max=200.0,
        n_savings=20,
        state_grid_sizes=(2, 2, 2),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.25),
        n_z=1,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=(2, 2),
        n_state_quad_nodes=(2, 2, 2),
    )
    var_config = build_real_full_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    return model, pc


# ---------------------------------------------------------------------------
# Gate 1: bit-identity (structural — same code path twice)
# ---------------------------------------------------------------------------

def test_lifecycle_bit_identity_back_to_back():
    """Solving the same tiny config twice must yield identical policies.

    The diagnostic-only nature of the Newton-failure wiring is such that the
    only way to break bit-identity is to accidentally edit the policy-math
    path. A back-to-back determinism check catches that as long as the
    diagnostic plumbing itself doesn't introduce nondeterminism (e.g. a
    different-dtype concatenate that flows back into the policy math).
    """
    model, pc = _tiny_lifecycle_setup()
    sc = CANONICAL_SOLVER._replace(max_iter=50, max_iter_unconstrained=50)

    C1, S1, B1, _ = run_lifecycle_solver(model, pc, sc, verbose=0)
    C2, S2, B2, _ = run_lifecycle_solver(model, pc, sc, verbose=0)

    np.testing.assert_array_equal(C1, C2)
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(B1, B2)


# ---------------------------------------------------------------------------
# Gate 2: lifecycle failure-count smoke
# ---------------------------------------------------------------------------

def test_lifecycle_under_budgeted_newton_registers_failures():
    """max_iter=3 is far below what cold-start Newton needs at most cells —
    several should fail to converge and total_newton_failures must be > 0.
    """
    model, pc = _tiny_lifecycle_setup()
    sc = CANONICAL_SOLVER._replace(max_iter=3, max_iter_unconstrained=3)

    _, _, _, diag = run_lifecycle_solver(model, pc, sc, verbose=0)

    n_fail = int(diag["total_newton_failures"])
    age_fail = np.asarray(diag["age_newton_fail"], dtype=np.int64)

    assert n_fail > 0, (
        f"Expected non-zero Newton failures with max_iter=3, got {n_fail}. "
        "Either the per-savings exit codes aren't propagating, or the smoke "
        "config is too easy for cold-start Newton."
    )
    assert age_fail.sum() == n_fail, (
        "total_newton_failures must equal the sum of age_newton_fail."
    )
    # Sanity: the count should be a real integer (not NaN/None) and sit in a
    # plausible range (not bigger than the total cell-savings count).
    n_age = pc.n_age
    cells_per_age = pc.n_z * pc.N_state * pc.n_s
    upper = n_age * cells_per_age
    assert 0 < n_fail <= upper


def test_lifecycle_well_budgeted_newton_has_few_failures():
    """max_iter=200 should leave the well-budgeted Newton with at most a
    handful of structurally-doomed cells (those where the FOC residual
    scale dips below fp64 precision relative to tol*scale; see
    docs/scans/NEWTON_FAILURE_STRUCTURE_2026-05-08.md).

    The threshold is intentionally generous (< 5% of cells): the test's
    intent is to verify failure-count plumbing — well-budgeted runs must
    register far fewer failures than under-budgeted runs — not to pin a
    specific number that depends on the calibration. The companion
    under-budgeted test (max_iter=3) confirms the failure count grows
    when the budget is tight.
    """
    model, pc = _tiny_lifecycle_setup()
    sc = CANONICAL_SOLVER._replace(max_iter=200, max_iter_unconstrained=200)

    _, _, _, diag = run_lifecycle_solver(model, pc, sc, verbose=0)

    n_fail = int(diag["total_newton_failures"])
    n_age = pc.n_age
    total_cells = n_age * pc.n_z * pc.N_state * pc.n_s
    fail_frac = n_fail / max(total_cells, 1)
    assert fail_frac < 0.05, (
        f"Well-budgeted Newton (max_iter=200) registered {n_fail} failures "
        f"({fail_frac:.1%} of {total_cells} cells). Either the threshold "
        "check is mis-wired or the smoke config is far harder than expected."
    )


# ---------------------------------------------------------------------------
# Gate 3: inf-horizon failure-count smoke
# ---------------------------------------------------------------------------

def _inf_horizon_solver_config(max_iter_newton: int) -> SolverConfig:
    return SolverConfig(
        wealth_dynamics_spec="ccv_log",
        max_iter=max_iter_newton,
        max_iter_unconstrained=max_iter_newton,
        delta_bequest=0.0,
        gather_precision="f32",
        cell_vmap_chunks=1,
    )


def test_inf_horizon_under_budgeted_newton_registers_failures():
    """Tiny inf-horizon config with max_iter Newton=3 — total_newton_failures
    must be > 0 and newton_failures_per_iter must be a per-iter array.
    """
    model, pc = _tiny_inf_horizon_setup()
    sc = _inf_horizon_solver_config(max_iter_newton=3)

    _, _, _, diag = ihs.run_infinite_horizon_solver(
        model, pc,
        solver_config=sc,
        max_iter=5, tol=1e-15, damping=1.0,
        verbose=False, show_progress=False,
    )

    n_fail = int(diag["total_newton_failures"])
    per_iter = np.asarray(diag["newton_failures_per_iter"], dtype=np.int64)

    assert n_fail > 0, (
        f"Expected non-zero Newton failures with under-budgeted Newton, "
        f"got {n_fail}. Inf-horizon exit-code wiring may be broken."
    )
    assert per_iter.shape == (diag["n_iter"],), (
        f"newton_failures_per_iter shape {per_iter.shape} does not match "
        f"n_iter={diag['n_iter']}."
    )
    assert int(per_iter.sum()) == n_fail


def test_inf_horizon_well_budgeted_newton_has_few_failures():
    """Generous Newton budget — failures should drop to zero (or near-zero;
    fixed-point iteration may still exhibit transient ill-conditioning early
    on, so we tolerate a small count).
    """
    model, pc = _tiny_inf_horizon_setup()
    sc = _inf_horizon_solver_config(max_iter_newton=200)

    _, _, _, diag = ihs.run_infinite_horizon_solver(
        model, pc,
        solver_config=sc,
        max_iter=5, tol=1e-15, damping=1.0,
        verbose=False, show_progress=False,
    )

    n_fail = int(diag["total_newton_failures"])
    # Allow up to a handful of failures: cold-start Newton on iteration 1 of
    # an inf-horizon loop can occasionally hit edge cases. The point is that
    # the *count* is now real, not the constant 0 of the pre-fix bug.
    assert n_fail < 50, (
        f"Expected near-zero Newton failures with max_iter=200, got {n_fail}."
    )
