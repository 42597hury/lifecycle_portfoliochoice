"""Regression test: V'(W') quadrature floor helper + smoke test.

Pre-fix (handoff: HANDOFF_IH_LIFT_KINK_STRUCTURAL_FIX.md):
  At g=7, n_w=100, wmin=0.01 the IH solver oscillates with stop sup-norm in
  the 5-10 range and alpha_b corner extremes blow up to +/- 10. The kink in
  the lifted policy at the constrained-corner boundary leaks slope-1 MPC into
  Newton's Jacobian every outer iter; the operator stops being a contraction.

Post-fix:
  retirement_foc_jac_ccv applies x_next = jnp.maximum(x_next, w_floor) before
  the V'(W') interp; the IH outer loop computes w_floor each iter from the
  current C_old (see lifecycle.inf_horizon_solver._compute_w_floor_from_policy).
  Lifecycle still sees w_floor=0.0 and is bit-identical (W' >= 0 always).

Why the contraction-trajectory test from the handoff isn't here:
  The handoff suggested asserting stop[10] < stop[5] < stop[1] at g=4, n_w=20.
  Empirically that grid does NOT exhibit the constrained-corner band (the FOC
  produces interior solutions at every wealth_grid point because the savings
  grid is too coarse to resolve the tiny-savings boundary). With no
  constrained band there is no kink for the fix to address, and the small-grid
  trajectory still oscillates from inherent quadrature noise unrelated to the
  kink. Behavioral validation happens at g=7 in the cloud verify run
  (verify/test_inf_horizon_g7_theta000_wmin001_postfix.py).

What this test DOES catch:
  1. Helper logic regression — wrong index axis, wrong equality, all-interior
     edge case, all-constrained edge case.
  2. Kernel-plumbing regression — w_floor parameter wired through pmap/vmap,
     default lifecycle path returns finite arrays.
  3. Failure to call _compute_w_floor_from_policy at all (the IH outer loop
     would silently use w_floor=0 without the helper invocation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle import inf_horizon_solver as ihs
from lifecycle.inf_horizon_solver import (
    run_infinite_horizon_solver,
    _compute_w_floor_from_policy,
)


def test_w_floor_helper_partial_constrained_band():
    """Helper returns wealth_grid[k] where k is the first non-constrained index.
    Constrained cells are detected by exact equality with wealth_grid (the
    jnp.where clamp on solver.py:1413 produces bit-identical wealth_grid values).
    """
    wealth_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.0])
    C = np.zeros((1, 1, wealth_grid.size))
    C[0, 0, :2] = wealth_grid[:2]              # cells 0,1 constrained
    C[0, 0, 2:] = 0.5 * wealth_grid[2:]        # cells 2+ interior (c < wealth)
    assert _compute_w_floor_from_policy(C, wealth_grid) == wealth_grid[2]


def test_w_floor_helper_all_interior_returns_grid_zero():
    """No constrained cells -> argmax over all-True returns 0 -> floor =
    wealth_grid[0], i.e. an effective no-op clip. This is the "small grid"
    regime where the FOC's interior solves reach all the way down to
    wealth_min and there's no kink to address."""
    wealth_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.0])
    C = np.full((1, 1, wealth_grid.size), 0.5 * wealth_grid)
    assert _compute_w_floor_from_policy(C, wealth_grid) == wealth_grid[0]


def test_w_floor_helper_global_max_across_cells():
    """Different cells have different first-non-constrained indices; global
    floor is the max across cells (per handoff Open-question 2: 'start
    global, refine if needed')."""
    wealth_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.0])
    C = np.zeros((1, 2, wealth_grid.size))
    C[0, 0, :1] = wealth_grid[:1]              # cell (0,0): 1 constrained
    C[0, 0, 1:] = 0.5 * wealth_grid[1:]
    C[0, 1, :3] = wealth_grid[:3]              # cell (0,1): 3 constrained
    C[0, 1, 3:] = 0.5 * wealth_grid[3:]
    # Per-cell: cell (0,0) -> wealth_grid[1]; cell (0,1) -> wealth_grid[3].
    # Global max -> wealth_grid[3].
    assert _compute_w_floor_from_policy(C, wealth_grid) == wealth_grid[3]


def _tiny_model_pc():
    disc = DiscretizationConfig(
        n_wealth=20, wealth_min=0.01, wealth_max=50.0,
        n_savings=20,
        state_grid_sizes=(4, 4, 4),
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


def _tiny_solver_config():
    return SolverConfig(
        wealth_dynamics_spec="ccv_log",
        max_iter=50,
        delta_bequest=0.0,
        gather_precision="f32",
        cell_vmap_chunks=1,
        use_line_search=True,
    )


def test_inf_horizon_smoke_with_w_floor_plumbed():
    """Plumbing test: the IH solver runs end-to-end at wmin=0.01, the
    _compute_w_floor_from_policy helper is invoked every iter, the kernel
    accepts the w_floor argument, and the run produces finite output.

    Does NOT assert convergence or alpha_b bounds — those are exercised at
    full resolution (g=7) in the cloud verify run."""
    model, pc = _tiny_model_pc()
    sc = _tiny_solver_config()

    # Spy on the helper to verify it actually fires per-iter.
    call_log = []
    orig = ihs._compute_w_floor_from_policy
    def spy(C_old, wealth_grid):
        f = orig(C_old, wealth_grid)
        call_log.append(f)
        return f
    ihs._compute_w_floor_from_policy = spy
    try:
        C, S, B, diag = run_infinite_horizon_solver(
            model, pc,
            solver_config=sc,
            max_iter=5, tol=1e-15, damping=0.5,
            verbose=False, show_progress=False,
        )
    finally:
        ihs._compute_w_floor_from_policy = orig

    # Sanity.
    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    assert C.shape == expected_shape
    assert not np.isnan(C).any() and not np.isinf(C).any()
    assert not np.isnan(S).any() and not np.isinf(S).any()
    assert not np.isnan(B).any() and not np.isinf(B).any()
    assert diag["n_iter"] == 5

    # Helper fired once per iter.
    assert len(call_log) == 5, f"helper invoked {len(call_log)} times, expected 5"
    for f in call_log:
        assert f >= float(pc.wealth_grid[0]) - 1e-12, (
            f"w_floor must be >= wealth_grid[0]; got {f}"
        )
