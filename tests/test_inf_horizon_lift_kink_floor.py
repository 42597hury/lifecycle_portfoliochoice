"""Regression test: per-cell V'(W') quadrature floor helper + smoke test.

History
-------
v1 (commit ``2d33c95``, reverted at ``5dafd0b``): the helper returned a global
``max`` across cells. At ``wmin=0.10`` that homogenised a rare-cell floor onto
every other cell and broke convergence (alpha_b corner explosion). See
``docs/handoff/HANDOFF_KINK_FIX_REGRESSION_AT_WMIN_010.md``.

The v1 regression test asserted ``_compute_w_floor_from_policy(C_all_interior)
== wealth_grid[0]`` — codifying Bug 1 (the wrong sentinel) as expected
behavior. This file replaces those assertions with the correct contract.

v2 contract
-----------
``_compute_w_floor_from_policy(C_old, wealth_grid)`` returns an
``(n_z, N_state)`` float64 array where each entry is the per-cell w_floor:
- Cells with a constrained-corner band (some ``c[i_w] == wealth_grid[i_w]``)
  get ``wealth_grid[first_unc]`` — the smallest wealth-grid point above the
  constrained band, i.e. the smallest W' realisation that misses the
  kink-edge cell pair entirely.
- Cells with no constrained band get ``0.0`` — true no-op against
  ``jnp.maximum(x_next, 0.0)`` since W' >= 0 always.

Gates this file enforces
------------------------
1. Helper: all-interior input -> all-zeros output (sentinel correctness).
2. Helper: partial constrained band in one cell -> that cell's floor is
   wealth_grid[first_unc]; OTHER cells are 0 (Bug-2 specific test —
   heterogeneous cells were the case that bit v1).
3. Helper: heterogeneous bands across cells -> each cell gets its OWN
   floor; no global homogenisation.
4. Plumbing smoke: IH solver runs end-to-end with the v2 helper invoked
   every iter, produces finite output, w_floor_arr is shape (n_z, N_state).

What this file does NOT exercise
---------------------------------
- Convergence at wmin=0.01 (the kink-fix target) — that's
  verify/test_inf_horizon_g7_theta000_wmin001_postfix.py on the cluster.
- Convergence at wmin=0.10 across the theta sweep — that's
  verify/test_inf_horizon_g10_ccv_nominal_theta_sweep.py on the cluster.
- Lifecycle parity — that's verify/test_retire_g11.py before/after bit-identity.
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


# -----------------------------------------------------------------------------
# Helper-level unit tests
# -----------------------------------------------------------------------------

def test_w_floor_helper_all_interior_returns_zeros():
    """v2 sentinel correctness: no constrained band anywhere -> all zeros.

    v1 returned wealth_grid[0] here, which became a non-no-op floor at wmin=0.10
    and degraded Newton's Jacobian in low-savings cells. v2 must return 0.0
    so jnp.maximum(x_next, 0.0) is bit-identical to no-clip.
    """
    wealth_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.0])
    C = np.full((2, 3, wealth_grid.size), 0.5 * wealth_grid)  # all interior
    out = _compute_w_floor_from_policy(C, wealth_grid)
    assert out.shape == (2, 3)
    assert out.dtype == np.float64
    assert np.all(out == 0.0), (
        f"all-interior input must produce all-zero floor; got {out!r}"
    )


def test_w_floor_helper_partial_band_in_single_cell_isolated():
    """Bug-2 specific test: ONE cell has a constrained band, ALL OTHERS are
    fully interior. The lone cell's floor must equal wealth_grid[first_unc];
    every other cell must be 0.0.

    This is the case v1 globalisation got wrong. If v2 were to leak any
    non-zero floor to other cells (e.g. via a max-reduce), this test catches it.
    """
    wealth_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.0])
    n_z, N_state = 2, 7
    C = np.full((n_z, N_state, wealth_grid.size), 0.5 * wealth_grid)
    # One cell (i_z=0, i_s=5) has 2 constrained-corner cells (c == wealth_grid)
    C[0, 5, :2] = wealth_grid[:2]
    out = _compute_w_floor_from_policy(C, wealth_grid)

    # The flagged cell carries its own floor.
    assert out[0, 5] == wealth_grid[2], (
        f"cell (0,5) should carry wealth_grid[2]={wealth_grid[2]}, got {out[0,5]}"
    )
    # Every OTHER cell must be 0.0 — this is the property v1 violated.
    mask = np.ones_like(out, dtype=bool)
    mask[0, 5] = False
    assert np.all(out[mask] == 0.0), (
        f"only cell (0,5) should be non-zero; got non-zero at "
        f"{list(zip(*np.where((out != 0.0) & mask)))}"
    )


def test_w_floor_helper_heterogeneous_cells_get_own_floor():
    """Multiple cells each with their own constrained band -> each entry of
    the returned array is THIS cell's wealth_grid[first_unc], not a homogenised
    max. v1's bug was the global-max reduction; v2 keeps cells independent.
    """
    wealth_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.0])
    C = np.full((1, 3, wealth_grid.size), 0.5 * wealth_grid)
    C[0, 0, :1] = wealth_grid[:1]   # cell 0: 1 constrained -> floor wealth_grid[1]
    C[0, 1, :3] = wealth_grid[:3]   # cell 1: 3 constrained -> floor wealth_grid[3]
    # cell 2: all interior -> 0.0
    out = _compute_w_floor_from_policy(C, wealth_grid)
    assert out[0, 0] == wealth_grid[1]
    assert out[0, 1] == wealth_grid[3]
    assert out[0, 2] == 0.0
    # Confirm we did NOT collapse to max across cells (which v1 would have).
    global_max_would_be = float(wealth_grid[3])
    assert not np.all(out == global_max_would_be), (
        "v2 must not homogenise to a global max — this is the v1 regression"
    )


def test_w_floor_helper_dtype_and_shape_contract():
    """Contract: float64, shape (n_z, N_state), contiguous."""
    wealth_grid = np.array([0.01, 0.05, 0.10])
    C = np.zeros((3, 4, 3))
    out = _compute_w_floor_from_policy(C, wealth_grid)
    assert out.shape == (3, 4)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]


def test_w_floor_helper_tolerates_fp32_roundtrip_noise():
    """v3 regression test: when ``gather_precision='f32'`` the kernel
    re-materialises ``c_w`` through an fp32 path before returning to the IH
    outer loop, so bit-equality with the fp64 ``wealth_grid`` is lost on the
    round-trip. The helper must still detect constrained cells.

    Reproduces the Gate A symptom: at wmin=0.01 v2's strict-equality check
    yielded ``has_band.mean() = 0`` universally, making the entire fix a no-op.
    v3 detects within ``atol=1e-8`` of ``wealth_grid``, which catches values
    quantised by an fp32 round-trip at the low-W range where the kink lives.
    """
    wealth_grid = np.array([0.01, 0.07984, 0.1543, 0.2390, 0.5, 1.0])
    n_z, N_state = 1, 3
    C = np.full((n_z, N_state, wealth_grid.size), 0.5 * wealth_grid)
    # Cell (0,0) has 2 constrained cells, but the values are fp32-noised — NOT
    # bit-equal to wealth_grid in fp64. Simulates the real round-trip the
    # kernel produces.
    fp32_noise = wealth_grid[:2].astype(np.float32).astype(np.float64)
    C[0, 0, :2] = fp32_noise
    # Confirm the simulated noise actually breaks strict equality (so the test
    # is exercising the v3 fix, not silently degenerating to v2's contract).
    assert not np.array_equal(C[0, 0, :2], wealth_grid[:2]), (
        "fp32 round-trip should yield non-bit-equal values; test is mis-set"
    )
    out = _compute_w_floor_from_policy(C, wealth_grid)
    assert out[0, 0] == wealth_grid[2], (
        f"v3 must detect the fp32-noised constrained band; got {out[0,0]}, "
        f"expected wealth_grid[2]={wealth_grid[2]}"
    )
    assert out[0, 1] == 0.0
    assert out[0, 2] == 0.0


def test_w_floor_helper_does_not_misclassify_interior_at_low_w():
    """Tolerance trade-off check: an interior cell with ``c_opt`` close-but-not-
    equal to ``wealth_grid`` (e.g. very low W where CRRA gives c~0.05W, so
    c_opt and W differ by 95%) must NOT be classified as constrained.

    The fp32-noise tolerance is 1e-8; interior c_opt is on the order of
    0.05*wealth_grid (typically >=1e-3 different from wealth_grid at low W) —
    several orders of magnitude above the tolerance. False-positive risk
    should be zero.
    """
    wealth_grid = np.array([0.01, 0.08, 0.15, 0.25])
    n_z, N_state = 1, 2
    # Cell (0,0): all interior, c = 0.05 * W (CRRA-like). Differences from
    # wealth_grid are 0.0095, 0.076, 0.1425, 0.2375 — all >> atol=1e-8.
    C = np.full((n_z, N_state, wealth_grid.size), 0.05 * wealth_grid)
    out = _compute_w_floor_from_policy(C, wealth_grid)
    assert np.all(out == 0.0), (
        f"interior cells must not be misclassified as constrained; got {out!r}"
    )


# -----------------------------------------------------------------------------
# End-to-end plumbing smoke
# -----------------------------------------------------------------------------

def _tiny_model_pc():
    """Smallest config that runs the inf-horizon kernel on CPU."""
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
    """End-to-end: IH solver runs at wmin=0.01, helper fires per iter with
    correct shape, kernel accepts the array w_floor, output is finite.

    Does NOT assert convergence — that lives in cloud verify runs.
    """
    model, pc = _tiny_model_pc()
    sc = _tiny_solver_config()

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

    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    assert C.shape == expected_shape
    assert not np.isnan(C).any() and not np.isinf(C).any()
    assert not np.isnan(S).any() and not np.isinf(S).any()
    assert not np.isnan(B).any() and not np.isinf(B).any()
    assert diag["n_iter"] == 5
    assert len(call_log) == 5, f"helper called {len(call_log)} times, expected 5"
    for f in call_log:
        assert f.shape == (pc.n_z, pc.N_state), (
            f"w_floor must be shape ({pc.n_z}, {pc.N_state}); got {f.shape}"
        )
        assert f.dtype == np.float64
        assert np.all(f >= 0.0), "w_floor entries must be non-negative"
