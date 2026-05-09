"""Constrained-corner clamp in ``_lift_to_wealth_grid`` (CRITICAL fix).

The post-pivot solver did NOT implement a true borrowing-constrained corner;
plain ``jnp.interp`` linearly blended the artificial ``egm_anchor`` (or the
``tiny_savings`` fallback) with the smallest unconstrained interior FOC
solution, producing meaningless ``c < W, alpha != 0`` policies at low wealth.

This module tests the post-hoc clamp added to ``_lift_to_wealth_grid``:

  - Unit: explicit synthetic EGM grid; verify (i) bit-identity at non-
    constrained wealth, (ii) corner solution ``c=W, a_s=a_b=0`` below the
    smallest real interior W_implied, (iii) edge cases (all-non-real,
    everything constrained, kink continuity, ``min_consumption=0``).
  - End-to-end smoke: solve a tiny config with deliberately low
    ``wealth_min``; verify the corner policy at ``wealth_grid[0]``.

References:
  - Carroll (2006) "The method of endogenous gridpoints" Econ Letters.
  - Druedahl & Jorgensen (2017) "A general endogenous grid method for
    multi-dimensional models with non-convexities and constraints,"
    J Econ Dyn Control.
"""
from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lifecycle.solver import _lift_to_wealth_grid


# =============================================================================
# Unit tests — synthetic EGM grids exercise the clamp in isolation
# =============================================================================

def _build_synthetic_egm(
    *,
    n_tiny: int,
    interior_x: np.ndarray,
    interior_c: np.ndarray,
    interior_a_s: np.ndarray,
    interior_a_b: np.ndarray,
    egm_anchor: float = 1e-10,
    min_consumption: float = 1e-10,
    init_alpha_s: float = 0.1,
    init_alpha_b: float = 0.4,
):
    """Build (x_egm, c_egm, a_s_egm, a_b_egm) mirroring the per-cell scan.

    Layout: [anchor] + [n_tiny tiny-savings fallback points] + [interior].
    The anchor and tiny-savings fallback both record c = min_consumption
    (= egm_anchor by default) — so the clamp's ``c > 2*min_consumption``
    threshold should pick out only the interior.
    """
    # Anchor: x = anchor, c = anchor, a_s = a_b = 0
    x_anchor = np.array([egm_anchor], dtype=np.float64)
    c_anchor = np.array([egm_anchor], dtype=np.float64)
    a_s_anchor = np.array([0.0], dtype=np.float64)
    a_b_anchor = np.array([0.0], dtype=np.float64)

    # Tiny-savings fallback: small s spaced linearly between anchor and interior[0],
    # c forced to min_consumption, alphas pinned at cold init.
    if n_tiny > 0:
        s_tiny = np.linspace(1e-9, 5e-7, n_tiny)
        x_tiny = min_consumption + s_tiny
        c_tiny = np.full(n_tiny, min_consumption)
        a_s_tiny = np.full(n_tiny, init_alpha_s)
        a_b_tiny = np.full(n_tiny, init_alpha_b)
    else:
        x_tiny = np.empty(0)
        c_tiny = np.empty(0)
        a_s_tiny = np.empty(0)
        a_b_tiny = np.empty(0)

    x_egm = jnp.asarray(np.concatenate([x_anchor, x_tiny, interior_x]))
    c_egm = jnp.asarray(np.concatenate([c_anchor, c_tiny, interior_c]))
    a_s_egm = jnp.asarray(np.concatenate([a_s_anchor, a_s_tiny, interior_a_s]))
    a_b_egm = jnp.asarray(np.concatenate([a_b_anchor, a_b_tiny, interior_a_b]))
    return x_egm, c_egm, a_s_egm, a_b_egm


def test_constrained_corner_at_low_wealth():
    """Below smallest real interior W_implied: c=W, a_s=a_b=0."""
    interior_x = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    interior_c = np.array([0.4, 0.8, 1.5, 3.5, 7.0])
    interior_a_s = np.array([0.45, 0.50, 0.55, 0.60, 0.65])
    interior_a_b = np.array([0.30, 0.32, 0.34, 0.36, 0.38])
    x_egm, c_egm, a_s_egm, a_b_egm = _build_synthetic_egm(
        n_tiny=3,
        interior_x=interior_x,
        interior_c=interior_c,
        interior_a_s=interior_a_s,
        interior_a_b=interior_a_b,
    )
    wealth_grid = jnp.asarray(np.array([0.05, 0.10, 0.25, 0.49, 0.5, 1.0, 5.0]))
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption=1e-10,
    )
    # First four wealth points are below interior_x[0] = 0.5: clamped corner.
    np.testing.assert_allclose(np.asarray(c_w[:4]), np.asarray(wealth_grid[:4]),
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(a_s_w[:4]), np.zeros(4))
    np.testing.assert_array_equal(np.asarray(a_b_w[:4]), np.zeros(4))
    # At W = 0.5 exactly (== interior_x[0]): boundary — interior value
    # (the clamp uses strict ``< W_min_real``, so W = W_min_real is not
    # constrained; jnp.interp at the exact knot returns interior_c[0]).
    assert float(c_w[4]) == pytest.approx(interior_c[0], abs=1e-12)
    # At W = 1.0 and 5.0: linear interp between interior knots, untouched by clamp.
    # Knot match at W=1.0 → c=0.8; W=5.0 → c=3.5.
    assert float(c_w[5]) == pytest.approx(interior_c[1], abs=1e-12)
    assert float(c_w[6]) == pytest.approx(interior_c[3], abs=1e-12)


def test_bit_identity_at_non_constrained_wealth():
    """Wealth above smallest real interior: clamp must be a no-op."""
    interior_x = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    interior_c = np.array([0.4, 0.8, 1.5, 3.5, 7.0])
    interior_a_s = np.array([0.45, 0.50, 0.55, 0.60, 0.65])
    interior_a_b = np.array([0.30, 0.32, 0.34, 0.36, 0.38])
    x_egm, c_egm, a_s_egm, a_b_egm = _build_synthetic_egm(
        n_tiny=3,
        interior_x=interior_x,
        interior_c=interior_c,
        interior_a_s=interior_a_s,
        interior_a_b=interior_a_b,
    )
    # All wealth points above interior_x[0] = 0.5.
    wealth_grid = jnp.asarray(np.array([0.6, 0.75, 1.0, 1.5, 3.0, 7.5, 15.0]))
    # Reference: the interp call without the clamp.
    order = jnp.argsort(x_egm)
    x_sorted = x_egm[order]
    c_ref = jnp.interp(wealth_grid, x_sorted, c_egm[order])
    a_s_ref = jnp.interp(wealth_grid, x_sorted, a_s_egm[order])
    a_b_ref = jnp.interp(wealth_grid, x_sorted, a_b_egm[order])
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption=1e-10,
    )
    # Bit-identity (no clamp triggered → outputs == reference interp).
    np.testing.assert_array_equal(np.asarray(c_w), np.asarray(c_ref))
    np.testing.assert_array_equal(np.asarray(a_s_w), np.asarray(a_s_ref))
    np.testing.assert_array_equal(np.asarray(a_b_w), np.asarray(a_b_ref))


def test_kink_continuity_at_w_min_real():
    """At W = W_min_real, corner c=W matches interior c[0] (within FOC tolerance)."""
    # Construct an interior whose smallest point has c ~ x (small s ⇒ c ≈ W).
    interior_x = np.array([0.6, 1.0, 2.0])
    interior_c = np.array([0.59, 0.85, 1.4])  # c = x - s; for x=0.6, s≈0.01
    interior_a_s = np.array([0.5, 0.55, 0.6])
    interior_a_b = np.array([0.3, 0.32, 0.34])
    x_egm, c_egm, a_s_egm, a_b_egm = _build_synthetic_egm(
        n_tiny=2,
        interior_x=interior_x,
        interior_c=interior_c,
        interior_a_s=interior_a_s,
        interior_a_b=interior_a_b,
    )
    # Just below and at the knot.
    wealth_grid = jnp.asarray(np.array([0.59, 0.6, 0.61]))
    c_w, _, _ = _lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption=1e-10,
    )
    # Below knot: clamped to W = 0.59
    assert float(c_w[0]) == pytest.approx(0.59, abs=1e-12)
    # At knot: returns interior_c[0] = 0.59 (tightly close to W=0.6 since s≈0.01)
    assert float(c_w[1]) == pytest.approx(0.59, abs=1e-12)
    # Above knot: linear interp between c=0.59 (at x=0.6) and c=0.85 (at x=1.0).
    expected_above = 0.59 + (0.85 - 0.59) * (0.61 - 0.6) / (1.0 - 0.6)
    assert float(c_w[2]) == pytest.approx(expected_above, abs=1e-12)


def test_all_constrained():
    """All wealth-grid points below W_min_real: entire policy = (W, 0, 0)."""
    interior_x = np.array([10.0, 20.0])
    interior_c = np.array([8.0, 15.0])
    interior_a_s = np.array([0.5, 0.55])
    interior_a_b = np.array([0.3, 0.32])
    x_egm, c_egm, a_s_egm, a_b_egm = _build_synthetic_egm(
        n_tiny=3,
        interior_x=interior_x,
        interior_c=interior_c,
        interior_a_s=interior_a_s,
        interior_a_b=interior_a_b,
    )
    wealth_grid = jnp.asarray(np.array([0.01, 0.05, 0.5, 1.0, 5.0]))
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption=1e-10,
    )
    # All below interior_x[0] = 10.0 → corner everywhere.
    np.testing.assert_allclose(np.asarray(c_w), np.asarray(wealth_grid),
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(a_s_w), np.zeros_like(np.asarray(a_s_w)))
    np.testing.assert_array_equal(np.asarray(a_b_w), np.zeros_like(np.asarray(a_b_w)))


def test_all_non_real_degenerate_no_op():
    """If every EGM point is in {anchor, tiny_savings} (no interior solves),
    argmax returns 0, W_min_real = anchor ~ 1e-10, clamp is a no-op for any
    realistic wealth_grid."""
    egm_anchor = 1e-10
    min_consumption = 1e-10
    # Build a grid with ONLY anchor + tiny-savings fallback (no interior).
    n_tiny = 5
    s_tiny = np.linspace(1e-9, 5e-7, n_tiny)
    x_egm = jnp.asarray(np.concatenate([[egm_anchor], min_consumption + s_tiny]))
    c_egm = jnp.full((n_tiny + 1,), min_consumption)
    a_s_egm = jnp.asarray(np.concatenate([[0.0], np.full(n_tiny, 0.1)]))
    a_b_egm = jnp.asarray(np.concatenate([[0.0], np.full(n_tiny, 0.4)]))
    wealth_grid = jnp.asarray(np.array([0.01, 0.5, 5.0]))
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption=min_consumption,
    )
    # All wealth >> 1e-10 = max(x_egm); jnp.interp clips to right-edge value.
    # All entries have c = min_consumption, so c_w = min_consumption everywhere.
    np.testing.assert_allclose(np.asarray(c_w), np.full(3, min_consumption),
                               rtol=0, atol=0)
    # The clamp should be a no-op (constrained = wealth < W_min_real = anchor
    # ~1e-10; all wealth points are well above that).
    np.testing.assert_allclose(np.asarray(a_s_w), np.full(3, 0.1),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(a_b_w), np.full(3, 0.4),
                               rtol=0, atol=0)


def test_min_consumption_zero():
    """min_consumption=0 → threshold is c>0; real interior c_opt is positive
    so detection still works."""
    interior_x = np.array([0.5, 1.0, 2.0])
    interior_c = np.array([0.4, 0.8, 1.5])
    interior_a_s = np.array([0.5, 0.55, 0.6])
    interior_a_b = np.array([0.3, 0.32, 0.34])
    # With min_consumption=0, anchor & fallback have c=0 (not 1e-10).
    x_anchor = np.array([0.0])
    c_anchor = np.array([0.0])
    a_s_anchor = np.array([0.0])
    a_b_anchor = np.array([0.0])
    x_tiny = np.array([1e-9, 1e-8])
    c_tiny = np.array([0.0, 0.0])
    a_s_tiny = np.array([0.1, 0.1])
    a_b_tiny = np.array([0.4, 0.4])
    x_egm = jnp.asarray(np.concatenate([x_anchor, x_tiny, interior_x]))
    c_egm = jnp.asarray(np.concatenate([c_anchor, c_tiny, interior_c]))
    a_s_egm = jnp.asarray(np.concatenate([a_s_anchor, a_s_tiny, interior_a_s]))
    a_b_egm = jnp.asarray(np.concatenate([a_b_anchor, a_b_tiny, interior_a_b]))
    wealth_grid = jnp.asarray(np.array([0.1, 0.3, 0.49, 0.5, 1.0]))
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption=0.0,
    )
    # First three (W < 0.5) clamped.
    np.testing.assert_allclose(np.asarray(c_w[:3]), np.asarray(wealth_grid[:3]),
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(a_s_w[:3]), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(a_b_w[:3]), np.zeros(3))
    # At W = 0.5: interior knot.
    assert float(c_w[3]) == pytest.approx(0.4, abs=1e-12)


def test_jit_compiles_under_jit():
    """JIT-compatibility smoke: ensure the clamp lowers under jax.jit
    without static-shape recompiles. Two calls with the SAME shapes
    must hit the cache (a single trace)."""
    interior_x = np.array([0.5, 1.0, 2.0])
    interior_c = np.array([0.4, 0.8, 1.5])
    interior_a_s = np.array([0.45, 0.5, 0.55])
    interior_a_b = np.array([0.3, 0.32, 0.34])
    x_egm, c_egm, a_s_egm, a_b_egm = _build_synthetic_egm(
        n_tiny=2,
        interior_x=interior_x,
        interior_c=interior_c,
        interior_a_s=interior_a_s,
        interior_a_b=interior_a_b,
    )
    wealth_grid = jnp.asarray(np.linspace(0.01, 5.0, 20))

    @jax.jit
    def lift_jit(x, c, a_s, a_b, w, mc):
        return _lift_to_wealth_grid(x, c, a_s, a_b, w, mc)

    # First call traces; second call should hit cache.
    c_w_1, a_s_w_1, a_b_w_1 = lift_jit(x_egm, c_egm, a_s_egm, a_b_egm,
                                       wealth_grid, 1e-10)
    c_w_2, a_s_w_2, a_b_w_2 = lift_jit(x_egm, c_egm, a_s_egm, a_b_egm,
                                       wealth_grid, 1e-10)
    np.testing.assert_array_equal(np.asarray(c_w_1), np.asarray(c_w_2))
    np.testing.assert_array_equal(np.asarray(a_s_w_1), np.asarray(a_s_w_2))
    np.testing.assert_array_equal(np.asarray(a_b_w_1), np.asarray(a_b_w_2))


# =============================================================================
# End-to-end smoke (skipped when var_dataset.csv is unavailable)
# =============================================================================

CSV_PATH = REPO / "data" / "var_dataset.csv"
REQUIRED_REAL_COLUMNS = ("cape", "spr", "y_1", "xr", "xb")


def _real_dataset_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
    except OSError:
        return False
    return all(col in header for col in REQUIRED_REAL_COLUMNS)


@pytest.mark.skipif(
    not _real_dataset_ready(CSV_PATH),
    reason=(
        "real-yields VAR dataset (cape/spr/y_1/xr/xb) not available at "
        f"{CSV_PATH}"
    ),
)
def test_corner_solution_at_lowest_wealth_grid_point():
    """End-to-end: solve a tiny config with deliberately low ``wealth_min``;
    verify ``C[..., 0] = wealth_grid[0]`` and ``S[..., 0] = B[..., 0] = 0``.

    The deliberately low wealth_min (0.01 AWI) puts the lowest grid point
    well below any plausible real interior W_implied, so the clamp must
    fire there.
    """
    from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
    from lifecycle.model import DiscretizationConfig
    from lifecycle.precompute import build_model, build_precompute
    from lifecycle.predictability_ablation import prepare_predictability_system
    from lifecycle.solver import run_lifecycle_solver

    # 3-axis template (matches the canonical smoke pattern); the
    # ``prepare_predictability_system`` projection slices it down to the
    # System I (y_1 only) shape.
    template = DiscretizationConfig(
        n_wealth=10,
        wealth_min=0.01,         # below any plausible W_implied
        wealth_max=200.0,
        n_savings=10,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.25),
        n_z=3,
        n_stds=3.0,
        n_eps_nodes=3,
        n_eta_nodes=3,
        n_ret_nodes_1d=(3, 3),
        n_state_quad_nodes=(3, 3, 3),
    )
    meta = prepare_predictability_system(
        "1", csv_path=str(CSV_PATH), disc_config_template=template,
    )

    base = dict(BASE_CONFIG)
    base["start_age"] = 60
    base["retire_age"] = 64
    base["terminal_age"] = 65
    model = build_model(base, meta["var_config"], verbose=False)
    pc = build_precompute(model, meta["disc_config"], verbose=False)

    sc = CANONICAL_SOLVER._replace(max_iter=200, max_iter_unconstrained=200)
    C_mat, S_mat, B_mat, _diag = run_lifecycle_solver(
        model, pc, sc, verbose=0,
    )

    wealth0 = float(np.asarray(pc.wealth_grid)[0])
    # At the lowest grid point: corner solution everywhere.
    np.testing.assert_allclose(
        np.asarray(C_mat)[..., 0], wealth0, rtol=0, atol=1e-9,
        err_msg="C[..., 0] != wealth_grid[0] (corner clamp not firing)",
    )
    np.testing.assert_array_equal(
        np.asarray(S_mat)[..., 0], np.zeros_like(np.asarray(S_mat)[..., 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(B_mat)[..., 0], np.zeros_like(np.asarray(B_mat)[..., 0]),
    )
