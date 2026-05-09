"""Phase 6 of the real-yields pivot: end-to-end smoke.

For each of System 1 / System 2 / Full System: build precompute, run the
lifecycle solver to convergence on a tiny config, save the bundle, reload
it, and verify the policies are finite and within sensible bounds. This
is the per-handoff Phase 2/Phase 6 acceptance smoke.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.policy_io import save_policy_bundle, load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.predictability_ablation import prepare_predictability_system
from lifecycle.simulation import simulate_lifecycle
from lifecycle.solver import run_lifecycle_solver


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


pytestmark = pytest.mark.skipif(
    not _real_dataset_ready(CSV_PATH),
    reason=(
        "real-yields VAR dataset (cape/spr/y_1/xr/xb) not yet committed at "
        f"{CSV_PATH}"
    ),
)


def _smoke_disc_template():
    """3-axis (cape, spr, y_1) tiny-smoke template per the pivot handoff."""
    return DiscretizationConfig(
        n_wealth=20,
        wealth_min=0.13,
        wealth_max=200.0,
        n_savings=20,
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


def _smoke_base_config():
    base = dict(BASE_CONFIG)
    # Tiny-smoke shrink: 5-period lifecycle, retire 1 period before terminal.
    base["start_age"] = 60
    base["retire_age"] = 64
    base["terminal_age"] = 65
    return base


def _smoke_solver_config():
    return CANONICAL_SOLVER._replace(max_iter=200)


@pytest.mark.parametrize(
    "code, expected_state_names, expected_grid_sizes",
    [
        ("1",    ("y_1",),                 (3,)),
        ("2",    ("spr", "y_1"),           (3, 3)),
        ("full", ("cape", "spr", "y_1"),   (3, 3, 3)),
    ],
)
def test_build_solve_save_reload(
    code, expected_state_names, expected_grid_sizes, tmp_path,
):
    """End-to-end: build + solve + save + reload, no NaN, sensible alphas."""
    template = _smoke_disc_template()
    meta = prepare_predictability_system(
        code, csv_path=str(CSV_PATH), disc_config_template=template,
    )

    assert meta["state_names"] == expected_state_names

    model = build_model(_smoke_base_config(), meta["var_config"], verbose=False)
    pc = build_precompute(model, meta["disc_config"], verbose=False)

    n_state = len(expected_state_names)
    N_state = int(np.prod(expected_grid_sizes))
    assert pc.N_state == N_state
    assert model.n_state == n_state
    assert model.rtb_index_in_state is None
    assert model.y_1_index_in_state is not None

    sc = _smoke_solver_config()
    C_mat, S_mat, B_mat, diag = run_lifecycle_solver(
        model, pc, sc, verbose=0,
    )

    # No NaN anywhere; alphas in a wide-but-bounded leverage window.
    assert np.all(np.isfinite(C_mat)), "C_mat has NaN/inf"
    assert np.all(np.isfinite(S_mat)), "S_mat has NaN/inf"
    assert np.all(np.isfinite(B_mat)), "B_mat has NaN/inf"
    assert np.all(C_mat >= 0.0), "negative consumption"
    assert -10.0 < float(S_mat.min()) and float(S_mat.max()) < 10.0, (
        f"alpha_s out of [-10, 10]: [{S_mat.min():.3f}, {S_mat.max():.3f}]"
    )
    assert -10.0 < float(B_mat.min()) and float(B_mat.max()) < 10.0, (
        f"alpha_b out of [-10, 10]: [{B_mat.min():.3f}, {B_mat.max():.3f}]"
    )

    # Save + reload through the bundle layer.
    bundle_dir = tmp_path / f"smoke_system_{code}"
    save_policy_bundle(
        bundle_dir,
        C_mat, S_mat, B_mat,
        diagnostics=diag,
        wealth_grid=np.asarray(pc.wealth_grid, dtype=np.float64),
        overwrite=True,
    )

    # Round-trip: bundle layer should hand back identical arrays.
    Cc, Sc, Bc, ckpt_diag, _meta = load_policy_bundle(bundle_dir)
    np.testing.assert_array_equal(Cc, C_mat)
    np.testing.assert_array_equal(Sc, S_mat)
    np.testing.assert_array_equal(Bc, B_mat)


def test_full_system_smoke_simulator_runs():
    """Full System tiny-smoke: build + solve, then simulate a small panel.

    The simulator-side correctness on the bill leg is locked in by
    test_simulator_real_yields_returns; here we just confirm the panel
    pipeline doesn't throw end-to-end on a real solved policy.
    """
    template = _smoke_disc_template()
    meta = prepare_predictability_system(
        "full", csv_path=str(CSV_PATH), disc_config_template=template,
    )
    model = build_model(_smoke_base_config(), meta["var_config"], verbose=False)
    pc = build_precompute(model, meta["disc_config"], verbose=False)

    sc = _smoke_solver_config()
    C_mat, S_mat, B_mat, _diag = run_lifecycle_solver(
        model, pc, sc, verbose=0,
    )

    panel = simulate_lifecycle(
        C_mat=C_mat, S_mat=S_mat, B_mat=B_mat,
        pc=pc, model=model,
        n_simulations=8, seed=7,
        initial_wealth=10.0,
        initial_state="median",
        initial_z="median",
        return_draw_mode="quadrature",
        verbose=False,
    )
    R_port = np.asarray(panel["R_port"])
    alive = np.asarray(panel["alive"]).astype(bool)
    assert R_port.shape[0] == 8
    if alive.any():
        # Realised R_port positive everywhere alive (CCV log-portfolio).
        assert float(R_port[alive].min()) > 0.0
        assert np.all(np.isfinite(R_port[alive]))
