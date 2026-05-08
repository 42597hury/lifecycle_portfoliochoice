"""Phase 2 of the real-yields pivot: precompute on 3-axis state.

Verifies that build_model + build_precompute work for each real-yields
system at a tiny config. Does not exercise the solver — that comes in
Phase 3 / Phase 6.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lifecycle.model import DiscretizationConfig
from lifecycle.precompute import build_model, build_precompute
from lifecycle.predictability_ablation import prepare_predictability_system

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


# Minimal economic calibration. Pulled from configs/_canonical.py but
# kept inline so the test does not depend on the canonical config that
# Phase 5 will rewrite.
_BASE_CONFIG = {
    "beta": 0.96, "gamma": 5.0, "b_bar": 10,
    "start_age": 22, "retire_age": 67, "terminal_age": 99,
    "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
    "rho": 0.991, "pz": 0.176,
    "mu_eta1": -0.524, "sigma_eta1": 0.113,
    "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524), "sigma_eta2": 0.046,
    "pe": 0.044,
    "mu_eps1": 0.134, "sigma_eps1": 0.762,
    "mu_eps2": 0.0, "sigma_eps2": 0.055,
}


def _tiny_template_disc():
    """3-axis (cape, spr, y_1) tiny-config template for precompute smoke."""
    return DiscretizationConfig(
        n_wealth=10,
        wealth_min=0.13,
        wealth_max=200.0,
        n_savings=10,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.0, 2.0),
        n_z=3,
        n_stds=3.0,
        n_eps_nodes=3,
        n_eta_nodes=3,
        n_ret_nodes_1d=(3, 3),
        n_state_quad_nodes=(3, 3, 3),
    )


@pytest.mark.parametrize(
    "code, expected_state_names, expected_grid_sizes, expected_quad_nodes",
    [
        ("1",    ("y_1",),                (3,),     (3,)),
        ("2",    ("spr", "y_1"),          (3, 3),   (3, 3)),
        ("full", ("cape", "spr", "y_1"),  (3, 3, 3), (3, 3, 3)),
    ],
)
def test_precompute_builds_for_each_system(
    code, expected_state_names, expected_grid_sizes, expected_quad_nodes,
):
    template = _tiny_template_disc()
    meta = prepare_predictability_system(
        code,
        csv_path=str(CSV_PATH),
        disc_config_template=template,
    )
    assert meta["state_names"] == expected_state_names

    disc = meta["disc_config"]
    assert disc.state_grid_sizes == expected_grid_sizes
    assert disc.n_state_quad_nodes == expected_quad_nodes

    model = build_model(_BASE_CONFIG, meta["var_config"], verbose=False)
    assert model.n_state == len(expected_state_names)
    assert model.n_ret == 2  # xr, xb
    assert tuple(model.state_names) == expected_state_names
    assert tuple(model.ret_names) == ("xr", "xb")

    # Real-yields invariants: rtb axis is gone, y_1 lives on the state grid
    # as the bill anchor.
    assert model.rtb_index_in_state is None
    assert model.y_1_index_in_state is not None
    assert model.state_names[model.y_1_index_in_state] == "y_1"

    # Sigma sizes match the new partition.
    assert model.Phi_11.shape == (len(expected_state_names), len(expected_state_names))
    assert model.Sigma_ss.shape == (len(expected_state_names), len(expected_state_names))
    assert model.Sigma_rr.shape == (2, 2)

    pc = build_precompute(model, disc, verbose=False)

    n_state = len(expected_state_names)
    N_state = int(np.prod(expected_grid_sizes))
    n_state_quad = int(np.prod(expected_quad_nodes))

    assert pc.N_state == N_state
    assert pc.state_grid.shape == (N_state, n_state)
    assert pc.Pi_state.shape == (N_state, N_state)
    assert pc.mu_r.shape == (N_state, N_state, 2)
    assert pc.v_nodes.shape == (n_state_quad, n_state)
    assert pc.M_v_nodes.shape == (n_state_quad, 2)
    assert pc.annuity_factors.shape == (N_state,)
    assert np.all(np.isfinite(pc.state_grid))
    assert np.all(np.isfinite(pc.mu_r))


def test_build_model_rejects_missing_y_1_in_real_yields_setup():
    """If rtb_index_in_state=None then y_1 MUST live on the grid."""
    # Build a deliberately broken var_config: real-yields shape but y_1 is
    # supplied only as a scalar fallback. The model has no bill anchor on
    # the grid, so build_model should refuse.
    bad_var_config = {
        "z_bar": np.zeros(5),
        "Phi": np.zeros((5, 5)),
        "Omega": np.eye(5),
        "const": np.zeros(5),
        "variable_names": ["cape", "spr", "y_1", "xr", "xb"],
        "state_indices": [0, 1, 2],
        "return_indices": [3, 4],
        "y_1_index_in_state": None,            # broken: y_1 not on grid
        "spr_index_in_state": 1,
        "rtb_index_in_state": None,            # real-yields setup
        "y_1_scalar_fallback": 0.012,
        "spr_scalar_fallback": None,
    }
    with pytest.raises(ValueError, match="y_1 to live on the state grid"):
        build_model(_BASE_CONFIG, bad_var_config, verbose=False)
