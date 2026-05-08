"""Phase 3 of the real-yields pivot: solver FOC math on 3-axis state.

Pins the central invariant of the new model: the bill leg is deterministic
given the current state. Specifically, for every i_s,

    log_R_bill[i_s, k_v, k_r] == state_grid[i_s, y_1_idx]

i.e. constant across both the state-innovation and residual quadrature axes.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import build_model, build_precompute
from lifecycle.predictability_ablation import prepare_predictability_system
from lifecycle.solver import _all_is_log_returns_numpy, _pc_to_jnp

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

_SOLVER_CONFIG = SolverConfig()  # default tuning is fine for shape-checks


def _tiny_disc():
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


def _make_pc(code):
    template = _tiny_disc()
    meta = prepare_predictability_system(
        code, csv_path=str(CSV_PATH), disc_config_template=template,
    )
    model = build_model(_BASE_CONFIG, meta["var_config"], verbose=False)
    pc = build_precompute(model, meta["disc_config"], verbose=False)
    return model, pc


@pytest.mark.parametrize("code", ["1", "2", "full"])
def test_pcjax_y_1_idx_matches_model(code):
    _, pc = _make_pc(code)
    pcj = _pc_to_jnp(pc, delta=0.0)
    assert int(pcj.y_1_idx) == int(pc.model.y_1_index_in_state)
    # State name at that index is "y_1" (sanity).
    assert pc.model.state_names[pcj.y_1_idx] == "y_1"


@pytest.mark.parametrize("code", ["1", "2", "full"])
def test_log_R_bill_is_deterministic_per_state(code):
    """log_R_bill[i_s, k_v, k_r] == state_grid[i_s, y_1_idx] for all (k_v, k_r)."""
    _, pc = _make_pc(code)
    pcj = _pc_to_jnp(pc, delta=0.0)
    log_R_bill, _, _ = _all_is_log_returns_numpy(pcj)

    state_grid = np.asarray(pc.state_grid, dtype=float)
    expected_per_is = state_grid[:, int(pcj.y_1_idx)]              # (N_state,)

    # Bill leg is constant across (k_v, k_r) for each i_s.
    expected = np.broadcast_to(
        expected_per_is[:, None, None], log_R_bill.shape
    )
    assert np.allclose(log_R_bill, expected, atol=1e-15)

    # Specifically: variance across the (k_v, k_r) axes is zero per i_s.
    flat = log_R_bill.reshape(log_R_bill.shape[0], -1)
    assert np.all(flat.std(axis=1) < 1e-15)


def test_pc_to_jnp_rejects_n_state_4():
    """Real-yields solver supports n_state in {1, 2, 3}; 4-axis must raise."""
    # Cheapest way to exercise this: monkey-patch a precompute's bracket-grid
    # tuple to length 4 and confirm _pc_to_jnp raises.
    _, pc = _make_pc("full")
    fake_pc = pc._replace(state_bracket_grids=tuple(pc.state_bracket_grids) + (np.array([0.0]),))
    with pytest.raises(NotImplementedError, match="n_state in"):
        _pc_to_jnp(fake_pc, delta=0.0)
