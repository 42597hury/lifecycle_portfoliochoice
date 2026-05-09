"""Phase 4 of the real-yields pivot: simulator return formulas.

Pins the simulator-side mirror of the solver bill-leg invariant: when the
portfolio is 100% in bills (alpha_s = alpha_b = 0 everywhere), the realised
gross portfolio return equals exp(s_t[y_1_idx]) — i.e. the real bill rate
read off the CURRENT-period state.
"""
from __future__ import annotations

import inspect
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
from lifecycle import simulation as sim_mod

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


# mu_eta2 / mu_eps2 are derived inside `build_model` from the zero-mean
# constraint E[eta] = E[eps] = 0 (Fix A in
# docs/scans/INCOME_PIPELINE_REVIEW_2026-05-09.md), so we don't pass them.
_BASE_CONFIG = {
    "beta": 0.96, "gamma": 5.0, "b_bar": 10,
    "start_age": 22, "retire_age": 67, "terminal_age": 99,
    "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
    "rho": 0.991, "pz": 0.176,
    "mu_eta1": -0.524, "sigma_eta1": 0.113, "sigma_eta2": 0.046,
    "pe": 0.044,
    "mu_eps1": 0.134, "sigma_eps1": 0.762, "sigma_eps2": 0.055,
}


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


def test_build_simulate_kernel_signature_uses_y_1_idx():
    """rtb_idx parameter is gone; y_1_idx replaces it."""
    sig = inspect.signature(sim_mod._build_simulate_kernel)
    params = sig.parameters
    assert "y_1_idx" in params
    assert "rtb_idx" not in params


def test_simulator_n_state_4_rejected():
    """Real-yields simulator only supports n_state in {1, 2, 3}."""
    template = _tiny_disc()
    meta = prepare_predictability_system(
        "full", csv_path=str(CSV_PATH), disc_config_template=template,
    )
    model = build_model(_BASE_CONFIG, meta["var_config"], verbose=False)
    pc = build_precompute(model, meta["disc_config"], verbose=False)

    # Spoof an n_state=4 model by replacing the n_state field on the model
    # NamedTuple. The guard fires at the entry of simulate_lifecycle before
    # any kernel build, so the inconsistent precompute shape is not exercised.
    fake_model = model._replace(n_state=4)

    n_age = pc.n_age
    n_z = pc.n_z
    N_state = pc.N_state
    n_w = pc.n_w
    C = np.zeros((n_age, n_z, N_state, n_w))
    S = np.zeros_like(C)
    B = np.zeros_like(C)

    with pytest.raises(NotImplementedError, match="n_state in"):
        sim_mod.simulate_lifecycle(
            C_mat=C, S_mat=S, B_mat=B, pc=pc, model=fake_model,
            n_simulations=2, seed=0, verbose=False,
        )


@pytest.mark.parametrize("code", ["1", "2", "full"])
def test_bill_only_portfolio_realises_y_1(code):
    """Bill-only policy (alpha_s=alpha_b=0) yields R_port == exp(s_t[y_1_idx]).

    This is the simulator-side mirror of the solver's deterministic-bill
    invariant: under a 100%-bill portfolio the Itô correction terms zero
    out, so log_R_port collapses to log_R_bill = s_t[y_1_idx]. We sample
    several households over the full lifecycle and confirm the per-period
    R_port matches exp(state_coords[h, t, y_1_idx]) at every alive cell.
    """
    template = _tiny_disc()
    meta = prepare_predictability_system(
        code, csv_path=str(CSV_PATH), disc_config_template=template,
    )
    model = build_model(_BASE_CONFIG, meta["var_config"], verbose=False)
    pc = build_precompute(model, meta["disc_config"], verbose=False)

    # Bill-only policy across the full grid: alpha_s = alpha_b = 0,
    # consume half of cash-on-hand so wealth stays positive throughout.
    n_age = pc.n_age
    n_z = pc.n_z
    N_state = pc.N_state
    n_w = pc.n_w
    wealth_grid = np.asarray(pc.wealth_grid)
    C = np.broadcast_to(0.5 * wealth_grid, (n_age, n_z, N_state, n_w)).copy()
    S = np.zeros((n_age, n_z, N_state, n_w))
    B = np.zeros((n_age, n_z, N_state, n_w))

    panel = sim_mod.simulate_lifecycle(
        C_mat=C, S_mat=S, B_mat=B, pc=pc, model=model,
        n_simulations=4, seed=42,
        # Use a deterministic single-state init so the test is reproducible.
        initial_wealth=10.0,
        initial_state="median",
        initial_z="median",
        return_draw_mode="quadrature",
        verbose=False,
    )

    R_port = np.asarray(panel["R_port"])              # (n_simulations, n_age)
    state_coords = np.asarray(panel["state_coords"])  # (n_simulations, n_age, n_state)
    alive = np.asarray(panel["alive"]).astype(bool)   # (n_simulations, n_age)

    y_1_idx = int(model.y_1_index_in_state)
    expected_log_R = state_coords[:, :, y_1_idx]
    actual_log_R = np.log(np.maximum(R_port, 1e-300))

    if alive.any():
        diff = np.abs(actual_log_R[alive] - expected_log_R[alive])
        assert float(diff.max()) < 1e-12, (
            f"R_port deviates from exp(s_t[y_1_idx]) under a 100%-bill policy: "
            f"max|diff| = {diff.max():.3e}"
        )
