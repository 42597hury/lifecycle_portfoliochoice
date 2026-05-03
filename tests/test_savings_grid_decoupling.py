"""
Regression tests for decoupling the EGM savings grid from the wealth grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from lifecycle.model import DiscretizationConfig
from lifecycle.policy_io import load_policy_bundle, save_policy_bundle
from lifecycle.precompute import Precompute, build_model
from lifecycle.var import build_nominal_system1_var_config


def _reference_base_config() -> dict:
    return {
        "beta": 0.96,
        "gamma": 3.0,
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


def _build_reference_model():
    var_config, _, _ = build_nominal_system1_var_config(csv_path=str(ROOT / "data" / "var_dataset.csv"))
    return build_model(_reference_base_config(), var_config, verbose=False)


def test_default_savings_grid_still_uses_wealth_max():
    model = _build_reference_model()
    disc = DiscretizationConfig(n_wealth=12, wealth_max=200.0, n_savings=10)
    pc = Precompute(model, disc, verbose=False)

    assert abs(pc.wealth_grid[-1] - 200.0) < 1e-12
    assert abs(pc.s_grid[-1] - 200.0) < 1e-12
    assert abs(pc.savings_max - 200.0) < 1e-12


def test_explicit_savings_max_decouples_grids_and_regeneration():
    model = _build_reference_model()
    disc = DiscretizationConfig(
        n_wealth=12,
        wealth_max=500.0,
        n_savings=10,
        savings_min=1e-8,
        savings_max=200.0,
    )
    pc = Precompute(model, disc, verbose=False)

    assert abs(pc.wealth_grid[-1] - 500.0) < 1e-12
    assert abs(pc.s_grid[-1] - 200.0) < 1e-12

    regen = pc.regenerate_savings_grid(7)
    assert len(regen) == 7
    assert abs(regen[0] - disc.savings_min) < 1e-20
    assert abs(regen[-1] - 200.0) < 1e-12


def test_invalid_savings_max_above_wealth_max_is_rejected():
    model = _build_reference_model()
    disc = DiscretizationConfig(wealth_max=200.0, savings_max=250.0)

    try:
        Precompute(model, disc, verbose=False)
    except ValueError as exc:
        assert "savings_max cannot exceed wealth_max" in str(exc)
    else:
        raise AssertionError("Expected ValueError when savings_max > wealth_max")


def test_policy_bundle_roundtrip_preserves_decoupled_savings_max(tmp_path):
    disc = DiscretizationConfig(
        n_wealth=12,
        wealth_max=500.0,
        n_savings=10,
        savings_min=1e-8,
        savings_max=200.0,
        state_grid_sizes=(2, 2, 2),
        n_z=5,
        n_eps_nodes=1,
        n_eta_nodes=1,
        n_ret_nodes_1d=(1, 1, 1),
        n_state_quad_nodes=1,
    )

    bundle_dir = tmp_path / "decoupled_bundle"
    shape = (3, disc.n_z, 8, disc.n_wealth)
    C = np.zeros(shape, dtype=np.float64)
    S = np.zeros(shape, dtype=np.float64)
    B = np.zeros(shape, dtype=np.float64)

    save_policy_bundle(
        bundle_dir,
        C,
        S,
        B,
        run_config={"discretization_config": disc},
        overwrite=False,
    )

    _C, _S, _B, _diag, meta = load_policy_bundle(bundle_dir)
    disc_meta = meta["run_config"]["discretization_config"]

    assert disc_meta["wealth_max"] == 500.0
    assert disc_meta["savings_max"] == 200.0

    rebuilt_disc = DiscretizationConfig(
        n_wealth=disc_meta["n_wealth"],
        wealth_min=disc_meta["wealth_min"],
        wealth_max=disc_meta["wealth_max"],
        n_savings=disc_meta["n_savings"],
        savings_min=disc_meta["savings_min"],
        savings_max=disc_meta.get("savings_max"),
        state_grid_sizes=tuple(disc_meta["state_grid_sizes"]),
        state_grid_mode=disc_meta["state_grid_mode"],
        state_n_stds=disc_meta["state_n_stds"],
        n_z=disc_meta["n_z"],
        n_stds=disc_meta["n_stds"],
        n_eps_nodes=disc_meta["n_eps_nodes"],
        n_eta_nodes=disc_meta["n_eta_nodes"],
        n_ret_nodes_1d=tuple(disc_meta["n_ret_nodes_1d"]),
        n_state_quad_nodes=disc_meta["n_state_quad_nodes"],
    )

    assert rebuilt_disc.savings_max == 200.0
