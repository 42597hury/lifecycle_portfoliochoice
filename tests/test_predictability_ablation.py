from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model import DiscretizationConfig, annuity_factor
from predictability_ablation import (
    get_predictability_system_spec,
    prepare_predictability_system,
)
from precompute import Precompute, build_model
from simulation import simulate_lifecycle
from solver import run_lifecycle_solver
from var import (
    build_iid_var_config,
    build_no_cy_var_config,
    build_nominal_system1_var_config,
    build_nominal_system1_var_config_hardcoded,
    build_y1_only_var_config,
)


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = str(ROOT / "data" / "var_dataset.csv")


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


def _small_disc_3d() -> DiscretizationConfig:
    return DiscretizationConfig(
        n_wealth=10,
        n_savings=10,
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


def _notebook_disc_template() -> DiscretizationConfig:
    return DiscretizationConfig(
        n_wealth=150,
        n_savings=150,
        state_grid_sizes=(5, 5, 5),
        state_grid_mode="principal",
        state_n_stds=(0.6, 1.75, 2.0),
        n_z=9,
        n_stds=3.0,
        n_eps_nodes=3,
        n_eta_nodes=3,
        n_ret_nodes_1d=(3, 5, 3),
        n_state_quad_nodes=(2, 2, 5),
    )


def _small_retired_base_config() -> dict:
    base = _reference_base_config()
    base["start_age"] = 94
    base["retire_age"] = 95
    base["terminal_age"] = 99
    return base


def _disc_for_builder(builder_name: str) -> DiscretizationConfig:
    common = dict(
        n_wealth=8,
        n_savings=8,
        wealth_max=30.0,
        savings_max=30.0,
        state_grid_mode="principal",
        state_n_stds=2.0,
        n_z=3,
        n_stds=2.0,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=1,
    )
    if builder_name == "build_iid_var_config":
        return DiscretizationConfig(
            state_grid_sizes=(1,),
            n_state_quad_nodes=(1,),
            **common,
        )
    if builder_name == "build_y1_only_var_config":
        return DiscretizationConfig(
            state_grid_sizes=(3,),
            n_state_quad_nodes=2,
            **common,
        )
    if builder_name == "build_no_cy_var_config":
        return DiscretizationConfig(
            state_grid_sizes=(3, 3),
            n_state_quad_nodes=(2, 2),
            **common,
        )
    if builder_name == "build_nominal_system1_var_config":
        return DiscretizationConfig(
            state_grid_sizes=(3, 3, 3),
            n_state_quad_nodes=2,
            **common,
        )
    raise ValueError(f"unknown builder {builder_name}")


def _clone_cfg(cfg: dict) -> dict:
    out = dict(cfg)
    for key in ("Phi", "Omega", "z_bar", "const"):
        if key in out and out[key] is not None:
            out[key] = np.array(out[key], copy=True)
    return out


@pytest.mark.parametrize(
    "token, expected_code, expected_builder_name",
    [
        ("I", "I", "build_iid_var_config"),
        ("iid", "I", "build_iid_var_config"),
        ("2", "II", "build_y1_only_var_config"),
        ("system_iii", "III", "build_no_cy_var_config"),
        ("baseline", "IV", "build_nominal_system1_var_config"),
    ],
)
def test_predictability_system_aliases(token, expected_code, expected_builder_name):
    spec = get_predictability_system_spec(token)
    assert spec.code == expected_code
    assert spec.builder.__name__ == expected_builder_name


@pytest.mark.parametrize(
    "system, expected_state_names, expected_grid_sizes, expected_quad_nodes, expected_state_n_stds",
    [
        ("I", ("dummy",), (1,), (1,), (1.0,)),
        ("II", ("y_1",), (5,), (5,), (2.0,)),
        ("III", ("spr", "y_1"), (5, 5), (2, 5), (1.75, 2.0)),
        ("IV", ("cy", "spr", "y_1"), (5, 5, 5), (2, 2, 5), (0.6, 1.75, 2.0)),
    ],
)
def test_prepare_predictability_system_projects_notebook_disc_template(
    system,
    expected_state_names,
    expected_grid_sizes,
    expected_quad_nodes,
    expected_state_n_stds,
):
    setup = prepare_predictability_system(
        system,
        csv_path=CSV_PATH,
        disc_config_template=_notebook_disc_template(),
    )
    disc = setup["disc_config"]

    assert setup["state_names"] == expected_state_names
    assert disc.state_grid_sizes == expected_grid_sizes
    assert disc.n_state_quad_nodes == expected_quad_nodes
    assert disc.state_n_stds == expected_state_n_stds


def test_prepare_predictability_system_i_precompute_succeeds():
    base = _small_retired_base_config()
    setup = prepare_predictability_system(
        "I",
        csv_path=CSV_PATH,
        disc_config_template=_notebook_disc_template(),
    )

    model = build_model(base, setup["var_config"], verbose=False)
    pc = Precompute(model, setup["disc_config"], verbose=False)

    assert model.state_names == ("dummy",)
    assert pc.N_state == 1
    np.testing.assert_allclose(pc.state_grid[:, 0], 0.0, atol=1e-12)


def test_build_model_accepts_system_i_ii_iii_configs():
    base = _reference_base_config()
    cases = [
        (build_iid_var_config, 1, ("dummy",), None, None),
        (build_y1_only_var_config, 1, ("y_1",), 0, None),
        (build_no_cy_var_config, 2, ("spr", "y_1"), 1, 0),
    ]

    for builder, n_state, state_names, y1_idx, spr_idx in cases:
        cfg, _, _ = builder(csv_path=CSV_PATH)
        model = build_model(base, cfg, verbose=False)
        assert model.n_state == n_state
        assert model.state_names == state_names
        assert model.y_1_index_in_state == y1_idx
        assert model.spr_index_in_state == spr_idx


def test_builder_scalar_fallbacks_match_dataset_means():
    df = pd.read_csv(CSV_PATH)
    y_1_mean = float(df["y_1"].mean())
    spr_mean = float(df["spr"].mean())

    cfg_i, _, _ = build_iid_var_config(csv_path=CSV_PATH)
    assert cfg_i["y_1_scalar_fallback"] == pytest.approx(y_1_mean, abs=1e-12)
    assert cfg_i["spr_scalar_fallback"] == pytest.approx(spr_mean, abs=1e-12)

    cfg_ii, _, _ = build_y1_only_var_config(csv_path=CSV_PATH)
    assert cfg_ii["y_1_scalar_fallback"] is None
    assert cfg_ii["spr_scalar_fallback"] == pytest.approx(spr_mean, abs=1e-12)

    cfg_iii, _, _ = build_no_cy_var_config(csv_path=CSV_PATH)
    assert cfg_iii["y_1_scalar_fallback"] is None
    assert cfg_iii["spr_scalar_fallback"] is None


def test_iid_var_has_zero_predictive_channels_and_sample_covariance():
    cfg, _, _ = build_iid_var_config(csv_path=CSV_PATH)

    state_idx = np.asarray(cfg["state_indices"], dtype=int)
    ret_idx = np.asarray(cfg["return_indices"], dtype=int)
    phi_21 = cfg["Phi"][np.ix_(ret_idx, state_idx)]
    sigma_rs = cfg["Omega"][np.ix_(ret_idx, state_idx)]

    np.testing.assert_array_equal(phi_21, 0.0)
    np.testing.assert_array_equal(sigma_rs, 0.0)

    df = pd.read_csv(CSV_PATH)
    sample_cov = df[["rtb", "xr", "xb"]].cov(ddof=1).to_numpy()
    sigma_rr = cfg["Omega"][np.ix_(ret_idx, ret_idx)]
    np.testing.assert_allclose(sigma_rr, sample_cov, rtol=1e-9, atol=1e-12)


def test_return_means_invariant_across_all_four_systems():
    cfgs = [
        build_iid_var_config(csv_path=CSV_PATH)[0],
        build_y1_only_var_config(csv_path=CSV_PATH)[0],
        build_no_cy_var_config(csv_path=CSV_PATH)[0],
        build_nominal_system1_var_config(csv_path=CSV_PATH)[0],
    ]
    z_bar_rets = [
        cfg["z_bar"][np.asarray(cfg["return_indices"], dtype=int)]
        for cfg in cfgs
    ]

    for idx in range(1, len(z_bar_rets)):
        np.testing.assert_allclose(
            z_bar_rets[idx],
            z_bar_rets[0],
            atol=1e-12,
            err_msg=f"system {idx + 1} return means differ from System I",
        )


def test_precompute_uses_scalar_annuity_fallbacks():
    cfg = _clone_cfg(build_nominal_system1_var_config_hardcoded())
    cfg["y_1_index_in_state"] = None
    cfg["spr_index_in_state"] = None
    cfg["y_1_scalar_fallback"] = 0.04
    cfg["spr_scalar_fallback"] = 0.01

    model = build_model(_reference_base_config(), cfg, verbose=False)
    pc = Precompute(model, _small_disc_3d(), verbose=False)

    expected = float(annuity_factor(0.04, 0.01, model.b_bar))
    assert pc.annuity_factors.shape == (pc.N_state,)
    np.testing.assert_allclose(pc.annuity_factors, expected, atol=1e-12)


def test_precompute_scalar_annuity_depends_on_spread():
    cfg_lo = _clone_cfg(build_nominal_system1_var_config_hardcoded())
    cfg_hi = _clone_cfg(build_nominal_system1_var_config_hardcoded())

    for cfg, spr in ((cfg_lo, 0.00), (cfg_hi, 0.03)):
        cfg["y_1_index_in_state"] = None
        cfg["spr_index_in_state"] = None
        cfg["y_1_scalar_fallback"] = 0.04
        cfg["spr_scalar_fallback"] = spr

    model_lo = build_model(_reference_base_config(), cfg_lo, verbose=False)
    model_hi = build_model(_reference_base_config(), cfg_hi, verbose=False)
    pc_lo = Precompute(model_lo, _small_disc_3d(), verbose=False)
    pc_hi = Precompute(model_hi, _small_disc_3d(), verbose=False)

    assert np.all(pc_lo.annuity_factors > 0.0)
    assert np.all(pc_hi.annuity_factors > 0.0)
    assert not np.allclose(pc_lo.annuity_factors, pc_hi.annuity_factors)


@pytest.mark.parametrize(
    "builder, expected_n_state",
    [
        (build_iid_var_config, 1),
        (build_y1_only_var_config, 1),
        (build_no_cy_var_config, 2),
        (build_nominal_system1_var_config, 3),
    ],
)
def test_each_system_solves_small_retired_problem(builder, expected_n_state):
    base = _small_retired_base_config()
    cfg, _, _ = builder(csv_path=CSV_PATH)
    model = build_model(base, cfg, verbose=False)
    disc = _disc_for_builder(builder.__name__)
    pc = Precompute(model, disc, verbose=False)

    assert model.n_state == expected_n_state
    C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(
        model,
        pc,
        verbose=0,
    )

    assert diagnostics["solve_status"] == "complete"
    assert np.all(np.isfinite(C_mat))
    assert np.all(np.isfinite(S_mat))
    assert np.all(np.isfinite(B_mat))
    assert np.all(S_mat >= -1e-10)
    assert np.all(B_mat >= -1e-10)
    assert np.all(S_mat + B_mat <= 1.0 + 1e-8)


@pytest.mark.parametrize(
    "builder",
    [build_iid_var_config, build_no_cy_var_config],
)
def test_low_dimensional_systems_simulate(builder):
    base = _small_retired_base_config()
    cfg, _, _ = builder(csv_path=CSV_PATH)
    model = build_model(base, cfg, verbose=False)
    disc = _disc_for_builder(builder.__name__)
    pc = Precompute(model, disc, verbose=False)
    C_mat, S_mat, B_mat, _ = run_lifecycle_solver(model, pc, verbose=0)

    sim = simulate_lifecycle(
        C_mat,
        S_mat,
        B_mat,
        pc,
        model,
        n_simulations=25,
        initial_z="median",
        initial_state="stationary",
        verbose=False,
    )

    assert sim["state_coords"].shape == (25, pc.n_age, model.n_state)
    assert sim["state_idx"].shape == (25, pc.n_age)
    assert np.all(np.isfinite(sim["estate"]))
    if builder is build_iid_var_config:
        np.testing.assert_allclose(sim["state_coords"], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (
            lambda cfg: (
                cfg.update(y_1_index_in_state=None, y_1_scalar_fallback=None),
                cfg,
            )[1],
            "either y_1_index_in_state or y_1_scalar_fallback",
        ),
        (
            lambda cfg: (
                cfg.update(spr_index_in_state=None, spr_scalar_fallback=None),
                cfg,
            )[1],
            "either spr_index_in_state or spr_scalar_fallback",
        ),
        (
            lambda cfg: (
                cfg.update(y_1_index_in_state=99, y_1_scalar_fallback=None),
                cfg,
            )[1],
            "out of bounds",
        ),
        (
            lambda cfg: (
                cfg.update(spr_index_in_state=99, spr_scalar_fallback=None),
                cfg,
            )[1],
            "out of bounds",
        ),
        (
            lambda cfg: (
                cfg.update(y_1_index_in_state=0, spr_index_in_state=0),
                cfg,
            )[1],
            "must be distinct",
        ),
    ],
)
def test_build_model_rejects_malformed_annuity_index_configs(mutator, match):
    cfg = _clone_cfg(build_nominal_system1_var_config_hardcoded())
    cfg = mutator(cfg)
    with pytest.raises(ValueError, match=match):
        build_model(_reference_base_config(), cfg, verbose=False)
