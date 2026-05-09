from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig
from lifecycle.precompute import build_model, build_precompute
from lifecycle.policy_io import save_policy_bundle
from lifecycle.solver import _ensure_checkpoint_wealth_grid_compatible
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.wealth_grid import (
    legacy_log1p_wealth_grid,
    load_wealth_grid_from_bundle,
    validate_wealth_grid,
    wealth_grid_hash,
)


def _tiny_disc(**overrides) -> DiscretizationConfig:
    base = DiscretizationConfig(
        n_wealth=12,
        wealth_min=0.13,
        wealth_max=200.0,
        n_savings=12,
        state_grid_sizes=(2, 2, 2),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.25),
        n_z=3,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=(2, 2),
        n_state_quad_nodes=(2, 2, 2),
    )
    return base._replace(**overrides)


def _tiny_model():
    base = dict(BASE_CONFIG)
    base.update(start_age=60, retire_age=63, terminal_age=65)
    var_config = build_real_full_var_config_hardcoded()
    return build_model(base, var_config, verbose=False)


def test_default_wealth_grid_matches_legacy_log1p():
    disc = _tiny_disc()
    pc = build_precompute(_tiny_model(), disc, verbose=False)

    expected = legacy_log1p_wealth_grid(
        disc.n_wealth, disc.wealth_min, disc.wealth_max
    )
    np.testing.assert_array_equal(pc.wealth_grid, expected)
    assert pc.wealth_grid_kind == "log1p"
    assert pc.wealth_grid_source is None
    assert pc.wealth_grid_hash == wealth_grid_hash(expected)


def test_custom_wealth_grid_from_npy_and_bundle_roundtrip(tmp_path):
    disc = _tiny_disc()
    u = np.linspace(0.0, 1.0, disc.n_wealth)
    custom = disc.wealth_min + (disc.wealth_max - disc.wealth_min) * u**1.35
    custom[0] = disc.wealth_min
    custom[-1] = disc.wealth_max
    grid_path = tmp_path / "custom_grid.npy"
    np.save(grid_path, custom)

    pc = build_precompute(
        _tiny_model(), disc._replace(wealth_grid_path=str(grid_path)), verbose=False
    )
    np.testing.assert_array_equal(pc.wealth_grid, custom)
    assert pc.wealth_grid_kind == "custom"
    assert pc.wealth_grid_hash == wealth_grid_hash(custom)

    C = np.zeros((1, 1, 1, disc.n_wealth), dtype=np.float64)
    out = save_policy_bundle(
        tmp_path / "bundle",
        C,
        C,
        C,
        run_config={
            "discretization_config": disc._replace(
                wealth_grid_path=str(grid_path)
            )._asdict()
        },
        overwrite=True,
        wealth_grid=pc.wealth_grid,
    )
    loaded = load_wealth_grid_from_bundle(out)
    np.testing.assert_array_equal(loaded, custom)

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["wealth_grid_file"] == "wealth_grid.npy"
    assert metadata["wealth_grid_hash"] == wealth_grid_hash(custom)
    assert metadata["wealth_grid_kind"] == "custom"


def test_validate_wealth_grid_rejects_float32_duplicate_gap():
    grid = np.array([0.13, 100.0, 100.0 + 1e-7, 200.0], dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        validate_wealth_grid(
            grid,
            n_wealth=4,
            wealth_min=0.13,
            wealth_max=200.0,
            source="bad test grid",
        )


def test_checkpoint_wealth_grid_guard_rejects_hash_mismatch(tmp_path):
    current = legacy_log1p_wealth_grid(5, 0.13, 200.0)
    pc = SimpleNamespace(
        wealth_grid=current,
        wealth_grid_hash=wealth_grid_hash(current),
        wealth_grid_kind="log1p",
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        _ensure_checkpoint_wealth_grid_compatible(
            tmp_path,
            {"wealth_grid_hash": "not-the-current-hash"},
            {},
            pc,
        )


def test_checkpoint_wealth_grid_guard_rejects_hashless_custom(tmp_path):
    grid = np.array([0.13, 1.0, 5.0, 200.0], dtype=np.float64)
    pc = SimpleNamespace(
        wealth_grid=grid,
        wealth_grid_hash=wealth_grid_hash(grid),
        wealth_grid_kind="custom",
    )
    with pytest.raises(RuntimeError, match="custom wealth grid"):
        _ensure_checkpoint_wealth_grid_compatible(tmp_path, {}, {}, pc)


def test_canonical_log1p_grid_passes_fp32_validator():
    """The canonical log1p constructor produces an fp32-safe grid at the
    current canonical (n_wealth=180, wealth_min=0.01, wealth_max=750.0).

    Regression guard for FP32_PLACEMENT_REVIEW_2026-05-09 finding F1: the
    log1p path in precompute.py used to skip validate_wealth_grid; now it
    runs the same gate as the custom-grid path. This test asserts the
    canonical passes with substantial headroom (not just barely)."""
    grid = legacy_log1p_wealth_grid(180, 0.01, 750.0)
    stats = validate_wealth_grid(
        grid,
        n_wealth=180,
        wealth_min=0.01,
        wealth_max=750.0,
        source="canonical log1p (regression guard)",
    )
    # Probe (FP32_PLACEMENT_REVIEW §F1) measured min_rel_diff32 = 3.77e-2
    # at canonical. Demand >= 1e-3 to catch a grid that *barely* passes
    # rather than one with the observed ~40,000x headroom.
    assert stats["min_rel_diff32"] > 1e-3, (
        f"canonical log1p min_rel_diff32 = {stats['min_rel_diff32']:.3e} "
        f"is too close to the f32 floor; investigate before lowering "
        f"wealth_min further or bumping n_wealth"
    )
    assert stats["n_nonpositive_diff32"] == 0
    assert stats["min_diff64"] > 1e-8
