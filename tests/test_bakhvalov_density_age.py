"""Tests for the configurable --density-age knob in optimal_wealth_grid.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lifecycle.wealth_grid import wealth_grid_hash  # noqa: E402

from scripts.analysis import optimal_wealth_grid as owg  # noqa: E402


def _stub_simulation(
    *,
    n_age: int,
    n_sim: int,
    n_z: int,
    n_state: int,
    w_lo: float,
    w_hi: float,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    alive = np.ones((n_sim, n_age), dtype=bool)
    age_centers = np.linspace(0.5 * (w_lo + w_hi) * 0.05, 0.6 * w_hi, n_age)
    age_stds = np.linspace(0.05 * w_hi, 0.25 * w_hi, n_age)
    x = np.empty((n_sim, n_age), dtype=np.float64)
    for t in range(n_age):
        x[:, t] = np.clip(
            age_centers[t] + age_stds[t] * rng.standard_normal(n_sim),
            w_lo + 1e-3,
            w_hi - 1e-3,
        )
    z_idx = rng.integers(0, n_z, size=(n_sim, n_age), dtype=np.int64)
    state_idx = rng.integers(0, n_state, size=(n_sim, n_age), dtype=np.int64)
    return {"alive": alive, "x": x, "z_idx": z_idx, "state_idx": state_idx}


def _stub_pc(n_w: int = 20, n_z: int = 3, n_state: int = 4) -> SimpleNamespace:
    wealth_grid = np.expm1(np.linspace(np.log1p(0.13), np.log1p(50.0), n_w))
    return SimpleNamespace(
        wealth_grid=wealth_grid,
        n_w=n_w,
        n_z=n_z,
        N_state=n_state,
    )


def _stub_policies(n_age: int, n_z: int, n_state: int, n_w: int) -> np.ndarray:
    return np.zeros((n_age, n_z, n_state, n_w), dtype=np.float64)


def _patch_simulate(monkeypatch: pytest.MonkeyPatch, sim: dict) -> None:
    monkeypatch.setattr(owg, "simulate_lifecycle", lambda *a, **kw: sim)


def test_parse_density_age_accepts_all_and_int():
    assert owg._parse_density_age("all") == "all"
    assert owg._parse_density_age("ALL") == "all"
    assert owg._parse_density_age("67") == 67
    assert owg._parse_density_age(" 67 ") == 67
    with pytest.raises(argparse.ArgumentTypeError):
        owg._parse_density_age("middle")


def test_resolve_density_age_idx_maps_age_to_t():
    assert owg._resolve_density_age_idx("all", start_age=22, terminal_age=99) is None
    assert owg._resolve_density_age_idx(67, start_age=22, terminal_age=99) == 45
    assert owg._resolve_density_age_idx(22, start_age=22, terminal_age=99) == 0
    assert owg._resolve_density_age_idx(99, start_age=22, terminal_age=99) == 77


def test_resolve_density_age_idx_rejects_out_of_range():
    with pytest.raises(ValueError, match="age range"):
        owg._resolve_density_age_idx(21, start_age=22, terminal_age=99)
    with pytest.raises(ValueError, match="age range"):
        owg._resolve_density_age_idx(100, start_age=22, terminal_age=99)


def test_all_ages_weights_distribute_across_lifecycle(monkeypatch: pytest.MonkeyPatch):
    pc = _stub_pc()
    n_age = 8
    sim = _stub_simulation(
        n_age=n_age, n_sim=500, n_z=pc.n_z, n_state=pc.N_state,
        w_lo=pc.wealth_grid[0], w_hi=pc.wealth_grid[-1],
    )
    _patch_simulate(monkeypatch, sim)
    C = _stub_policies(n_age, pc.n_z, pc.N_state, pc.n_w)

    weights = owg._density_weights_from_simulation(
        C, C, C, model=None, pc=pc,
        n_simulations=500, seed=42, initial_wealth=0.1, age_idx=None,
    )
    assert weights.shape == (n_age, pc.n_z, pc.N_state, pc.n_w - 2)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
    per_age = weights.sum(axis=(1, 2, 3))
    assert np.all(per_age > 0.0)


def test_snapshot_weights_concentrate_on_one_age(monkeypatch: pytest.MonkeyPatch):
    pc = _stub_pc()
    n_age = 8
    target_t = 5
    sim = _stub_simulation(
        n_age=n_age, n_sim=500, n_z=pc.n_z, n_state=pc.N_state,
        w_lo=pc.wealth_grid[0], w_hi=pc.wealth_grid[-1],
    )
    _patch_simulate(monkeypatch, sim)
    C = _stub_policies(n_age, pc.n_z, pc.N_state, pc.n_w)

    weights = owg._density_weights_from_simulation(
        C, C, C, model=None, pc=pc,
        n_simulations=500, seed=42, initial_wealth=0.1, age_idx=target_t,
    )
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
    per_age = weights.sum(axis=(1, 2, 3))
    assert per_age[target_t] == pytest.approx(1.0)
    other = np.delete(per_age, target_t)
    assert np.all(other == 0.0)


def test_snapshot_grid_differs_from_all_ages_grid(monkeypatch: pytest.MonkeyPatch):
    pc = _stub_pc(n_w=80)
    n_age = 8
    sim = _stub_simulation(
        n_age=n_age, n_sim=2000, n_z=pc.n_z, n_state=pc.N_state,
        w_lo=pc.wealth_grid[0], w_hi=pc.wealth_grid[-1],
    )
    _patch_simulate(monkeypatch, sim)
    C = _stub_policies(n_age, pc.n_z, pc.N_state, pc.n_w)

    w_all = owg._density_weights_from_simulation(
        C, C, C, model=None, pc=pc,
        n_simulations=2000, seed=42, initial_wealth=0.1, age_idx=None,
    )
    w_snap = owg._density_weights_from_simulation(
        C, C, C, model=None, pc=pc,
        n_simulations=2000, seed=42, initial_wealth=0.1, age_idx=n_age - 1,
    )

    interior = pc.wealth_grid[1:-1]
    centers = np.linspace(pc.wealth_grid[0] + 0.5, pc.wealth_grid[-1] - 0.5, n_age)
    curv = np.zeros((n_age, pc.n_z, pc.N_state, interior.size), dtype=np.float64)
    for t in range(n_age):
        curv[t] = np.exp(-((interior - centers[t]) / 1.5) ** 2)[None, None, :]

    axes = (0, 1, 2)
    monitor_all = np.maximum((curv * w_all).sum(axis=axes), 1e-14)
    monitor_snap = np.maximum((curv * w_snap).sum(axis=axes), 1e-14)

    grid_all, _ = owg._grid_from_monitor(
        pc.wealth_grid, monitor_all, n_wealth=40, window=9
    )
    grid_snap, _ = owg._grid_from_monitor(
        pc.wealth_grid, monitor_snap, n_wealth=40, window=9
    )

    assert wealth_grid_hash(grid_all) != wealth_grid_hash(grid_snap)
    wealth_range = float(pc.wealth_grid[-1] - pc.wealth_grid[0])
    assert np.max(np.abs(grid_all - grid_snap)) > 0.01 * wealth_range
