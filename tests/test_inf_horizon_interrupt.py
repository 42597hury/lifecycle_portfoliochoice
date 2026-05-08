"""Regression test: run_infinite_horizon_solver must return partial output on
KeyboardInterrupt so the launcher can call save_policy_bundle.

Pre-fix (2026-05-07 incident): an inf-horizon run was Ctrl-C'd at iter 21/50.
KeyboardInterrupt unwound past the solver's `return` and past the launcher's
save_policy_bundle, losing ~70 minutes of compute. Mirror of the lifecycle
solver's try/except pattern (lifecycle/solver.py:2755).

Two gates:
  1. Normal completion — the solver still returns and the bundle still saves.
  2. Synthetic interrupt — a monkey-patched kernel raises KeyboardInterrupt on
     the K-th call; the solver must return partial (C, S, B) of the right
     shape with diag["converged"] == False, and save_policy_bundle must
     succeed on that output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle import inf_horizon_solver as ihs
from lifecycle.policy_io import save_policy_bundle, load_policy_bundle


def _tiny_model_pc():
    """Smallest config that exercises the inf-horizon kernel on CPU.

    Real-yields pivot: 3-axis state vector (cape, spr, y_1).
    """
    disc = DiscretizationConfig(
        n_wealth=10, wealth_min=0.13, wealth_max=200.0,
        n_savings=10,
        state_grid_sizes=(2, 2, 2),
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
    # Mirrors verify/benchmark_inf_horizon.py knobs at minimum cost.
    return SolverConfig(
        wealth_dynamics_spec="ccv_log",
        max_iter=100,
        max_iter_unconstrained=100,
        delta_bequest=0.0,
        gather_precision="f32",
        cell_vmap_chunks=1,
    )


def test_inf_horizon_completes_and_saves_bundle(tmp_path):
    """Gate 1: normal max_iter exit returns valid (C, S, B) and the bundle
    serializes through save_policy_bundle / load_policy_bundle round-trip."""
    model, pc = _tiny_model_pc()
    sc = _tiny_solver_config()

    # tol=1e-15 forces hitting the max_iter cap; max_iter=3 keeps wall short.
    C, S, B, diag = ihs.run_infinite_horizon_solver(
        model, pc,
        solver_config=sc,
        max_iter=3, tol=1e-15, damping=1.0,
        verbose=False, show_progress=False,
    )

    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    assert C.shape == expected_shape
    assert S.shape == expected_shape
    assert B.shape == expected_shape
    assert not np.isnan(C).any()
    assert not np.isnan(S).any()
    assert not np.isnan(B).any()
    assert diag["n_iter"] == 3
    assert diag["converged"] is False  # tol unreachable in 3 iters

    bundle_dir = tmp_path / "gate1_bundle"
    out = save_policy_bundle(
        bundle_dir, C, S, B, diagnostics=diag, overwrite=True,
        wealth_grid=pc.wealth_grid,
    )
    assert (out / "policy_arrays.npz").exists()
    assert (out / "diagnostics.pkl").exists()
    assert (out / "metadata.json").exists()

    C_r, S_r, B_r, diag_r, _ = load_policy_bundle(out)
    np.testing.assert_array_equal(C, C_r)
    np.testing.assert_array_equal(S, S_r)
    np.testing.assert_array_equal(B, B_r)
    assert diag_r["n_iter"] == 3


def test_inf_horizon_interrupt_returns_partial_bundle(tmp_path, monkeypatch):
    """Gate 2: KeyboardInterrupt mid-loop must NOT propagate. The solver must
    return the policies as of the last fully-committed iteration plus a
    diagnostics dict with converged=False, and save_policy_bundle must accept
    the partial output."""
    model, pc = _tiny_model_pc()
    sc = _tiny_solver_config()

    # Wrap the real kernel: succeed for the first K_OK calls, then raise
    # KeyboardInterrupt on call K_OK+1. K_OK=3 means three iters fully commit
    # before interrupt — n_iter_done should equal 3 and (C_old, S_old, B_old)
    # should be those iter-3 policies.
    K_OK = 3
    real_build = ihs._build_per_age_retirement_kernel
    call_counter = {"n": 0}

    def build_raising_kernel(*args, **kwargs):
        real_kernel = real_build(*args, **kwargs)

        def kernel(*a, **kw):
            call_counter["n"] += 1
            if call_counter["n"] > K_OK:
                raise KeyboardInterrupt
            return real_kernel(*a, **kw)

        return kernel

    monkeypatch.setattr(ihs, "_build_per_age_retirement_kernel", build_raising_kernel)

    C, S, B, diag = ihs.run_infinite_horizon_solver(
        model, pc,
        solver_config=sc,
        max_iter=50, tol=1e-15, damping=1.0,
        verbose=False, show_progress=False,
    )

    expected_shape = (pc.n_z, pc.N_state, pc.n_w)
    assert C.shape == expected_shape
    assert S.shape == expected_shape
    assert B.shape == expected_shape
    assert not np.isnan(C).any()
    assert not np.isnan(S).any()
    assert not np.isnan(B).any()
    assert call_counter["n"] == K_OK + 1
    assert diag["n_iter"] == K_OK
    assert diag["converged"] is False

    # Histories must be truncated to n_iter_done — the in-flight 4th iteration
    # never appended its metrics, so length should be exactly K_OK.
    assert diag["xi_supnorm_history"].shape == (K_OK,)
    assert diag["share_supnorm_history"].shape == (K_OK,)
    assert diag["policy_supnorm_history"].shape == (K_OK,)

    bundle_dir = tmp_path / "gate2_partial_bundle"
    out = save_policy_bundle(
        bundle_dir, C, S, B, diagnostics=diag, overwrite=True,
        wealth_grid=pc.wealth_grid,
    )
    assert (out / "policy_arrays.npz").exists()

    C_r, S_r, B_r, diag_r, _ = load_policy_bundle(out)
    np.testing.assert_array_equal(C, C_r)
    np.testing.assert_array_equal(S, S_r)
    np.testing.assert_array_equal(B, B_r)
    assert diag_r["converged"] is False
    assert diag_r["n_iter"] == K_OK
