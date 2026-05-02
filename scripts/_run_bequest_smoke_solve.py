"""Standalone smoke solve for the bequest=0-on-bankruptcy patch.

Mirrors the `partial-solve-test` cell in main.ipynb but writes to a fresh
bundle suffix (`_log1p_pathB_v1`) so the prior baseline (`_log1p_v1`) is
preserved for diagnostic comparison.

Wall expectations (per handoff): ~5300s baseline, accept up to ~5400s.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import DiscretizationConfig, SolveControl, SolverConfig
from policy_io import save_policy_bundle
from precompute import Precompute, build_model
from predictability_ablation import prepare_predictability_system, project_predictability_disc_config
from solver import run_lifecycle_solver


def _base_config() -> dict:
    return {
        "beta": 0.96,
        "gamma": 5.0,
        "b_bar": 10,
        "start_age": 22,
        "retire_age": 67,
        "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991, "pz": 0.176,
        "mu_eta1": -0.524, "sigma_eta1": 0.113,
        "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524),
        "sigma_eta2": 0.046,
        "pe": 0.044,
        "mu_eps1": 0.134, "sigma_eps1": 0.762,
        "mu_eps2": 0.0, "sigma_eps2": 0.055,
        "constrained": False,
    }


def main():
    base_config = _base_config()
    csv_path = str(ROOT / "data" / "var_dataset.csv")

    disc_config_template = DiscretizationConfig(
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

    system_setup = prepare_predictability_system(
        "IV", csv_path=csv_path, disc_config_template=disc_config_template,
    )
    var_config = system_setup["var_config"]
    system_label = system_setup["system_label"]
    system_title = system_setup["system_title"]

    model = build_model(base_config, var_config, verbose=False)
    model_unc = model._replace(constrained=False)

    # --- Attribution disc-config: 7x7x7, wide support, log1p, kret 3,7,5 ---
    attribution_template = disc_config_template._replace(
        state_grid_sizes=(7, 7, 7),
        state_n_stds=(2.0, 2.25, 2.25),
        n_ret_nodes_1d=(3, 7, 5),
        n_state_quad_nodes=(2, 2, 5),
    )
    disc_config_attribution = project_predictability_disc_config(
        attribution_template, model_unc.state_names,
    )
    pc_attribution = Precompute(model_unc, disc_config_attribution, verbose=False)

    solver_config_unc = SolverConfig(
        tol=1e-7,
        max_iter=20,
        max_iter_unconstrained=8000,
        init_alpha_s=0.85,
        init_alpha_b=0.44,
        step_damp_constrained=0.2,
        step_damp_unconstrained=0.3,
        use_line_search=True,
    )

    # Trimmed to a 13-age retirement-only window (87-99) for a fast read on
    # the bequest patch. The 42% next_finer invalidity-rate signal is per-
    # quadrature-node, not age-cohort, so a small contiguous tail is plenty.
    youngest_partial_age = 87
    ns_label = 'x'.join(
        str(v).replace('.', 'p') for v in np.atleast_1d(disc_config_attribution.state_n_stds)
    )
    ret_label = 'x'.join(map(str, np.atleast_1d(disc_config_attribution.n_ret_nodes_1d)))
    bundle_path = ROOT / "saved_runs" / "checkpoints" / (
        f"{system_label}_unconstrained_{disc_config_attribution.state_grid_mode}"
        f"_grid{'x'.join(map(str, disc_config_attribution.state_grid_sizes))}"
        f"_nz{disc_config_attribution.n_z}_from_age{youngest_partial_age}_kret{ret_label}"
        f"_ns{ns_label}_log1p_pathB_v1"
    )

    solve_control = SolveControl(
        youngest_age_to_solve=youngest_partial_age,
        checkpoint_path=str(bundle_path),
        checkpoint_every_n_ages=1,
        save_on_interrupt=True,
        return_partial_on_interrupt=True,
    )

    print(f"[bequest_smoke] {system_title} unconstrained partial solve, ages "
          f"{youngest_partial_age}-{model_unc.terminal_age}", flush=True)
    print(f"[bequest_smoke] disc: {disc_config_attribution}", flush=True)
    print(f"[bequest_smoke] checkpoint: {bundle_path}", flush=True)
    print(f"[bequest_smoke] starting solver wall clock", flush=True)

    t0 = time.perf_counter()
    C, S, B, diag = run_lifecycle_solver(
        model_unc,
        pc_attribution,
        solver_config=solver_config_unc,
        solve_control=solve_control,
    )
    wall = time.perf_counter() - t0

    run_config = {
        "base_config": {**base_config, "constrained": False},
        "discretization_config": disc_config_attribution._asdict(),
        "solver_config": solver_config_unc._asdict(),
        "var_config": var_config,
        "predictability_ablation": {
            "system_code": system_setup["system_code"],
            "system_label": system_label,
            "system_title": system_title,
            "var_builder_name": system_setup["var_builder_name"],
            "state_names": list(system_setup["state_names"]),
        },
        "constrained": False,
        "solve_label": "unconstrained",
    }
    save_policy_bundle(
        bundle_path, C, S, B, diagnostics=diag,
        run_config=run_config, overwrite=True,
    )

    print(f"[bequest_smoke] wall = {wall:.1f}s", flush=True)
    print(f"[bequest_smoke] status = {diag.get('solve_status')}", flush=True)
    print(f"[bequest_smoke] partial = {diag.get('is_partial')}", flush=True)
    print(f"[bequest_smoke] ages solved: {diag.get('youngest_solved_age')}-{diag.get('oldest_solved_age')}", flush=True)
    print(f"[bequest_smoke] total_newton_failures = {diag.get('total_newton_failures')}", flush=True)
    print(f"[bequest_smoke] total_mono_violations = {diag.get('total_mono_violations')}", flush=True)
    print(f"[bequest_smoke] worst_foc_resid = {diag.get('worst_foc_resid')}", flush=True)
    print(f"[bequest_smoke] bundle saved to: {bundle_path}", flush=True)


if __name__ == "__main__":
    main()
