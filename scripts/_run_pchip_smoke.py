"""One-shot smoke solve for the PCHIP secondary-interp change.

Mirrors the saved baseline at
saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_v2/
- 5x5x5 state grid, n_z=9, n_wealth=150, n_savings=150, from age 65
- Same SolverConfig as that baseline (max_iter_unconstrained=8000,
  init_alpha_s=0.85, init_alpha_b=0.44).

Writes to a fresh suffix `_pchip_smoke` so the baseline is preserved for
side-by-side comparison.

Reports: total_newton_failures, worst_foc_resid, total_mono_violations,
wall_time_sec.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.model import DiscretizationConfig, SolveControl, SolverConfig
from lifecycle.precompute import Precompute, build_model
from lifecycle.predictability_ablation import prepare_predictability_system
from lifecycle.solver import run_lifecycle_solver


def _base_config() -> dict:
    pz = 0.176
    return {
        "beta": 0.96, "gamma": 5.0, "b_bar": 10,
        "start_age": 22, "retire_age": 67, "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991, "pz": pz,
        "mu_eta1": -0.524, "sigma_eta1": 0.113,
        "mu_eta2": -(pz / (1.0 - pz)) * (-0.524), "sigma_eta2": 0.046,
        "pe": 0.044,
        "mu_eps1": 0.134, "sigma_eps1": 0.762,
        "mu_eps2": 0.0, "sigma_eps2": 0.055,
        "constrained": False,
    }


def main() -> None:
    base_config = _base_config()
    csv_path = str(ROOT / "data" / "var_dataset.csv")

    disc_config_template = DiscretizationConfig(
        n_wealth=150,
        n_savings=150,
        state_grid_sizes=(5, 5, 5),
        state_grid_mode="cholesky",
        state_n_stds=(0.6, 1.75, 2.0),
        n_z=9,
        n_stds=3.0,
        n_eps_nodes=3,
        n_eta_nodes=3,
        n_ret_nodes_1d=(3, 5, 3),
        n_state_quad_nodes=(2, 2, 5),
    )

    # Match the baseline SolverConfig stored in the existing checkpoint
    # metadata so the comparison is apples-to-apples.
    solver_config = SolverConfig(
        tol=1e-7,
        max_iter=20,
        max_iter_unconstrained=8000,
        edge_max_iter=8,
        init_alpha_s=0.85,
        init_alpha_b=0.44,
        step_damp_constrained=0.2,
        step_damp_unconstrained=0.3,
        grad_step_size=0.05,
        use_line_search=True,
        max_backtrack_iter=10,
        line_search_max_step=2.0,
    )

    system_setup = prepare_predictability_system(
        "IV", csv_path=csv_path,
        disc_config_template=disc_config_template,
    )
    var_config = system_setup["var_config"]
    disc_config = system_setup["disc_config"]

    model = build_model(base_config, var_config, verbose=False)
    pc = Precompute(model, disc_config)

    solve_control = SolveControl(
        youngest_age_to_solve=65,
        checkpoint_path=None,  # no checkpointing for smoke
        checkpoint_every_n_ages=None,
        save_on_interrupt=False,
        return_partial_on_interrupt=False,
        progress_wealth_source="scf_median",
    )

    print("[smoke] kicking off 5x5x5 from-age-65 partial solve...")
    t0 = time.time()
    _C, _S, _B, diagnostics = run_lifecycle_solver(
        model, pc,
        solver_config=solver_config,
        solve_control=solve_control,
    )
    wall = time.time() - t0

    print()
    print(f"[smoke] wall_time_sec            = {wall:.1f}")
    print(f"[smoke] total_newton_failures    = {diagnostics['total_newton_failures']}")
    print(f"[smoke] worst_foc_resid          = {diagnostics['worst_foc_resid']:.6e}")
    print(f"[smoke] total_mono_violations    = {diagnostics['total_mono_violations']}")
    print(f"[smoke] worst_mono_drop          = {diagnostics['worst_mono_drop']:.6e}")
    print(f"[smoke] total_calls              = {diagnostics['total_calls']}")
    print(f"[smoke] total_newton_iter        = {diagnostics['total_newton_iter']}")
    print(f"[smoke] avg_newton_iter          = {diagnostics['avg_newton_iter']:.4f}")
    print(f"[smoke] max_newton_iter          = {diagnostics['max_newton_iter']}")
    print(f"[smoke] n_ages_solved            = {diagnostics['n_ages_solved']}")


if __name__ == "__main__":
    main()
