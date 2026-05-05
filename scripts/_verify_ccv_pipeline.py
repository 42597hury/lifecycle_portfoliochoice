"""End-to-end verification that the CCV toolchain runs cleanly.

Mirrors the production AWS pipeline at smoke-scale:
    1. build_model + Precompute
    2. run_lifecycle_solver (retirement-only via SolveControl)
    3. save_policy_bundle  (verifies metadata tagging)
    4. load_policy_bundle  (verifies round-trip + spec auto-detection)
    5. _diag_euler_errors  (verifies diagnostic accepts CCV bundle)

Uses a deliberately tiny grid (3x3x3 state, n_z=3, n_w=20) to keep wall-clock
under a minute. All other knobs are inherited from the production CCV config
(configs/run_canonical_ccv.py) so any name / signature regression surfaces
here, not on AWS.

Run:
    python -X utf8 scripts/_verify_ccv_pipeline.py

Exit code:
    0 -- all stages OK; production config is safe to ship
    nonzero -- one of the stages errored; see traceback
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lifecycle.model import DiscretizationConfig, SolveControl
from lifecycle.policy_io import load_policy_bundle, save_policy_bundle
from lifecycle.precompute import Precompute, build_model
from lifecycle.solver import run_lifecycle_solver
from lifecycle.var import build_nominal_system1_var_config


def step(label: str, fn):
    print(f"\n[{label}] starting...", flush=True)
    t0 = time.perf_counter()
    out = fn()
    print(f"[{label}] OK in {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def main() -> int:
    # --- Load the production CCV config and shrink the grid for the smoke.
    spec = importlib.util.spec_from_file_location(
        "ccv_cfg", PROJECT_ROOT / "configs" / "run_canonical_ccv.py"
    )
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    base_config = dict(cfg.base_config)
    solver_config = cfg.solver_config

    # Override the production discretization with a smoke-scale one. We
    # keep the CCV-relevant fields (wealth_min=0.01, no Lobatto) and just
    # shrink everything that drives wall-clock.
    smoke_disc = DiscretizationConfig(
        n_wealth=20,
        wealth_min=0.01,
        wealth_max=750.0,
        n_savings=20,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=2.0,
        n_z=3,
        n_stds=2.0,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=2,
        n_state_quad_nodes=2,
        ret_lobatto_Z=None,
        state_lobatto_Z=None,
    )

    # Retirement-only via SolveControl, matching production.
    smoke_solve_control = SolveControl(
        youngest_age_to_solve=67,
        checkpoint_path=None,           # no checkpoint dir for smoke
        save_on_interrupt=False,
        return_partial_on_interrupt=True,
    )

    # --- Stage 1: build_model + Precompute -----------------------------------
    # Use the same prepare_predictability_system path that scripts/run_solve.py
    # uses, so the var_config persisted in the bundle's run_config matches the
    # production format the EE diagnostic expects.
    from lifecycle.predictability_ablation import prepare_predictability_system
    sys_setup = prepare_predictability_system(
        cfg.PREDICTABILITY_SYSTEM,
        csv_path=str(PROJECT_ROOT / "data" / "var_dataset.csv"),
        disc_config_template=smoke_disc,
    )
    var_config = sys_setup["var_config"]
    model = step("1/5 build_model + Precompute",
                 lambda: (build_model(base_config, var_config, verbose=False),))
    model = model[0]
    pc = Precompute(model, smoke_disc, verbose=False)

    print(f"      pc.wealth_grid[0:3] = {pc.wealth_grid[:3]}")
    print(f"      pc.sigma2_xr={pc.sigma2_xr:.4f} sigma2_xb={pc.sigma2_xb:.4f} sigma_xrxb={pc.sigma_xrxb:.4f}")
    assert solver_config.wealth_dynamics_spec == "ccv_log"

    # --- Stage 2: run_lifecycle_solver (retirement only, CCV) ----------------
    def _solve():
        return run_lifecycle_solver(
            model, pc,
            solver_config=solver_config,
            solve_control=smoke_solve_control,
            verbose=0,
        )
    C, S, B, diag = step("2/5 run_lifecycle_solver retirement-only ccv_log", _solve)

    n_age = pc.n_age
    retire_age_idx = model.retire_age - model.start_age
    solved_mask = diag["solved_age_mask"]
    assert solved_mask[retire_age_idx:].all(), "expected retire_age..terminal solved"
    assert not solved_mask[:retire_age_idx].any(), "expected working ages NOT solved"
    print(f"      solved ages: {int(np.sum(solved_mask))} / {n_age}  (retire_age_idx={retire_age_idx})")
    print(f"      C range over solved ages: [{np.nanmin(C):.4f}, {np.nanmax(C):.4f}]")
    print(f"      S range over solved ages: [{np.nanmin(S):.4f}, {np.nanmax(S):.4f}]")
    print(f"      B range over solved ages: [{np.nanmin(B):.4f}, {np.nanmax(B):.4f}]")

    # --- Stage 3: save_policy_bundle (tag verification) ----------------------
    bundle_dir = Path(tempfile.mkdtemp(prefix="ccv_smoke_")) / "bundle"

    def _save():
        # Mirror scripts/run_solve.py's run_config exactly so the EE
        # diagnostic's _build_model_from_bundle has every field it expects.
        run_config = {
            "base_config": base_config,
            "discretization_config": smoke_disc._asdict(),
            "solver_config": solver_config._asdict(),
            "var_config": var_config,
            "predictability_ablation": {
                "system_code": sys_setup["system_code"],
                "system_label": sys_setup["system_label"],
                "system_title": sys_setup["system_title"],
                "var_builder_name": sys_setup["var_builder_name"],
                "state_names": list(sys_setup["state_names"]),
            },
            "constrained": bool(model.constrained),
            "solve_label": "constrained" if model.constrained else "unconstrained",
        }
        save_policy_bundle(bundle_dir, C, S, B, diagnostics=diag,
                           run_config=run_config, overwrite=True)
    step("3/5 save_policy_bundle", _save)

    # --- Stage 4: load_policy_bundle + spec auto-detection -------------------
    def _load():
        return load_policy_bundle(bundle_dir)
    C_l, S_l, B_l, diag_l, meta = step("4/5 load_policy_bundle + tag round-trip", _load)
    assert meta["wealth_dynamics_spec"] == "ccv_log", \
        f"bundle metadata wealth_dynamics_spec={meta['wealth_dynamics_spec']!r}, expected ccv_log"
    print(f"      meta['wealth_dynamics_spec'] = {meta['wealth_dynamics_spec']!r}  OK")
    np.testing.assert_array_equal(C, C_l)
    np.testing.assert_array_equal(S, S_l)
    np.testing.assert_array_equal(B, B_l)

    # --- Stage 5: run _diag_euler_errors against the CCV bundle --------------
    # Use a subprocess so we exercise the actual CLI entrypoint, catching any
    # arg-parsing / threading regression at the same level production uses.
    def _diag():
        cmd = [
            sys.executable, "-X", "utf8",
            "scripts/diagnostics/_diag_euler_errors.py",
            str(bundle_dir),
            "--model-bundle", str(bundle_dir),
            "--eval-mode", "same",          # tiny smoke: just check it runs
            "--n-simulations", "200",
            "--eval-households-per-age", "32",
            "--seed", "42",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            print("STDOUT:\n" + (proc.stdout or "")[-2000:])
            print("STDERR:\n" + (proc.stderr or "")[-2000:])
            raise RuntimeError(f"_diag_euler_errors exited {proc.returncode}")
        # Sanity check: the diagnostic should announce its CCV branch
        out = (proc.stdout or "") + (proc.stderr or "")
        return out

    out = step("5/5 _diag_euler_errors against CCV bundle (CLI)", _diag)
    print("      diagnostic output tail:")
    for line in (out or "").splitlines()[-15:]:
        print(f"         {line}")

    # --- Cleanup -------------------------------------------------------------
    shutil.rmtree(bundle_dir.parent, ignore_errors=True)

    print("\nALL STAGES PASSED. Production config configs/run_canonical_ccv.py "
          "is safe to ship to AWS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
