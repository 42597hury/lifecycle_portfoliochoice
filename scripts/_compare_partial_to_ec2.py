"""Partial resolve (last 10 ages) to compare against the EC2 bundle.

Solves ages 90-99 locally using configs/system_iv_5x5x5.py, then compares
the resulting policies to the corresponding slice of the EC2 bundle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lifecycle.model import SolveControl
from lifecycle.precompute import build_model, Precompute
from lifecycle.predictability_ablation import prepare_predictability_system
from lifecycle.solver import run_lifecycle_solver
from lifecycle.policy_io import load_policy_bundle


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "system_iv_5x5x5.py")

    var_csv = PROJECT_ROOT / "data" / "var_dataset.csv"
    system_setup = prepare_predictability_system(
        cfg.PREDICTABILITY_SYSTEM,
        csv_path=str(var_csv),
        disc_config_template=cfg.disc_config_template,
    )
    var_config = system_setup["var_config"]
    disc_config = system_setup["disc_config"]

    model = build_model(cfg.base_config, var_config, verbose=False)
    pc = Precompute(model, disc_config)

    # Solve ONLY ages 90 through 99 (last 10 ages)
    sc = SolveControl(youngest_age_to_solve=90)
    print("Running partial solve (ages 90-99)...")
    C_l, S_l, B_l, _ = run_lifecycle_solver(
        model, pc, solver_config=cfg.solver_config, solve_control=sc, verbose=1,
    )
    print()

    ec2_path = (
        PROJECT_ROOT / "saved_runs" /
        "system_iv_full_var_unconstrained_principal_grid5x5x5_nz9_v2"
    )
    print(f"Loading EC2 bundle from: {ec2_path.name}")
    C_e, S_e, B_e, _, _ = load_policy_bundle(ec2_path)

    # Age 90 = index 68 (since start_age=22, age k -> index k-22)
    start_idx = 90 - cfg.base_config["start_age"]
    sl = slice(start_idx, 78)

    print()
    print("Per-age comparison (local partial vs EC2 ages 90-99):")
    print("age | C max diff | S max diff | B max diff")
    print("----+------------+------------+-----------")
    for i in range(start_idx, 78):
        age = 22 + i
        cd = np.max(np.abs(C_e[i] - C_l[i]))
        sd = np.max(np.abs(S_e[i] - S_l[i]))
        bd = np.max(np.abs(B_e[i] - B_l[i]))
        print(f" {age:>2} | {cd:>10.3e} | {sd:>10.3e} | {bd:>10.3e}")

    print()
    print("Overall (ages 90-99):")
    print(f"  C max abs diff: {np.max(np.abs(C_e[sl] - C_l[sl])):.3e}")
    print(f"  S max abs diff: {np.max(np.abs(S_e[sl] - S_l[sl])):.3e}")
    print(f"  B max abs diff: {np.max(np.abs(B_e[sl] - B_l[sl])):.3e}")
    print(f"  C close (1e-5): {np.allclose(C_e[sl], C_l[sl], rtol=1e-5)}")
    print(f"  S close (1e-5): {np.allclose(S_e[sl], S_l[sl], rtol=1e-5)}")
    print(f"  B close (1e-5): {np.allclose(B_e[sl], B_l[sl], rtol=1e-5)}")


if __name__ == "__main__":
    main()
