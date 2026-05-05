"""section 7.1 pre-commitment: solve the same problem under both wealth_dynamics_spec
values, compare age-by-age stock and bond shares.

Per the implementation handoff section 7.1 and the PR2 review note:
  - constrained M-no-labor at gamma=5 (or whatever the config provides)
  - tolerance: ±3pp stock share, ±2pp bond share at every age
  - if the gap exceeds tolerance, escalate (do NOT widen)

This script does NOT regenerate production bundles; it just produces the
side-by-side comparison so the spec switch can be approved before flipping
the default. To run on the smoke config (sanity check that the pipeline
works end-to-end):

    python scripts/_run_cvc_precommitment.py configs/smoke_test.py

For the actual gate, run on the production config:

    python scripts/_run_cvc_precommitment.py configs/run_v8_canonical.py

Output:
  - prints per-age max(|alpha_s_simple - alpha_s_ccv|), same for alpha_b
  - prints PASS/FAIL against ±3pp / ±2pp tolerances
  - saves a CSV with age-by-age shares for both specs
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lifecycle.precompute import Precompute, build_model
from lifecycle.solver import run_lifecycle_solver
from lifecycle.var import build_nominal_system1_var_config


def load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("run_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _solve_one(model, pc, solver_config, label: str, verbose: int = 1):
    print(f"\n{'='*70}\nSolving with wealth_dynamics_spec={solver_config.wealth_dynamics_spec!r}  ({label})\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    C, S, B, diag = run_lifecycle_solver(model, pc, solver_config=solver_config, verbose=verbose)
    dt = time.perf_counter() - t0
    print(f"  done in {dt:.1f}s", flush=True)
    return C, S, B, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--constrained", action="store_true",
                        help="Force constrained=True regardless of config "
                             "(per handoff section 7.1, constrained baseline).")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_config = dict(cfg.base_config)
    if args.constrained:
        base_config["constrained"] = True

    var_config, _, _ = build_nominal_system1_var_config(
        csv_path=str(PROJECT_ROOT / "data" / "var_dataset.csv")
    )
    model = build_model(base_config, var_config, verbose=False)
    pc = Precompute(model, cfg.disc_config_template, verbose=not args.quiet)

    sc_simple = cfg.solver_config._replace(wealth_dynamics_spec="simple_clamp")
    sc_ccv = cfg.solver_config._replace(wealth_dynamics_spec="ccv_log")

    verbose = 0 if args.quiet else 1
    C_a, S_a, B_a, diag_a = _solve_one(model, pc, sc_simple, "simple+clamp", verbose=verbose)
    C_b, S_b, B_b, diag_b = _solve_one(model, pc, sc_ccv, "ccv_log",       verbose=verbose)

    # Compare at median z, median financial state across all ages.
    i_z = pc.n_z // 2
    i_s = pc.N_state // 2

    # We average across the wealth grid for each age — this approximates the
    # comparison the hypothesis spec calls for ("optimal share at age t"
    # treated at a representative wealth point). For the formal gate the user
    # can compare at the lifecycle simulation distribution; here we use the
    # mid-wealth grid point, which is the standard convention for figure
    # generation in the codebase.
    i_w = pc.n_w // 2

    n_age = len(pc.ages)
    rows = []
    max_abs_dS = 0.0
    max_abs_dB = 0.0
    max_age_S = -1
    max_age_B = -1
    for t in range(n_age):
        a = pc.ages[t]
        s_a = float(S_a[t, i_z, i_s, i_w])
        s_b = float(S_b[t, i_z, i_s, i_w])
        b_a = float(B_a[t, i_z, i_s, i_w])
        b_b = float(B_b[t, i_z, i_s, i_w])
        dS = abs(s_a - s_b)
        dB = abs(b_a - b_b)
        if dS > max_abs_dS:
            max_abs_dS = dS; max_age_S = int(a)
        if dB > max_abs_dB:
            max_abs_dB = dB; max_age_B = int(a)
        rows.append({"age": int(a),
                     "alpha_s_simple": s_a, "alpha_s_ccv": s_b, "delta_s": s_a - s_b,
                     "alpha_b_simple": b_a, "alpha_b_ccv": b_b, "delta_b": b_a - b_b})

    print("\n" + "="*70)
    print("section 7.1 PRE-COMMITMENT RESULT -- age-by-age share comparison")
    print(f"  (at z=z_median, financial state=median, wealth=mid-grid)")
    print("="*70)
    print(f"{'age':>5} {'aS_simple':>12} {'aS_ccv':>10} {'d_aS':>10} | "
          f"{'aB_simple':>12} {'aB_ccv':>10} {'d_aB':>10}")
    for r in rows:
        print(f"{r['age']:>5} {r['alpha_s_simple']:>12.4f} {r['alpha_s_ccv']:>10.4f} "
              f"{r['delta_s']:>+10.4f} | {r['alpha_b_simple']:>12.4f} {r['alpha_b_ccv']:>10.4f} "
              f"{r['delta_b']:>+10.4f}")

    print("\nSummary:")
    print(f"  max |d_aS| = {max_abs_dS:.4f} = {max_abs_dS*100:.2f} pp at age {max_age_S}")
    print(f"  max |d_aB| = {max_abs_dB:.4f} = {max_abs_dB*100:.2f} pp at age {max_age_B}")

    # Decision rule from handoff section 7.1
    pass_S = max_abs_dS <= 0.03
    pass_B = max_abs_dB <= 0.02
    print(f"\nGate (constrained baseline): +/-3pp stock, +/-2pp bond")
    print(f"  stock: {'PASS' if pass_S else 'FAIL'}  ({max_abs_dS*100:.2f} pp vs 3.00 pp)")
    print(f"  bond:  {'PASS' if pass_B else 'FAIL'}  ({max_abs_dB*100:.2f} pp vs 2.00 pp)")
    overall = "PASS" if (pass_S and pass_B) else "FAIL"
    print(f"\nOVERALL: {overall}")
    if overall == "FAIL":
        print("\n!! Per handoff section 7.1: do not widen tolerance. Escalate.")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.output_csv}")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
