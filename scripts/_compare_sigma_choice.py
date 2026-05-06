"""Side-by-side CCV solve under sigma_cond (current code) vs sigma_unc (CCV).

Solves the retirement-only CCV problem at smoke scale (3x3x3 state, n_z=3,
n_w=20, retire ages only) twice:
  A) sigma2_xr/xb/xrxb = Sigma_r_cond[1:,1:]      <- current production
  B) sigma2_xr/xb/xrxb = Sigma_rr[1:,1:]          <- CCV w8566 eq.(10)

Compares the converged optimal alpha policies across the full grid.

The patch is purely on the formula constants; the integration grid (state
quadrature, return quadrature) is identical in both runs. The joint
distribution being integrated has variance Sigma_rr in both cases — only
the constant offsets in the per-realization Itô term change.

Run:
    python -X utf8 scripts/_compare_sigma_choice.py
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lifecycle.model import DiscretizationConfig, SolveControl
from lifecycle.precompute import Precompute, build_model
from lifecycle.solver import run_lifecycle_solver


def step(label: str, fn):
    print(f"\n[{label}] starting...", flush=True)
    t0 = time.perf_counter()
    out = fn()
    print(f"[{label}] OK in {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def main():
    spec = importlib.util.spec_from_file_location(
        "ccv_cfg", PROJECT_ROOT / "configs" / "run_canonical_ccv.py"
    )
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    base_config = dict(cfg.base_config)
    solver_config = cfg.solver_config

    # Smoke-scale grid for fast turnaround.
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

    # IMPORTANT: when youngest_age_to_solve is set, the solver auto-generates
    # a default checkpoint_path even if we pass None (see solver.py:3601-3605).
    # We pass an explicit dedicated path per run, and wipe it before each
    # solve, to guarantee each run actually re-executes from scratch.
    ckpt_A = PROJECT_ROOT / "saved_runs" / "checkpoints" / "_compare_sigma_A"
    ckpt_B = PROJECT_ROOT / "saved_runs" / "checkpoints" / "_compare_sigma_B"
    smoke_solve_control_A = SolveControl(
        youngest_age_to_solve=67,
        checkpoint_path=str(ckpt_A),
        save_on_interrupt=False,
        return_partial_on_interrupt=True,
    )
    smoke_solve_control_B = SolveControl(
        youngest_age_to_solve=67,
        checkpoint_path=str(ckpt_B),
        save_on_interrupt=False,
        return_partial_on_interrupt=True,
    )

    def _wipe(p):
        if p.exists():
            shutil.rmtree(p)
    _wipe(ckpt_A); _wipe(ckpt_B)

    from lifecycle.predictability_ablation import prepare_predictability_system
    sys_setup = prepare_predictability_system(
        cfg.PREDICTABILITY_SYSTEM,
        csv_path=str(PROJECT_ROOT / "data" / "var_dataset.csv"),
        disc_config_template=smoke_disc,
    )
    var_config = sys_setup["var_config"]
    model = build_model(base_config, var_config, verbose=False)

    Sigma_rr = np.asarray(model.Sigma_rr)
    Sigma_r_cond = np.asarray(model.Sigma_r_cond)

    print("=" * 72)
    print("SIDE-BY-SIDE: sigma_cond (production) vs sigma_unc (CCV w8566)")
    print("=" * 72)
    print()
    print("Sigma_rr[1:, 1:]  (CCV's prescription)        Sigma_r_cond[1:, 1:]  (code)")
    print(f"  sigma2_xr  = {Sigma_rr[1,1]:.5e}             sigma2_xr  = {Sigma_r_cond[1,1]:.5e}  (ratio {Sigma_rr[1,1]/Sigma_r_cond[1,1]:.1f}x)")
    print(f"  sigma2_xb  = {Sigma_rr[2,2]:.5e}             sigma2_xb  = {Sigma_r_cond[2,2]:.5e}  (ratio {Sigma_rr[2,2]/Sigma_r_cond[2,2]:.1f}x)")
    print(f"  sigma_xrxb = {Sigma_rr[1,2]:+.5e}            sigma_xrxb = {Sigma_r_cond[1,2]:+.5e}")
    print()
    assert solver_config.wealth_dynamics_spec == "ccv_log"

    # --- Run A: production sigma_cond ---
    pc_cond = Precompute(model, smoke_disc, verbose=False)
    print(f"[A] pc_cond.sigma2_xr={pc_cond.sigma2_xr:.5e} (Sigma_r_cond)")

    def solve_cond():
        return run_lifecycle_solver(
            model, pc_cond,
            solver_config=solver_config,
            solve_control=smoke_solve_control_A,
            verbose=1,
        )
    C_A, S_A, B_A, diag_A = step("A: solve under sigma_cond", solve_cond)

    # --- Run B: CCV-prescribed sigma_unc (Sigma_rr) ---
    pc_unc = Precompute(model, smoke_disc, verbose=False)
    pc_unc.sigma2_xr  = float(Sigma_rr[1, 1])
    pc_unc.sigma2_xb  = float(Sigma_rr[2, 2])
    pc_unc.sigma_xrxb = float(Sigma_rr[1, 2])
    print(f"[B] pc_unc.sigma2_xr={pc_unc.sigma2_xr:.5e}  (Sigma_rr; {pc_unc.sigma2_xr/pc_cond.sigma2_xr:.1f}x larger)")

    def solve_unc():
        return run_lifecycle_solver(
            model, pc_unc,
            solver_config=solver_config,
            solve_control=smoke_solve_control_B,
            verbose=0,
        )
    C_B, S_B, B_B, diag_B = step("B: solve under sigma_unc (Sigma_rr)", solve_unc)

    # --- Compare ---
    solved_mask = diag_A["solved_age_mask"]
    age_solved_idx = np.where(solved_mask)[0]
    print()
    print("=" * 72)
    print("POLICY COMPARISON  A=sigma_cond (code)  vs  B=sigma_unc (CCV)")
    print("=" * 72)

    def _mask_finite(*arrs):
        m = np.ones_like(arrs[0], dtype=bool)
        for a in arrs:
            m &= np.isfinite(a)
        return m

    print()
    print(f"DEBUG: id(S_A)={id(S_A)}  id(S_B)={id(S_B)}  same={S_A is S_B}")
    print(f"DEBUG: pc_cond.sigma2_xr={pc_cond.sigma2_xr:.5e}  pc_unc.sigma2_xr={pc_unc.sigma2_xr:.5e}")
    # Pick one mid-grid interior cell and print its policy value
    age_idx = pc_cond.n_age - 5  # 5 ages from terminal
    z_idx = pc_cond.n_z // 2
    s_idx = pc_cond.N_state // 2
    w_idx = pc_cond.n_w // 2
    print(f"DEBUG: cell (age_idx={age_idx}, z={z_idx}, s={s_idx}, w={w_idx}):")
    print(f"  S_A[cell]={S_A[age_idx, z_idx, s_idx, w_idx]:.10f}  "
          f"S_B[cell]={S_B[age_idx, z_idx, s_idx, w_idx]:.10f}  "
          f"diff={S_A[age_idx, z_idx, s_idx, w_idx] - S_B[age_idx, z_idx, s_idx, w_idx]:+.4e}")
    print(f"  B_A[cell]={B_A[age_idx, z_idx, s_idx, w_idx]:.10f}  "
          f"B_B[cell]={B_B[age_idx, z_idx, s_idx, w_idx]:.10f}  "
          f"diff={B_A[age_idx, z_idx, s_idx, w_idx] - B_B[age_idx, z_idx, s_idx, w_idx]:+.4e}")

    fin = _mask_finite(S_A, S_B, B_A, B_B)
    fin_solved = fin & np.isfinite(C_A) & np.isfinite(C_B)
    n_finite = int(fin_solved.sum())
    print(f"finite-policy cells (both runs): {n_finite:,} / {fin.size:,} "
          f"({100*n_finite/fin.size:.1f}%)")

    dS = S_A[fin_solved] - S_B[fin_solved]
    dB = B_A[fin_solved] - B_B[fin_solved]
    print()
    print("Delta = A - B  (positive => sigma_cond yields LARGER alpha than CCV)")
    print(f"{'':>10} | {'mean':>9} | {'p50':>9} | {'p90':>9} | {'p99':>9} | {'max abs':>9}")
    for label, dx in [("delta_alpha_s", dS), ("delta_alpha_b", dB)]:
        print(f"{label:>10} | {dx.mean():+9.4f} | "
              f"{np.percentile(dx, 50):+9.4f} | "
              f"{np.percentile(dx, 90):+9.4f} | "
              f"{np.percentile(dx, 99):+9.4f} | "
              f"{np.abs(dx).max():9.4f}")

    print()
    print("Per-policy magnitude distributions (both runs):")
    for label, S, B in [("A: sigma_cond", S_A, B_A), ("B: sigma_unc (CCV)", S_B, B_B)]:
        s_f = S[np.isfinite(S)]
        b_f = B[np.isfinite(B)]
        mag = np.sqrt(s_f**2 + b_f**2)
        print(f"  {label:>22}:  |a_s| p99={np.percentile(np.abs(s_f), 99):5.2f}  "
              f"|a_b| p99={np.percentile(np.abs(b_f), 99):5.2f}  "
              f"|a|_2 p99={np.percentile(mag, 99):5.2f}")
        # Cap-binding fraction
        cap_s = (np.abs(s_f) >= 5.99).mean()
        cap_b = (np.abs(b_f) >= 5.99).mean()
        print(f"                          cap_bound (|a|>=5.99): "
              f"alpha_s {cap_s*100:.2f}%, alpha_b {cap_b*100:.2f}%")

    # --- Newton diag aggregates ---
    print()
    print("Solver diagnostics (aggregate counts across all retire-age cells):")
    for label, diag in [("A: sigma_cond", diag_A), ("B: sigma_unc", diag_B)]:
        agg = diag.get("aggregate_int") if isinstance(diag, dict) else None
        if agg is None:
            agg = diag["aggregate_int"]["values"]
        if agg is not None:
            agg = np.asarray(agg)
            interior = int(agg[6])
            newton_fail = int(agg[7])
            total = int(agg[10])
            ratio = interior / max(total, 1)
            print(f"  {label:>20}: interior={interior:,}  newton_fail={newton_fail:,}  "
                  f"total={total:,}  conv_rate={ratio*100:.1f}%")
        fmax = diag.get("aggregate_fmax")
        if fmax is None and isinstance(diag, dict):
            fmax = diag.get("aggregate_fmax")
        if fmax is not None:
            fmax = np.asarray(fmax)
            print(f"  {' ':>20}  worst_foc_resid={fmax[1]:.5f}  max_alpha_s={fmax[4]:.3f}  max_alpha_b={fmax[6]:.3f}")


if __name__ == "__main__":
    main()
