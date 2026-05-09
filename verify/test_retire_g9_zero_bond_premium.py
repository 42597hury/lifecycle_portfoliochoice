"""Retirement-only experiment: g9 with zero simple bond excess return.

Same disc + solver as test_retire_g9.py (state=(9,9,9), state_n_stds=(3,3,3),
n_z=10, max_iter=10, line search ON, gather_precision='f32') except the VAR
is the zero-simple-bond-premium variant: E[exp(xb)] = 1, so the bond's
unconditional excess return is exactly zero in level (simple) terms.

Bond risk/dynamics preserved (Phi xb-row, Sigma_rr, state dependence — all
unchanged); only the unconditional log-mean of xb is shifted via the
Jensen correction.

Hypothesis test:
  - Baseline (test_retire_g9 post-normalization): alpha_b ~ +0.94 at mean state
  - With zero bond premium: alpha_b should drop sharply (no more compensation
    for bond risk). Agent's terminal sanity showed:
      baseline:          alpha_s=0.396, alpha_b=+0.351, bill=0.253
      zero bond premium: alpha_s=0.406, alpha_b=-0.105, bill=0.699
    so retirement-wide we expect alpha_b near 0 or negative.

Compare against:
  - saved_runs/test_retire_g9/ (post-normalization, 9.31 min, 5900 failures)
"""
import os
import time

import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig, SolverConfig, SolveControl
from lifecycle.var import build_real_full_var_config_zero_bond_simple_excess_mean
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.policy_io import save_policy_bundle


disc = DiscretizationConfig(
    n_wealth=100,
    wealth_min=0.01,
    wealth_max=750.0,
    n_savings=100,
    state_grid_sizes=(9, 9, 9),
    state_grid_mode="cholesky",
    state_n_stds=(3.0, 3.0, 3.0),
    n_z=10,
    n_stds=3.0,
    n_eps_nodes=4,
    n_eta_nodes=3,
    n_ret_nodes_1d=(4, 4),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(4, 4, 4),
    state_lobatto_Z=None,
)

sc = SolverConfig(
    tol=1e-6,
    max_iter=10,
    max_backtrack_iter=5,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    use_line_search=True,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=2,
)


print("=" * 70, flush=True)
print("TEST g=9 RETIRE-ONLY  (zero simple bond excess return VAR)", flush=True)
print("  state_grid=(9,9,9), state_n_stds=(3,3,3), n_z=10, max_iter=10", flush=True)
print("=" * 70, flush=True)

import jax
print(f"JAX devices: {jax.devices()}", flush=True)

print("\nBuilding model + precompute (zero-bond-premium VAR)...", flush=True)
t0 = time.time()
var_config = build_real_full_var_config_zero_bond_simple_excess_mean(
    target_simple_mean=0.0
)
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=True)
print(f"Setup wall: {time.time() - t0:.1f}s", flush=True)
print(
    f"  n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}, "
    f"n_state_quad={pc.n_state_quad}, n_ret_quad={pc.n_ret_quad}",
    flush=True,
)

solve_control = SolveControl(
    youngest_age_to_solve=int(model.retire_age),
    save_on_interrupt=False,
    return_partial_on_interrupt=True,
)
print(f"\nSolveControl: youngest_age_to_solve={solve_control.youngest_age_to_solve}", flush=True)

print("\nSolving retirement only (zero bond premium) ...", flush=True)
t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1, solve_control=solve_control)
solve_wall = time.time() - t0
print(f"\nSolve wall: {solve_wall:.1f}s = {solve_wall / 60:.2f} min", flush=True)


print("\n" + "=" * 70, flush=True)
print("SANITY (retirement ages only)", flush=True)
print("=" * 70, flush=True)

solved_mask = np.asarray(diag.get("solved_age_mask", []), dtype=bool)
print(f"  solved ages: {int(solved_mask.sum())}/{len(solved_mask)}", flush=True)

C_solved = np.asarray(C)[solved_mask]
S_solved = np.asarray(S)[solved_mask]
B_solved = np.asarray(B)[solved_mask]
print(f"  NaN count : C={int(np.isnan(C_solved).sum())}  S={int(np.isnan(S_solved).sum())}  B={int(np.isnan(B_solved).sum())}", flush=True)
print(f"  Inf count : C={int(np.isinf(C_solved).sum())}  S={int(np.isinf(S_solved).sum())}  B={int(np.isinf(B_solved).sum())}", flush=True)
print(f"  alpha_s range (solved): [{float(S_solved.min()):.3f}, {float(S_solved.max()):.3f}]", flush=True)
print(f"  alpha_b range (solved): [{float(B_solved.min()):.3f}, {float(B_solved.max()):.3f}]", flush=True)
print(f"  C  range (solved)     : [{float(C_solved.min()):.4f}, {float(C_solved.max()):.4f}]", flush=True)

w0 = float(np.asarray(pc.wealth_grid)[0])
c_at_w0_max = float(np.asarray(C_solved)[..., 0].max())
c_at_w0_min = float(np.asarray(C_solved)[..., 0].min())
s_at_w0_abs = float(np.abs(np.asarray(S_solved)[..., 0]).max())
b_at_w0_abs = float(np.abs(np.asarray(B_solved)[..., 0]).max())
print(
    f"\n  Path B clamp at w0={w0:.4f}: "
    f"C[..., 0] in [{c_at_w0_min:.6f}, {c_at_w0_max:.6f}]  "
    f"|S[..., 0]|max={s_at_w0_abs:.2e}  |B[..., 0]|max={b_at_w0_abs:.2e}",
    flush=True,
)

print("\n" + "=" * 70, flush=True)
print("NEWTON / BACKTRACK HISTOGRAMS", flush=True)
print("=" * 70, flush=True)
nih = diag.get("newton_iter_histogram")
if nih is not None:
    print(
        f"  Newton iters    : p50={nih['p50']:.0f}  p95={nih['p95']:.0f}  "
        f"p99={nih['p99']:.0f}  max={nih['max']}  (cap=10)",
        flush=True,
    )
bth = diag.get("backtrack_iter_histogram")
if bth is not None:
    print(
        f"  Backtrack iters : p50={bth['p50']:.1f}  p95={bth['p95']:.1f}  "
        f"p99={bth['p99']:.1f}  max={bth['max']}  (cap=5)",
        flush=True,
    )
print(f"  Total Newton failures: {diag.get('total_newton_failures', '?')}", flush=True)


BUNDLE_NAME = "test_retire_g9_zero_bond_premium"
BUNDLE_DIR = os.path.join("saved_runs", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "solve_control": solve_control._asdict(),
    "predictability_ablation": {
        "system_code": "full",
        "system_label": "full_system_real",
        "system_title": "Full System (real, zero bond premium)",
        "state_names": ["cape", "spr", "y_1"],
    },
    "var_config_variant": "zero_bond_simple_excess_mean",
    "var_config_xb_target_simple_mean": 0.0,
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(solve_wall),
    "scope": "retirement_only_zero_bond_premium",
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
bundle_path = save_policy_bundle(
    BUNDLE_DIR, C, S, B,
    diagnostics=diag, run_config=run_config_snapshot,
    overwrite=True, wealth_grid=pc.wealth_grid,
)
print(f"  Saved: {bundle_path}", flush=True)


print("\n" + "=" * 70, flush=True)
print("DONE  —  retirement-only g=9 zero-bond-premium run", flush=True)
print("=" * 70, flush=True)
print(f"  Solve wall: {solve_wall / 60:.2f} min", flush=True)
print(f"  Bundle:     {BUNDLE_DIR}/", flush=True)
print(f"  Compare against: saved_runs/test_retire_g9/  (baseline VAR, 9.31 min)", flush=True)
