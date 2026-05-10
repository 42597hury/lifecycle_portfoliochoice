"""Retirement-only test at state_grid=(10,10,10), theta=0.

Mirrors test_retire_g11.py (post-normalization, n_z=10, max_iter=10,
line search ON, gather_precision='f32', cell_vmap_chunks=8) except:
  - state_grid=(10,10,10)  (1000 cells, between g9=729 and g11=1331)
  - VAR is the term-premium-theta=0.0 dataset (no estimated real term
    premium; bond's unconditional excess return is roughly zero in
    level terms)

Note on even-N: with grid_sizes=(10,10,10), the live probe at the
center cell is at one of the two centerline-adjacent cells (~0.1σ off
mu_s), not exactly on mu_s as it would be at odd-N (g9, g11). The
policy itself is unaffected; this is a probe-readout caveat only.

Wall estimate: g11 post-norm at ~29.1s/age × 33 retire ages = 16 min.
g10 has 1000/1331 = 0.75x the state cells, so ~21.9s/age × 33 = 12.0 min.

Compare against:
  - saved_runs/test_retire_g11/  (post-norm, empirical VAR, ~15.6 min)
  - saved_runs/test_retire_g9/   (post-norm, empirical VAR, 9.31 min, 5,900 failures)
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
from lifecycle.var import build_real_full_var_config_term_premium_theta
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.policy_io import save_policy_bundle


disc = DiscretizationConfig(
    n_wealth=100,
    wealth_min=0.01,
    wealth_max=750.0,
    n_savings=100,
    state_grid_sizes=(10, 10, 10),      # 1000 cells (vs 729 at g9, 1331 at g11)
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
    cell_vmap_chunks=8,                 # match g11 — A100-40GB safe at this state size
)

TERM_PREMIUM_THETA = 0.0


print("=" * 70, flush=True)
print(f"TEST g=10 RETIRE-ONLY  (theta={TERM_PREMIUM_THETA})", flush=True)
print(f"  state_grid=(10,10,10), state_n_stds=(3,3,3), n_z=10, max_iter=10", flush=True)
print("=" * 70, flush=True)

import jax
print(f"JAX devices: {jax.devices()}", flush=True)

print(f"\nBuilding model + precompute (theta={TERM_PREMIUM_THETA})...", flush=True)
t0 = time.time()
var_config, _fit_obj, _data = build_real_full_var_config_term_premium_theta(
    theta=TERM_PREMIUM_THETA,
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

print("\nSolving retirement only (theta=0) ...", flush=True)
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


BUNDLE_NAME = f"test_retire_g10_theta{int(TERM_PREMIUM_THETA*100):03d}"
BUNDLE_DIR = os.path.join("saved_runs", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "solve_control": solve_control._asdict(),
    "predictability_ablation": {
        "system_code": "full",
        "system_label": "full_system_real",
        "system_title": f"Full System (real, term_premium_theta={TERM_PREMIUM_THETA})",
        "state_names": ["cape", "spr", "y_1"],
    },
    "term_premium_theta": float(TERM_PREMIUM_THETA),
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(solve_wall),
    "scope": "retirement_only_theta0",
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
bundle_path = save_policy_bundle(
    BUNDLE_DIR, C, S, B,
    diagnostics=diag, run_config=run_config_snapshot,
    overwrite=True, wealth_grid=pc.wealth_grid,
)
print(f"  Saved: {bundle_path}", flush=True)


print("\n" + "=" * 70, flush=True)
print("DONE  —  retirement-only g=10 theta=0 run", flush=True)
print("=" * 70, flush=True)
print(f"  Solve wall: {solve_wall / 60:.2f} min", flush=True)
print(f"  Bundle:     {BUNDLE_DIR}/", flush=True)
