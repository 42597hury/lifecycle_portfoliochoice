"""Overnight lifecycle solve — App-C Shi-10 theta=0, full 78 ages.

Spec (locked with user 2026-05-10):
  state_grid=(12,12,12)  state_n_stds=(3,3,3)  n_z=30
  n_state_quad=(4,4,4)   n_ret=(4,4) GH        n_eps=4 n_eta=3
  n_wealth=100  wealth_min=0.01  wealth_max=500
  n_savings=100
  Newton: tol=1e-6, max_iter=15, max_backtrack=5, line_search=True
  gather_precision='f32'  cell_vmap_chunks=8
  delta_bequest=0.0
  use_backward_age_warm_start=True  failure_seed_from_neighbor=True
  checkpoint_every_n_ages=1

VAR: data/ccv_nominal_yield_scaling/var_dataset_theta_0p00.csv (Shi-10 canonical)

Wall: ~11-12 hr H100.
"""
import os, time
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from configs._canonical import BASE_CONFIG, resolve_var_csv_path
from lifecycle.model import DiscretizationConfig, SolverConfig, SolveControl
from lifecycle.var import build_real_full_var_config
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.policy_io import save_policy_bundle


TERM_PREMIUM_DATASET = "ccv_nominal"      # Shi-10 default family
TERM_PREMIUM_THETA = 0.0

disc = DiscretizationConfig(
    n_wealth=100,
    wealth_min=0.01,
    wealth_max=500.0,
    n_savings=100,
    state_grid_sizes=(12, 12, 12),
    state_grid_mode="cholesky",
    state_n_stds=(3.0, 3.0, 3.0),
    n_z=30,
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
    max_iter=15,
    max_backtrack_iter=5,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    use_line_search=True,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=8,
    wealth_dynamics_spec="ccv_log",
    use_backward_age_warm_start=True,
    failure_seed_from_neighbor=True,
)

solve_control = SolveControl(
    checkpoint_every_n_ages=1,
    save_on_interrupt=True,
    return_partial_on_interrupt=True,
)

print("=" * 70, flush=True)
print(f"OVERNIGHT LIFECYCLE — Shi-10 (CCV nominal) theta={TERM_PREMIUM_THETA}", flush=True)
print(f"  state=(12,12,12) n_z=30 n_wealth=100 n_savings=100 wmax=500", flush=True)
print(f"  Newton mi=15 chunks=8 gather=f32", flush=True)
print("=" * 70, flush=True)

import jax
print(f"JAX devices: {jax.devices()}", flush=True)

csv_path = resolve_var_csv_path(TERM_PREMIUM_THETA, term_premium_dataset=TERM_PREMIUM_DATASET)
print(f"\nVAR CSV: {csv_path}", flush=True)

t0 = time.time()
var_config, _, _ = build_real_full_var_config(csv_path=csv_path)
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=True)
print(f"Setup wall: {time.time() - t0:.1f}s", flush=True)
print(
    f"  n_age={pc.n_age}  n_z={pc.n_z}  N_state={pc.N_state}  n_w={pc.n_w}  "
    f"n_state_quad={pc.n_state_quad}  n_ret_quad={pc.n_ret_quad}",
    flush=True,
)

print("\nSolving full lifecycle (78 ages)...", flush=True)
t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1, solve_control=solve_control)
solve_wall = time.time() - t0
print(f"\nSolve wall: {solve_wall:.1f}s = {solve_wall / 60:.2f} min = {solve_wall / 3600:.2f} hr", flush=True)

print("\nSANITY", flush=True)
solved_mask = np.asarray(diag.get("solved_age_mask", []), dtype=bool)
print(f"  solved ages: {int(solved_mask.sum())}/{len(solved_mask)}", flush=True)
C_solved = np.asarray(C)[solved_mask]
S_solved = np.asarray(S)[solved_mask]
B_solved = np.asarray(B)[solved_mask]
print(f"  NaN: C={int(np.isnan(C_solved).sum())} S={int(np.isnan(S_solved).sum())} B={int(np.isnan(B_solved).sum())}", flush=True)
print(f"  Inf: C={int(np.isinf(C_solved).sum())} S={int(np.isinf(S_solved).sum())} B={int(np.isinf(B_solved).sum())}", flush=True)
print(f"  alpha_s range: [{float(S_solved.min()):.3f}, {float(S_solved.max()):.3f}]", flush=True)
print(f"  alpha_b range: [{float(B_solved.min()):.3f}, {float(B_solved.max()):.3f}]", flush=True)
print(f"  C  range:      [{float(C_solved.min()):.4f}, {float(C_solved.max()):.4f}]", flush=True)

nih = diag.get("newton_iter_histogram")
if nih is not None:
    print(f"  Newton iters: p50={nih['p50']:.0f} p95={nih['p95']:.0f} p99={nih['p99']:.0f} max={nih['max']}  (cap=15)", flush=True)
bth = diag.get("backtrack_iter_histogram")
if bth is not None:
    print(f"  Backtrack:    p50={bth['p50']:.1f} p95={bth['p95']:.1f} p99={bth['p99']:.1f} max={bth['max']}  (cap=5)", flush=True)
print(f"  Total Newton failures (FORI strict-tol): {diag.get('total_newton_failures', '?')}", flush=True)

BUNDLE_NAME = f"overnight_lifecycle_shi10_theta{int(TERM_PREMIUM_THETA*100):03d}"
BUNDLE_DIR = os.path.join("saved_runs", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "solve_control": solve_control._asdict(),
    "predictability_ablation": {
        "system_code": "full", "system_label": "full_system_real",
        "system_title": f"Full System (real, App-C Shi-10, theta={TERM_PREMIUM_THETA})",
        "state_names": ["cape", "spr", "y_1"],
    },
    "term_premium_dataset": TERM_PREMIUM_DATASET,
    "term_premium_theta": float(TERM_PREMIUM_THETA),
    "var_csv_path": csv_path,
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(solve_wall),
    "scope": "lifecycle_full_overnight",
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
save_policy_bundle(BUNDLE_DIR, C, S, B, diagnostics=diag, run_config=run_config_snapshot,
                   overwrite=True, wealth_grid=pc.wealth_grid)
print(f"DONE  wall={solve_wall / 3600:.2f} hr  bundle: {BUNDLE_DIR}/", flush=True)
