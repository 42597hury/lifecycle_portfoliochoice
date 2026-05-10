"""Inf-horizon g=9, theta=0, Newton max_iter=50, damping=0.5.

Same combo as test_inf_horizon_g7_theta000_mi50_damp05.py but at g=9
state grid (729 cells vs 343 at g7) — confirms whether the noise-suppression
benefit (if any) holds or shrinks at finer state discretization.

Wall budget: roughly 2-3x g7 (729 vs 343 cells). g7 wall is ~0.8 min,
so estimate ~1.5-2.5 min for g9.

Compare against:
  - test_inf_horizon_g7_theta000_mi50_damp05  (g7 same combo)
  - test_inf_horizon_g9                        (g9 baseline empirical, diverged stop=10.8)
"""
import os
import time

import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_real_full_var_config_term_premium_theta
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver
from lifecycle.policy_io import save_policy_bundle


disc = DiscretizationConfig(
    n_wealth=100,
    wealth_min=0.01,
    wealth_max=750.0,
    n_savings=100,
    state_grid_sizes=(9, 9, 9),
    state_grid_mode="cholesky",
    state_n_stds=(3.0, 3.0, 3.0),
    n_z=1,
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
    max_iter=50,                        # IH Newton cap rule
    max_backtrack_iter=5,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    use_line_search=True,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=4,                 # bumped from g7's 2 — 2.1x more state cells
    wealth_dynamics_spec="ccv_log",
)

IH_TOL = 1e-5
IH_MAX_ITER = 50
IH_DAMPING = 0.5
IH_PROGRESS_EVERY = 1
TERM_PREMIUM_THETA = 0.0


print("=" * 70, flush=True)
print(f"TEST INF-HORIZON g=9 (theta={TERM_PREMIUM_THETA})  mi=50, damp=0.5", flush=True)
print(f"  state_grid=(9,9,9), state_n_stds=(3,3,3), n_z=1", flush=True)
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

print(
    f"\nRunning inf-horizon fixed-point "
    f"(IH_TOL={IH_TOL}, IH_MAX_ITER={IH_MAX_ITER}, damping={IH_DAMPING}) ...",
    flush=True,
)
t0 = time.time()
C, S, B, diag = run_infinite_horizon_solver(
    model, pc,
    solver_config=sc,
    max_iter=IH_MAX_ITER,
    tol=IH_TOL,
    damping=IH_DAMPING,
    progress_every=IH_PROGRESS_EVERY,
    show_progress=True,
    verbose=True,
)
solve_wall = time.time() - t0
print(f"\nSolve wall: {solve_wall:.1f}s = {solve_wall / 60:.2f} min", flush=True)


print("\n" + "=" * 70, flush=True)
print("SANITY", flush=True)
print("=" * 70, flush=True)
print(f"  Converged       : {diag.get('converged', '?')}", flush=True)
print(f"  Iterations done : {diag.get('n_iter', '?')}", flush=True)
fss = diag.get('final_stopping_supnorm', None)
if fss is not None:
    print(f"  Final stop sup-norm: {fss:.3e}", flush=True)
print(f"  NaN count : C={int(np.isnan(C).sum())}  S={int(np.isnan(S).sum())}  B={int(np.isnan(B).sum())}", flush=True)
print(f"  Inf count : C={int(np.isinf(C).sum())}  S={int(np.isinf(S).sum())}  B={int(np.isinf(B).sum())}", flush=True)
print(f"  alpha_s range : [{float(S.min()):.3f}, {float(S.max()):.3f}]", flush=True)
print(f"  alpha_b range : [{float(B.min()):.3f}, {float(B.max()):.3f}]", flush=True)
print(f"  C  range      : [{float(C.min()):.4f}, {float(C.max()):.4f}]", flush=True)

w0 = float(np.asarray(pc.wealth_grid)[0])
c_at_w0_max = float(np.asarray(C)[..., 0].max())
c_at_w0_min = float(np.asarray(C)[..., 0].min())
s_at_w0_abs = float(np.abs(np.asarray(S)[..., 0]).max())
b_at_w0_abs = float(np.abs(np.asarray(B)[..., 0]).max())
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
        f"p99={nih['p99']:.0f}  max={nih['max']}  (cap=50)",
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


BUNDLE_NAME = f"test_inf_horizon_g9_theta{int(TERM_PREMIUM_THETA*100):03d}_mi50_damp05"
BUNDLE_DIR = os.path.join("saved_runs", "inf_horizon", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "inf_horizon_params": {
        "tol": IH_TOL,
        "max_iter": IH_MAX_ITER,
        "damping": IH_DAMPING,
    },
    "predictability_ablation": {
        "system_code": "full",
        "system_label": "full_system_real",
        "system_title": f"Full System (real, theta={TERM_PREMIUM_THETA}, mi=50, damp=0.5)",
        "state_names": ["cape", "spr", "y_1"],
    },
    "term_premium_theta": float(TERM_PREMIUM_THETA),
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(solve_wall),
    "solver_kind": "infinite_horizon",
    "ablation_axis": "newton_mi50_damp05",
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
bundle_path = save_policy_bundle(
    BUNDLE_DIR, C, S, B,
    diagnostics=diag, run_config=run_config_snapshot,
    overwrite=True, wealth_grid=pc.wealth_grid,
)
print(f"  Saved: {bundle_path}", flush=True)


print("\n" + "=" * 70, flush=True)
print("DONE  —  inf-horizon g=9 theta=0 mi=50 damp=0.5 run", flush=True)
print("=" * 70, flush=True)
print(f"  Solve wall: {solve_wall / 60:.2f} min", flush=True)
print(f"  Bundle:     {BUNDLE_DIR}/", flush=True)
