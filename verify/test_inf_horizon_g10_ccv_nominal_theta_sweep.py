"""IH g=10 NEW VAR (CCV nominal) theta sweep — parameterized.

Usage:
    python verify/test_inf_horizon_g10_ccv_nominal_theta_sweep.py 0.25
    python verify/test_inf_horizon_g10_ccv_nominal_theta_sweep.py 0.50
    python verify/test_inf_horizon_g10_ccv_nominal_theta_sweep.py 0.75

Same disc/solver as test_inf_horizon_g10_ccv_nominal_theta000_wmin01.py
and test_inf_horizon_g10_ccv_nominal_theta100_wmin01.py — only theta varies.

Together with the prior theta=0.0 and theta=1.0 bundles, this fills out
a clean 5-point theta sweep: {0.00, 0.25, 0.50, 0.75, 1.00}.

Bundle: saved_runs/inf_horizon/test_inf_horizon_g10_ccv_nominal_theta{NNN}_wmin01_qr33/
where NNN is the integer percentage (e.g. theta=0.25 → theta025).

Wall: ~3.4 min each.
"""
import os, sys, time
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from configs._canonical import BASE_CONFIG, resolve_var_csv_path
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_real_full_var_config
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver
from lifecycle.policy_io import save_policy_bundle


if len(sys.argv) != 2:
    print("usage: theta_sweep.py <theta>  (theta in 0.00..1.00)", file=sys.stderr)
    sys.exit(2)
TERM_PREMIUM_THETA = float(sys.argv[1])
TERM_PREMIUM_DATASET = "ccv_nominal"
THETA_TAG = f"theta{int(round(TERM_PREMIUM_THETA * 100)):03d}"
SWEEP_TAG = f"ccv_nominal_{THETA_TAG}"

disc = DiscretizationConfig(
    n_wealth=100, wealth_min=0.10, wealth_max=750.0, n_savings=100,
    state_grid_sizes=(10, 10, 10), state_grid_mode="cholesky",
    state_n_stds=(3.0, 3.0, 3.0), n_z=1, n_stds=3.0,
    n_eps_nodes=4, n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3), ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 3, 5), state_lobatto_Z=None,
)

sc = SolverConfig(tol=1e-6, max_iter=10, max_backtrack_iter=5,
    init_alpha_s=0.85, init_alpha_b=0.44, use_line_search=True,
    delta_bequest=0.0, gather_precision="f32",
    cell_vmap_chunks=8, wealth_dynamics_spec="ccv_log")

IH_TOL = 1e-5
IH_MAX_ITER = 200
IH_DAMPING = 0.5

print("=" * 70, flush=True)
print(f"IH g=10 NEW VAR ({TERM_PREMIUM_DATASET} theta={TERM_PREMIUM_THETA})", flush=True)
print(f"  qs={disc.n_state_quad_nodes}  qr={disc.n_ret_nodes_1d}  wmin={disc.wealth_min}", flush=True)
print(f"  bundle suffix: {SWEEP_TAG}", flush=True)
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
print(f"  n_state_quad={pc.n_state_quad}  n_ret_quad={pc.n_ret_quad}", flush=True)

t0 = time.time()
C, S, B, diag = run_infinite_horizon_solver(model, pc, solver_config=sc,
    max_iter=IH_MAX_ITER, tol=IH_TOL, damping=IH_DAMPING,
    progress_every=10, show_progress=True, verbose=True)
solve_wall = time.time() - t0
print(f"\nSolve wall: {solve_wall:.1f}s = {solve_wall / 60:.2f} min", flush=True)

print("\nSANITY", flush=True)
print(f"  Converged: {diag.get('converged', '?')}  n_iter: {diag.get('n_iter', '?')}", flush=True)
fss = diag.get('final_stopping_supnorm', None)
if fss is not None: print(f"  Final stop: {fss:.3e}", flush=True)
print(f"  alpha_s range: [{float(S.min()):.3f}, {float(S.max()):.3f}]", flush=True)
print(f"  alpha_b range: [{float(B.min()):.3f}, {float(B.max()):.3f}]", flush=True)
print(f"  C  range:      [{float(C.min()):.4f}, {float(C.max()):.4f}]", flush=True)
nih = diag.get("newton_iter_histogram")
if nih is not None:
    print(f"  Newton iters: p50={nih['p50']:.0f} p95={nih['p95']:.0f} p99={nih['p99']:.0f} max={nih['max']}", flush=True)
print(f"  Total Newton failures: {diag.get('total_newton_failures', '?')}", flush=True)
print(f"  Total Newton strict-tol misses: {diag.get('total_newton_strict_tol_misses', '?')}", flush=True)

# Center-cell readout (g=10 → flat=555)
state_axes = (10, 10, 10)
mid_per_axis = (5, 5, 5)
i_s_center = int(np.ravel_multi_index(mid_per_axis, state_axes))
i_w_mid = 50
print(f"\nCENTER-CELL POLICY (i_z=0, i_s={i_s_center}=({mid_per_axis}), i_w={i_w_mid}, W={float(pc.wealth_grid[i_w_mid]):.2f}):", flush=True)
print(f"  C={float(C[0, i_s_center, i_w_mid]):.4f}  alpha_s={float(S[0, i_s_center, i_w_mid]):.4f}  alpha_b={float(B[0, i_s_center, i_w_mid]):.4f}", flush=True)

BUNDLE_NAME = f"test_inf_horizon_g10_{SWEEP_TAG}_wmin01_qr33"
BUNDLE_DIR = os.path.join("saved_runs", "inf_horizon", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "inf_horizon_params": {"tol": IH_TOL, "max_iter": IH_MAX_ITER, "damping": IH_DAMPING},
    "predictability_ablation": {"system_code": "full", "system_label": "full_system_real",
        "system_title": f"Full System (real, CCV nominal, theta={TERM_PREMIUM_THETA})",
        "state_names": ["cape", "spr", "y_1"]},
    "term_premium_dataset": TERM_PREMIUM_DATASET,
    "term_premium_theta": float(TERM_PREMIUM_THETA),
    "var_csv_path": csv_path,
    "bundle_name": BUNDLE_NAME, "wall_time_seconds": float(solve_wall),
    "solver_kind": "infinite_horizon", "ablation_axis": SWEEP_TAG,
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
save_policy_bundle(BUNDLE_DIR, C, S, B, diagnostics=diag, run_config=run_config_snapshot,
                   overwrite=True, wealth_grid=pc.wealth_grid)
print(f"DONE in {solve_wall / 60:.2f} min  bundle: {BUNDLE_DIR}/", flush=True)
