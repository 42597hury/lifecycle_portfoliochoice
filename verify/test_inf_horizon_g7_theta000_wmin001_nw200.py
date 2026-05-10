"""IH g=7, theta=0, wmin=0.01 BACK to baseline, n_wealth=200.

Probes whether doubling wealth-grid density (more points in the W[0..1]
AWI danger zone where the W_min_real kink lives) tames the IH outer-loop
oscillation that wmin=0.10 papered over.

log1p spacing with n_wealth=200, wmin=0.01 puts ~16 points in [0, 1] AWI
(vs 10 at n_wealth=100). The kink in the lifted MPC at W_min_real ~ 0.10
is then sampled finer in the next-iter V'(W') quadrature.

If this converges (stop drops monotonically like wmin=0.10 did): wealth
resolution alone fixes the IH instability — no need for wmin bump or
structural lift fix. If it still oscillates like the wmin=0.01 baseline:
the kink is genuinely discontinuous, more points just sample it finer
without smoothing, and the structural fix (floor V'(W') quadrature) is
the right path.

All else baseline: g=7, theta=0, wmin=0.01, qs=(4,4,4), qr=(4,4) GH,
mi=10, damp=1.0 (no damping), IH_MAX_ITER=200.

Wall: ~1 min wall (n_wealth=100 baseline was 0.81 min for 50 iters; at
MI=200 = ~3.2 min; n_wealth=200 adds ~10-20% lift cost = ~3.5-4 min).

Compare:
  - test_inf_horizon_g7_theta000               (wmin=0.01, nw=100, MI=50, stop=4.82, diverges)
  - test_inf_horizon_g7_theta000_wmin01_mi200  (wmin=0.10, nw=100, MI=200, stop=6.95e-5, converges)
"""
import os, time
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_real_full_var_config_term_premium_theta
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver
from lifecycle.policy_io import save_policy_bundle


disc = DiscretizationConfig(
    n_wealth=200,                       # bumped 100 -> 200
    wealth_min=0.01,                    # back to baseline (the danger-zone exposure)
    wealth_max=750.0,
    n_savings=100,
    state_grid_sizes=(7, 7, 7),
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
    tol=1e-6, max_iter=10, max_backtrack_iter=5,
    init_alpha_s=0.85, init_alpha_b=0.44, use_line_search=True,
    delta_bequest=0.0, gather_precision="f32",
    cell_vmap_chunks=2, wealth_dynamics_spec="ccv_log",
)

IH_TOL = 1e-5
IH_MAX_ITER = 200
IH_DAMPING = 1.0                        # baseline no-damping (test if wealth points alone fix it)
IH_PROGRESS_EVERY = 10
TERM_PREMIUM_THETA = 0.0


print("=" * 70, flush=True)
print(f"IH g=7 theta=0 wmin=0.01 n_wealth=200 (danger-zone resolution test)", flush=True)
print(f"  damping={IH_DAMPING}  MI={IH_MAX_ITER}", flush=True)
print("=" * 70, flush=True)

import jax
print(f"JAX devices: {jax.devices()}", flush=True)

t0 = time.time()
var_config, _, _ = build_real_full_var_config_term_premium_theta(theta=TERM_PREMIUM_THETA)
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=True)
print(f"Setup wall: {time.time() - t0:.1f}s", flush=True)
w = np.asarray(pc.wealth_grid)
print(f"  wealth grid: W[0]={w[0]:.4f}, W[5]={w[5]:.4f}, W[10]={w[10]:.4f}, W[15]={w[15]:.4f}, W[20]={w[20]:.4f}", flush=True)

t0 = time.time()
C, S, B, diag = run_infinite_horizon_solver(model, pc, solver_config=sc,
    max_iter=IH_MAX_ITER, tol=IH_TOL, damping=IH_DAMPING,
    progress_every=IH_PROGRESS_EVERY, show_progress=True, verbose=True)
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

BUNDLE_NAME = f"test_inf_horizon_g7_theta000_wmin001_nw200"
BUNDLE_DIR = os.path.join("saved_runs", "inf_horizon", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "inf_horizon_params": {"tol": IH_TOL, "max_iter": IH_MAX_ITER, "damping": IH_DAMPING},
    "predictability_ablation": {"system_code": "full", "system_label": "full_system_real",
        "system_title": f"Full System (real, theta={TERM_PREMIUM_THETA}, wmin=0.01, nw=200)",
        "state_names": ["cape", "spr", "y_1"]},
    "term_premium_theta": float(TERM_PREMIUM_THETA),
    "bundle_name": BUNDLE_NAME, "wall_time_seconds": float(solve_wall),
    "solver_kind": "infinite_horizon", "ablation_axis": "n_wealth_200_wmin_001",
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
save_policy_bundle(BUNDLE_DIR, C, S, B, diagnostics=diag, run_config=run_config_snapshot,
                   overwrite=True, wealth_grid=pc.wealth_grid)
print(f"DONE in {solve_wall / 60:.2f} min  bundle: {BUNDLE_DIR}/", flush=True)
