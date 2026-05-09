"""Test baseline: full-lifecycle solve at the user's test config.

(4,4,4) state, (4,4,4) state_quad, (4,4) ret, n_z=10, max_iter=100,
max_backtrack_iter=5, gather_precision='f32', full age range 22..99.

Wall budget: ~2.3 h on a single A100.

Saves bundle to saved_runs/test_baseline/ and a 5000-household sim panel
alongside. Prints the Newton + backtrack iter histograms so we can see
where the actual p99 lands relative to the (100, 5) caps.

This is ONE point. Convergence requires a second point at (5,5,5) or
similar — fire after this lands.
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
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.simulation import simulate_lifecycle
from lifecycle.policy_io import save_policy_bundle


# =============================================================================
# Test-baseline config
# =============================================================================

disc = DiscretizationConfig(
    n_wealth=100,
    wealth_min=0.01,
    wealth_max=750.0,
    n_savings=100,
    state_grid_sizes=(4, 4, 4),         # 64 cells
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=10,
    n_stds=3.0,
    n_eps_nodes=4,
    n_eta_nodes=3,
    n_ret_nodes_1d=(4, 4),              # 16 nodes
    ret_lobatto_Z=None,
    n_state_quad_nodes=(4, 4, 4),       # 64 nodes (uniform 4 — no K-bump in this baseline)
    state_lobatto_Z=None,
)

sc = SolverConfig(
    tol=1e-6,
    max_iter=100,
    max_backtrack_iter=5,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    use_line_search=True,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=2,
)


# =============================================================================
# Solve
# =============================================================================

print("=" * 70, flush=True)
print("TEST BASELINE — full lifecycle at (4,4,4)/(4,4,4)/(4,4)/n_z=10/mb=5", flush=True)
print("=" * 70, flush=True)

import jax
print(f"JAX devices: {jax.devices()}", flush=True)

print("\nBuilding model + precompute...", flush=True)
t0 = time.time()
var_config = build_real_full_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=True)
print(f"Setup wall: {time.time() - t0:.1f}s", flush=True)
print(
    f"  n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}, "
    f"n_s={pc.n_s}, n_state_quad={pc.n_state_quad}, n_ret_quad={pc.n_ret_quad}",
    flush=True,
)

print("\nSolving full lifecycle...", flush=True)
t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1)
solve_wall = time.time() - t0
print(f"\nSolve wall: {solve_wall:.1f}s = {solve_wall / 60:.2f} min = {solve_wall / 3600:.2f} h", flush=True)


# =============================================================================
# Sanity + diagnostics
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("SANITY", flush=True)
print("=" * 70, flush=True)
print(f"  NaN count : C={int(np.isnan(C).sum())}  S={int(np.isnan(S).sum())}  B={int(np.isnan(B).sum())}", flush=True)
print(f"  Inf count : C={int(np.isinf(C).sum())}  S={int(np.isinf(S).sum())}  B={int(np.isinf(B).sum())}", flush=True)
print(f"  alpha_s range: [{float(S.min()):.3f}, {float(S.max()):.3f}]", flush=True)
print(f"  alpha_b range: [{float(B.min()):.3f}, {float(B.max()):.3f}]", flush=True)
print(f"  C  range: [{float(C.min()):.4f}, {float(C.max()):.4f}]", flush=True)

# Path B regression
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
        f"p99={nih['p99']:.0f}  max={nih['max']}  (cap=100)",
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


# =============================================================================
# Save bundle
# =============================================================================

BUNDLE_NAME = "test_baseline"
BUNDLE_DIR = os.path.join("saved_runs", BUNDLE_NAME)
run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc._asdict(),
    "solver_config": sc._asdict(),
    "predictability_ablation": {
        "system_code": "full",
        "system_label": "full_system_real",
        "system_title": "Full System (real)",
        "state_names": ["cape", "spr", "y_1"],
    },
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(solve_wall),
}
print(f"\nSaving bundle to {BUNDLE_DIR}/ ...", flush=True)
bundle_path = save_policy_bundle(
    BUNDLE_DIR, C, S, B,
    diagnostics=diag, run_config=run_config_snapshot,
    overwrite=True, wealth_grid=pc.wealth_grid,
)
print(f"  Saved: {bundle_path}", flush=True)


# =============================================================================
# Sim
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("SIMULATION (5000 households, seed=42)", flush=True)
print("=" * 70, flush=True)
t0 = time.time()
sim = simulate_lifecycle(C, S, B, pc, model, n_simulations=5000, seed=42, verbose=False)
sim_wall = time.time() - t0
print(f"  sim wall: {sim_wall:.1f}s", flush=True)
print(f"  sim NaN: x={int(np.isnan(sim['x']).sum())}  c={int(np.isnan(sim['c']).sum())}", flush=True)
print(f"  survival to terminal: {float(sim['alive'][:, -1].mean()):.1%}", flush=True)
print(f"  median death age: {int(np.median(sim['death_age']))}", flush=True)
print(
    f"  alive mean wealth at age 22/44/67/87: "
    f"{float(sim['x'][sim['alive'][:, 0], 0].mean()):.2f}  "
    f"{float(sim['x'][sim['alive'][:, 22], 22].mean()):.2f}  "
    f"{float(sim['x'][sim['alive'][:, 45], 45].mean()):.2f}  "
    f"{float(sim['x'][sim['alive'][:, 65], 65].mean()):.2f}",
    flush=True,
)

sim_path = os.path.join(BUNDLE_DIR, "test_baseline_sim.npz")
np.savez(sim_path, **{k: v for k, v in sim.items() if isinstance(v, np.ndarray)})
print(f"  saved sim: {sim_path}", flush=True)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("DONE", flush=True)
print("=" * 70, flush=True)
print(f"  Solve wall: {solve_wall / 3600:.2f} h", flush=True)
print(f"  Sim wall:   {sim_wall:.1f} s", flush=True)
print(f"  Bundle:     {BUNDLE_DIR}/", flush=True)
print(f"\nNext: read newton/backtrack histograms; run EE residuals on the bundle:", flush=True)
print(f"  python verify/invalid_cells.py {BUNDLE_NAME}", flush=True)
print(f"  python verify/ee_residuals.py {BUNDLE_NAME} --metric consumption_ee", flush=True)
print(f"  python verify/ee_simpath.py {BUNDLE_NAME} --eval-mode next_finer", flush=True)
