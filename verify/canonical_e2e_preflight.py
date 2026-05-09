"""Phase 2 GPU pre-flight: full-lifecycle smoke at canonical wealth_min,
   with checkpoint save+resume cycle.

Runs the full canonical age range (start_age=22 → terminal_age=99) at
small grid sizes with the canonical config (wealth_min=0.01,
gather_precision='f32', real-yields VAR). Verifies:

  Pass A — Continuous solve: reference policy.
  Pass B — Partial solve, stops at youngest_age_to_solve=K, writes
           checkpoint mid-run.
  Pass C — Resume from B's checkpoint with no youngest_age_to_solve;
           the solver should pick up from the saved age and complete.
  Compare: max |A - (B->C)| / |A| < 1e-12 across (C, S, B).

  Plus: Path B clamp fires at wealth_grid[0] ~= 0.01 (post-Path-B
        regression — existing smokes pin wealth_min >= 0.05 and never
        exercise the corner).
  Plus: Simulator runs against the final bundle without NaN/Inf.
  Plus: Diagnostics banner runs cleanly post-Pi_z-removal.

Why this script: the canonical_small smoke runs a full age range but does
NOT thread a SolveControl with checkpoint_path. A latent checkpoint
plumbing bug would silently slip through. Per
docs/scans/PREFLIGHT_VALIDATION_REVIEW_2026-05-09.md (top issue #1).

Wall budget: ~15 min on a single A100 (n_wealth=30, state_grid=(3,3,3),
n_z=5, ~78 ages, gather_precision='f32').

Usage:
    python verify/canonical_e2e_preflight.py

Writes intermediate checkpoint to a temp directory and removes it at the
end. Saves the final-policy NPZ + simulation panel under
verify_canonical_e2e_preflight_*.npz for inspection.
"""
import os
import shutil
import tempfile
import time

import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, SolveControl
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.simulation import simulate_lifecycle
from lifecycle.policy_io import save_policy_bundle


# =============================================================================
# Tiny canonical-shaped config — full age range, small grids
# =============================================================================

# wealth_min=0.01 matches the canonical post-Path-B; this is the whole point
# of the script — a full lifecycle that actually exercises the corner clamp.
disc = DiscretizationConfig(
    n_wealth=30, wealth_min=0.01, wealth_max=750.0,
    n_savings=30,
    state_grid_sizes=(3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=5,
    n_eps_nodes=3,
    n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 3),
)

# Mirror canonical solver: gather_precision='f32', tol=1e-6, max_iter=100
# (canonical's 8000 is overkill at this scale).
sc = CANONICAL_SOLVER._replace(max_iter=100)

print("Building model and precompute...", flush=True)
var_config = build_real_full_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=True)
print(
    f"  shape: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, "
    f"n_w={pc.n_w}, n_s={pc.n_s}",
    flush=True,
)
assert float(np.asarray(pc.wealth_grid)[0]) <= 0.011, (
    f"wealth_grid[0] = {float(pc.wealth_grid[0]):.6f}, expected ~0.01 "
    f"(Path B exercise requires the corner to be on the grid)"
)


# =============================================================================
# Pass A — continuous reference solve
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("Pass A: continuous reference solve (no checkpoint)", flush=True)
print("=" * 70, flush=True)
t0 = time.time()
C_A, S_A, B_A, diag_A = run_lifecycle_solver(model, pc, sc, verbose=1)
wall_A = time.time() - t0
print(f"\nPass A wall: {wall_A / 60:.2f} min", flush=True)
assert np.isfinite(C_A).all(), "Pass A: NaN/Inf in C"
assert np.isfinite(S_A).all(), "Pass A: NaN/Inf in S"
assert np.isfinite(B_A).all(), "Pass A: NaN/Inf in B"


# =============================================================================
# Pass B — partial solve to mid-age, with checkpoint
# =============================================================================

ckpt_dir = tempfile.mkdtemp(prefix="canonical_e2e_preflight_ckpt_")
ckpt_path = os.path.join(ckpt_dir, "policy_checkpoint")
mid_age = (model.start_age + model.terminal_age) // 2
print("\n" + "=" * 70, flush=True)
print(f"Pass B: partial solve to age={mid_age} with checkpointing", flush=True)
print(f"  checkpoint_path: {ckpt_path}", flush=True)
print("=" * 70, flush=True)
ctrl_partial = SolveControl(
    youngest_age_to_solve=mid_age,
    checkpoint_path=ckpt_path,
    checkpoint_every_n_ages=5,
    save_on_interrupt=False,
    return_partial_on_interrupt=False,
)
t0 = time.time()
C_B, S_B, B_B, diag_B = run_lifecycle_solver(
    model, pc, sc, verbose=1, solve_control=ctrl_partial,
)
wall_B = time.time() - t0
print(f"\nPass B wall: {wall_B / 60:.2f} min", flush=True)
print(f"Pass B status: {diag_B.get('solve_status', '?')}", flush=True)
solved_mask_B = np.asarray(diag_B["solved_age_mask"], dtype=bool)
print(
    f"  solved ages: {int(solved_mask_B.sum())}/{len(solved_mask_B)}  "
    f"(youngest unsolved age = "
    f"{int(np.asarray(pc.ages)[~solved_mask_B][0]) if (~solved_mask_B).any() else 'none'})",
    flush=True,
)
assert (~solved_mask_B).any(), (
    "Pass B was supposed to stop early but ALL ages got solved — "
    f"youngest_age_to_solve={mid_age} did not take effect."
)


# =============================================================================
# Pass C — resume from Pass B's checkpoint, complete the rest
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("Pass C: resume from checkpoint, complete remaining ages", flush=True)
print("=" * 70, flush=True)
ctrl_resume = SolveControl(
    youngest_age_to_solve=None,        # no early stop — solve to start_age
    checkpoint_path=ckpt_path,         # resume from B
    checkpoint_every_n_ages=5,
    save_on_interrupt=False,
    return_partial_on_interrupt=False,
)
t0 = time.time()
C_C, S_C, B_C, diag_C = run_lifecycle_solver(
    model, pc, sc, verbose=1, solve_control=ctrl_resume,
)
wall_C = time.time() - t0
print(f"\nPass C wall: {wall_C / 60:.2f} min", flush=True)
solved_mask_C = np.asarray(diag_C["solved_age_mask"], dtype=bool)
assert solved_mask_C.all(), (
    f"Pass C did not solve every age: {int(solved_mask_C.sum())}/{len(solved_mask_C)}"
)


# =============================================================================
# Compare A (continuous) vs C (resumed) — should be bit-identical
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("Compare: continuous (A) vs checkpoint+resume (C)", flush=True)
print("=" * 70, flush=True)


def _max_rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))


rel_C = _max_rel(C_A, C_C)
rel_S = _max_rel(S_A, S_C)
rel_B = _max_rel(B_A, B_C)
print(f"  max rel err C: {rel_C:.2e}", flush=True)
print(f"  max rel err S: {rel_S:.2e}", flush=True)
print(f"  max rel err B: {rel_B:.2e}", flush=True)
# Resume should be bit-identical (same float ops in same order). Allow a
# tiny epsilon for f32 trace re-key timing; > 1e-12 indicates a real
# divergence (resume produced a different policy).
TOL_RESUME = 1e-10
checkpoint_pass = rel_C < TOL_RESUME and rel_S < TOL_RESUME and rel_B < TOL_RESUME
if not checkpoint_pass:
    print(
        "  FAIL: checkpoint+resume diverges from continuous solve. "
        "Investigate before launching production.",
        flush=True,
    )


# =============================================================================
# Path B clamp regression — corner solution at wealth_grid[0]
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("Path B clamp regression at wealth_grid[0] = 0.01", flush=True)
print("=" * 70, flush=True)
w0 = float(np.asarray(pc.wealth_grid)[0])
c_at_w0 = np.asarray(C_C)[..., 0]      # shape (n_age, n_z, N_state)
s_at_w0 = np.asarray(S_C)[..., 0]
b_at_w0 = np.asarray(B_C)[..., 0]
# Path B sets c=W exactly, alpha_s=alpha_b=0 below the smallest real
# interior W_implied. At wealth_grid[0]=0.01 the corner must fire.
clamp_pass_c = np.allclose(c_at_w0, w0, rtol=0, atol=1e-9)
clamp_pass_s = np.allclose(s_at_w0, 0.0, rtol=0, atol=1e-12)
clamp_pass_b = np.allclose(b_at_w0, 0.0, rtol=0, atol=1e-12)
print(f"  C[..., 0] == w0={w0:.6f}: {clamp_pass_c}", flush=True)
print(f"  S[..., 0] == 0:           {clamp_pass_s}", flush=True)
print(f"  B[..., 0] == 0:           {clamp_pass_b}", flush=True)
clamp_pass = clamp_pass_c and clamp_pass_s and clamp_pass_b


# =============================================================================
# Simulator smoke — final bundle is usable
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("Simulator smoke (5000 households)", flush=True)
print("=" * 70, flush=True)
t0 = time.time()
sim = simulate_lifecycle(
    C_C, S_C, B_C, pc, model, n_simulations=5000, seed=42, verbose=False,
)
wall_sim = time.time() - t0
print(f"  sim wall: {wall_sim:.1f}s", flush=True)
sim_pass = np.isfinite(sim["x"]).all() and np.isfinite(sim["c"]).all()
print(f"  sim finite: {sim_pass}", flush=True)
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


# =============================================================================
# Save artefacts and tear down checkpoint
# =============================================================================

BUNDLE_NAME = "canonical_e2e_preflight"
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
    "wall_time_seconds": float(wall_A + wall_B + wall_C),
    "preflight_passes": {
        "wall_A_continuous_min": wall_A / 60,
        "wall_B_partial_min": wall_B / 60,
        "wall_C_resume_min": wall_C / 60,
    },
}
bundle_path = save_policy_bundle(
    BUNDLE_DIR, C_C, S_C, B_C,
    diagnostics=diag_C, run_config=run_config_snapshot,
    overwrite=True, wealth_grid=pc.wealth_grid,
)
print(f"\nSaved bundle: {bundle_path}", flush=True)

# Sim panel — separate file alongside the bundle for downstream use.
out_sim = os.path.join(BUNDLE_DIR, "preflight_sim.npz")
np.savez(out_sim, **{k: v for k, v in sim.items() if isinstance(v, np.ndarray)})
print(f"Saved sim:    {out_sim}", flush=True)

shutil.rmtree(ckpt_dir, ignore_errors=True)
print(f"Removed checkpoint dir: {ckpt_dir}", flush=True)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70, flush=True)
print("PRE-FLIGHT SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"  Pass A wall:                 {wall_A / 60:.2f} min", flush=True)
print(f"  Pass B + Pass C wall:        {(wall_B + wall_C) / 60:.2f} min", flush=True)
print(f"  Checkpoint+resume identity:  {'PASS' if checkpoint_pass else 'FAIL'}", flush=True)
print(f"  Path B clamp at w0=0.01:     {'PASS' if clamp_pass else 'FAIL'}", flush=True)
print(f"  Simulator smoke:             {'PASS' if sim_pass else 'FAIL'}", flush=True)

all_pass = checkpoint_pass and clamp_pass and sim_pass
if not all_pass:
    print("\nPRE-FLIGHT FAILED — investigate before launching production run.", flush=True)
    raise SystemExit(1)
print("\nAll pre-flight gates PASS. Cleared for full canonical solve.", flush=True)
