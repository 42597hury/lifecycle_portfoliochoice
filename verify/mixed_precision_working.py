"""Phase 3 GPU pre-flight: f32 vs f64 agreement on a working-age slice.

Why this script
---------------
verify/mixed_precision_tiny.py covers the retirement kernel only. The
working kernel uses a different gather chain
(_interp_c_and_mpc_at_cell, with eta/eps quadratures) than the inline
per_kv_kr in retirement_foc_jac_ccv. A bug isolated to the working-kernel
cast pattern would not be caught by mixed_precision_tiny.

This script runs the FULL canonical age range (working + retirement) at
small scale once at gather_precision='f64' and once at 'f32', then
compares (C, S, B) split by phase:
  - Working ages       (start_age .. retire_age - 1)
  - Retirement ages    (retire_age .. terminal_age - 1)

Passes if max relative error in each phase < 1e-4 (the documented
gather_precision agreement target in lifecycle/model.py).

Wall budget
-----------
~5-10 min on a single A100 with n_wealth=15, state_grid=(2,2,2), n_z=3,
canonical age range (ages 22..99 = 78 ages).
"""
import time

import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver


# =============================================================================
# Tiny config — small grids, FULL canonical age range so working ages hit
# =============================================================================

# Use canonical wealth_min=0.01 so the working kernel sees the same
# Path B-clamped corner region as production.
disc = DiscretizationConfig(
    n_wealth=15, wealth_min=0.01, wealth_max=750.0,
    n_savings=15,
    state_grid_sizes=(2, 2, 2),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=3,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(2, 2, 2),
)

print("Building model and precompute...", flush=True)
var_config = build_real_full_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=False)
print(
    f"  shape: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, "
    f"n_w={pc.n_w}, n_s={pc.n_s}",
    flush=True,
)


# =============================================================================
# Solve once at f64 and once at f32
# =============================================================================

def solve_at(gather_precision):
    sc = CANONICAL_SOLVER._replace(
        max_iter=100, gather_precision=gather_precision,
    )
    t0 = time.time()
    C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=0)
    wall = time.time() - t0
    return C, S, B, diag, wall


print("\n[f64] solving full lifecycle...", flush=True)
C64, S64, B64, diag64, wall_f64 = solve_at("f64")
print(f"  f64 wall: {wall_f64:.1f}s", flush=True)
assert np.isfinite(C64).all(), "f64: NaN/Inf in C"
assert np.isfinite(S64).all(), "f64: NaN/Inf in S"
assert np.isfinite(B64).all(), "f64: NaN/Inf in B"

print("\n[f32] solving full lifecycle...", flush=True)
C32, S32, B32, diag32, wall_f32 = solve_at("f32")
print(f"  f32 wall: {wall_f32:.1f}s", flush=True)
assert np.isfinite(C32).all(), "f32: NaN/Inf in C"
assert np.isfinite(S32).all(), "f32: NaN/Inf in S"
assert np.isfinite(B32).all(), "f32: NaN/Inf in B"


# =============================================================================
# Compare split by phase: working (pre-retire) vs retirement
# =============================================================================

ages = np.asarray(pc.ages)
working_mask = ages < model.retire_age
retire_mask = (ages >= model.retire_age) & (ages < model.terminal_age)

# Drop the terminal age — it's z-invariant by construction and doesn't
# exercise either kernel meaningfully.
n_working = int(working_mask.sum())
n_retire = int(retire_mask.sum())
print(
    f"\nAge partition: working={n_working} (ages {int(ages[working_mask][0])}..."
    f"{int(ages[working_mask][-1])}), retire={n_retire} (ages "
    f"{int(ages[retire_mask][0])}...{int(ages[retire_mask][-1])})",
    flush=True,
)


def _max_rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-10)))


print("\nWorking-age agreement (the primary target of this script):", flush=True)
rel_C_w = _max_rel(C32[working_mask], C64[working_mask])
rel_S_w = _max_rel(S32[working_mask], S64[working_mask])
rel_B_w = _max_rel(B32[working_mask], B64[working_mask])
print(f"  C: {rel_C_w:.2e}  S: {rel_S_w:.2e}  B: {rel_B_w:.2e}", flush=True)

print("\nRetirement-age agreement (sanity — already covered by mixed_precision_tiny):", flush=True)
rel_C_r = _max_rel(C32[retire_mask], C64[retire_mask])
rel_S_r = _max_rel(S32[retire_mask], S64[retire_mask])
rel_B_r = _max_rel(B32[retire_mask], B64[retire_mask])
print(f"  C: {rel_C_r:.2e}  S: {rel_S_r:.2e}  B: {rel_B_r:.2e}", flush=True)


# =============================================================================
# Verdict
# =============================================================================

# Agreement target documented in lifecycle/model.py SolverConfig.gather_precision
# docstring: "agreement test at 1e-4 relative".
TOL = 1e-4
working_pass = max(rel_C_w, rel_S_w, rel_B_w) < TOL
retire_pass = max(rel_C_r, rel_S_r, rel_B_r) < TOL

print("\n" + "=" * 70, flush=True)
print("MIXED-PRECISION (working+retirement) PRE-FLIGHT", flush=True)
print("=" * 70, flush=True)
print(f"  Working-age f32 vs f64: {'PASS' if working_pass else 'FAIL'} "
      f"(max rel = {max(rel_C_w, rel_S_w, rel_B_w):.2e}, tol = {TOL:.0e})", flush=True)
print(f"  Retirement   f32 vs f64: {'PASS' if retire_pass else 'FAIL'} "
      f"(max rel = {max(rel_C_r, rel_S_r, rel_B_r):.2e}, tol = {TOL:.0e})", flush=True)
print(f"  Wall: f64={wall_f64:.1f}s  f32={wall_f32:.1f}s  "
      f"speedup={wall_f64 / wall_f32:.2f}x", flush=True)

if not (working_pass and retire_pass):
    raise SystemExit(1)
print("\nAll mixed-precision gates PASS.", flush=True)
