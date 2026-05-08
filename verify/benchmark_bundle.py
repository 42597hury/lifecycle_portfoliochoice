"""Benchmark JAX retirement-only solve against the Numba reference bundle.

Mirrors thesisscripts/configs/run_ccv_wide9_gh_k4.py — the most recent bundle
at the time of writing (system_iv_full_var_unconstrained_cholesky_grid9x9x9_nz11_ccv_wide9_gh_k4).
That run took 1341.9s = 22.4 min on Numba (likely c6i.4xlarge / 16 vCPU).

Config (post rtb-as-state — state vector is 4-D (cy, spr, rtb, y_1)):
    state_grid_sizes = (9, 9, 9, 9)            # N_state = 6561
    state_n_stds     = (2.93, 2.93, 2.93, 2.93)
    n_z              = 11
    n_wealth         = 180,  wealth_min = 0.05
    n_savings        = 180
    n_eps_nodes      = 4,    n_eta_nodes = 4
    n_ret_nodes_1d   = (5, 5)                  # (xr, xb) only; 25 return quad points
    n_state_quad_nodes = (3, 4, 3, 4)          # 144 state quad points
    youngest_age_to_solve = 67  →  retirement-only (33 ages: 67..99)
    wealth_dynamics_spec = "ccv_log"
    max_iter = 400                              # tighter than canonical 8000

Outputs:
    ./saved_runs/<bundle-name>/                — local bundle (npz + metadata)
    s3://${S3_BUCKET}/saved_runs/<bundle-name>/ — if S3_BUCKET env var is set

Plus a post-solve terminal-portfolio diagnostic printed to stdout for a quick
sanity check (FOC residuals at the solved policy).
"""
import os
import shutil
import subprocess
import sys
sys.path.insert(0, ".")

import time
import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.model import SolveControl
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.policy_io import save_policy_bundle
from lifecycle.diagnostics import diagnose_terminal_portfolio_states

BUNDLE_NAME = "system_iv_full_var_unconstrained_cholesky_grid9x9x9x9_nz11_jax_benchmark"
BUNDLE_DIR = os.path.join("saved_runs", BUNDLE_NAME)

# 4-D state grid post rtb-as-state. axes are (cy, spr, rtb, y_1); 9-points
# per axis matches the previous CCV-wide setting on the three retained axes
# and adds a matching width on the new rtb axis.
disc_config = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(9, 9, 9, 9),
    state_n_stds=(2.93, 2.93, 2.93, 2.93),
    n_ret_nodes_1d=(5, 5),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 4, 3, 4),
    state_lobatto_Z=None,
)
solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
    # Tighten Newton cap from 8000 → 400. The smoke (6-age) showed identical
    # policies at max_iter=100 vs 8000; 400 leaves a 4-13× margin over typical
    # convergence (5-30 iters/cell). lax.while_loop's per-iter overhead is
    # higher than Numba's plain Python loop, so 8000 is wasteful on JAX.
    # Newton failures will show in diag["total_newton_failures"] if any cell
    # actually needs more.
    max_iter=400,
    max_iter_unconstrained=400,
)
solve_control = SolveControl(
    youngest_age_to_solve=67,
    save_on_interrupt=False,
    return_partial_on_interrupt=True,
)

print("=" * 70, flush=True)
print("JAX BENCHMARK: 9x9x9x9 retirement-only run", flush=True)
print(f"Numba reference: 1341.9s = 22.4 min", flush=True)
print("=" * 70, flush=True)

print("\nBuilding model + precompute...", flush=True)
t0 = time.time()
var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc_config, verbose=True)
print(f"Setup wall: {time.time() - t0:.1f}s", flush=True)
print(
    f"  N_state={pc.N_state}, n_z={pc.n_z}, n_w={pc.n_w}, n_s={pc.n_s}, "
    f"n_state_quad={pc.n_state_quad}, n_ret_quad={pc.n_ret_quad}",
    flush=True,
)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()[:3]}{'...' if len(jax.devices()) > 3 else ''}", flush=True)

print("\nSolving retirement-only (ages 67..99, 33 ages)...", flush=True)
print("This includes JIT compile cost; first WORK or RETIRE age in each kernel", flush=True)
print("will be slow then subsequent ages reuse the compiled trace.", flush=True)
t0 = time.time()
C, S, B, diag = run_lifecycle_solver(
    model, pc, solver_config, verbose=1, solve_control=solve_control,
)
wall = time.time() - t0

print("\n" + "=" * 70, flush=True)
print("BENCHMARK RESULTS", flush=True)
print("=" * 70, flush=True)
print(f"  Wall time      : {wall:.1f}s = {wall/60:.2f} min", flush=True)
print(f"  Numba reference: 1341.9s = 22.36 min", flush=True)
ratio = wall / 1341.9
if ratio < 1:
    print(f"  JAX is {1/ratio:.2f}x FASTER than Numba", flush=True)
else:
    print(f"  JAX is {ratio:.2f}x SLOWER than Numba", flush=True)

solved = diag["n_ages_solved"]
print(f"\n  Ages solved    : {solved}/78 (expected 34: terminal + ages 67..99)", flush=True)
print(f"  Solve status   : {diag.get('solve_status', 'unknown')}", flush=True)
print(f"  Newton fails   : {diag.get('total_newton_failures', 'unknown')}", flush=True)

# Sanity on solved-age slices.
solved_mask = diag["solved_age_mask"]
C_solved = C[solved_mask]
S_solved = S[solved_mask]
B_solved = B[solved_mask]
print(
    f"  NaN check (solved ages only): C={int(np.isnan(C_solved).sum())}  "
    f"S={int(np.isnan(S_solved).sum())}  B={int(np.isnan(B_solved).sum())}",
    flush=True,
)
if S_solved.size:
    print(
        f"  alpha_s range  : [{float(S_solved.min()):.3f}, {float(S_solved.max()):.3f}]",
        flush=True,
    )
    print(
        f"  alpha_b range  : [{float(B_solved.min()):.3f}, {float(B_solved.max()):.3f}]",
        flush=True,
    )

# Spot-check terminal-age policy sanity (matches earlier verify_smoke pattern).
i_z = pc.n_z // 2; i_s = pc.N_state // 2; i_w = pc.n_w // 2
print(
    f"\n  Mid-cell terminal policy (z={i_z}, state={i_s}, w={i_w}, W={pc.wealth_grid[i_w]:.2f}):",
    flush=True,
)
print(
    f"    c={C[-1, i_z, i_s, i_w]:.4f}  alpha_s={S[-1, i_z, i_s, i_w]:.4f}  "
    f"alpha_b={B[-1, i_z, i_s, i_w]:.4f}",
    flush=True,
)
print("=" * 70, flush=True)


# =============================================================================
# Save bundle locally (and optionally upload to S3)
# =============================================================================

print(f"\n=== Saving bundle ===", flush=True)
print(f"  Local target: {BUNDLE_DIR}", flush=True)
if os.path.exists(BUNDLE_DIR):
    print(f"  (overwriting existing bundle)", flush=True)
    shutil.rmtree(BUNDLE_DIR)

run_config_snapshot = {
    "base_config": dict(BASE_CONFIG),
    "discretization_config": disc_config._asdict(),
    "solver_config": solver_config._asdict(),
    "solve_control": solve_control._asdict(),
    "predictability_ablation": {
        "system_label": "system_iv_full_var",
        "system_title": "System IV (full VAR baseline)",
    },
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(wall),
}

bundle_path = save_policy_bundle(
    BUNDLE_DIR,
    C, S, B,
    diagnostics=diag,
    run_config=run_config_snapshot,
    overwrite=True,
    wealth_grid=pc.wealth_grid,
)
print(f"  Saved local bundle: {bundle_path}", flush=True)

s3_bucket = os.environ.get("S3_BUCKET")
if s3_bucket:
    s3_uri = f"s3://{s3_bucket}/saved_runs/{BUNDLE_NAME}/"
    print(f"\n  S3_BUCKET set; uploading to {s3_uri}", flush=True)
    rc = subprocess.run(
        ["aws", "s3", "sync", BUNDLE_DIR, s3_uri,
         "--region", os.environ.get("AWS_REGION", "eu-north-1")],
        check=False,
    ).returncode
    if rc == 0:
        print(f"  S3 upload OK", flush=True)
    else:
        print(f"  S3 upload FAILED (rc={rc}) — bundle remains in {BUNDLE_DIR}", flush=True)
else:
    print(f"\n  (S3_BUCKET not set; skipping S3 upload)", flush=True)


# =============================================================================
# Post-solve diagnostic
# =============================================================================

print(f"\n=== Post-solve terminal-portfolio diagnostic ===", flush=True)
try:
    term_result = diagnose_terminal_portfolio_states(
        model, pc, C, S, B,
        solver_config=solver_config,
        max_fail_rows=12,
        print_report=True,
    )
    summary = term_result["summary"]
    print(
        f"\n  Summary: {summary['n_interior']}/{summary['n_states']} states interior, "
        f"{summary['n_fail']} fails. "
        f"Resid median={summary['resid_median']:.2e} max={summary['resid_max']:.2e}",
        flush=True,
    )
except Exception as exc:
    print(f"  diagnose_terminal_portfolio_states raised: {type(exc).__name__}: {exc}", flush=True)

print("\n=== ALL DONE ===", flush=True)
