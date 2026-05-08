"""8x8x8 infinite-horizon (stationary Bellman) benchmark on the Full real-yields system.

Solves the no-income, no-mortality, no-bequest stationary Merton-with-VAR
benchmark by fixed-point iteration of the JAX retirement kernel. With pension=0
and psi=1 across all z, all income states produce identical policies — so n_z
is set to 1 (the minimum).

Post real-yields pivot (2026-05-08): state vector is 3-axis (cape, spr, y_1).

Config rationale:
    state_grid_sizes  = (8, 8, 8)                # N_state = 512
    state_n_stds      = (2.0, 2.25, 2.25)        # cape, spr, y_1
    n_stds            = 2.25                     # z-axis (no income, but precompute needs it)
    n_z               = 1                        # z is inert under pension=0/psi=1
    n_wealth          = 180,  wealth_min = 0.05
    n_savings         = 180
    n_state_quad_nodes= (3, 3, 5)                # K-bump on y_1 axis (last)
    state_lobatto_Z   = (None, None, 2.93)       # Lobatto on bumped y_1 axis
    n_ret_nodes_1d    = (3, 3)                   # 9 nodes
    ret_lobatto_Z     = None
    n_eps_nodes       = 4, n_eta_nodes = 3       # not exercised in inf-horizon
    max_iter (Newton) = 100                      # per fixed-point iter
    delta_bequest     = 0.0                      # also forced to 0 by inf-horizon (b_bar=0)
    gather_precision  = "f32"
    cell_vmap_chunks  = 1                        # n_z=1 keeps memory comfortable

Outputs:
    ./saved_runs/<bundle-name>/                — local bundle (npz + metadata)
    s3://${S3_BUCKET}/saved_runs/<bundle-name>/ — if S3_BUCKET env var is set
"""
import os
import shutil
import subprocess
import sys
sys.path.insert(0, ".")

import time
import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.model import SolverConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver
from lifecycle.policy_io import save_policy_bundle

BUNDLE_NAME = "full_system_inf_horizon_grid8x8x8_nz1_y1lob_calib1"
BUNDLE_DIR = os.path.join("saved_runs", "inf_horizon", BUNDLE_NAME)

disc_config = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(8, 8, 8),
    state_n_stds=(2.0, 2.25, 2.25),
    n_stds=2.25,
    n_z=1,
    n_eta_nodes=3,
    n_state_quad_nodes=(3, 3, 5),
    state_lobatto_Z=(None, None, 2.93),
    n_ret_nodes_1d=(3, 3),
    ret_lobatto_Z=None,
)

solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
    max_iter=100,
    max_iter_unconstrained=100,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=1,
)

# Inf-horizon convergence parameters
IH_TOL = 1e-5             # stopping criterion on policy sup-norm
IH_MAX_ITER = 50          # safety cap on fixed-point iterations
IH_DAMPING = 1.0          # full Bellman update; lower if divergence appears
IH_PROGRESS_EVERY = 1     # log every iter

print("=" * 70, flush=True)
print("JAX BENCHMARK: 8x8x8 infinite-horizon stationary Bellman", flush=True)
print("Full real-yields state (cape, spr, y_1); no income, no mortality, no bequest.", flush=True)
print("=" * 70, flush=True)

print("\nBuilding model + precompute...", flush=True)
t0 = time.time()
var_config = build_real_full_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc_config, verbose=True)
print(f"Setup wall: {time.time() - t0:.1f}s", flush=True)
print(
    f"  N_state={pc.N_state}, n_z={pc.n_z}, n_w={pc.n_w}, n_s={pc.n_s}, "
    f"n_state_quad={pc.n_state_quad}, n_ret_quad={pc.n_ret_quad}",
    flush=True,
)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()}", flush=True)

print(
    f"\nRunning fixed-point loop "
    f"(tol={IH_TOL}, max_iter={IH_MAX_ITER}, damping={IH_DAMPING})...",
    flush=True,
)
t0 = time.time()
C, S, B, diag = run_infinite_horizon_solver(
    model, pc,
    solver_config=solver_config,
    max_iter=IH_MAX_ITER,
    tol=IH_TOL,
    damping=IH_DAMPING,
    progress_every=IH_PROGRESS_EVERY,
    show_progress=True,
    verbose=True,
)
wall = time.time() - t0

print("\n" + "=" * 70, flush=True)
print("INF-HORIZON RESULTS", flush=True)
print("=" * 70, flush=True)
print(f"  Wall time      : {wall:.1f}s = {wall/60:.2f} min", flush=True)
print(f"  Converged      : {diag.get('converged', '?')}", flush=True)
print(f"  Iterations done: {diag.get('n_iter', '?')}", flush=True)
print(f"  Final stop sup-norm: {diag.get('final_stopping_supnorm', '?'):.3e}", flush=True)
print(f"  Newton fails   : {diag.get('total_newton_failures', '?')}", flush=True)

nih = diag.get("newton_iter_histogram")
if nih is not None:
    print(
        f"  Newton iters   : p50={nih['p50']:.0f}  p95={nih['p95']:.0f}  "
        f"p99={nih['p99']:.0f}  max={nih['max']}",
        flush=True,
    )
bth = diag.get("backtrack_iter_histogram")
if bth is not None:
    print(
        f"  Backtrack iters: p50={bth['p50']:.1f}  p95={bth['p95']:.1f}  "
        f"p99={bth['p99']:.1f}  max={bth['max']}",
        flush=True,
    )

print(
    f"  NaN check: C={int(np.isnan(C).sum())}  S={int(np.isnan(S).sum())}  "
    f"B={int(np.isnan(B).sum())}",
    flush=True,
)
print(f"  alpha_s range  : [{float(S.min()):.3f}, {float(S.max()):.3f}]", flush=True)
print(f"  alpha_b range  : [{float(B.min()):.3f}, {float(B.max()):.3f}]", flush=True)
print(f"  C shape: {C.shape}", flush=True)
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
    "inf_horizon_params": {
        "tol": IH_TOL,
        "max_iter": IH_MAX_ITER,
        "damping": IH_DAMPING,
    },
    "predictability_ablation": {
        "system_code": "full",
        "system_label": "full_system_real",
        "system_title": "Full System (real)",
        "state_names": ["cape", "spr", "y_1"],
    },
    "bundle_name": BUNDLE_NAME,
    "wall_time_seconds": float(wall),
    "solver_kind": "infinite_horizon",
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
    s3_uri = f"s3://{s3_bucket}/saved_runs/inf_horizon/{BUNDLE_NAME}/"
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

print("\n=== ALL DONE ===", flush=True)
