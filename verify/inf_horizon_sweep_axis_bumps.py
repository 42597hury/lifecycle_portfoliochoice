"""Inf-horizon Sweep B/C combined: per-axis quadrature bump sensitivity.

Holds state grid at 5⁴ (the safe-enough cell from Sweep A's analysis)
and varies one quadrature axis at a time to isolate which axis bumps
materially change the policy.

Cells:
    1. (3,3,3,3) state / (3,3) ret  — cheap baseline
    2. (3,3,3,5) state / (3,3) ret  — y_1 (axis 3) state bump
    3. (5,3,3,3) state / (3,3) ret  — dp (axis 0) state bump
    4. (3,5,3,3) state / (3,3) ret  — spr (axis 1) state bump
    5. (3,3,3,3) state / (3,5) ret  — xb (axis 1) ret bump
    6. (3,3,3,3) state / (5,3) ret  — xr (axis 0) ret bump

All cells share shape (1, 625, 180) — direct element-wise pairwise
sup-norm comparison vs cell 1, no interpolation needed.

Per-bundle path:
    saved_runs/inf_horizon/system_iv_inf_axisbump_run<N>_<sq>_<rq>_calib1/

Wall projection on 1× GH200 (rough; first cell measures the truth):
    Cell 1: ~16 min
    Cells 2-6: ~27 min each = ~135 min
    Compile (~6 distinct kernel shapes): ~7 min
    Total: ~158 min ≈ 2.6 h ≈ ~$5
"""
import os
import shutil
import subprocess
import sys
sys.path.insert(0, ".")

import time
import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver
from lifecycle.policy_io import save_policy_bundle


# =============================================================================
# Sweep cells — state grid fixed at 5⁴; one quadrature axis bumped per run.
# =============================================================================

# (run_tag, state_quad, ret_quad, purpose)
SWEEP_CELLS = (
    ("run1_sq3333_rq33", (3, 3, 3, 3), (3, 3), "cheap baseline"),
    ("run2_sq3335_rq33", (3, 3, 3, 5), (3, 3), "y_1 state bump"),
    ("run3_sq5333_rq33", (5, 3, 3, 3), (3, 3), "dp state bump"),
    ("run4_sq3533_rq33", (3, 5, 3, 3), (3, 3), "spr state bump"),
    ("run5_sq3333_rq35", (3, 3, 3, 3), (3, 5), "xb ret bump"),
    ("run6_sq3333_rq53", (3, 3, 3, 3), (5, 3), "xr ret bump"),
)

STATE_GRID_SIZES = (5, 5, 5, 5)
STATE_LOBATTO_Z = None
RET_LOBATTO_Z = None

# Inf-horizon convergence parameters
IH_TOL = 1e-5
IH_MAX_ITER = 100
IH_DAMPING = 1.0
IH_PROGRESS_EVERY = 1


def make_disc(state_quad, ret_quad):
    return CANONICAL_DISC._replace(
        wealth_min=0.05,
        n_wealth=180,
        n_savings=180,
        state_grid_sizes=STATE_GRID_SIZES,
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_stds=2.25,
        n_z=1,
        n_eta_nodes=3,
        n_eps_nodes=4,
        n_state_quad_nodes=state_quad,
        state_lobatto_Z=STATE_LOBATTO_Z,
        n_ret_nodes_1d=ret_quad,
        ret_lobatto_Z=RET_LOBATTO_Z,
    )


SOLVER_CONFIG = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
    max_iter=100,
    max_iter_unconstrained=100,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=1,
)


# =============================================================================
# Sweep runner
# =============================================================================

print("=" * 70, flush=True)
print("INF-HORIZON SWEEP — per-axis quadrature bump sensitivity", flush=True)
print(f"  state_grid={STATE_GRID_SIZES}  tol={IH_TOL}  max_iter={IH_MAX_ITER}", flush=True)
print("=" * 70, flush=True)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()}", flush=True)

print("\nBuilding shared model + VAR config...", flush=True)
t0 = time.time()
var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
print(f"  Model build wall: {time.time() - t0:.1f}s", flush=True)


def run_one_cell(tag, state_quad, ret_quad, purpose):
    print("\n" + "=" * 70, flush=True)
    print(f"CELL {tag}: state_quad={state_quad} ret_quad={ret_quad}  ({purpose})", flush=True)
    print("=" * 70, flush=True)

    bundle_name = f"system_iv_inf_axisbump_{tag}_calib1"
    bundle_dir = os.path.join("saved_runs", "inf_horizon", bundle_name)

    disc_config = make_disc(state_quad, ret_quad)
    print(f"  Bundle name: {bundle_name}", flush=True)

    t_pc = time.time()
    pc = build_precompute(model, disc_config, verbose=False)
    print(
        f"  Precompute wall: {time.time() - t_pc:.1f}s  "
        f"N_state={pc.N_state}  n_z={pc.n_z}  n_w={pc.n_w}  "
        f"n_state_quad={pc.n_state_quad}  n_ret_quad={pc.n_ret_quad}",
        flush=True,
    )

    t_solve = time.time()
    C, S, B, diag = run_infinite_horizon_solver(
        model, pc,
        solver_config=SOLVER_CONFIG,
        max_iter=IH_MAX_ITER,
        tol=IH_TOL,
        damping=IH_DAMPING,
        progress_every=IH_PROGRESS_EVERY,
        show_progress=True,
        verbose=True,
    )
    wall = time.time() - t_solve

    print("\n--- RESULTS ---", flush=True)
    print(f"  Wall time      : {wall:.1f}s = {wall/60:.2f} min", flush=True)
    print(f"  Converged      : {diag.get('converged', '?')}", flush=True)
    print(f"  Iterations done: {diag.get('n_iter', '?')}", flush=True)
    fss = diag.get('final_stopping_supnorm', None)
    if fss is not None:
        print(f"  Final stop sup-norm: {fss:.3e}", flush=True)
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

    # Save bundle
    print(f"\n--- Saving {bundle_dir} ---", flush=True)
    if os.path.exists(bundle_dir):
        print(f"  (overwriting existing bundle)", flush=True)
        shutil.rmtree(bundle_dir)

    run_config_snapshot = {
        "base_config": dict(BASE_CONFIG),
        "discretization_config": disc_config._asdict(),
        "solver_config": SOLVER_CONFIG._asdict(),
        "inf_horizon_params": {
            "tol": IH_TOL,
            "max_iter": IH_MAX_ITER,
            "damping": IH_DAMPING,
        },
        "predictability_ablation": {
            "system_label": "system_iv_full_var",
            "system_title": "System IV (full VAR baseline)",
        },
        "sweep": {
            "name": "inf_horizon_axis_bump_sweep",
            "cell_tag": tag,
            "purpose": purpose,
            "state_grid_sizes": list(STATE_GRID_SIZES),
            "n_state_quad_nodes": list(state_quad),
            "n_ret_nodes_1d": list(ret_quad),
        },
        "bundle_name": bundle_name,
        "wall_time_seconds": float(wall),
        "solver_kind": "infinite_horizon",
    }

    save_policy_bundle(
        bundle_dir, C, S, B,
        diagnostics=diag,
        run_config=run_config_snapshot,
        overwrite=True,
    )
    print(f"  Saved local bundle: {bundle_dir}", flush=True)

    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_uri = f"s3://{s3_bucket}/saved_runs/inf_horizon/{bundle_name}/"
        print(f"  Uploading to {s3_uri}", flush=True)
        rc = subprocess.run(
            ["aws", "s3", "sync", bundle_dir, s3_uri,
             "--region", os.environ.get("AWS_REGION", "eu-north-1")],
            check=False,
        ).returncode
        if rc == 0:
            print(f"  S3 upload OK", flush=True)
        else:
            print(f"  S3 upload FAILED (rc={rc}) — bundle remains local", flush=True)

    return wall, diag


# Run sequentially: cell 1 first (cheap baseline), then bumps in order.
sweep_t0 = time.time()
walls = {}
for tag, state_quad, ret_quad, purpose in SWEEP_CELLS:
    cell_wall, _ = run_one_cell(tag, state_quad, ret_quad, purpose)
    walls[tag] = cell_wall

print("\n" + "=" * 70, flush=True)
print("AXIS-BUMP SWEEP COMPLETE", flush=True)
print("=" * 70, flush=True)
total = time.time() - sweep_t0
for tag, _, _, purpose in SWEEP_CELLS:
    print(f"  {tag}: {walls[tag]/60:.2f} min   ({purpose})", flush=True)
print(f"  TOTAL: {total/60:.2f} min", flush=True)
print("=" * 70, flush=True)
