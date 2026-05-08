"""Inf-horizon Sweep A: state-grid density sensitivity.

Sweeps state_grid_sizes ∈ {(3,3,3,3), (4,4,4,4), (5,5,5,5)} at fixed
quadrature floor (state_quad=(3,3,3,4) with K-bump on y_1; ret=(4,4)).
The bumped floor is conservative — ret-quad and y_1-axis state-quad both at
4 — so any state-grid divergence we observe is attributable to the grid,
not to under-resolved orthogonal quadratures.

Per-bundle path:
    saved_runs/inf_horizon/system_iv_inf_grid_g{N}_quad3334_ret44_calib1/

Each bundle contains policy_arrays.npz, metadata.json, diagnostics.pkl with
the post-fix per-savings Newton-iter histogram, full per-iter convergence
trajectory, and total_newton_failures (post the per-cell exit-code wiring).

Wall projection on 1× A100 SXM4 (rough; first cell measures the truth):
    g3 (3⁴ =  81 cells): ~17 min
    g4 (4⁴ = 256 cells): ~30 min
    g5 (5⁴ = 625 cells): ~48 min
    total: ~95 min ≈ 1.6 h ≈ ~$2

(g6 = 6⁴ dropped per user direction — adds ~90 min for marginal extra
information beyond the g4-vs-g5 verdict.)
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
# Sweep cells — state grid varies; everything else held to the bumped floor.
# =============================================================================

SWEEP_CELLS = (
    # tag,        state_grid_sizes
    ("g3", (3, 3, 3, 3)),
    ("g4", (4, 4, 4, 4)),
    ("g5", (5, 5, 5, 5)),
)

# Bumped floor: ret nodes (4,4); state quad (3,3,3,4) with K-bump on y_1.
N_STATE_QUAD = (3, 3, 3, 4)
N_RET_NODES = (4, 4)
STATE_LOBATTO_Z = None
RET_LOBATTO_Z = None

# Inf-horizon convergence parameters
IH_TOL = 1e-5
IH_MAX_ITER = 100
IH_DAMPING = 1.0
IH_PROGRESS_EVERY = 1


def make_disc(state_grid_sizes):
    return CANONICAL_DISC._replace(
        wealth_min=0.05,
        n_wealth=180,
        n_savings=180,
        state_grid_sizes=state_grid_sizes,
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_stds=2.25,
        n_z=1,
        n_eta_nodes=3,
        n_eps_nodes=4,
        n_state_quad_nodes=N_STATE_QUAD,
        state_lobatto_Z=STATE_LOBATTO_Z,
        n_ret_nodes_1d=N_RET_NODES,
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
print("INF-HORIZON SWEEP A — state-grid density at bumped quad floor", flush=True)
print(f"  state_quad={N_STATE_QUAD}  ret_quad={N_RET_NODES}  "
      f"tol={IH_TOL}  max_iter={IH_MAX_ITER}", flush=True)
print("=" * 70, flush=True)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()}", flush=True)

print("\nBuilding shared model + VAR config...", flush=True)
t0 = time.time()
var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
print(f"  Model build wall: {time.time() - t0:.1f}s", flush=True)


def run_one_cell(tag, state_grid_sizes):
    print("\n" + "=" * 70, flush=True)
    print(f"CELL {tag}: state_grid_sizes={state_grid_sizes}", flush=True)
    print("=" * 70, flush=True)

    bundle_name = (
        f"system_iv_inf_grid_{tag}_quad"
        f"{''.join(str(k) for k in N_STATE_QUAD)}_"
        f"ret{N_RET_NODES[0]}{N_RET_NODES[1]}_calib1"
    )
    bundle_dir = os.path.join("saved_runs", "inf_horizon", bundle_name)

    # Per-cell unique checkpoint path so cells with shape-equivalent disc
    # don't collide (lesson from the eta-eps sweep).
    solver_config = SOLVER_CONFIG._replace(
        wealth_dynamics_spec="ccv_log",
    )

    disc_config = make_disc(state_grid_sizes)
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
        solver_config=solver_config,
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
        "solver_config": solver_config._asdict(),
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
            "name": "inf_horizon_state_grid_sweep_A",
            "cell_tag": tag,
            "state_grid_sizes": list(state_grid_sizes),
            "n_state_quad_nodes": list(N_STATE_QUAD),
            "n_ret_nodes_1d": list(N_RET_NODES),
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
        wealth_grid=pc.wealth_grid,
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


# Run sequentially: smallest first, fail-fast on any blowup.
sweep_t0 = time.time()
walls = {}
for tag, state_grid_sizes in SWEEP_CELLS:
    cell_wall, _ = run_one_cell(tag, state_grid_sizes)
    walls[tag] = cell_wall

print("\n" + "=" * 70, flush=True)
print("SWEEP A COMPLETE", flush=True)
print("=" * 70, flush=True)
total = time.time() - sweep_t0
for tag, _ in SWEEP_CELLS:
    print(f"  {tag}: {walls[tag]/60:.2f} min", flush=True)
print(f"  TOTAL: {total/60:.2f} min", flush=True)
print("=" * 70, flush=True)
