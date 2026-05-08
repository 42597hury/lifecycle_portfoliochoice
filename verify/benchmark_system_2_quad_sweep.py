"""System 2 (spr, y_1) lifecycle full-solve quadrature sensitivity sweep.

Post real-yields pivot (2026-05-08): System 2 state = (spr, y_1); cape and
the legacy rtb axis are dropped. The template here is the 3-axis Full
template (cape, spr, y_1); prepare_predictability_system projects out
cape and keeps the (spr, y_1) entries.

Sweeps 4 (state_quad, ret_quad) configurations at fixed n_z=15 to characterize
policy sensitivity to quadrature density in the 2-D-state ablation system.

Sweep:
  Run 1: state_quad=(3,3), ret=(3,3)   - baseline (81 quad pts/cell)
  Run 2: state_quad=(4,4), ret=(3,3)   - uniform state refinement (144 pts)
  Run 3: state_quad=(3,3), ret=(4,4)   - ret refinement (144 pts)
  Run 4: state_quad=(3,5), ret=(3,3)   - y_1-axis refinement only (135 pts)

Other knobs fixed across all 4 runs:
  System 2  (state = spr + y_1)
  n_z = 15
  (n_eta, n_eps) = (3, 4)               - System 1 sweep showed this is adequate
  state_lobatto_Z = None                - plain GH; no Lobatto on any axis
  ret_lobatto_Z   = None
  delta_bequest = 0.0
  gather_precision = "f32"
  cell_vmap_chunks = 1
  youngest_age_to_solve = 22            - full lifecycle
  per-run unique checkpoint_path        - prevents the cross-run reload bug

Outputs:
  saved_runs/ablations/system_2_grid7x7_nz15_sq<NSQ>_rq<NRQ>_calib1/
  s3://${S3_BUCKET}/saved_runs/ablations/<bundle-name>/
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
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.policy_io import save_policy_bundle
from lifecycle.predictability_ablation import prepare_predictability_system

CSV_PATH = "data/var_dataset.csv"
N_Z_FIXED = 15

# (state_quad_template_3tuple, ret_quad_2tuple) — state quad axes are
# (cape, spr, y_1) at the template; project_predictability_disc_config
# will pick the (spr, y_1) entries for System 2.
QUAD_SWEEP = (
    # (label, state_quad_template, ret_quad)
    ("sq3x3_rq3x3", (3, 3, 3), (3, 3)),
    ("sq4x4_rq3x3", (4, 4, 4), (3, 3)),
    ("sq3x3_rq4x4", (3, 3, 3), (4, 4)),
    ("sq3x5_rq3x3", (3, 3, 5), (3, 3)),    # K-bump on y_1 axis only
)

ABLATIONS_DIR = os.path.join("saved_runs", "ablations")

# Template config — state axes are (cape, spr, y_1) at template; System 2
# projection keeps (spr, y_1) and drops cape.
template_disc_base = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(7, 7, 7),                   # System 2 projects to (7, 7)
    state_n_stds=(2.0, 2.25, 2.25),
    n_stds=2.25,
    n_z=N_Z_FIXED,
    n_eps_nodes=4,
    n_eta_nodes=3,
    state_lobatto_Z=None,                         # plain GH on all state axes
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

print("=" * 70, flush=True)
print("JAX SWEEP: System 2 (spr, y_1) lifecycle full-solve, quadrature sensitivity", flush=True)
print(f"Fixed n_z = {N_Z_FIXED}", flush=True)
print(f"4 quad configs: {[label for label, _, _ in QUAD_SWEEP]}", flush=True)
print("=" * 70, flush=True)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()}", flush=True)
print(f"Output folder: {ABLATIONS_DIR}/", flush=True)
print(flush=True)

sweep_results = []
sweep_t0 = time.time()

for label, state_quad_template, ret_quad in QUAD_SWEEP:
    bundle_name = f"system_2_grid7x7_nz{N_Z_FIXED}_{label}_calib1"
    bundle_dir = os.path.join(ABLATIONS_DIR, bundle_name)
    s3_uri = None
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_uri = f"s3://{s3_bucket}/saved_runs/ablations/{bundle_name}/"

    # Per-run unique checkpoint path so different (sq, rq) variants don't
    # collide (the bug from the eta-eps sweep).
    checkpoint_tag = (
        f"jax_cholesky_grid7x7_nz{N_Z_FIXED}_{label}_to_age22"
    )
    solve_control = SolveControl(
        youngest_age_to_solve=22,
        checkpoint_every_n_ages=10,
        save_on_interrupt=True,
        return_partial_on_interrupt=True,
        checkpoint_path=os.path.join("saved_runs", "checkpoints", checkpoint_tag),
    )

    print("=" * 70, flush=True)
    print(f"  RUN: {label}", flush=True)
    print(f"  state_quad (template) = {state_quad_template}, ret_quad = {ret_quad}", flush=True)
    print(f"  Bundle: {bundle_name}", flush=True)
    print(f"  S3:     {s3_uri or '(no S3 upload — S3_BUCKET not set)'}", flush=True)
    print("=" * 70, flush=True)

    # Override quadrature in the template
    template_disc = template_disc_base._replace(
        n_state_quad_nodes=state_quad_template,
        n_ret_nodes_1d=ret_quad,
    )
    meta = prepare_predictability_system(
        "2",
        csv_path=CSV_PATH,
        disc_config_template=template_disc,
    )
    var_config = meta["var_config"]
    disc_config = meta["disc_config"]
    state_names = meta["state_names"]

    print(f"\n  state_names      = {state_names}", flush=True)
    print(f"  state_grid_sizes = {disc_config.state_grid_sizes}", flush=True)
    print(f"  n_state_quad     = {disc_config.n_state_quad_nodes}", flush=True)
    print(f"  n_ret_nodes_1d   = {disc_config.n_ret_nodes_1d}", flush=True)
    print(f"  n_z              = {disc_config.n_z}", flush=True)

    print("\n  Building model + precompute...", flush=True)
    t0 = time.time()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc_config, verbose=False)
    setup_wall = time.time() - t0
    print(
        f"  Setup wall: {setup_wall:.1f}s  "
        f"(N_state={pc.N_state}, n_z={pc.n_z}, n_w={pc.n_w}, n_s={pc.n_s})",
        flush=True,
    )

    print("\n  Solving full lifecycle (ages 22..99)...", flush=True)
    t0 = time.time()
    C, S, B, diag = run_lifecycle_solver(
        model, pc, solver_config, verbose=1, solve_control=solve_control,
    )
    solve_wall = time.time() - t0

    sm = np.asarray(diag["solved_age_mask"], dtype=bool)
    ages_solved = int(sm.sum())
    nan_c = int(np.isnan(C[sm]).sum())
    nf = diag.get("total_newton_failures", "?")

    nih = diag.get("newton_iter_histogram", {}) or {}
    bth = diag.get("backtrack_iter_histogram", {}) or {}

    print(f"\n  RESULT {label}", flush=True)
    print(f"    Solve wall    : {solve_wall:.1f}s = {solve_wall/60:.2f} min", flush=True)
    print(f"    Ages solved   : {ages_solved}", flush=True)
    print(f"    NaN check     : C={nan_c}", flush=True)
    print(f"    Newton fails  : {nf}", flush=True)
    if "p99" in nih:
        print(
            f"    Newton iters  : p50={nih.get('p50','?')}  p95={nih.get('p95','?')}  "
            f"p99={nih.get('p99','?')}  max={nih.get('max','?')}",
            flush=True,
        )
    if "p99" in bth:
        print(
            f"    Backtrack     : p50={bth.get('p50','?')}  p95={bth.get('p95','?')}  "
            f"p99={bth.get('p99','?')}  max={bth.get('max','?')}",
            flush=True,
        )
    if S[sm].size:
        print(
            f"    alpha_s range : [{float(S[sm].min()):.3f}, {float(S[sm].max()):.3f}]",
            flush=True,
        )
        print(
            f"    alpha_b range : [{float(B[sm].min()):.3f}, {float(B[sm].max()):.3f}]",
            flush=True,
        )

    if os.path.exists(bundle_dir):
        print(f"  (overwriting existing {bundle_dir})", flush=True)
        shutil.rmtree(bundle_dir)
    os.makedirs(ABLATIONS_DIR, exist_ok=True)

    run_config_snapshot = {
        "base_config": dict(BASE_CONFIG),
        "discretization_config": disc_config._asdict(),
        "solver_config": solver_config._asdict(),
        "solve_control": solve_control._asdict(),
        "predictability_ablation": {
            "system_code": meta["system_code"],
            "system_label": meta["system_label"],
            "system_title": meta["system_title"],
            "system_description": meta.get("system_description", ""),
            "state_names": list(state_names),
        },
        "bundle_name": bundle_name,
        "wall_time_seconds": float(solve_wall),
        "solver_kind": "lifecycle_full",
        "sweep_dimension": "state_ret_quad",
        "label": label,
        "state_quad_template": list(state_quad_template),
        "ret_quad": list(ret_quad),
    }

    bundle_path = save_policy_bundle(
        bundle_dir,
        C, S, B,
        diagnostics=diag,
        run_config=run_config_snapshot,
        overwrite=True,
        wealth_grid=pc.wealth_grid,
    )
    print(f"\n  Saved local bundle: {bundle_path}", flush=True)

    if s3_uri:
        rc = subprocess.run(
            ["aws", "s3", "sync", bundle_dir, s3_uri,
             "--region", os.environ.get("AWS_REGION", "eu-north-1")],
            check=False,
        ).returncode
        print(
            f"  S3 upload {'OK' if rc == 0 else 'FAILED rc=' + str(rc)}",
            flush=True,
        )

    sweep_results.append({
        "label": label,
        "state_quad": disc_config.n_state_quad_nodes,
        "ret_quad": disc_config.n_ret_nodes_1d,
        "wall_sec": solve_wall,
        "ages_solved": ages_solved,
        "newton_p99": nih.get("p99"),
        "alpha_s_range": (float(S[sm].min()), float(S[sm].max())) if S[sm].size else None,
        "alpha_b_range": (float(B[sm].min()), float(B[sm].max())) if S[sm].size else None,
        "bundle_path": str(bundle_path),
    })

    print(flush=True)

sweep_total_wall = time.time() - sweep_t0

print("=" * 70, flush=True)
print("SWEEP COMPLETE", flush=True)
print("=" * 70, flush=True)
print(f"  Total wall: {sweep_total_wall:.1f}s = {sweep_total_wall/60:.2f} min", flush=True)
print(flush=True)
print(f"  {'label':<14}  {'state_q':>8}  {'ret_q':>6}  {'wall(s)':>8}  {'ages':>5}  "
      f"{'p99 N':>6}  {'alpha_s_range':>20}  {'alpha_b_range':>20}", flush=True)
for r in sweep_results:
    a_s = f"[{r['alpha_s_range'][0]:.2f},{r['alpha_s_range'][1]:.2f}]" if r["alpha_s_range"] else "?"
    a_b = f"[{r['alpha_b_range'][0]:.2f},{r['alpha_b_range'][1]:.2f}]" if r["alpha_b_range"] else "?"
    p99 = r['newton_p99'] if r['newton_p99'] is not None else "?"
    print(
        f"  {r['label']:<14}  {str(r['state_quad']):>8}  {str(r['ret_quad']):>6}  "
        f"{r['wall_sec']:>8.1f}  {r['ages_solved']:>5}  {p99:>6}  "
        f"{a_s:>20}  {a_b:>20}",
        flush=True,
    )
print("\n=== ALL DONE ===", flush=True)
