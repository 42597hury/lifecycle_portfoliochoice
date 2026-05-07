"""System I (rtb-only) lifecycle full-solve sensitivity sweep over n_z.

Sweeps n_z in {10, 15, 30, 70} to characterize policy sensitivity to the
labour-income-state discretization in the simplest predictability system
(System I: state vector = (rtb,), interest rate only). Each run produces
a separate bundle in saved_runs/ablations/.

System I is the smallest of the four ablation systems:
  - state_names: ('rtb',) — 1-D state
  - n_state_quad: projects to (3,) — 45x cheaper per-cell FOC than System IV
  - VAR builder restricts to interest-rate-only predictability

Per-run wall on 2x H100 (estimated): 1-6 min depending on n_z.
Total sweep wall: ~15-25 min, ~$2-3.

Folder convention going forward:
  saved_runs/full/        — full-canonical System IV publication artifacts
  saved_runs/ablations/   — Systems I/II/III + parameter sweeps
  saved_runs/inf_horizon/ — stationary Bellman benchmarks

Outputs (one per n_z):
  saved_runs/ablations/system_i_grid7_nz<N>_calib1/
  s3://${S3_BUCKET}/saved_runs/ablations/system_i_grid7_nz<N>_calib1/
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
N_Z_SWEEP = (10, 15, 30, 70)
ABLATIONS_DIR = os.path.join("saved_runs", "ablations")

# Template config — only state_grid_sizes, state_n_stds, n_state_quad_nodes get
# projected by prepare_predictability_system. n_z is the variable we sweep.
# state_lobatto_Z left None: System I has only the rtb axis, no y_1 to K-bump.
template_disc_base = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(7, 7, 7, 7),                # rtb axis -> projects to (7,)
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_stds=2.25,
    n_eta_nodes=3,
    n_eps_nodes=4,
    n_state_quad_nodes=(3, 3, 3, 5),              # rtb axis -> projects to (3,)
    state_lobatto_Z=None,                         # off; not applicable to 1-D state
    n_ret_nodes_1d=(3, 3),
    ret_lobatto_Z=None,
)

solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
    max_iter=100,
    max_iter_unconstrained=100,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=1,                           # System I cells are tiny; no chunking needed
)

solve_control = SolveControl(
    youngest_age_to_solve=22,                     # full lifecycle (no slicing)
    checkpoint_every_n_ages=10,
    save_on_interrupt=True,
    return_partial_on_interrupt=True,
)

print("=" * 70, flush=True)
print("JAX SWEEP: System I (rtb-only) lifecycle full-solve, n_z sensitivity", flush=True)
print(f"n_z values: {N_Z_SWEEP}", flush=True)
print("=" * 70, flush=True)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()}", flush=True)
print(f"Output folder: {ABLATIONS_DIR}/", flush=True)
print(flush=True)

sweep_results = []
sweep_t0 = time.time()

for n_z in N_Z_SWEEP:
    bundle_name = f"system_i_grid7_nz{n_z}_calib1"
    bundle_dir = os.path.join(ABLATIONS_DIR, bundle_name)
    s3_uri = None
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_uri = f"s3://{s3_bucket}/saved_runs/ablations/{bundle_name}/"

    print("=" * 70, flush=True)
    print(f"  RUN: n_z = {n_z}", flush=True)
    print(f"  Bundle: {bundle_name}", flush=True)
    print(f"  S3:     {s3_uri or '(no S3 upload — S3_BUCKET not set)'}", flush=True)
    print("=" * 70, flush=True)

    # Project the template config onto System I and override n_z
    template_disc = template_disc_base._replace(n_z=n_z)
    meta = prepare_predictability_system(
        "I",
        csv_path=CSV_PATH,
        disc_config_template=template_disc,
    )
    var_config = meta["var_config"]
    disc_config = meta["disc_config"]
    state_names = meta["state_names"]

    print(f"\n  state_names      = {state_names}", flush=True)
    print(f"  state_grid_sizes = {disc_config.state_grid_sizes}", flush=True)
    print(f"  n_state_quad     = {disc_config.n_state_quad_nodes}", flush=True)
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

    print(f"\n  RESULT n_z={n_z}", flush=True)
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

    # Save bundle
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
        "sweep_dimension": "n_z",
        "n_z_value": n_z,
    }

    bundle_path = save_policy_bundle(
        bundle_dir,
        C, S, B,
        diagnostics=diag,
        run_config=run_config_snapshot,
        overwrite=True,
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
        "n_z": n_z,
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
print(f"  {'n_z':>5}  {'wall(s)':>8}  {'ages':>5}  {'p99 N':>6}  "
      f"{'alpha_s_range':>20}  {'alpha_b_range':>20}", flush=True)
for r in sweep_results:
    a_s = f"[{r['alpha_s_range'][0]:.2f},{r['alpha_s_range'][1]:.2f}]" if r["alpha_s_range"] else "?"
    a_b = f"[{r['alpha_b_range'][0]:.2f},{r['alpha_b_range'][1]:.2f}]" if r["alpha_b_range"] else "?"
    p99 = r['newton_p99'] if r['newton_p99'] is not None else "?"
    print(
        f"  {r['n_z']:>5}  {r['wall_sec']:>8.1f}  {r['ages_solved']:>5}  "
        f"{p99:>6}  {a_s:>20}  {a_b:>20}",
        flush=True,
    )
print("\n=== ALL DONE ===", flush=True)
