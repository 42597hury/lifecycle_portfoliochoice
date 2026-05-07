"""System I (rtb-only) lifecycle full-solve sensitivity sweep over (n_eta, n_eps).

Sweeps the working-age income-shock quadrature density at fixed n_z=30 to
characterize policy sensitivity to the income-quadrature resolution.
Convention: n_eps >= n_eta (transitory shocks have more support than
persistent shock innovations in this model).

Sweep:
  (n_eta, n_eps) = (3, 4) | eta * eps = 12 (baseline, matches earlier sweep)
  (n_eta, n_eps) = (4, 5) | eta * eps = 20
  (n_eta, n_eps) = (6, 6) | eta * eps = 36

Per-run wall on 2x H100 (estimated): 4-9 min depending on (eta, eps).
Total sweep wall: ~19 min, ~$2.60.

Outputs (one per (eta, eps) pair):
  saved_runs/ablations/system_i_grid7_nz30_eta<E>eps<P>_calib1/
  s3://${S3_BUCKET}/saved_runs/ablations/system_i_grid7_nz30_eta<E>eps<P>_calib1/
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

# (n_eta, n_eps) — eps >= eta always
ETA_EPS_SWEEP = (
    (3, 4),
    (4, 5),
    (6, 6),
)

ABLATIONS_DIR = os.path.join("saved_runs", "ablations")
N_Z_FIXED = 30                                    # high-res z grid

# Template config — only state_grid_sizes, state_n_stds, n_state_quad_nodes
# get projected by prepare_predictability_system. n_eta/n_eps are not
# projected (they're eta/eps-axis quadrature, not state-axis).
template_disc_base = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(7, 7, 7, 7),                # rtb axis -> projects to (7,)
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_stds=2.25,
    n_z=N_Z_FIXED,
    n_state_quad_nodes=(3, 3, 3, 5),              # rtb axis -> projects to (3,)
    state_lobatto_Z=None,
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

solve_control = SolveControl(
    youngest_age_to_solve=22,                     # full lifecycle
    checkpoint_every_n_ages=10,
    save_on_interrupt=True,
    return_partial_on_interrupt=True,
)

print("=" * 70, flush=True)
print("JAX SWEEP: System I (rtb-only) lifecycle full-solve, (n_eta, n_eps) sensitivity", flush=True)
print(f"Fixed n_z = {N_Z_FIXED}", flush=True)
print(f"(n_eta, n_eps) values: {ETA_EPS_SWEEP}", flush=True)
print("=" * 70, flush=True)

import jax
print(f"\nJAX devices: {len(jax.devices())} -> {jax.devices()}", flush=True)
print(f"Output folder: {ABLATIONS_DIR}/", flush=True)
print(flush=True)

sweep_results = []
sweep_t0 = time.time()

for n_eta, n_eps in ETA_EPS_SWEEP:
    bundle_name = f"system_i_grid7_nz{N_Z_FIXED}_eta{n_eta}eps{n_eps}_calib1"
    bundle_dir = os.path.join(ABLATIONS_DIR, bundle_name)
    s3_uri = None
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_uri = f"s3://{s3_bucket}/saved_runs/ablations/{bundle_name}/"

    print("=" * 70, flush=True)
    print(f"  RUN: (n_eta, n_eps) = ({n_eta}, {n_eps}) | product = {n_eta * n_eps}", flush=True)
    print(f"  Bundle: {bundle_name}", flush=True)
    print(f"  S3:     {s3_uri or '(no S3 upload — S3_BUCKET not set)'}", flush=True)
    print("=" * 70, flush=True)

    # Override n_eta and n_eps in the template; n_z stays fixed
    template_disc = template_disc_base._replace(n_eta_nodes=n_eta, n_eps_nodes=n_eps)
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
    print(f"  n_eta            = {disc_config.n_eta_nodes}", flush=True)
    print(f"  n_eps            = {disc_config.n_eps_nodes}", flush=True)

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

    print(f"\n  RESULT (n_eta, n_eps) = ({n_eta}, {n_eps})", flush=True)
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
        "sweep_dimension": "n_eta_n_eps",
        "n_eta_value": n_eta,
        "n_eps_value": n_eps,
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
        "n_eta": n_eta,
        "n_eps": n_eps,
        "product": n_eta * n_eps,
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
print(f"  {'eta':>3}  {'eps':>3}  {'prod':>4}  {'wall(s)':>8}  {'ages':>5}  "
      f"{'p99 N':>6}  {'alpha_s_range':>20}  {'alpha_b_range':>20}", flush=True)
for r in sweep_results:
    a_s = f"[{r['alpha_s_range'][0]:.2f},{r['alpha_s_range'][1]:.2f}]" if r["alpha_s_range"] else "?"
    a_b = f"[{r['alpha_b_range'][0]:.2f},{r['alpha_b_range'][1]:.2f}]" if r["alpha_b_range"] else "?"
    p99 = r['newton_p99'] if r['newton_p99'] is not None else "?"
    print(
        f"  {r['n_eta']:>3}  {r['n_eps']:>3}  {r['product']:>4}  "
        f"{r['wall_sec']:>8.1f}  {r['ages_solved']:>5}  {p99:>6}  "
        f"{a_s:>20}  {a_b:>20}",
        flush=True,
    )
print("\n=== ALL DONE ===", flush=True)
