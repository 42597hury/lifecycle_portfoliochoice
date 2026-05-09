"""Diagnose Newton-iter histogram counting across ages (handoff Step 2-3).

Builds a tiny solve and prints, per age:
  - mean / max / unique values of per-cell n_iters
  - count of cells at max_iter
  - same for backtrack iters

If every age shows mean ~ max_iter and unique == {max_iter} the histogram
counting is bugged. If there's variance the counter is fine.
"""
import os

os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")

import sys
import time
import numpy as np
import jax

# path bootstrap
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

assert len(jax.devices()) == 1

small_disc = DiscretizationConfig(
    n_wealth=15, wealth_min=0.13, wealth_max=200.0,
    n_savings=15,
    state_grid_sizes=(2, 2, 2, 2),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=4,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(2, 2, 2, 2),
)

small_base = dict(BASE_CONFIG)
small_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(small_base, var_config, verbose=False)
pc = build_precompute(model, small_disc, verbose=False)

MAX_ITER = 50
sc = CANONICAL_SOLVER._replace(max_iter=MAX_ITER)

print(f"Probe config: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, "
      f"n_w={pc.n_w}, n_savings={pc.n_s}, max_iter={MAX_ITER}")
print()

# Monkey-patch _build_iter_histograms so we can grab the per-age arrays.
import lifecycle.solver as solver_mod
captured = {}
_orig_build = solver_mod._build_iter_histograms

def _capture_build(newton_per_age, backtrack_per_age, ages, mask):
    captured["ni"] = list(newton_per_age)
    captured["nb"] = list(backtrack_per_age)
    captured["ages"] = np.asarray(ages).copy()
    captured["mask"] = np.asarray(mask).copy()
    return _orig_build(newton_per_age, backtrack_per_age, ages, mask)

solver_mod._build_iter_histograms = _capture_build

t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1)
wall = time.time() - t0

print(f"\nSolve wall: {wall:.1f}s\n")

print("=" * 100)
print(f"Per-age Newton-iter distribution  (max_iter = {MAX_ITER})")
print("=" * 100)
print(f"{'idx':>3} {'age':>3}  {'shape':>10}  {'mean':>8} {'max':>5} {'p99':>6} "
      f"{'#@max':>6} {'#cells':>6}  unique (first 10)")
for t, arr in enumerate(captured["ni"]):
    if arr is None:
        continue
    a = np.asarray(arr).ravel()
    if a.size == 0:
        continue
    age = int(captured["ages"][t])
    n_at_max = int((a == MAX_ITER).sum())
    uniq = np.unique(a)
    uniq_show = ",".join(str(int(x)) for x in uniq[:10])
    if uniq.size > 10:
        uniq_show += f",...({uniq.size} total)"
    print(f"{t:>3} {age:>3}  {str(np.asarray(arr).shape):>10}  "
          f"{a.mean():>8.2f} {a.max():>5d} {int(np.percentile(a, 99)):>6d} "
          f"{n_at_max:>6d} {a.size:>6d}  {uniq_show}")

print()
print("=" * 100)
print(f"Per-age backtrack-iter distribution")
print("=" * 100)
print(f"{'idx':>3} {'age':>3}  {'shape':>10}  {'mean':>8} {'max':>5} {'p99':>6} "
      f"{'#cells':>6}  unique (first 10)")
for t, arr in enumerate(captured["nb"]):
    if arr is None:
        continue
    a = np.asarray(arr).ravel()
    if a.size == 0:
        continue
    age = int(captured["ages"][t])
    uniq = np.unique(a)
    uniq_show = ",".join(str(int(x)) for x in uniq[:10])
    if uniq.size > 10:
        uniq_show += f",...({uniq.size} total)"
    print(f"{t:>3} {age:>3}  {str(np.asarray(arr).shape):>10}  "
          f"{a.mean():>8.2f} {a.max():>5d} {int(np.percentile(a, 99)):>6d} "
          f"{a.size:>6d}  {uniq_show}")

print()
print("=" * 100)
print("Aggregate histogram from diag")
print("=" * 100)
ni_h = diag["newton_iter_histogram"]
nb_h = diag["backtrack_iter_histogram"]
print(f"newton_iter_histogram:    p50={ni_h['p50']}  p95={ni_h['p95']}  "
      f"p99={ni_h['p99']}  max={ni_h['max']}  n_cells={ni_h['n_cells']}")
print(f"backtrack_iter_histogram: p50={nb_h['p50']}  p95={nb_h['p95']}  "
      f"p99={nb_h['p99']}  max={nb_h['max']}  n_cells={nb_h['n_cells']}")
