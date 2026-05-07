"""Bit-identity test for SolverConfig.cell_vmap_chunks on the pmap path.

Forces 4 virtual CPU devices and solves a tiny lifecycle window that exercises
terminal, retirement, work->retirement boundary, and normal working kernels.
The math should be identical for all chunk counts; only dispatch shape changes.
"""
import os

# Must be set before importing lifecycle/JAX. This verifier is specifically
# for the pmap path, so force the environment rather than inheriting a caller's
# single-device/debug settings.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["LIFECYCLE_DISABLE_VIRTUAL_CPUS"] = "0"

import time
import numpy as np

import jax  # noqa: E402

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.var import build_nominal_system1_var_config_hardcoded  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.solver import run_lifecycle_solver  # noqa: E402


assert len(jax.devices()) == 4, (
    f"Expected 4 devices for pmap path; got {len(jax.devices())}. "
    "Run in a fresh process so XLA_FLAGS takes effect before JAX import."
)

tiny_disc = DiscretizationConfig(
    n_wealth=6, wealth_min=0.13, wealth_max=50.0,
    n_savings=6,
    state_grid_sizes=(2, 2, 2, 2),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=3,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(1, 1, 1, 1),
)

tiny_base = dict(BASE_CONFIG)
tiny_base.update(start_age=62, retire_age=64, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(tiny_base, var_config, verbose=False)
pc = build_precompute(model, tiny_disc, verbose=False)

print(f"Devices: {jax.devices()}  (pmap path)")
print(
    f"Built model: n_age={pc.n_age}, n_z={pc.n_z}, "
    f"N_state={pc.N_state}, n_w={pc.n_w}"
)
print("Kernel coverage: terminal, retirement, boundary, working")

base_sc = CANONICAL_SOLVER._replace(
    max_iter=5,
    max_iter_unconstrained=5,
    max_backtrack_iter=2,
    gather_precision="f64",
)


def solve_with(n_chunks):
    sc = base_sc._replace(cell_vmap_chunks=n_chunks)
    t0 = time.time()
    C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=0)
    wall = time.time() - t0
    solved = np.asarray(diag["solved_age_mask"], dtype=bool)
    return C[solved], S[solved], B[solved], wall


print("\nBaseline (cell_vmap_chunks=1)...")
C1, S1, B1, t1 = solve_with(1)
print(f"  wall = {t1:.1f}s")

results = []
for n_chunks in (2, 4, 7):
    print(f"\nChunked (cell_vmap_chunks={n_chunks})...")
    C, S, B, t = solve_with(n_chunks)
    print(f"  wall = {t:.1f}s")
    eq_C = np.array_equal(C1, C, equal_nan=True)
    eq_S = np.array_equal(S1, S, equal_nan=True)
    eq_B = np.array_equal(B1, B, equal_nan=True)
    max_C = float(np.nanmax(np.abs(C1 - C)))
    max_S = float(np.nanmax(np.abs(S1 - S)))
    max_B = float(np.nanmax(np.abs(B1 - B)))
    print(
        f"  equal: C={eq_C} S={eq_S} B={eq_B}  "
        f"max|delta|=({max_C:.2e}, {max_S:.2e}, {max_B:.2e})"
    )
    results.append((n_chunks, eq_C, eq_S, eq_B))

print("\n" + "=" * 70)
print("Pmap chunking bit-identity verdict:")
all_pass = True
for n_chunks, eq_C, eq_S, eq_B in results:
    ok = eq_C and eq_S and eq_B
    all_pass = all_pass and ok
    print(
        f"  chunks={n_chunks}: C={'ok' if eq_C else 'FAIL'} "
        f"S={'ok' if eq_S else 'FAIL'} "
        f"B={'ok' if eq_B else 'FAIL'}  [{'PASS' if ok else 'FAIL'}]"
    )

if not all_pass:
    raise SystemExit("Pmap chunking is NOT bit-identical -- see deltas above.")
print("All pmap chunked runs are bit-identical to baseline.")
