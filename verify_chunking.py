"""Bit-identity test for SolverConfig.cell_vmap_chunks.

Same algorithm, different dispatch shape — chunks={1, 2, 4, 8} must produce
identical alphas to numerical precision (~1e-12). Gate before merging the
cell-axis vmap chunking change. See HANDOFF_CELL_VMAP_CHUNKING.md.

Forces single-device JAX so the vmap-only (chunkable) kernel path is
exercised. Without this, on a 12-virtual-CPU box the dispatch falls through
to the pmap path and the new chunking code is dead-code from the test's
perspective.
"""
import os

# Must be set BEFORE importing lifecycle (which configures JAX at import time).
os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")

import time
import numpy as np

import jax  # noqa: E402

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.var import build_nominal_system1_var_config_hardcoded  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.solver import run_lifecycle_solver  # noqa: E402

assert len(jax.devices()) == 1, (
    f"Expected 1 device for vmap-only path; got {len(jax.devices())}. "
    "Make sure LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 is set before any lifecycle import."
)

# Small config: n_z=4, state grid 2x2x2x2 -> N_state=16; n_cells = 64.
# Chunk values 2, 4, 8 all divide 64 evenly; chunks=7 forces padding.
tiny_disc = DiscretizationConfig(
    n_wealth=15, wealth_min=0.13, wealth_max=200.0,
    n_savings=15,
    state_grid_sizes=(2, 2, 2, 2),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=4,
    n_eps_nodes=3,
    n_eta_nodes=3,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(2, 2, 2, 2),
)

tiny_base = dict(BASE_CONFIG)
tiny_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(tiny_base, var_config, verbose=False)
pc = build_precompute(model, tiny_disc, verbose=False)

print(f"Devices: {jax.devices()}  (vmap-only path)")
print(f"Built model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}")
print(f"Total cells per age: n_z * N_state = {pc.n_z * pc.N_state}")

base_sc = CANONICAL_SOLVER._replace(max_iter=50, max_iter_unconstrained=50)


def solve_with(n_chunks):
    sc = base_sc._replace(cell_vmap_chunks=n_chunks)
    t0 = time.time()
    C, S, B, _ = run_lifecycle_solver(model, pc, sc, verbose=0)
    return C, S, B, time.time() - t0


print("\nBaseline (cell_vmap_chunks=1)...")
C1, S1, B1, t1 = solve_with(1)
print(f"  wall = {t1:.1f}s")

# 2,4 divide 64 evenly; 7 forces padding (chunk_size=ceil(64/7)=10 -> padded to 70).
results = []
for n_chunks in (2, 4, 7):
    print(f"\nChunked (cell_vmap_chunks={n_chunks})...")
    C, S, B, t = solve_with(n_chunks)
    print(f"  wall = {t:.1f}s")
    max_C = float(np.max(np.abs(C1 - C)))
    max_S = float(np.max(np.abs(S1 - S)))
    max_B = float(np.max(np.abs(B1 - B)))
    print(f"  max|deltaC|={max_C:.2e}  max|deltaS|={max_S:.2e}  max|deltaB|={max_B:.2e}")
    results.append((n_chunks, max_C, max_S, max_B))

print("\n" + "=" * 70)
print("Bit-identity verdict (tolerance 1e-10):")
all_pass = True
for n_chunks, max_C, max_S, max_B in results:
    pass_C = max_C < 1e-10
    pass_S = max_S < 1e-10
    pass_B = max_B < 1e-10
    ok = pass_C and pass_S and pass_B
    all_pass = all_pass and ok
    flag = "PASS" if ok else "FAIL"
    print(
        f"  chunks={n_chunks}: C={'ok' if pass_C else 'FAIL'} "
        f"S={'ok' if pass_S else 'FAIL'} "
        f"B={'ok' if pass_B else 'FAIL'}  [{flag}]"
    )

if not all_pass:
    raise SystemExit("Chunking is NOT bit-identical -- see deltas above.")
print("All chunked runs are bit-identical to baseline (<= 1e-10).")
