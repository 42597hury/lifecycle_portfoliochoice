"""Quick debug: isolate which chunk count first deviates."""
import os
os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")

import time
import numpy as np
import jax  # noqa

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

assert len(jax.devices()) == 1

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

print(f"n_cells = {pc.n_z * pc.N_state}")

base_sc = CANONICAL_SOLVER._replace(max_iter=50, max_iter_unconstrained=50)


def solve_with(n_chunks):
    sc = base_sc._replace(cell_vmap_chunks=n_chunks)
    t0 = time.time()
    C, S, B, _ = run_lifecycle_solver(model, pc, sc, verbose=0)
    return C, S, B, time.time() - t0


print("Baseline (chunks=1)...", flush=True)
C1, S1, B1, t1 = solve_with(1)
print(f"  wall = {t1:.1f}s", flush=True)

# 64 / 8 = 8 (no padding)
print("\nChunks=8 (no padding, 8 cells/chunk)...", flush=True)
C, S, B, t = solve_with(8)
print(f"  wall = {t:.1f}s")
print(f"  max|deltaC|={np.max(np.abs(C1 - C)):.2e}  "
      f"max|deltaS|={np.max(np.abs(S1 - S)):.2e}  "
      f"max|deltaB|={np.max(np.abs(B1 - B)):.2e}")

# 64 / 7 = 9.14 -> chunk_size=10, pad_count=6 (padded)
print("\nChunks=7 (padding, 10 cells/chunk, 6 padded)...", flush=True)
C, S, B, t = solve_with(7)
print(f"  wall = {t:.1f}s")
deltaS = np.abs(S1 - S)
print(f"  max|deltaC|={np.max(np.abs(C1 - C)):.2e}  "
      f"max|deltaS|={np.max(deltaS):.2e}  "
      f"max|deltaB|={np.max(np.abs(B1 - B)):.2e}")

# Locate the failing cell.
i_age, i_z, i_state, i_w = np.unravel_index(np.argmax(deltaS), S1.shape)
print(f"  worst-S cell: age_idx={i_age} z_idx={i_z} i_s={i_state} i_w={i_w}")
print(f"    S1={S1[i_age, i_z, i_state, i_w]:.6e}  S7={S[i_age, i_z, i_state, i_w]:.6e}")
print(f"    delta = {S1[i_age, i_z, i_state, i_w] - S[i_age, i_z, i_state, i_w]:.6e}")

# Per-age max deltas
print("\nPer-age max deltaS:")
for t in range(S1.shape[0]):
    age_max = float(np.max(np.abs(S1[t] - S[t])))
    print(f"  t={t} (age {int(np.asarray(pc.ages)[t])}): max|deltaS|={age_max:.2e}")
