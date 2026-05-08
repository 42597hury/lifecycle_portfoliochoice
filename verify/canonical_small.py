"""Phase 3 canonical-small: 78-age lifecycle on (3,3,3) state grid + small wealth/savings/z.

Post real-yields pivot (2026-05-08): 3-axis state vector (cape, spr, y_1).
"""
import time
import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.simulation import simulate_lifecycle

disc = DiscretizationConfig(
    n_wealth=40, wealth_min=0.13, wealth_max=200.0,
    n_savings=40,
    state_grid_sizes=(3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=5,
    n_eps_nodes=3,
    n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 3),
)

print("Building model and precompute...", flush=True)
var_config = build_real_full_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=True)

# max_iter=100 (canonical's 8000 was overkill; chunk-2 verified same policy)
sc = CANONICAL_SOLVER._replace(max_iter=100, max_iter_unconstrained=100)

print("\nSolving full lifecycle (~7-12 min expected)...", flush=True)
t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1)
solve_time = time.time() - t0
print(f"\nSolve wall time: {solve_time / 60:.2f} min", flush=True)

# Save policies for later comparison.
np.savez(
    "verify_canonical_small_policies.npz",
    C=C, S=S, B=B,
    wealth_grid=np.asarray(pc.wealth_grid),
    z_grid=np.asarray(pc.z_grid),
    ages=np.asarray(pc.ages),
)
print("Saved policies to verify_canonical_small_policies.npz", flush=True)

print("\nSimulating 5000 households...", flush=True)
t0 = time.time()
sim = simulate_lifecycle(C, S, B, pc, model, n_simulations=5000, seed=42, verbose=True)
sim_time = time.time() - t0
print(f"Simulation wall time: {sim_time:.1f}s", flush=True)

# Save sim panel.
np.savez(
    "verify_canonical_small_sim.npz",
    **{k: v for k, v in sim.items() if isinstance(v, np.ndarray)},
)
print("Saved simulation to verify_canonical_small_sim.npz", flush=True)

# Top-line sanity numbers.
print("\n--- Sanity ---", flush=True)
print(f"NaN: C={int(np.isnan(C).sum())}  S={int(np.isnan(S).sum())}  B={int(np.isnan(B).sum())}", flush=True)
print(f"alpha_s range: [{S.min():.3f}, {S.max():.3f}]", flush=True)
print(f"alpha_b range: [{B.min():.3f}, {B.max():.3f}]", flush=True)
print(f"Lifecycle wealth (alive mean): "
      f"age 22={sim['x'][sim['alive'][:,0],0].mean():.2f}  "
      f"age 44={sim['x'][sim['alive'][:,22],22].mean():.2f}  "
      f"age 67={sim['x'][sim['alive'][:,45],45].mean():.2f}  "
      f"age 87={sim['x'][sim['alive'][:,65],65].mean():.2f}", flush=True)
print(f"survival to terminal: {sim['alive'][:,-1].mean():.1%}", flush=True)
print(f"median death age: {int(np.median(sim['death_age']))}", flush=True)
