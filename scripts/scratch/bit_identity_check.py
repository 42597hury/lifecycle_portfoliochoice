"""Save C/S/B policies from the current solver run; another run loads and diffs."""
import os
os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")

import sys, time, numpy as np, jax
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

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

sc = CANONICAL_SOLVER._replace(max_iter=50)

t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=0)
print(f"Wall: {time.time()-t0:.1f}s")

label = sys.argv[1] if len(sys.argv) > 1 else "current"
out_path = os.path.join(HERE, f"_policies_{label}.npz")
np.savez(out_path, C=C, S=S, B=B)
print(f"Saved {out_path}")
print(f"  C: shape={C.shape} sum={C.sum():.10e}")
print(f"  S: shape={S.shape} sum={S.sum():.10e}")
print(f"  B: shape={B.shape} sum={B.sum():.10e}")
