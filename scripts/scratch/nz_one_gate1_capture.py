"""Gate 1 bit-identity capture: lifecycle smoke at canonical n_z=3.

Saves SHA-256 hashes of (C, S, B) plus their min/max/mean to a small JSON
file. Run before and after the discretization n_z=1 patch; compare for
bit-identity.

Clears the checkpoint cache to force a fresh solve so the hash reflects
the actual computation, not a cached blob.
"""
import hashlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, SolveControl
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver


CKPT_DIR = os.path.join("saved_runs", "checkpoints", "jax_cholesky_grid2x3x2x3_nz3_to_age62")
OUT_FILE = os.path.join("docs", "scans", "nz_one_gate1_capture.json")


def _hash_array(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main(label: str):
    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)

    tiny_disc = DiscretizationConfig(
        n_wealth=12, wealth_min=0.13, wealth_max=200.0,
        n_savings=12,
        state_grid_sizes=(2, 3, 2, 3),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_z=3,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=(2, 2),
        n_state_quad_nodes=(2, 3, 2, 3),
    )
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, tiny_disc, verbose=False)

    sc = CANONICAL_SOLVER._replace(max_iter=30)
    solve_control = SolveControl(youngest_age_to_solve=62)

    t0 = time.time()
    C, S, B, diag = run_lifecycle_solver(
        model, pc, sc, verbose=0, solve_control=solve_control,
    )
    wall = time.time() - t0
    solved_mask = diag["solved_age_mask"]

    payload = {
        "label": label,
        "wall_sec": wall,
        "shape_C": list(C.shape),
        "shape_S": list(S.shape),
        "shape_B": list(B.shape),
        "n_solved_ages": int(solved_mask.sum()),
        "C_min": float(C[solved_mask].min()),
        "C_max": float(C[solved_mask].max()),
        "C_mean": float(C[solved_mask].mean()),
        "S_min": float(S[solved_mask].min()),
        "S_max": float(S[solved_mask].max()),
        "S_mean": float(S[solved_mask].mean()),
        "B_min": float(B[solved_mask].min()),
        "B_max": float(B[solved_mask].max()),
        "B_mean": float(B[solved_mask].mean()),
        "C_sha256": _hash_array(C),
        "S_sha256": _hash_array(S),
        "B_sha256": _hash_array(B),
    }
    print(json.dumps(payload, indent=2))

    if os.path.exists(OUT_FILE):
        existing = json.loads(open(OUT_FILE).read())
    else:
        existing = {}
    existing[label] = payload
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {OUT_FILE} under key {label!r}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "default"
    main(label)
