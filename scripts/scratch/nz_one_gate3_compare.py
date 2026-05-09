"""Gate 3: inf-horizon at n_z=1 vs n_z=2 must match within 1e-10 across
C, S, B at the (only-shared) z=0 slice.

z is mathematically inert under inf-horizon (pension=0, psi=1) so both
runs are computing the same fixed-point map. Any difference would point
to a model-semantic divergence rather than mere quadrature scheduling drift.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver


def _solve(n_z: int):
    disc = CANONICAL_DISC._replace(
        n_wealth=10,
        wealth_min=0.05,
        n_savings=10,
        state_grid_sizes=(2, 2, 2, 2),
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_stds=2.25,
        n_z=n_z,
        n_eta_nodes=3,
        n_state_quad_nodes=(2, 2, 2, 2),
        state_lobatto_Z=None,
        n_ret_nodes_1d=(2, 2),
        ret_lobatto_Z=None,
    )
    sc = CANONICAL_SOLVER._replace(
        wealth_dynamics_spec="ccv_log",
        max_iter=20,
        delta_bequest=0.0,
        gather_precision="f32",
        cell_vmap_chunks=1,
    )
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    print(f"\n=== n_z={n_z}  pc.z_grid={pc.z_grid}  pc.dz={pc.dz}  pc.init_z_probs={pc.init_z_probs}")

    C, S, B, diag = run_infinite_horizon_solver(
        model, pc,
        solver_config=sc,
        max_iter=15,
        tol=1e-8,
        damping=1.0,
        progress_every=5,
        show_progress=True,
        verbose=True,
    )
    print(f"   converged={diag['converged']}  iters={diag['n_iter']}  "
          f"final_stop={diag['final_stopping_supnorm']:.3e}")
    return C, S, B


def main():
    C1, S1, B1 = _solve(n_z=1)
    C2, S2, B2 = _solve(n_z=2)

    # n_z=1 has shape (1, N_state, n_w); n_z=2 has shape (2, N_state, n_w).
    # Compare the n_z=1 only-slice against each z-slice of n_z=2.
    print("\n--- comparing n_z=1[0] vs n_z=2[*] ---")
    for k in range(C2.shape[0]):
        dC = float(np.max(np.abs(C1[0] - C2[k])))
        dS = float(np.max(np.abs(S1[0] - S2[k])))
        dB = float(np.max(np.abs(B1[0] - B2[k])))
        print(f"  z={k}:  max|dC|={dC:.3e}  max|dS|={dS:.3e}  max|dB|={dB:.3e}")
        if k == 0:
            dC0, dS0, dB0 = dC, dS, dB

    print("\n--- z-slice invariance within n_z=2 ---")
    print(f"  max|C2[0]-C2[1]| = {float(np.max(np.abs(C2[0] - C2[1]))):.3e}")
    print(f"  max|S2[0]-S2[1]| = {float(np.max(np.abs(S2[0] - S2[1]))):.3e}")
    print(f"  max|B2[0]-B2[1]| = {float(np.max(np.abs(B2[0] - B2[1]))):.3e}")

    # Pass criterion from the handoff: < 1e-10 across C, S, B for the
    # n_z=1 vs n_z=2 [z=0] comparison.
    TOL = 1e-10
    if max(dC0, dS0, dB0) < TOL:
        print(f"\nGate 3: PASS (max diff {max(dC0, dS0, dB0):.3e} < {TOL:.0e})")
    else:
        print(f"\nGate 3: REPORT (max diff {max(dC0, dS0, dB0):.3e} >= {TOL:.0e})")
        print("  Tightest constraint in handoff is < 1e-10. If diff is 1e-10..1e-7 the")
        print("  cause is likely JAX scheduling drift across shape variation; if larger,")
        print("  surface to user — possible model-semantic difference.")


if __name__ == "__main__":
    main()
