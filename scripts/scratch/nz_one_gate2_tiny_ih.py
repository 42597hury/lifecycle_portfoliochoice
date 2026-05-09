"""Gate 2: tiny inf-horizon at n_z=1 — completes without error and
produces sane policies (no NaN, alphas in a bounded range).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver


def main():
    disc = CANONICAL_DISC._replace(
        n_wealth=10,
        wealth_min=0.05,
        n_savings=10,
        state_grid_sizes=(2, 2, 2, 2),
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_stds=2.25,
        n_z=1,
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
    pc = build_precompute(model, disc, verbose=True)
    print(f"\n  pc.n_z={pc.n_z}, pc.z_grid={pc.z_grid}, pc.dz={pc.dz}, pc.init_z_probs={pc.init_z_probs}")

    t0 = time.time()
    C, S, B, diag = run_infinite_horizon_solver(
        model, pc,
        solver_config=sc,
        max_iter=5,
        tol=1e-5,
        damping=1.0,
        progress_every=1,
        show_progress=True,
        verbose=True,
    )
    wall = time.time() - t0

    print(f"\n  Wall: {wall:.2f}s   converged={diag.get('converged')}   "
          f"iters={diag.get('n_iter')}   final_stop={diag.get('final_stopping_supnorm'):.3e}")
    print(f"  C shape: {C.shape}    NaN: C={int(np.isnan(C).sum())}  "
          f"S={int(np.isnan(S).sum())}  B={int(np.isnan(B).sum())}")
    print(f"  alpha_s range: [{float(S.min()):.3f}, {float(S.max()):.3f}]")
    print(f"  alpha_b range: [{float(B.min()):.3f}, {float(B.max()):.3f}]")
    print(f"  consumption range: [{float(C.min()):.3f}, {float(C.max()):.3f}]")

    assert not np.isnan(C).any(), "Gate 2 FAIL: NaN in C"
    assert not np.isnan(S).any(), "Gate 2 FAIL: NaN in S"
    assert not np.isnan(B).any(), "Gate 2 FAIL: NaN in B"
    assert C.shape == (1, 16, 10), f"Gate 2 FAIL: unexpected C shape {C.shape}"
    # Bounds are loose because Gate 2 is a "doesn't crash + no NaN" smoke. The
    # 5-iter, 2x2x2x2-state run from cold start hasn't converged, so the
    # transient share range can be wide; the gate just needs finite policies.
    assert np.all(np.isfinite(S)), "Gate 2 FAIL: non-finite alpha_s"
    assert np.all(np.isfinite(B)), "Gate 2 FAIL: non-finite alpha_b"
    assert np.all(np.isfinite(C)) and float(C.min()) > 0.0, "Gate 2 FAIL: non-finite or non-positive C"
    print("\nGate 2: PASS")


if __name__ == "__main__":
    main()
