"""Time a few retirement periods at different grid sizes."""

import time
import numpy as np
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import build_model, Precompute
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.solver import solve_terminal_age, solve_retirement_step_quad

# --- Build model ---
var_config, _, _ = build_nominal_system1_var_config("data/var_dataset.csv")
base_config = {
    "beta": 0.96, "gamma": 3.0, "b_bar": 10,
    "start_age": 22, "retire_age": 67, "terminal_age": 99,
    "b0": -6.142, "b1": 0.304, "b2": -0.051, "b3": 0.002586,
    "rho": 0.991, "pz": 0.176,
    "mu_eta1": -0.524, "sigma_eta1": 0.113,
    "mu_eta2": -(0.176/(1-0.176))*(-0.524), "sigma_eta2": 0.046,
    "pe": 0.044, "mu_eps1": 0.134, "sigma_eps1": 0.762,
    "mu_eps2": 0.0, "sigma_eps2": 0.055, "constrained": True,
}

configs = [
    ("5x5x5", DiscretizationConfig(state_grid_sizes=(5,5,5), n_z=7)),
    ("7x7x7", DiscretizationConfig(state_grid_sizes=(7,7,7), n_z=7)),
]

N_RETIRE_PERIODS = 5   # solve 5 retirement periods (ages 94->98, backward from terminal)

for label, disc in configs:
    model = build_model(base_config, var_config)
    pc = Precompute(model, disc_config=disc)
    sc = SolverConfig()

    N_state = pc.N_state
    n_z = pc.n_z
    n_w = len(pc.wealth_grid)

    print(f"\n{'='*60}")
    print(f"Grid: {label}  |  N_state={N_state}, n_z={n_z}, n_w={n_w}")
    print(f"State quad nodes: {len(pc.v_weights)}, Return quad nodes: {len(pc.ret_weights)}")
    print(f"{'='*60}")

    # Terminal age
    t0 = time.time()
    c_T, a_s_T, a_b_T, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, N_state, n_z,
        constrained=model.constrained, solver_config=sc)
    t_term = time.time() - t0
    print(f"  Terminal age:  {t_term:.2f}s")

    # Retirement periods
    C_prev = c_T  # shape (n_z, N_state, n_w)
    times = []
    for i in range(N_RETIRE_PERIODS):
        age_idx = pc.n_age - 2 - i   # second-to-last, going backward
        psi = pc.survival_probs_2d[age_idx, :]
        pension = pc.pension_after_tax[age_idx + 1, :]

        t0 = time.time()
        c, a_s, a_b, _di, _df = solve_retirement_step_quad(
            pc.wealth_grid, pc.s_grid, pc.z_grid, N_state,
            C_prev, pension,
            pc.annuity_factors,
            pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
            pc.v_nodes, pc.v_weights, pc.M_v_nodes, pc.const_r, pc.A_r,
            model.Phi_0_state, model.Phi_11,
            pc.exp_ret_bill, pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
            model.gamma, psi, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
        elapsed = time.time() - t0
        times.append(elapsed)
        C_prev = c

        age = pc.ages[age_idx]
        print(f"  Age {age}:  {elapsed:.2f}s")

    avg = np.mean(times)
    print(f"\n  Avg retirement period: {avg:.2f}s")
    print(f"  Per-state (N_state*n_z*n_s): {avg / (N_state * n_z * 150) * 1e6:.1f} us")
    print(f"  Projected 32 retirement periods: {32 * avg:.0f}s")
    print(f"  Projected 45 working periods (est ~36x): {45 * avg * 36:.0f}s")
