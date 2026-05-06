"""Sanity: confirm the JIT kernel actually uses sigma2_xr argument.

Calls compute_foc_jac_retirement_quad twice at the SAME (state, alpha)
with sigma2_xr swapped between Sigma_r_cond[1,1] and Sigma_rr[1,1].
If the FOC values differ, the kernel is honoring the argument.
If they're identical, something upstream is overriding.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import Precompute, build_model
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.solver import compute_foc_jac_retirement_quad


def main():
    var_config, _, _ = build_nominal_system1_var_config(
        csv_path=str(PROJECT_ROOT / "data" / "var_dataset.csv")
    )
    from configs._canonical import BASE_CONFIG
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    disc = DiscretizationConfig(
        n_wealth=20, wealth_min=0.01, wealth_max=750.0,
        n_savings=20, state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky", state_n_stds=2.0,
        n_z=3, n_stds=2.0, n_eps_nodes=2, n_eta_nodes=2,
        n_ret_nodes_1d=2, n_state_quad_nodes=2,
    )
    pc = Precompute(model, disc, verbose=False)

    # Build a dummy c_next array (constant policy as next-period continuation)
    c_next = np.full((pc.N_state, pc.n_w), 1.0)

    # Pick a state and savings level
    i_s = pc.N_state // 2
    z_idx = pc.n_z // 2
    s_val = 5.0
    pension_next_scalar = 1.0
    annuity_factor_is = float(pc.annuity_factors[i_s])
    state_grid_i = pc.state_grid[i_s]
    base_mu_r_i = pc.const_r + pc.A_r @ state_grid_i

    alpha_s, alpha_b = 0.5, 0.3  # interior, sigma2 effects should show

    common_args = (
        alpha_s, alpha_b, s_val, z_idx, i_s,
        pc.wealth_grid, c_next, pension_next_scalar,
        annuity_factor_is,
        pc.v_nodes, pc.v_weights, pc.M_v_nodes,
        base_mu_r_i,
        model.Phi_0_state, model.Phi_11, state_grid_i,
        pc.state_bracket_shift, pc.state_bracket_L_inv,
        pc.state_bracket_grids[0], pc.state_bracket_grids[1], pc.state_bracket_grids[2],
        len(pc.state_bracket_grids[1]), len(pc.state_bracket_grids[2]),
        pc.exp_ret_bill, pc.exp_ret_stock, pc.exp_ret_bond,
        pc.ret_weights, pc.ret_nodes,
    )

    Sigma_rr = np.asarray(model.Sigma_rr)
    Sigma_r_cond = np.asarray(model.Sigma_r_cond)

    # Call A: sigma_cond
    s2_xr_a, s2_xb_a, s_xrxb_a = float(Sigma_r_cond[1,1]), float(Sigma_r_cond[2,2]), float(Sigma_r_cond[1,2])
    foc_s_a, foc_b_a, J_ss_a, J_bb_a, J_sb_a, eu_a = compute_foc_jac_retirement_quad(
        *common_args,
        s2_xr_a, s2_xb_a, s_xrxb_a, True,  # use_ccv=True
        5.0,  # gamma
        0.95, 0.96, 10,  # psi, beta, b_bar
    )

    # Call B: sigma_unc (Sigma_rr)
    s2_xr_b, s2_xb_b, s_xrxb_b = float(Sigma_rr[1,1]), float(Sigma_rr[2,2]), float(Sigma_rr[1,2])
    foc_s_b, foc_b_b, J_ss_b, J_bb_b, J_sb_b, eu_b = compute_foc_jac_retirement_quad(
        *common_args,
        s2_xr_b, s2_xb_b, s_xrxb_b, True,  # use_ccv=True
        5.0,
        0.95, 0.96, 10,
    )

    print("Direct kernel test at (alpha_s, alpha_b) = (0.5, 0.3), i_s=midgrid")
    print()
    print(f"sigma_cond:  s2_xr={s2_xr_a:.5e}  s2_xb={s2_xb_a:.5e}  s_xrxb={s_xrxb_a:+.5e}")
    print(f"sigma_unc:   s2_xr={s2_xr_b:.5e}  s2_xb={s2_xb_b:.5e}  s_xrxb={s_xrxb_b:+.5e}")
    print()
    print(f"          {'A: sigma_cond':>18} | {'B: sigma_unc':>18} | {'B - A':>12}")
    for label, va, vb in [
        ("foc_s",      foc_s_a,  foc_s_b),
        ("foc_b",      foc_b_a,  foc_b_b),
        ("J_ss",       J_ss_a,   J_ss_b),
        ("J_bb",       J_bb_a,   J_bb_b),
        ("J_sb",       J_sb_a,   J_sb_b),
        ("euler_sum",  eu_a,     eu_b),
    ]:
        print(f"{label:>9}  {va:18.10e} | {vb:18.10e} | {vb-va:+12.4e}")


if __name__ == "__main__":
    main()
