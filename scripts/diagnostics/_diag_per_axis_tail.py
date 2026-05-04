"""Per-axis tail-node coverage diagnostic.

For each state-quadrature axis (0, 1, 2), compute how many EE-pathological cells
get fixed by adding ONE tail node at +/-L_sigma on that axis alone.

Answers: do we need Lobatto on state axes other than axis 2?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics._diag_arbitrage_quadsweep import _make_pc
from scripts.diagnostics._diag_policy_convergence import (
    _build_disc_config, _build_model_from_bundle, _extract_disc_config
)
from lifecycle.policy_io import load_policy_bundle


def cloud_grossR(pc, M):
    state_grid = np.asarray(pc.state_grid)
    base_mu_r = np.asarray(pc.const_r)[None, :] + state_grid @ np.asarray(pc.A_r).T
    M_v = pc.v_nodes @ M.T
    log_r = base_mu_r[:, None, None, :] + M_v[None, :, None, :] + pc.ret_nodes[None, None, :, :]
    R_bill = np.exp(log_r[..., 0])
    R_stock = R_bill * np.exp(log_r[..., 1])
    R_bond = R_bill * np.exp(log_r[..., 2])
    return R_bill, R_stock, R_bond


def main():
    bundle_path = Path('saved_runs/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_ret_v1')
    model, _ = _build_model_from_bundle(bundle_path)
    C, S, B, _, meta = load_policy_bundle(bundle_path)
    disc_raw = _extract_disc_config(meta)
    base_disc = _build_disc_config(disc_raw)
    pc_solver = _make_pc(model, base_disc)
    pc_eval = _make_pc(model, base_disc, ret_nodes=(5, 7, 9), state_nodes=(4, 5, 8))

    L_state = np.linalg.cholesky(0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T))
    M = np.asarray(model.M)

    print('M-loadings on R_bond from each state axis (M[xb, k] * L_state[k, k]):')
    for k in [0, 1, 2]:
        shift = M[2, k] * L_state[k, k]
        print(f'  state axis {k}: {shift:+.5f} per sigma')

    print()
    print(f'L_state diagonal: {[round(float(L_state[k,k]), 5) for k in range(3)]}')
    print(f'M[xb, :] = {[round(float(M[2,k]), 4) for k in range(3)]}')

    R_bill_s, R_stock_s, R_bond_s = cloud_grossR(pc_solver, M)
    R_bill_e, R_stock_e, R_bond_e = cloud_grossR(pc_eval, M)

    ret_age_start = int(model.retire_age)
    solved_mask = np.all(np.isfinite(C), axis=(1, 2, 3))
    ret_ages = [a for a in range(ret_age_start, int(model.terminal_age))
                if solved_mask[a - int(model.start_age)]]
    z_idx = [0, 5, 10]
    wealth_idx = [0, 15, 75, 134, 149]
    L_sigma_grid = np.arange(0.25, 8.01, 0.25)

    results = {0: {'short': [], 'long': [], 'unfix_short': 0, 'unfix_long': 0},
               1: {'short': [], 'long': [], 'unfix_short': 0, 'unfix_long': 0},
               2: {'short': [], 'long': [], 'unfix_short': 0, 'unfix_long': 0}}
    n_patho_total = 0

    for age in ret_ages:
        t = age - int(model.start_age)
        for iz in z_idx:
            for iw in wealth_idx:
                c_vec = C[t, iz, :, iw]
                a_s_vec = S[t, iz, :, iw]
                a_b_vec = B[t, iz, :, iw]
                x = pc_solver.wealth_grid[iw]
                s_val = np.maximum(x - c_vec, 0.0)
                valid_savings = s_val > 1e-8
                a_bill_vec = 1.0 - a_s_vec - a_b_vec
                Rp_s = (a_s_vec[:, None, None] * R_stock_s
                        + a_b_vec[:, None, None] * R_bond_s
                        + a_bill_vec[:, None, None] * R_bill_s)
                min_Rp_s = Rp_s.reshape(pc_solver.N_state, -1).min(axis=1)
                Rp_e = (a_s_vec[:, None, None] * R_stock_e
                        + a_b_vec[:, None, None] * R_bond_e
                        + a_bill_vec[:, None, None] * R_bill_e)
                min_Rp_e = Rp_e.reshape(pc_eval.N_state, -1).min(axis=1)
                patho = valid_savings & (min_Rp_s > 0) & (min_Rp_e < 0)
                n_patho_total += int(patho.sum())
                patho_states = np.flatnonzero(patho)

                for i_s in patho_states:
                    a_s_v = float(a_s_vec[i_s])
                    a_b_v = float(a_b_vec[i_s])
                    a_bill_v = 1.0 - a_s_v - a_b_v
                    base_mu_r = np.asarray(pc_solver.const_r) + np.asarray(pc_solver.A_r) @ pc_solver.state_grid[i_s]
                    for k in [0, 1, 2]:
                        L_axis = L_state[k, k]
                        M_col_k = M[:, k]
                        fix_short_L = None
                        fix_long_L = None
                        for sign in [+1, -1]:
                            new_M_v = sign * L_sigma_grid[:, None] * (M_col_k * L_axis)[None, :]
                            log_r_new = base_mu_r[None, None, :] + new_M_v[:, None, :] + pc_solver.ret_nodes[None, :, :]
                            Rb_new = np.exp(log_r_new[..., 0])
                            Rs_new = Rb_new * np.exp(log_r_new[..., 1])
                            Rd_new = Rb_new * np.exp(log_r_new[..., 2])
                            Rp_new = a_s_v * Rs_new + a_b_v * Rd_new + a_bill_v * Rb_new
                            min_Rp_new = Rp_new.min(axis=1)
                            augmented = np.minimum(min_Rp_s[i_s], min_Rp_new)
                            flipped = augmented < 0
                            if flipped.any():
                                smallest_L = L_sigma_grid[flipped].min()
                                if a_b_v < 0:
                                    if fix_short_L is None or smallest_L < fix_short_L:
                                        fix_short_L = smallest_L
                                else:
                                    if fix_long_L is None or smallest_L < fix_long_L:
                                        fix_long_L = smallest_L
                        if a_b_v < 0:
                            if fix_short_L is not None:
                                results[k]['short'].append(fix_short_L)
                            else:
                                results[k]['unfix_short'] += 1
                        else:
                            if fix_long_L is not None:
                                results[k]['long'].append(fix_long_L)
                            else:
                                results[k]['unfix_long'] += 1

    print(f'\nTotal EE-pathological cells: {n_patho_total:,}')
    print(f'\n=== Per-axis coverage at L=+/-7sigma ===')
    print(f'{"axis":<10} {"short fix%":>10} {"long fix%":>10} {"med L_short":>13} {"med L_long":>12}')
    for k in [0, 1, 2]:
        n_short = len(results[k]['short']) + results[k]['unfix_short']
        n_long = len(results[k]['long']) + results[k]['unfix_long']
        cov_short = sum(1 for L in results[k]['short'] if L <= 7.0) / n_short * 100 if n_short else 0
        cov_long = sum(1 for L in results[k]['long'] if L <= 7.0) / n_long * 100 if n_long else 0
        med_short = np.median(results[k]['short']) if results[k]['short'] else float('nan')
        med_long = np.median(results[k]['long']) if results[k]['long'] else float('nan')
        print(f'state[{k}]   {cov_short:>9.1f}% {cov_long:>9.1f}% {med_short:>12.2f}sigma {med_long:>11.2f}sigma')

    print(f'\n=== Detailed coverage per axis ===')
    for k in [0, 1, 2]:
        print(f'\n--- state[{k}]   M[xb,{k}]={M[2,k]:+.3f}   L_state[{k},{k}]={L_state[k,k]:.4f} ---')
        n_short = len(results[k]['short']) + results[k]['unfix_short']
        n_long = len(results[k]['long']) + results[k]['unfix_long']
        for thr_L in [3, 4, 5, 6, 7, 8]:
            cov_s = sum(1 for L in results[k]['short'] if L <= thr_L) / n_short * 100 if n_short else 0
            cov_l = sum(1 for L in results[k]['long'] if L <= thr_L) / n_long * 100 if n_long else 0
            print(f'  L=+/-{thr_L}sigma:  short {cov_s:>5.1f}%   long {cov_l:>5.1f}%')


if __name__ == '__main__':
    main()
