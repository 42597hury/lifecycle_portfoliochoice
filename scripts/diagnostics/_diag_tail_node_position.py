"""Compute the smallest |L_sigma| (in standard-normal sigma units) at which a single
added Lobatto-style tail node on the bond-axis quadrature would flip the solver cloud
from "certified safe" to "bankruptcy detected" at the saved policy alpha, for each
currently-pathological retirement cell.

Answers: how far out does the fixed Lobatto endpoint need to be on the bond-Cholesky axis
to fix the discrete-free-lunch at the saved policy?

For SHORT-bond cells (alpha_b < 0), bankruptcy comes from HIGH bond return -> need POSITIVE z_2 endpoint.
For LONG-bond cells  (alpha_b > 0), bankruptcy comes from LOW bond return  -> need NEGATIVE z_2 endpoint.
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
from scipy.special import roots_hermite


def main():
    bundle_path = Path('saved_runs/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_ret_v1')
    print(f'Bundle: {bundle_path.name}\n')

    model, ages = _build_model_from_bundle(bundle_path)
    C, S, B, _diag, meta = load_policy_bundle(bundle_path)
    disc_raw = _extract_disc_config(meta)
    base_disc = _build_disc_config(disc_raw)
    pc = _make_pc(model, base_disc)

    Sigma_r = np.asarray(model.Sigma_r_cond)
    L_chol = np.linalg.cholesky(0.5 * (Sigma_r + Sigma_r.T))
    L00, L11, L22 = L_chol[0,0], L_chol[1,1], L_chol[2,2]
    L20, L21 = L_chol[2,0], L_chol[2,1]
    print(f'Cholesky diag (L00, L11, L22) = ({L00:.4f}, {L11:.4f}, {L22:.4f})')
    print(f'  std of orth-bond residual axis: {L22:.4f} log-return units')

    z, w = roots_hermite(7)
    z = z * np.sqrt(2.0)
    print(f'\nExisting K=7 GH z_2 nodes (standard-normal sigma):')
    print(f'  positive abscissae: {[round(float(v), 3) for v in z[z>0]]}')
    print(f'  max |z_2| in existing cloud = {z.max():.3f}sigma')
    print(f'  in bond-residual log units: max +xb_resid = {z.max()*L22:.4f}')
    print(f'  R_bond tail multiplier from residual alone:')
    print(f'    exp(+{z.max()*L22:.4f}) = {np.exp(z.max()*L22):.4f}  (HIGH bond return tail node)')
    print(f'    exp(-{z.max()*L22:.4f}) = {np.exp(-z.max()*L22):.4f}  (LOW bond return tail node)')

    v_nodes = pc.v_nodes
    v_weights = pc.v_weights
    ret_nodes = pc.ret_nodes
    ret_weights = pc.ret_weights

    state_grid = np.asarray(pc.state_grid)
    const_r = np.asarray(pc.const_r)
    A_r = np.asarray(pc.A_r)
    M = np.asarray(model.M)

    base_mu_r_per_state = const_r[None, :] + state_grid @ A_r.T
    M_v_per_state = v_nodes @ M.T

    N_state = pc.N_state
    n_v = len(v_weights)
    n_r = len(ret_weights)
    print(f'\nN_state={N_state}, n_v={n_v}, n_r={n_r}, joint cloud per state = {n_v*n_r}')

    log_r_mat = (base_mu_r_per_state[:, None, None, :]
                 + M_v_per_state[None, :, None, :]
                 + ret_nodes[None, None, :, :])
    R_bill = np.exp(log_r_mat[..., 0])
    R_stock = R_bill * np.exp(log_r_mat[..., 1])
    R_bond  = R_bill * np.exp(log_r_mat[..., 2])
    print(f'R_bond range across full v1 cloud: [{R_bond.min():.4f}, {R_bond.max():.4f}]')

    ret_age_start = int(model.retire_age)
    solved_mask_v3 = np.all(np.isfinite(C), axis=(1, 2, 3))
    ret_ages = [a for a in range(ret_age_start, int(model.terminal_age))
                if solved_mask_v3[a - int(model.start_age)]]
    print(f'Retirement ages: {ret_ages[0]}..{ret_ages[-1]}\n')

    z_idx = [0, 5, 10]
    wealth_idx = [0, 15, 75, 134, 149]

    L_sigma_grid = np.arange(-7.0, 7.01, 0.25)

    required_L_for_short = []
    required_L_for_long = []
    unfixable_short = 0
    unfixable_long = 0

    for age in ret_ages:
        t = age - int(model.start_age)
        for iz in z_idx:
            for iw in wealth_idx:
                c_vec = C[t, iz, :, iw]
                a_s = S[t, iz, :, iw]
                a_b = B[t, iz, :, iw]
                x = pc.wealth_grid[iw]
                s_val = np.maximum(x - c_vec, 0.0)
                if not np.any(s_val > 1e-8):
                    continue
                a_bill = 1.0 - a_s - a_b
                R_p_existing = (a_s[:, None, None] * R_stock
                                + a_b[:, None, None] * R_bond
                                + a_bill[:, None, None] * R_bill)
                min_Rp_existing = R_p_existing.reshape(N_state, -1).min(axis=1)
                patho_mask = (s_val > 1e-8) & (min_Rp_existing > 0.0)
                if not np.any(patho_mask):
                    continue
                patho_states = np.flatnonzero(patho_mask)
                for i_s in patho_states:
                    base_mu_r = base_mu_r_per_state[i_s]
                    base_log = base_mu_r[None, None, :] + M_v_per_state[None, :, :]
                    added = np.zeros((len(L_sigma_grid), 1, 3))
                    added[:, 0, 2] = L22 * L_sigma_grid
                    new_log_r = base_log + added
                    R_bill_new = np.exp(new_log_r[..., 0])
                    R_stock_new = R_bill_new * np.exp(new_log_r[..., 1])
                    R_bond_new  = R_bill_new * np.exp(new_log_r[..., 2])
                    R_p_new = a_s[i_s] * R_stock_new + a_b[i_s] * R_bond_new + a_bill[i_s] * R_bill_new
                    min_Rp_new_per_L = R_p_new.min(axis=1)
                    augmented_min_per_L = np.minimum(min_Rp_existing[i_s], min_Rp_new_per_L)
                    if a_b[i_s] < 0:
                        mask = (L_sigma_grid > 0) & (augmented_min_per_L < 0)
                        if mask.any():
                            smallest_L = L_sigma_grid[mask].min()
                            required_L_for_short.append((smallest_L, age, iz, int(i_s), iw, float(a_s[i_s]), float(a_b[i_s])))
                        else:
                            unfixable_short += 1
                    else:
                        mask = (L_sigma_grid < 0) & (augmented_min_per_L < 0)
                        if mask.any():
                            largest_negL = L_sigma_grid[mask].max()
                            required_L_for_long.append((largest_negL, age, iz, int(i_s), iw, float(a_s[i_s]), float(a_b[i_s])))
                        else:
                            unfixable_long += 1

    print(f'=== Results ===')
    print(f'Pathological short-bond cells fixed by some +L: {len(required_L_for_short)}')
    print(f'Pathological long-bond cells fixed by some -L:  {len(required_L_for_long)}')
    print(f'Unfixable within |L_sigma|<=7 (short bond): {unfixable_short}')
    print(f'Unfixable within |L_sigma|<=7 (long bond):  {unfixable_long}')

    if required_L_for_short:
        L_short = np.array([r[0] for r in required_L_for_short])
        print(f'\nSHORT-bond pathology (need POSITIVE z_2 tail node):')
        print(f'  Required +L (sigma units of standard-normal z_2):')
        for q in [50, 75, 90, 95, 99, 100]:
            print(f'    p{q} = +{np.percentile(L_short, q):.2f}sigma')
        print(f'  In bond-residual log-return units (multiplied by L22={L22:.4f}):')
        for q in [50, 90, 99, 100]:
            Lr = np.percentile(L_short, q) * L22
            print(f'    p{q} = +{Lr:.4f}  (R_bond multiplier exp(+{Lr:.4f}) = {np.exp(Lr):.3f})')

    if required_L_for_long:
        L_long = np.array([r[0] for r in required_L_for_long])
        print(f'\nLONG-bond pathology (need NEGATIVE z_2 tail node):')
        print(f'  Required -L (closest to zero, sigma units):')
        for q in [50, 75, 90, 95, 99, 100]:
            print(f'    p{q} = {np.percentile(L_long, 100-q):.2f}sigma')
        print(f'  In bond-residual log-return units:')
        for q in [50, 90, 99, 100]:
            Lr = np.percentile(L_long, 100-q) * L22
            print(f'    p{q} = {Lr:.4f}  (R_bond multiplier exp({Lr:.4f}) = {np.exp(Lr):.3f})')

    if required_L_for_short:
        print(f'\nTop-5 short-bond cells requiring largest +L_sigma (i.e. furthest tail node needed):')
        sorted_short = sorted(required_L_for_short, key=lambda r: -r[0])[:5]
        for L_, age, iz, i_s, iw, a_s, a_b in sorted_short:
            i0 = i_s // 49; i1 = (i_s // 7) % 7; i2 = i_s % 7
            print(f'  age={age} iz={iz} i_s={i_s} ({i0},{i1},{i2}) iw={iw}  alpha=({a_s:+.3f},{a_b:+.3f})  needs +{L_:.2f}sigma')


if __name__ == '__main__':
    main()
