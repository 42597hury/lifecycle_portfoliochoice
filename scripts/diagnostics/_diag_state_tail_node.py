"""Same calculation as _diag_tail_node_position but for the STATE-quadrature axis 2 instead of
the return-quadrature bond axis. State axis 2 carries the M[xb, y_1] = -8.7 loading onto bonds,
so a tail node on the state axis shifts the conditional bond MEAN (much larger lever than the
bond-residual axis).
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

    Sigma_ss = np.asarray(model.Sigma_ss)
    L_state = np.linalg.cholesky(0.5 * (Sigma_ss + Sigma_ss.T))
    L22_state = L_state[2, 2]
    print(f'Cholesky diag of state innovation cov:')
    print(f'  L_state diag = ({L_state[0,0]:.4f}, {L_state[1,1]:.4f}, {L_state[2,2]:.4f})')
    print(f'  std of orth-axis 2 (residual y_1 innovation): {L22_state:.4f}')

    M = np.asarray(model.M)
    print(f'\nM matrix loadings on bond return (M[xb, :]):')
    print(f'  M[xb, axis0] = {M[2,0]:+.4f}')
    print(f'  M[xb, axis1] = {M[2,1]:+.4f}')
    print(f'  M[xb, axis2] = {M[2,2]:+.4f}  (the dominant channel)')

    # Adding a single tail node at state-axis z_2 = L_sigma_state shifts:
    #   v_added = (0, 0, L22_state * L_sigma_state)
    #   M @ v_added = (M[0,2]*L22_state*L, M[1,2]*L22_state*L, M[2,2]*L22_state*L)
    # so R_bond multiplier from this state-shift alone = exp(M[2,2] * L22_state * L_sigma_state)
    print(f'\nR_bond multiplier from a single state-axis-2 tail node at z_2 = +/-L_sigma:')
    for L in [3, 4, 5, 6, 7]:
        mult_pos = np.exp(M[2,2] * L22_state * L)
        mult_neg = np.exp(M[2,2] * L22_state * (-L))
        print(f'  L = +{L}sigma: M-shift = {M[2,2]*L22_state*L:+.4f}, R_bond_mult = {mult_pos:.4f}')
        print(f'  L = -{L}sigma: M-shift = {M[2,2]*L22_state*(-L):+.4f}, R_bond_mult = {mult_neg:.4f}')

    # Existing state quadrature: K_axis2 = 7 (per v1 config)
    z, w = roots_hermite(7)
    z = z * np.sqrt(2.0)
    print(f'\nExisting state-axis-2 GH K=7 nodes (sigma):')
    print(f'  positive abscissae: {[round(float(v), 3) for v in z[z>0]]}')
    print(f'  max |z_2_state| = {z.max():.3f}sigma -> M-shift = {M[2,2]*L22_state*z.max():+.4f}')

    v_nodes = pc.v_nodes
    v_weights = pc.v_weights
    ret_nodes = pc.ret_nodes
    ret_weights = pc.ret_weights

    state_grid = np.asarray(pc.state_grid)
    const_r = np.asarray(pc.const_r)
    A_r = np.asarray(pc.A_r)

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

    ret_age_start = int(model.retire_age)
    solved_mask_v3 = np.all(np.isfinite(C), axis=(1, 2, 3))
    ret_ages = [a for a in range(ret_age_start, int(model.terminal_age))
                if solved_mask_v3[a - int(model.start_age)]]

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
                    # New STATE tail node: v_added = (0, 0, L22_state * L_sigma)
                    # When tensor-producted with existing K=105 ret nodes, gives 105 new joint nodes per L
                    # log_r per k_r = base_mu_r + (M @ v_added) + ret_nodes[k_r]
                    new_M_v = np.zeros((len(L_sigma_grid), 3))
                    new_M_v[:, 0] = M[0, 2] * L22_state * L_sigma_grid
                    new_M_v[:, 1] = M[1, 2] * L22_state * L_sigma_grid
                    new_M_v[:, 2] = M[2, 2] * L22_state * L_sigma_grid
                    # log_r_new shape (len(L), n_r, 3)
                    log_r_new = base_mu_r[None, None, :] + new_M_v[:, None, :] + ret_nodes[None, :, :]
                    R_bill_new = np.exp(log_r_new[..., 0])
                    R_stock_new = R_bill_new * np.exp(log_r_new[..., 1])
                    R_bond_new  = R_bill_new * np.exp(log_r_new[..., 2])
                    R_p_new = a_s[i_s] * R_stock_new + a_b[i_s] * R_bond_new + a_bill[i_s] * R_bill_new
                    min_Rp_new_per_L = R_p_new.min(axis=1)
                    augmented_min_per_L = np.minimum(min_Rp_existing[i_s], min_Rp_new_per_L)
                    if a_b[i_s] < 0:
                        # Short bond: bankruptcy from HIGH R_bond.
                        # M[2,2]<0 means HIGH state z_2 -> LOW R_bond (bad for long bond, good for short bond)
                        # so for short-bond pathology, we need LOW state z_2 (negative L) -> HIGH R_bond
                        mask_neg = (L_sigma_grid < 0) & (augmented_min_per_L < 0)
                        mask_pos = (L_sigma_grid > 0) & (augmented_min_per_L < 0)
                        candidates = []
                        if mask_neg.any():
                            candidates.append(L_sigma_grid[mask_neg].max())  # closest to 0
                        if mask_pos.any():
                            candidates.append(L_sigma_grid[mask_pos].min())  # closest to 0
                        if candidates:
                            # Pick the one closest to zero in absolute value
                            best = min(candidates, key=abs)
                            required_L_for_short.append((best, age, iz, int(i_s), iw, float(a_s[i_s]), float(a_b[i_s])))
                        else:
                            unfixable_short += 1
                    else:
                        mask_neg = (L_sigma_grid < 0) & (augmented_min_per_L < 0)
                        mask_pos = (L_sigma_grid > 0) & (augmented_min_per_L < 0)
                        candidates = []
                        if mask_neg.any():
                            candidates.append(L_sigma_grid[mask_neg].max())
                        if mask_pos.any():
                            candidates.append(L_sigma_grid[mask_pos].min())
                        if candidates:
                            best = min(candidates, key=abs)
                            required_L_for_long.append((best, age, iz, int(i_s), iw, float(a_s[i_s]), float(a_b[i_s])))
                        else:
                            unfixable_long += 1

    print(f'\n=== Results: STATE-axis-2 tail node ===')
    print(f'Pathological short-bond cells fixed by some L: {len(required_L_for_short)}')
    print(f'Pathological long-bond cells fixed by some L:  {len(required_L_for_long)}')
    print(f'Unfixable within |L_sigma|<=7 (short bond): {unfixable_short}')
    print(f'Unfixable within |L_sigma|<=7 (long bond):  {unfixable_long}')

    if required_L_for_short:
        L_short = np.abs(np.array([r[0] for r in required_L_for_short]))
        signs = np.sign([r[0] for r in required_L_for_short])
        n_neg = int((signs < 0).sum())
        n_pos = int((signs > 0).sum())
        print(f'\nSHORT-bond pathology cells: {n_neg} fixed by NEGATIVE state z_2, {n_pos} by POSITIVE state z_2')
        print(f'  |L| (sigma units of standard-normal state z_2):')
        for q in [50, 75, 90, 95, 99, 100]:
            print(f'    p{q} = {np.percentile(L_short, q):.2f}sigma')

    if required_L_for_long:
        L_long = np.abs(np.array([r[0] for r in required_L_for_long]))
        signs = np.sign([r[0] for r in required_L_for_long])
        n_neg = int((signs < 0).sum())
        n_pos = int((signs > 0).sum())
        print(f'\nLONG-bond pathology cells: {n_neg} fixed by NEGATIVE state z_2, {n_pos} by POSITIVE state z_2')
        print(f'  |L| (sigma units of standard-normal state z_2):')
        for q in [50, 75, 90, 95, 99, 100]:
            print(f'    p{q} = {np.percentile(L_long, q):.2f}sigma')


if __name__ == '__main__':
    main()
