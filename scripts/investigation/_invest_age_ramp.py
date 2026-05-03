"""Verify that the age 50→40→22 transition is a transition from
interior optimum to no-bankruptcy boundary, by tracking min R_port at saved policy.

Cell of interest: (state=62, iz=4, W=1.694, iw=100)."""
import numpy as np
import sys, json
sys.path.insert(0, '.')

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)
vc = meta['run_config']['var_config']
bc = meta['run_config']['base_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from lifecycle.discretization import build_state_grid, get_state_quadrature, get_return_quadrature

Phi_11 = Phi_full[np.ix_([0,1,2],[0,1,2])]
Phi_21 = Phi_full[np.ix_([3,4,5],[0,1,2])]
Sigma_ss = Omega[np.ix_([0,1,2],[0,1,2])]
Sigma_rr = Omega[np.ix_([3,4,5],[3,4,5])]
Sigma_rs = Omega[np.ix_([3,4,5],[0,1,2])]
M = Sigma_rs @ np.linalg.inv(Sigma_ss)
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[:3]
Phi_0_ret = z_bar[3:] - Phi_21 @ z_bar[:3]


class MM:
    pass


mm = MM()
mm.n_state = 3; mm.n_ret = 3
mm.Phi_0_state = Phi_0_state; mm.Phi_0_ret = Phi_0_ret
mm.Phi_11 = Phi_11; mm.Phi_21 = Phi_21
mm.Sigma_ss = Sigma_ss; mm.Sigma_rr = Sigma_rr; mm.Sigma_rs = Sigma_rs
mm.M = M; mm.Sigma_r_cond = Sigma_rr - M @ Sigma_rs.T

v_nodes, v_weights = get_state_quadrature(mm, n_nodes=2)
ret_nodes, ret_weights = get_return_quadrature(mm, n_nodes=(3, 5, 3))

info = build_state_grid(N_vec=[5,5,5], mu_intercept=Phi_0_state,
                         Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='principal')
state_grid = info['state_grid']

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']; B = d['B_mat']; C = d['C_mat']

wealth_grid = np.geomspace(0.0001, 200.0, 150)
i_s = 62
iz = 4
sv = state_grid[i_s]
base_mu_r = Phi_0_ret + Phi_21 @ sv

def min_Rp(a_s, a_b):
    a_bill = 1.0 - a_s - a_b
    m = np.inf
    for kv in range(v_nodes.shape[0]):
        mu_r_k = base_mu_r + M @ v_nodes[kv]
        for kr in range(ret_nodes.shape[0]):
            r_log = mu_r_k + ret_nodes[kr]
            R_bill = np.exp(r_log[0])
            R_s = np.exp(r_log[0] + r_log[1])
            R_b = np.exp(r_log[0] + r_log[2])
            R_p = a_s * R_s + a_b * R_b + a_bill * R_bill
            if R_p < m:
                m = R_p
    return m

# Track policy and min R_port across ages at multiple wealth points
print("=== State 62, iz=4: Policy and min(R_port) by age ===")
print()
for iw, label in [(40, 'W=0.005 (very low)'),
                   (60, 'W=0.034'),
                   (80, 'W=0.24'),
                   (100, 'W=1.69 (mid)'),
                   (130, 'W=31.4'),
                   (149, 'W=200 (high)')]:
    print(f"--- {label} ---")
    print(f"  Age |  alpha_s    alpha_b    a_bill   |  min R_port  |   c        c/W")
    for age in [22, 30, 40, 50, 55, 60, 65, 67, 80, 99]:
        t = age - 22
        a_s = float(S[t, iz, i_s, iw])
        a_b = float(B[t, iz, i_s, iw])
        c = float(C[t, iz, i_s, iw])
        m = min_Rp(a_s, a_b)
        W = wealth_grid[iw]
        cw = c / W
        marker = " <BOUND" if m < 1e-4 else ""
        print(f"  {age:3d} | {a_s:+8.3f}  {a_b:+8.3f}  {1-a_s-a_b:+8.3f} | {m:+10.4e} | {c:7.4f} {cw:.4f}{marker}")
    print()
