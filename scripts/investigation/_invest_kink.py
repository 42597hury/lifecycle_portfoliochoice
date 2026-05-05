"""Show that the plateau alpha is set by R_port_min crossing zero in the discrete quadrature."""
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
                         Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='cholesky')
state_grid = info['state_grid']

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']; B = d['B_mat']

print("=== Q11: For each state, compute saved (a_s, a_b) at age 22, iz=4, W=200")
print("    then compute MIN R_port across all 360 quadrature nodes.")
print()
print("    If MIN R_port is ~0 (within ~1e-3), the plateau is set by the")
print("    no-bankruptcy boundary in the discrete quadrature.")
print()
print(f"  i_s |  alpha_s  alpha_b  alpha_bill |  min R_port  |  saved alpha plus 1%   min R_port (after)")

results = []
for i_s in range(125):
    sv = state_grid[i_s]
    base_mu_r = Phi_0_ret + Phi_21 @ sv
    a_s = float(S[0, 4, i_s, 149])
    a_b = float(B[0, 4, i_s, 149])
    a_bill = 1.0 - a_s - a_b

    min_Rp = np.inf
    for kv in range(v_nodes.shape[0]):
        mu_r_k = base_mu_r + M @ v_nodes[kv]
        for kr in range(ret_nodes.shape[0]):
            r_log = mu_r_k + ret_nodes[kr]
            R_bill = np.exp(r_log[0])
            R_s = np.exp(r_log[0] + r_log[1])
            R_b = np.exp(r_log[0] + r_log[2])
            R_p = a_s * R_s + a_b * R_b + a_bill * R_bill
            if R_p < min_Rp:
                min_Rp = R_p

    # Increase a_s and a_b by 1% (sign-preserving) and recompute
    a_s_new = a_s * 1.01 if abs(a_s) > 0.1 else a_s + 0.01
    a_b_new = a_b * 1.01 if abs(a_b) > 0.1 else a_b + 0.01
    a_bill_new = 1.0 - a_s_new - a_b_new
    min_Rp_new = np.inf
    for kv in range(v_nodes.shape[0]):
        mu_r_k = base_mu_r + M @ v_nodes[kv]
        for kr in range(ret_nodes.shape[0]):
            r_log = mu_r_k + ret_nodes[kr]
            R_bill = np.exp(r_log[0])
            R_s = np.exp(r_log[0] + r_log[1])
            R_b = np.exp(r_log[0] + r_log[2])
            R_p = a_s_new * R_s + a_b_new * R_b + a_bill_new * R_bill
            if R_p < min_Rp_new:
                min_Rp_new = R_p
    results.append((i_s, a_s, a_b, a_bill, min_Rp, min_Rp_new))

# Print summary
print()
n_zero_floor = sum(1 for r in results if abs(r[4]) < 1e-2)
print(f"States with min R_port within 1e-2 of zero (boundary): {n_zero_floor} / 125")
print(f"States where +1% perturbation drives min R_port < 0: {sum(1 for r in results if r[5] < 0)} / 125")
print()
# Distribution
mins = np.array([r[4] for r in results])
print(f"min R_port percentiles: p1={np.percentile(mins, 1):.4e}, p10={np.percentile(mins, 10):.4e}, p50={np.percentile(mins, 50):.4e}, p90={np.percentile(mins, 90):.4e}")
print()
# Show some interesting cases
print("Examples:")
print(f"  {'i_s':>4} {'alpha_s':>8} {'alpha_b':>8} {'a_bill':>8} {'min Rp':>10}  +1%min_Rp")
for i in [0, 12, 24, 36, 48, 62, 84, 96, 120, 124]:
    r = results[i]
    print(f"  {r[0]:>4} {r[1]:+8.3f} {r[2]:+8.3f} {r[3]:+8.3f} {r[4]:+10.4e} {r[5]:+10.4e}")
