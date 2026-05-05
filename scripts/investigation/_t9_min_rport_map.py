"""T9 (full): min(R_port) at saved policy across (age, z, state, wealth).

Direct test of H1b. Diagnoses where the no-bankruptcy boundary binds.
"""
import json
import sys
import numpy as np

sys.path.insert(0, '.')

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)
vc = meta['run_config']['var_config']
bc = meta['run_config']['base_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from lifecycle.discretization import build_state_grid, get_state_quadrature, get_return_quadrature

Phi_11 = Phi_full[np.ix_([0, 1, 2], [0, 1, 2])]
Phi_21 = Phi_full[np.ix_([3, 4, 5], [0, 1, 2])]
Sigma_ss = Omega[np.ix_([0, 1, 2], [0, 1, 2])]
Sigma_rr = Omega[np.ix_([3, 4, 5], [3, 4, 5])]
Sigma_rs = Omega[np.ix_([3, 4, 5], [0, 1, 2])]
M = Sigma_rs @ np.linalg.inv(Sigma_ss)
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[:3]
Phi_0_ret = z_bar[3:] - Phi_21 @ z_bar[:3]


class MM:
    pass


mm = MM()
mm.n_state = 3
mm.n_ret = 3
mm.Phi_0_state = Phi_0_state
mm.Phi_0_ret = Phi_0_ret
mm.Phi_11 = Phi_11
mm.Phi_21 = Phi_21
mm.Sigma_ss = Sigma_ss
mm.Sigma_rr = Sigma_rr
mm.Sigma_rs = Sigma_rs
mm.M = M
mm.Sigma_r_cond = Sigma_rr - M @ Sigma_rs.T

v_nodes, v_weights = get_state_quadrature(mm, n_nodes=2)
ret_nodes, ret_weights = get_return_quadrature(mm, n_nodes=(3, 5, 3))

info = build_state_grid(N_vec=[5, 5, 5], mu_intercept=Phi_0_state,
                       Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='cholesky')
state_grid = info['state_grid']  # (125, 3)

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']  # (78, 9, 125, 150)
B = d['B_mat']

print(f"Policy shape: {S.shape}")
print(f"v_nodes: {v_nodes.shape}, ret_nodes: {ret_nodes.shape}")

n_T, n_z, n_S, n_W = S.shape
n_kv = v_nodes.shape[0]
n_kr = ret_nodes.shape[0]
n_quad = n_kv * n_kr  # 360

# Pre-compute R_bill, R_stock, R_bond for each (i_s, kv, kr)
# r_log_n[i_s, n, :] = Phi_0_ret + Phi_21 @ s_grid[i_s] + M @ v_nodes[kv] + ret_nodes[kr]
# Note: r_log is 3-vector: [rtb, xr, xb]
# R_bill = exp(rtb), R_stock = exp(rtb+xr), R_bond = exp(rtb+xb)

# Build base mu_r per state
base_mu_r = Phi_0_ret[None, :] + state_grid @ Phi_21.T  # (125, 3)

# Add M @ v_nodes per kv: (kv, 3)
mv = v_nodes @ M.T  # (kv, 3)

# Total log-returns per (i_s, kv, kr) of shape (125, n_kv, n_kr, 3)
r_log = base_mu_r[:, None, None, :] + mv[None, :, None, :] + ret_nodes[None, None, :, :]
# Reshape to (125, n_quad, 3)
r_log = r_log.reshape(n_S, n_quad, 3)

R_bill = np.exp(r_log[..., 0])               # (125, 360)
R_stock = np.exp(r_log[..., 0] + r_log[..., 1])
R_bond = np.exp(r_log[..., 0] + r_log[..., 2])

print(f"R_bill range across states/nodes: [{R_bill.min():.4f}, {R_bill.max():.4f}]")
print(f"R_stock range: [{R_stock.min():.4f}, {R_stock.max():.4f}]")
print(f"R_bond range: [{R_bond.min():.4f}, {R_bond.max():.4f}]")

# Now compute min_R_port[t, iz, i_s, iw] over the 360 quadrature nodes.
# min over n of: a_s * R_stock + a_b * R_bond + (1-a_s-a_b) * R_bill
#             = R_bill + a_s*(R_stock-R_bill) + a_b*(R_bond-R_bill)

DRs = R_stock - R_bill  # (125, 360)  excess stock return
DRb = R_bond - R_bill   # (125, 360)  excess bond return

# Memory check: 78*9*125*150 = 13,162,500 cells (about 100 MB for float64)
min_R_port = np.empty((n_T, n_z, n_S, n_W), dtype=np.float64)
which_n = np.empty((n_T, n_z, n_S, n_W), dtype=np.int32)  # which quadrature node achieves min

print(f"Computing min_R_port across {n_T*n_z*n_S*n_W:,} cells with {n_quad} quadrature nodes each...")

for i_s in range(n_S):
    Rb_s = R_bill[i_s]      # (n_quad,)
    DRs_s = DRs[i_s]        # (n_quad,)
    DRb_s = DRb[i_s]        # (n_quad,)

    for t in range(n_T):
        for iz in range(n_z):
            a_s = S[t, iz, i_s, :]  # (n_W,)
            a_b = B[t, iz, i_s, :]
            # broadcast: (n_W, n_quad)
            R_port = Rb_s[None, :] + a_s[:, None] * DRs_s[None, :] + a_b[:, None] * DRb_s[None, :]
            min_R_port[t, iz, i_s, :] = R_port.min(axis=1)
            which_n[t, iz, i_s, :] = R_port.argmin(axis=1)

print(f"Done. min_R_port range: [{min_R_port.min():.4e}, {min_R_port.max():.4f}]")

# ================================================================
# Diagnostic summaries
# ================================================================
print()
print("=" * 78)
print("T9 FULL MAP — min_R_port at saved policy")
print("=" * 78)

# Boundary: min_R_port < 1e-3
boundary = min_R_port < 1e-3
n_total = boundary.size
n_bound = boundary.sum()
print(f"Total cells: {n_total:,}")
print(f"Boundary cells (min_R_port < 1e-3): {n_bound:,} ({100*n_bound/n_total:.2f}%)")

print()
print("Distribution of min_R_port across all cells:")
qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
qvals = np.percentile(min_R_port, qs)
for q, v in zip(qs, qvals):
    print(f"  p{q:>3}: {v:+10.4e}")

# By age (marginalize z, s, w)
print()
print("=" * 78)
print("BOUNDARY BINDING BY AGE (fraction of (z, s, w) cells with min_R_port < 1e-3)")
print("=" * 78)
print("  age   frac_boundary   median_minRp   p10_minRp     p1_minRp")
for t in range(n_T):
    age = 22 + t
    if age % 5 == 0 or age in (22, 23, 24, 99, 67, 66):
        slc = min_R_port[t]  # (n_z, n_S, n_W)
        frac = (slc < 1e-3).mean()
        med = np.median(slc)
        p10 = np.percentile(slc, 10)
        p1 = np.percentile(slc, 1)
        print(f"  {age:>3}    {frac:>12.4f}   {med:+10.4e}   {p10:+10.4e}   {p1:+10.4e}")

# By wealth bracket (marginalize age, z, s)
print()
print("=" * 78)
print("BOUNDARY BINDING BY WEALTH (fraction of (t, z, s) cells)")
print("=" * 78)
wealth_grid = np.geomspace(0.0001, 200.0, 150)
print("  iw   W_value   frac_boundary   median_minRp   p10_minRp")
for iw in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 149]:
    slc = min_R_port[:, :, :, iw]
    frac = (slc < 1e-3).mean()
    med = np.median(slc)
    p10 = np.percentile(slc, 10)
    print(f"  {iw:>3}  {wealth_grid[iw]:>9.4f}    {frac:>12.4f}   {med:+10.4e}   {p10:+10.4e}")

# Joint age x wealth heatmap
print()
print("=" * 78)
print("BOUNDARY BINDING — AGE x WEALTH (frac of (z,s) cells with min_Rp < 1e-3)")
print("=" * 78)
ages_show = [22, 25, 30, 35, 40, 45, 50, 55, 60, 67, 75, 90, 99]
iw_show = [0, 30, 60, 80, 100, 120, 140, 149]
W_labels = [f"{wealth_grid[i]:.2f}" for i in iw_show]
header = "  age" + "".join(f"  W={w:>7}" for w in W_labels)
print(header)
for age in ages_show:
    t = age - 22
    if t < 0 or t >= n_T:
        continue
    row = f"  {age:>3}"
    for iw in iw_show:
        slc = min_R_port[t, :, :, iw]
        frac = (slc < 1e-3).mean()
        row += f"   {frac:>7.3f}"
    print(row)

# Save the array for downstream tests
np.savez_compressed('_t9_min_rport_map.npz',
                    min_R_port=min_R_port,
                    which_n=which_n,
                    R_bill=R_bill,
                    R_stock=R_stock,
                    R_bond=R_bond)
print()
print("Saved: _t9_min_rport_map.npz")
