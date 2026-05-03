"""Locate the (z, s, w) cells where the saved policy is still pathological."""
import json
import numpy as np

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)

import sys
sys.path.insert(0, '.')

vc = meta['run_config']['var_config']
bc = meta['run_config']['base_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from lifecycle.discretization import build_state_grid, discretize_income_ar1_mixture

Phi_11 = Phi_full[np.ix_([0, 1, 2], [0, 1, 2])]
Sigma_ss = Omega[np.ix_([0, 1, 2], [0, 1, 2])]
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[:3]

info = build_state_grid(N_vec=[5, 5, 5], mu_intercept=Phi_0_state,
                       Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='principal')
state_grid = info['state_grid']
state_indices = info['state_indices']

z_grid, _ = discretize_income_ar1_mixture(
    rho=bc['rho'], p=bc['pz'], mu1=bc['mu_eta1'], sigma1=bc['sigma_eta1'],
    mu2=bc['mu_eta2'], sigma2=bc['sigma_eta2'], N=9, n_stds=3.0,
)

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']
B = d['B_mat']
wealth_grid = np.geomspace(0.0001, 200.0, 150)

# Find the top-leverage cells
print("=" * 78)
print("TOP-10 cells by |alpha_b| at age 22")
print("=" * 78)
slc_b = B[0]  # (9, 125, 150)
flat = np.abs(slc_b).reshape(-1)
order = np.argsort(flat)[::-1]
print("  iz  i_s   iw    W     |  s_idx (i,j,k)  |   alpha_s   alpha_b   alpha_bill")
for k in order[:10]:
    iz = k // (125 * 150)
    rest = k - iz * (125 * 150)
    i_s = rest // 150
    iw = rest - i_s * 150
    si = state_indices[i_s]
    print(f"  {iz}   {i_s:>3}   {iw:>3}  {wealth_grid[iw]:>8.4f} | ({si[0]},{si[1]},{si[2]})        | "
          f"{S[0,iz,i_s,iw]:+8.3f}  {B[0,iz,i_s,iw]:+8.3f}  {1-S[0,iz,i_s,iw]-B[0,iz,i_s,iw]:+8.3f}")

print()
print("=" * 78)
print("TOP-10 cells by |alpha_s| at age 22")
print("=" * 78)
slc_s = S[0]
flat = np.abs(slc_s).reshape(-1)
order = np.argsort(flat)[::-1]
print("  iz  i_s   iw    W     |  s_idx (i,j,k)  |   alpha_s   alpha_b   alpha_bill")
for k in order[:10]:
    iz = k // (125 * 150)
    rest = k - iz * (125 * 150)
    i_s = rest // 150
    iw = rest - i_s * 150
    si = state_indices[i_s]
    print(f"  {iz}   {i_s:>3}   {iw:>3}  {wealth_grid[iw]:>8.4f} | ({si[0]},{si[1]},{si[2]})        | "
          f"{S[0,iz,i_s,iw]:+8.3f}  {B[0,iz,i_s,iw]:+8.3f}  {1-S[0,iz,i_s,iw]-B[0,iz,i_s,iw]:+8.3f}")

print()
print("=" * 78)
print("Distribution of |alpha_b| by state grid principal-coord index")
print("=" * 78)
print("Histogram by min(any_principal_idx) and max(any_principal_idx)")

# alpha_b at age 22, all (z, w) for each state
extremes = []
for i_s in range(125):
    si = state_indices[i_s]
    extreme = np.max(np.abs(B[0, :, i_s, :]))
    extremes.append((i_s, si, extreme))

# Sort by extreme
extremes.sort(key=lambda x: -x[2])
print()
print("State-by-state max|alpha_b| at age 22, top 20:")
print("  i_s  (i, j, k)   state_grid value     max|alpha_b|")
for i_s, si, ex in extremes[:20]:
    s_vals = state_grid[i_s]
    print(f"  {i_s:>3}  ({si[0]},{si[1]},{si[2]})  ({s_vals[0]:+.3f},{s_vals[1]:+.3f},{s_vals[2]:+.3f})  {ex:>10.3f}")

# How many cells have |alpha_b| > 5?
count_gt5 = (np.abs(B[0]) > 5.0).sum()
count_gt10 = (np.abs(B[0]) > 10.0).sum()
count_gt20 = (np.abs(B[0]) > 20.0).sum()
total = B[0].size
print()
print(f"|alpha_b| > 5  : {count_gt5} / {total} ({100*count_gt5/total:.2f}%)")
print(f"|alpha_b| > 10 : {count_gt10} / {total} ({100*count_gt10/total:.2f}%)")
print(f"|alpha_b| > 20 : {count_gt20} / {total} ({100*count_gt20/total:.2f}%)")

count_gt5s = (np.abs(S[0]) > 5.0).sum()
count_gt10s = (np.abs(S[0]) > 10.0).sum()
print(f"|alpha_s| > 5  : {count_gt5s} / {total} ({100*count_gt5s/total:.2f}%)")
print(f"|alpha_s| > 10 : {count_gt10s} / {total} ({100*count_gt10s/total:.2f}%)")
