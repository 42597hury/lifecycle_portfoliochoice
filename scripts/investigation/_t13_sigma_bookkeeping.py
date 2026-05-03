"""T13: Sigma_total vs Sigma_rr bookkeeping check + per-state Markowitz allocation.

Tests H8 (artifactual conditional Sharpe). Computes:
  Sigma_total = M @ Sigma_ss @ M.T + Sigma_r_cond     (constant in s; should ~= Sigma_rr)
  Markowitz alpha at each state under Sigma_total vs Sigma_r_cond
"""
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

from lifecycle.discretization import build_state_grid

Phi_11 = Phi_full[np.ix_([0, 1, 2], [0, 1, 2])]
Phi_21 = Phi_full[np.ix_([3, 4, 5], [0, 1, 2])]
Sigma_ss = Omega[np.ix_([0, 1, 2], [0, 1, 2])]
Sigma_rr = Omega[np.ix_([3, 4, 5], [3, 4, 5])]
Sigma_rs = Omega[np.ix_([3, 4, 5], [0, 1, 2])]
M = Sigma_rs @ np.linalg.inv(Sigma_ss)
Sigma_r_cond = Sigma_rr - M @ Sigma_rs.T
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[:3]
Phi_0_ret = z_bar[3:] - Phi_21 @ z_bar[:3]

print("=" * 78)
print("T13: Sigma bookkeeping")
print("=" * 78)

Sigma_total = M @ Sigma_ss @ M.T + Sigma_r_cond
print()
print("Sigma_rr (full residual cov):")
print(Sigma_rr)
print()
print("M @ Sigma_ss @ M.T + Sigma_r_cond  (should equal Sigma_rr):")
print(Sigma_total)
print()
print(f"Max absolute difference: {np.max(np.abs(Sigma_total - Sigma_rr)):.2e}")
print(f"Max relative difference: {np.max(np.abs(Sigma_total - Sigma_rr) / (np.abs(Sigma_rr) + 1e-15)):.2e}")
print()
print("=> H8 framing check: Sigma_total == Sigma_rr to machine precision (expected)")
print()

# --- Markowitz alphas per state ---
gamma = 3.0
info = build_state_grid(N_vec=[5, 5, 5], mu_intercept=Phi_0_state,
                       Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='principal')
state_grid = info['state_grid']

# excess returns: indices 1 (xr) and 2 (xb) in [rtb, xr, xb]
S_total = Sigma_total[1:, 1:]
S_cond = Sigma_r_cond[1:, 1:]
S_rr = Sigma_rr[1:, 1:]

S_total_inv = np.linalg.inv(S_total)
S_cond_inv = np.linalg.inv(S_cond)
S_rr_inv = np.linalg.inv(S_rr)

print("=" * 78)
print("Markowitz alpha = (1/gamma) * Sigma^-1 * mu_excess at each state")
print("=" * 78)
mu_e_per_state = (Phi_0_ret[None, :] + state_grid @ Phi_21.T)[:, 1:]  # (125, 2)

alpha_total = (mu_e_per_state @ S_total_inv.T) / gamma  # (125, 2)
alpha_cond = (mu_e_per_state @ S_cond_inv.T) / gamma
alpha_rr = (mu_e_per_state @ S_rr_inv.T) / gamma

print()
print("Distribution of Markowitz alpha_s under Sigma_total (~ Sigma_rr):")
qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
for q in qs:
    print(f"  p{q:>3}: alpha_s = {np.percentile(alpha_total[:,0],q):+.3f}  "
          f"alpha_b = {np.percentile(alpha_total[:,1],q):+.3f}")
print()
print("Distribution of Markowitz alpha_s under Sigma_r_cond (NO state innov uncertainty):")
for q in qs:
    print(f"  p{q:>3}: alpha_s = {np.percentile(alpha_cond[:,0],q):+.3f}  "
          f"alpha_b = {np.percentile(alpha_cond[:,1],q):+.3f}")

# Compare to saved policy at age 22, median z (iz=4), median wealth (iw=149)
d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S_pol = d['S_mat']
B_pol = d['B_mat']

print()
print("=" * 78)
print("Saved policy (age=22, iz=4, iw=149) vs Markowitz at each state")
print("=" * 78)
print(f"  i_s | s=(y1,spr,cy)         | mu_e=(xr,xb)         | "
      f"Markowitz_total      | Markowitz_cond       | saved policy")

interesting = [0, 4, 12, 20, 24, 36, 45, 48, 62, 70, 84, 95, 96, 104, 120, 124]
for i_s in interesting:
    sv = state_grid[i_s]
    mu = mu_e_per_state[i_s]
    a_total = alpha_total[i_s]
    a_cond = alpha_cond[i_s]
    a_save_s = S_pol[0, 4, i_s, 149]
    a_save_b = B_pol[0, 4, i_s, 149]
    print(f"  {i_s:>3} | ({sv[0]:+.3f},{sv[1]:+.3f},{sv[2]:+.3f}) | "
          f"({mu[0]:+.4f},{mu[1]:+.4f}) | "
          f"({a_total[0]:+6.2f},{a_total[1]:+6.2f}) | "
          f"({a_cond[0]:+6.2f},{a_cond[1]:+6.2f}) | "
          f"({a_save_s:+6.2f},{a_save_b:+6.2f})")

# Sharpe-ratio analysis
print()
print("=" * 78)
print("Conditional Sharpe ratio per state")
print("=" * 78)
sigma_total_diag = np.sqrt(np.diag(S_total))
sigma_cond_diag = np.sqrt(np.diag(S_cond))
sharpe_total = mu_e_per_state / sigma_total_diag[None, :]
sharpe_cond = mu_e_per_state / sigma_cond_diag[None, :]
print(f"sigma diag (Sigma_total): stock={sigma_total_diag[0]:.4f}, bond={sigma_total_diag[1]:.4f}")
print(f"sigma diag (Sigma_cond):  stock={sigma_cond_diag[0]:.4f}, bond={sigma_cond_diag[1]:.4f}")
print()
print("Sharpe_total distribution:")
for q in qs:
    print(f"  p{q:>3}: stock={np.percentile(sharpe_total[:,0],q):+.3f}, bond={np.percentile(sharpe_total[:,1],q):+.3f}")
print()
print("Sharpe_cond distribution:")
for q in qs:
    print(f"  p{q:>3}: stock={np.percentile(sharpe_cond[:,0],q):+.3f}, bond={np.percentile(sharpe_cond[:,1],q):+.3f}")

# Acceptance per spec: are saved alphas closer to alpha_total or alpha_cond?
print()
print("=" * 78)
print("ACCEPTANCE: which Markowitz formula matches the saved policy more closely?")
print("=" * 78)
saved_pol = np.stack([S_pol[0, 4, :, 149], B_pol[0, 4, :, 149]], axis=1)  # (125, 2)
err_total = np.abs(saved_pol - alpha_total).mean(axis=0)
err_cond = np.abs(saved_pol - alpha_cond).mean(axis=0)
print(f"  Mean |saved - alpha_total|: stock={err_total[0]:.3f}, bond={err_total[1]:.3f}")
print(f"  Mean |saved - alpha_cond|:  stock={err_cond[0]:.3f}, bond={err_cond[1]:.3f}")
print()

print("Per-cell ratio saved/alpha_Markowitz (selected states):")
print(f"  i_s | saved_s/Mkw_total_s   saved_s/Mkw_cond_s   saved_b/Mkw_total_b   saved_b/Mkw_cond_b")
for i_s in interesting:
    a_save_s = S_pol[0, 4, i_s, 149]
    a_save_b = B_pol[0, 4, i_s, 149]
    rat_total_s = a_save_s / (alpha_total[i_s, 0] + 1e-9)
    rat_cond_s = a_save_s / (alpha_cond[i_s, 0] + 1e-9)
    rat_total_b = a_save_b / (alpha_total[i_s, 1] + 1e-9)
    rat_cond_b = a_save_b / (alpha_cond[i_s, 1] + 1e-9)
    print(f"  {i_s:>3} |  {rat_total_s:+8.2f}            {rat_cond_s:+8.2f}            "
          f"{rat_total_b:+8.2f}            {rat_cond_b:+8.2f}")
