"""How did +0.064 spr at the grid corner correspond to '~4 sigma'? Reconcile units."""
import json
import numpy as np
import sys
sys.path.insert(0, '.')

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)

vc = meta['run_config']['var_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from lifecycle.discretization import build_state_grid, stationary_covariance

# Honor the saved bundle's state ordering (legacy (0,1,2)=y_1/spr/cy
# vs default (2,1,0)=cy/spr/y_1 after 2026-04-30 reorder).
_state_idx = list(meta['run_config']['var_config'].get('state_indices', [0, 1, 2]))
_var_names = ['y_1', 'spr', 'cy', 'rtb', 'xr', 'xb']
names = [_var_names[i] for i in _state_idx]

Phi_11 = Phi_full[np.ix_(_state_idx, _state_idx)]
Sigma_ss = Omega[np.ix_(_state_idx, _state_idx)]
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[_state_idx]

# Innovation sigma (per-period state shock)
sigma_innov = np.sqrt(np.diag(Sigma_ss))
mu_s = z_bar[_state_idx]

# Stationary covariance (steady-state variance of state)
Sigma_z = stationary_covariance(Phi_11, Sigma_ss)
sigma_stat = np.sqrt(np.diag(Sigma_z))

# Cholesky used to build the cholesky-mode grid
L = np.linalg.cholesky(Sigma_z)

# Build the actual production grid
info = build_state_grid(N_vec=[5,5,5], mu_intercept=Phi_0_state,
                       Phi=Phi_11, Sigma_innov=Sigma_ss,
                       n_stds=2.0, mode='cholesky')
state_grid = info['state_grid']

print("=" * 78)
print("PER-AXIS UNIT REFERENCES")
print("=" * 78)
print(f"  {'var':<6}  {'mean':>9}  {'sigma_innov':>11}  {'sigma_stat':>11}  {'stat/innov':>10}")
for d in range(3):
    ratio = sigma_stat[d] / sigma_innov[d]
    print(f"  {names[d]:<6}  {mu_s[d]:>+9.4f}  {sigma_innov[d]:>11.5f}  "
          f"{sigma_stat[d]:>11.5f}  {ratio:>10.3f}")
print()
print("Key: 'stat/innov' is how many innovation sigmas fit inside one stationary sigma.")
print("  This is determined by persistence — high persistence -> stationary sigma >> innovation sigma.")
print()

# Cholesky structure relevant to spr corner
print("=" * 78)
print("CHOLESKY OF STATIONARY COV (cholesky mode uses this to project)")
print("=" * 78)
print(f"L =\n{L}")
print()
print("L row for spr (axis 1):", L[1, :])
print(f"  L[spr, 0] = {L[1,0]:+.5f}  (loading from cholesky axis 0)")
print(f"  L[spr, 1] = {L[1,1]:+.5f}  (loading from cholesky axis 1)")
print(f"  L[spr, 2] = {L[1,2]:+.5f}  (Cholesky is lower-tri, must be 0)")
print()
print(f"  sqrt(L[1,0]^2 + L[1,1]^2) = {np.sqrt(L[1,0]**2 + L[1,1]**2):.5f}  (= sigma_stat[spr] by construction)")
print(f"  |L[1,0]| + |L[1,1]|       = {abs(L[1,0]) + abs(L[1,1]):.5f}  (worst-case projection at +/-1 in u)")
print(f"  ratio                      = {(abs(L[1,0]) + abs(L[1,1])) / np.sqrt(L[1,0]**2 + L[1,1]**2):.4f}  "
      f"(<=sqrt(2)=1.414 by L1<=sqrt(d)*L2)")
print()

# Now compute, for each axis, the corner extreme on the physical grid
print("=" * 78)
print("EACH PHYSICAL AXIS — grid corner reach in different units")
print("=" * 78)
print(f"  {'var':<6}  {'min phys':>10}  {'max phys':>10}  {'min-mean':>9}  {'max-mean':>9}  "
      f"{'in stat sigma':>14}  {'in innov sigma':>14}")
for d in range(3):
    smin = state_grid[:, d].min()
    smax = state_grid[:, d].max()
    dmin = smin - mu_s[d]
    dmax = smax - mu_s[d]
    in_stat_min = dmin / sigma_stat[d]
    in_stat_max = dmax / sigma_stat[d]
    in_innov_max = dmax / sigma_innov[d]
    in_innov_min = dmin / sigma_innov[d]
    print(f"  {names[d]:<6}  {smin:>+10.5f}  {smax:>+10.5f}  {dmin:>+9.5f}  {dmax:>+9.5f}  "
          f"{in_stat_min:>+6.2f} / {in_stat_max:<+6.2f}  {in_innov_min:>+6.2f} / {in_innov_max:<+6.2f}")
print()
print("=> The grid is built at +/- n_stds (=2.0) in stationary-standardized cholesky coords.")
print("=> 'sigma' on the right column means the per-period innovation sigma — same physical")
print("   point looks like more sigmas because the innovation sigma is smaller than the")
print("   stationary sigma (the latter accounts for persistence accumulation).")
