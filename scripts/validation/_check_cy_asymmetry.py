"""Is the alpha-extreme asymmetric in cy direction or symmetric in |cy - mean|?"""
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

from discretization import build_state_grid

Phi_11 = Phi_full[np.ix_([0,1,2], [0,1,2])]
Phi_21 = Phi_full[np.ix_([3,4,5], [0,1,2])]
Sigma_ss = Omega[np.ix_([0,1,2], [0,1,2])]
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[:3]
Phi_0_ret = z_bar[3:] - Phi_21 @ z_bar[:3]

info = build_state_grid(N_vec=[5,5,5], mu_intercept=Phi_0_state,
                       Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='principal')
state_grid = info['state_grid']

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']
B = d['B_mat']

# NOTE: this script was written assuming state ordering (y_1, spr, cy) where cy was at index 2.
# With the post-2026-04-30 default ordering (cy, spr, y_1) cy is at state index 0.
# Look up cy by name in case the script is rerun under either ordering.
_state_names_now = list(meta['run_config']['var_config'].get('state_predictor_columns', ['y_1', 'spr', 'cy']))
cy_state_idx = _state_names_now.index('cy')
cy_mean = z_bar[2]                   # z_bar always uses the FULL VAR ordering (y_1, spr, cy, rtb, xr, xb)
cy_grid = state_grid[:, cy_state_idx]
print(f"cy mean (unconditional): {cy_mean:.3f}")
print(f"cy range across grid:    [{cy_grid.min():.3f}, {cy_grid.max():.3f}]")
print(f"cy spans:                {cy_grid.min()-cy_mean:+.3f} to {cy_grid.max()-cy_mean:+.3f} from mean")
print()

# For each state, take max|alpha| across (z, w) at age 22
max_abs_b = np.max(np.abs(B[0, :, :, :]), axis=(0, 2))   # (125,)
max_abs_s = np.max(np.abs(S[0, :, :, :]), axis=(0, 2))

# Compute conditional excess returns at each state for context
mu_e = (Phi_0_ret[None, :] + state_grid @ Phi_21.T)[:, 1:]   # (125, 2)  [xr, xb]

# Sort by cy
order = np.argsort(cy_grid)

print("=" * 110)
print("STATE TABLE sorted by physical cy")
print("=" * 110)
print(f"  {'i_s':>3} | {'cy':>6} | {'cy-mean':>8} | {'y_1':>6}  {'spr':>6} | "
      f"{'mu_xr':>7} {'mu_xb':>7} | "
      f"{'max|alpha_s|':>12} {'max|alpha_b|':>12}")
print("-" * 110)
for i in order:
    sv = state_grid[i]
    print(f"  {i:>3} | {sv[2]:>+6.2f} | {sv[2]-cy_mean:>+8.2f} | "
          f"{sv[0]:>+6.3f}  {sv[1]:>+6.3f} | "
          f"{mu_e[i,0]:>+7.4f} {mu_e[i,1]:>+7.4f} | "
          f"{max_abs_s[i]:>12.2f} {max_abs_b[i]:>12.2f}")

# Group by cy direction
print()
print("=" * 78)
print("GROUPED summaries (across age=22, all z, all w)")
print("=" * 78)
cy_low = cy_grid < cy_mean - 0.5
cy_mid = (cy_grid >= cy_mean - 0.5) & (cy_grid <= cy_mean + 0.5)
cy_hi  = cy_grid > cy_mean + 0.5

print(f"  cy < mean - 0.5  (low cy):  {cy_low.sum()} states, "
      f"max|alpha_s|={max_abs_s[cy_low].max():.2f}, max|alpha_b|={max_abs_b[cy_low].max():.2f}")
print(f"  near cy mean +/- 0.5:        {cy_mid.sum()} states, "
      f"max|alpha_s|={max_abs_s[cy_mid].max():.2f}, max|alpha_b|={max_abs_b[cy_mid].max():.2f}")
print(f"  cy > mean + 0.5  (high cy):  {cy_hi.sum()} states, "
      f"max|alpha_s|={max_abs_s[cy_hi].max():.2f}, max|alpha_b|={max_abs_b[cy_hi].max():.2f}")

print()
# Same for stock loading
print(f"In the cy_low group ({cy_low.sum()} states):  alpha_s percentiles "
      f"p50={np.percentile(max_abs_s[cy_low],50):.2f}, p90={np.percentile(max_abs_s[cy_low],90):.2f}, max={max_abs_s[cy_low].max():.2f}")
print(f"In the cy_low group ({cy_low.sum()} states):  alpha_b percentiles "
      f"p50={np.percentile(max_abs_b[cy_low],50):.2f}, p90={np.percentile(max_abs_b[cy_low],90):.2f}, max={max_abs_b[cy_low].max():.2f}")
print(f"In the cy_hi  group ({cy_hi.sum()} states):  alpha_s percentiles "
      f"p50={np.percentile(max_abs_s[cy_hi],50):.2f}, p90={np.percentile(max_abs_s[cy_hi],90):.2f}, max={max_abs_s[cy_hi].max():.2f}")
print(f"In the cy_hi  group ({cy_hi.sum()} states):  alpha_b percentiles "
      f"p50={np.percentile(max_abs_b[cy_hi],50):.2f}, p90={np.percentile(max_abs_b[cy_hi],90):.2f}, max={max_abs_b[cy_hi].max():.2f}")
