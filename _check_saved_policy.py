"""Check actual saved policy values: is the 4.92 leverage still there?"""
import json
import numpy as np

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']  # (78, 9, 125, 150)
B = d['B_mat']
C = d['C_mat']

n_T, n_z, n_S, n_W = S.shape
print(f"Bundle shape: {S.shape}")
print(f"Bundle created: {meta['created_utc']}")
print(f"Solver init_alpha: ({meta['run_config']['solver_config']['init_alpha_s']}, "
      f"{meta['run_config']['solver_config']['init_alpha_b']})")
print()

# Canonical test cell
t = 0  # age 22
iz = 4  # median z
i_s = 62  # median state
iw = 149  # W = 200
print(f"Canonical test cell: age=22, iz=4, i_s=62, iw=149 (W=200)")
print(f"  alpha_s = {S[t, iz, i_s, iw]:+.4f}")
print(f"  alpha_b = {B[t, iz, i_s, iw]:+.4f}")
print(f"  alpha_bill = {1 - S[t, iz, i_s, iw] - B[t, iz, i_s, iw]:+.4f}")
print(f"  c       = {C[t, iz, i_s, iw]:+.4f}")
print()

# Distribution of alpha_s at age 22 across all (z, s, w)
print("=" * 70)
print("alpha_s distribution at age 22 (across z, s, w)")
print("=" * 70)
slc = S[0]  # (9, 125, 150)
qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
for q in qs:
    print(f"  p{q:>3}: {np.percentile(slc, q):+.4f}")

print()
print("=" * 70)
print("alpha_b distribution at age 22 (across z, s, w)")
print("=" * 70)
slc = B[0]
for q in qs:
    print(f"  p{q:>3}: {np.percentile(slc, q):+.4f}")

print()
print("=" * 70)
print("alpha_s by age (median across z, s, w)")
print("=" * 70)
print("  age   median_alpha_s   p10_alpha_s   p90_alpha_s   median_alpha_b")
for t in range(n_T):
    age = 22 + t
    if age in (22, 25, 30, 40, 50, 60, 65, 67, 75, 90, 99):
        med_s = np.median(S[t])
        p10_s = np.percentile(S[t], 10)
        p90_s = np.percentile(S[t], 90)
        med_b = np.median(B[t])
        print(f"  {age:>3}    {med_s:+13.4f}   {p10_s:+11.4f}   {p90_s:+11.4f}   {med_b:+14.4f}")

# Median wealth-grid alpha_s at the cell we use
print()
print("=" * 70)
print("alpha_s at (age=22, iz=4, i_s=62) across wealth grid")
print("=" * 70)
wealth_grid = np.geomspace(0.0001, 200.0, 150)
print("  iw   W       alpha_s   alpha_b   alpha_bill")
for iw in [0, 30, 60, 80, 100, 120, 140, 149]:
    print(f"  {iw:>3}  {wealth_grid[iw]:>8.4f}   "
          f"{S[0, 4, 62, iw]:+8.4f}  {B[0, 4, 62, iw]:+8.4f}  "
          f"{1 - S[0, 4, 62, iw] - B[0, 4, 62, iw]:+8.4f}")

# Across states at fixed (age, iz, iw)
print()
print("=" * 70)
print("alpha_s at (age=22, iz=4, iw=149) across all 125 states")
print("=" * 70)
slc_s = S[0, 4, :, 149]
slc_b = B[0, 4, :, 149]
print(f"  alpha_s range: [{slc_s.min():+.3f}, {slc_s.max():+.3f}]")
print(f"  alpha_b range: [{slc_b.min():+.3f}, {slc_b.max():+.3f}]")
print(f"  alpha_s p10/p50/p90: {np.percentile(slc_s,10):+.3f} / "
      f"{np.percentile(slc_s,50):+.3f} / {np.percentile(slc_s,90):+.3f}")
print(f"  alpha_b p10/p50/p90: {np.percentile(slc_b,10):+.3f} / "
      f"{np.percentile(slc_b,50):+.3f} / {np.percentile(slc_b,90):+.3f}")
