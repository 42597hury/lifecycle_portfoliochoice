"""Compare yesterday's 5^4 (delta=0.001) vs today's 6^4 (delta=0.0) on the
common solved ages 95-99. The two only differ in delta_bequest and grid size,
so the comparison isolates the effect of removing the luxury shift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

REPO = Path.cwd()
B5 = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark"
B6 = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid6x6x6x6_nz11_y1lob_calib1_partial_age95"

a5 = np.load(B5 / "policy_arrays.npz")
a6 = np.load(B6 / "policy_arrays.npz")

C5, S5, B5a = a5["C_mat"], a5["S_mat"], a5["B_mat"]
C6, S6, B6a = a6["C_mat"], a6["S_mat"], a6["B_mat"]

print(f"5^4 (delta=0.001):  {C5.shape}    N_state={C5.shape[2]}")
print(f"6^4 (delta=0.000):  {C6.shape}    N_state={C6.shape[2]}")

solved = [73, 74, 75, 76, 77]   # ages 95..99
ages = [95, 96, 97, 98, 99]
print()
print(f"  {'age':>4}  | {'5^4 (delta=.001)':<40}  | {'6^4 (delta=0.0)':<40}")
print(f"  {'':>4}  | "
      f"{'min α_b':>8} {'max α_b':>8} {'P>3':>5} {'P>5':>5} {'p99':>9}  | "
      f"{'min α_b':>8} {'max α_b':>8} {'P>3':>5} {'P>5':>5} {'p99':>9}")
for t, age in zip(solved, ages):
    a_bill5 = 1.0 - S5[t] - B5a[t]
    a_bill6 = 1.0 - S6[t] - B6a[t]

    p3_5 = float(np.mean(np.abs(a_bill5) > 3))
    p5_5 = float(np.mean(np.abs(a_bill5) > 5))
    p99_5 = float(np.percentile(a_bill5, 99))
    minb5, maxb5 = B5a[t].min(), B5a[t].max()

    p3_6 = float(np.mean(np.abs(a_bill6) > 3))
    p5_6 = float(np.mean(np.abs(a_bill6) > 5))
    p99_6 = float(np.percentile(a_bill6, 99))
    minb6, maxb6 = B6a[t].min(), B6a[t].max()

    print(f"  {age:>4}  | "
          f"{minb5:+8.3f} {maxb5:+8.3f} {100*p3_5:>4.0f}% {100*p5_5:>4.0f}% {p99_5:+9.3f}  | "
          f"{minb6:+8.3f} {maxb6:+8.3f} {100*p3_6:>4.0f}% {100*p5_6:>4.0f}% {p99_6:+9.3f}")

print()
print("Conclusions (terminal age 99 is z-invariant and pure-bequest -- best test):")

# Compare terminal slice at z=mid, state-grid midpoint (interpretation of i_s
# isn't the same in both bundles since N_state differs, but the AVERAGE policy
# behavior is comparable).
# The cleaner test: for each bundle, compute the fraction of (state, w) cells
# where |alpha_bill| > 5 at terminal age z=mid -- this is the pure-bequest
# degeneracy intensity.
n_z5 = C5.shape[1]
n_z6 = C6.shape[1]
imid5 = n_z5 // 2
imid6 = n_z6 // 2

a_bill_T5 = 1.0 - S5[-1, imid5] - B5a[-1, imid5]
a_bill_T6 = 1.0 - S6[-1, imid6] - B6a[-1, imid6]

print(f"  Terminal age 99, z=mid, all (state, w) cells:")
print(f"    5^4 (delta=0.001):  P(|α_bill|>3) = {np.mean(np.abs(a_bill_T5)>3):.1%}, "
      f"  P(|α_bill|>5) = {np.mean(np.abs(a_bill_T5)>5):.1%}, "
      f"  median = {np.median(a_bill_T5):+.3f}")
print(f"    6^4 (delta=0.000):  P(|α_bill|>3) = {np.mean(np.abs(a_bill_T6)>3):.1%}, "
      f"  P(|α_bill|>5) = {np.mean(np.abs(a_bill_T6)>5):.1%}, "
      f"  median = {np.median(a_bill_T6):+.3f}")
print()

# W-homothetic test:  at terminal age + delta=0, alpha should be EXACTLY independent of W
# The 5^4 has delta=0.001 -- still close to homothetic for W >> delta*A.  Compare wealth-axis
# variation at terminal age.
print(f"  W-axis variation of α_b at terminal age (z=mid, state-grid midpoint):")
imid_state5 = C5.shape[2] // 2
imid_state6 = C6.shape[2] // 2
a_b_T5_slice = B5a[-1, imid5, imid_state5, :]
a_b_T6_slice = B6a[-1, imid6, imid_state6, :]
print(f"    5^4 (delta=0.001):  α_b range over W = [{a_b_T5_slice.min():+.3f}, {a_b_T5_slice.max():+.3f}], "
      f"std = {a_b_T5_slice.std():.4f}")
print(f"    6^4 (delta=0.000):  α_b range over W = [{a_b_T6_slice.min():+.3f}, {a_b_T6_slice.max():+.3f}], "
      f"std = {a_b_T6_slice.std():.4f}")
print(f"  Pure CRRA (homothetic): range should be ~ machine eps; range > 0.1 = numerical failure.")
