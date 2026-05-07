"""Map out the structure of the bequest-degeneracy region.

Questions:
  1. Is α_b vs W a smooth curve or a cliff?
  2. Where is the W-threshold separating "Markowitz" from "degenerate"?
  3. What about state-axis variation -- is it tied to spread / y_1 / rtb / dp?
  4. Newton-residual probe: at the terminal-age policy, what is the actual
     FOC residual at each (state, w)?  If the bundle reports
     worst_foc_resid=0.0 but the policy is W-non-homothetic, Newton must be
     converging to a SPURIOUS root.
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
sys.path.insert(0, str(REPO))

B6 = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid6x6x6x6_nz11_y1lob_calib1_partial_age95"
arr = np.load(B6 / "policy_arrays.npz")
C, S, B = arr["C_mat"], arr["S_mat"], arr["B_mat"]

# Reconstruct the wealth grid the solver used.
# config: n_w=180, wealth_min=0.05, wealth_max=750.  Most JAX runs use a
# log-spaced grid; we infer it via solver.precompute (skip-import, use guess).
# DiscretizationConfig.wealth_grid is build by precompute -> log spacing
# typically.  For visualization, just emit values without exact x.
n_age, n_z, N_state, n_w = C.shape
i_z_med = n_z // 2

# 1. α_b at terminal across W, for several state-grid points
print("1. α_b at terminal (age 99, z=mid) across W, for several states")
print("-" * 78)
sample_states = [0, 200, 400, 648, 900, 1100, 1295]
print(f"   iw idx  | " + " | ".join([f"i_s={s:>4}" for s in sample_states]))
print("   " + "-" * 75)
for iw in [0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 170, 175, 179]:
    row = [f"{B[-1, i_z_med, s, iw]:+7.3f}" for s in sample_states]
    print(f"   iw={iw:>3}   | " + " | ".join(row))

print()
print("2. Hard cliff or smooth transition?")
print("-" * 78)
i_s_med = N_state // 2
a_b_w = B[-1, i_z_med, i_s_med, :]
# Find largest |delta_b| step
abs_delta = np.abs(np.diff(a_b_w))
peak_jump = abs_delta.max()
peak_idx = abs_delta.argmax()
print(f"   At (z=mid, state=mid): max single-step jump in α_b = {peak_jump:.3f} between iw={peak_idx} and {peak_idx+1}")
print(f"     iw={peak_idx}: α_b = {a_b_w[peak_idx]:+.3f}, iw={peak_idx+1}: α_b = {a_b_w[peak_idx+1]:+.3f}")
# print full smooth curve
print()
print(f"   Full α_b(W) profile (every 10th wealth point):")
for iw in range(0, n_w, 10):
    print(f"     iw={iw:>3}   α_b={a_b_w[iw]:+.3f}   α_s={S[-1, i_z_med, i_s_med, iw]:+.3f}   "
          f"c={C[-1, i_z_med, i_s_med, iw]:9.4f}")

# Boundary between regimes
print()
print(f"   First iw where α_b > -2.0:  ", end="")
above = np.where(a_b_w > -2.0)[0]
print(above[0] if above.size else "never")

print()
print("3. State-axis structure at terminal (fixed iw=mid, iz=mid)")
print("-" * 78)
i_w_mid = n_w // 2
print(f"   Profiling i_s=0..{N_state-1} at iw={i_w_mid}, iz={i_z_med}.")
a_b_s = B[-1, i_z_med, :, i_w_mid]
print(f"   α_b across state grid: min={a_b_s.min():+.3f}  max={a_b_s.max():+.3f}  "
      f"median={np.median(a_b_s):+.3f}  std={a_b_s.std():.4f}")
# Count fraction in degenerate band
n_deg = int(np.sum(a_b_s < -3.0))
print(f"   Fraction of states with α_b < -3.0 at iw=mid:  {n_deg}/{N_state} = {n_deg/N_state:.1%}")

# 4. Compare to 5^4 baseline at same location
print()
print("4. Cross-check vs 5^4 (delta=0.001) at the same z, w-fraction")
print("-" * 78)
B5_path = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark"
a5 = np.load(B5_path / "policy_arrays.npz")
B5 = a5["B_mat"]
S5 = a5["S_mat"]
n_age5, n_z5, N_state5, n_w5 = B5.shape
i_z_med5 = n_z5 // 2
i_s_med5 = N_state5 // 2

# Use same iw indices (same wealth grid since n_wealth=180 in both)
print(f"   At terminal age, z=mid, state-mid:")
print(f"   {'iw':>4}  {'5^4 α_b':>10}  {'6^4 α_b':>10}  {'5^4 α_s':>10}  {'6^4 α_s':>10}")
for iw in [0, 30, 60, 90, 120, 150, 170, 179]:
    print(f"   {iw:>4}  "
          f"{B5[-1, i_z_med5, i_s_med5, iw]:>+10.4f}  "
          f"{B[-1, i_z_med, i_s_med, iw]:>+10.4f}  "
          f"{S5[-1, i_z_med5, i_s_med5, iw]:>+10.4f}  "
          f"{S[-1, i_z_med, i_s_med, iw]:>+10.4f}")
