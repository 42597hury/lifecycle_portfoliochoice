"""Redo the bequest-review at the TRUE joint state midpoint for 6^4.

Earlier review used i_s = N_state // 2 = 648 for 6^4, which decodes to
(dp=3, spr=0, rtb=0, y_1=0) — three axes pinned at low corners (~ -2sigma).
Not a mid state. The TRUE midpoint pair for 6^4 is i_s=518 ((2,2,2,2)) and
i_s=777 ((3,3,3,3)).

This script redoes the comparisons:
  1. α(W) at terminal age, true midpoint
  2. Pervasiveness re-stratified by per-axis distance from joint midpoint
  3. 5^4 vs 6^4 at correctly-paired joint midpoints
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
B5_path = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark"
B6_path = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid6x6x6x6_nz11_y1lob_calib1_partial_age95"

a5 = np.load(B5_path / "policy_arrays.npz")
a6 = np.load(B6_path / "policy_arrays.npz")
C5, S5, B5 = a5["C_mat"], a5["S_mat"], a5["B_mat"]
C6, S6, B6 = a6["C_mat"], a6["S_mat"], a6["B_mat"]

print("Bundles:")
print(f"  5^4 shape={C5.shape}  delta_bequest=0.001")
print(f"  6^4 shape={C6.shape}  delta_bequest=0.0")
print()


def decode(i, sizes):
    coords = []
    strides = np.cumprod([1] + list(sizes[::-1]))[:-1][::-1]
    rem = i
    for s in strides:
        c = rem // s
        rem -= c * s
        coords.append(int(c))
    return tuple(coords)


def encode(coords, sizes):
    strides = np.cumprod([1] + list(sizes[::-1]))[:-1][::-1]
    return int(np.dot(coords, strides))


# True joint midpoints
sizes6 = (6, 6, 6, 6)
sizes5 = (5, 5, 5, 5)
mid6_a = encode((2, 2, 2, 2), sizes6)   # 518
mid6_b = encode((3, 3, 3, 3), sizes6)   # 777
mid5 = encode((2, 2, 2, 2), sizes5)     # 312

print(f"True joint midpoints:")
print(f"  6^4 lower-mid (2,2,2,2) -> i_s = {mid6_a}")
print(f"  6^4 upper-mid (3,3,3,3) -> i_s = {mid6_b}")
print(f"  5^4 mid       (2,2,2,2) -> i_s = {mid5}")

# -----------------------------------------------------------------------------
# 1. Terminal-age slice at TRUE midpoints, across W
# -----------------------------------------------------------------------------
print()
print("=" * 78)
print("1. Terminal age (z=mid), α_b across W at TRUE joint midpoints")
print("=" * 78)
i_z6 = C6.shape[1] // 2
i_z5 = C5.shape[1] // 2

print(f"  {'iw':>4}  {'5^4 (2,2,2,2)':>14}  {'6^4 (2,2,2,2)':>14}  {'6^4 (3,3,3,3)':>14}")
for iw in [0, 30, 60, 90, 120, 140, 150, 160, 170, 175, 179]:
    print(f"  {iw:>4}  "
          f"{B5[-1, i_z5, mid5, iw]:>+14.4f}  "
          f"{B6[-1, i_z6, mid6_a, iw]:>+14.4f}  "
          f"{B6[-1, i_z6, mid6_b, iw]:>+14.4f}")

print()
print(f"  {'iw':>4}  {'5^4 α_s':>14}  {'6^4 α_s (lower)':>16}  {'6^4 α_s (upper)':>16}")
for iw in [0, 30, 90, 150, 179]:
    print(f"  {iw:>4}  "
          f"{S5[-1, i_z5, mid5, iw]:>+14.4f}  "
          f"{S6[-1, i_z6, mid6_a, iw]:>+16.4f}  "
          f"{S6[-1, i_z6, mid6_b, iw]:>+16.4f}")

# -----------------------------------------------------------------------------
# 2. Pervasiveness stratified by joint state distance-from-midpoint
# -----------------------------------------------------------------------------
print()
print("=" * 78)
print("2. Pervasiveness of |α_bill|>3 by per-axis distance from joint midpoint")
print("=" * 78)
print("  6^4 grid is 6 per axis (indices 0..5). Distance from joint midpoint")
print("  defined as max axis-distance from {2, 3} (the central pair).")

n_age, n_z, N_state, n_w = C6.shape
# Compute per-state axis-coords and distance-from-center
all_coords = np.zeros((N_state, 4), dtype=int)
for i in range(N_state):
    all_coords[i] = decode(i, sizes6)

# Distance from center: max of |coord - 2.5| (centered at 2.5 for 6 cells)
dist_from_mid = np.max(np.abs(all_coords - 2.5), axis=1)
# Buckets: 0.5 (cells at 2 or 3 on every axis -> innermost 16 states),
#          1.5 (cells with at least one axis at 1 or 4),
#          2.5 (cells with at least one axis at 0 or 5).
print(f"  Distance buckets:")
print(f"    0.5  (axes all in {{2,3}}, innermost {np.sum(dist_from_mid==0.5)} states)")
print(f"    1.5  (some axis in {{1,4}}, {np.sum(dist_from_mid==1.5)} states)")
print(f"    2.5  (some axis in {{0,5}}, outermost {np.sum(dist_from_mid==2.5)} states)")
print()
print("  Pervasiveness of |α_bill| > 3 at terminal age, z=mid, by bucket:")
print(f"    {'bucket':<10}  {'#states':>8}  {'P(|α_bill|>3)':>14}  "
      f"{'P(|α_bill|>5)':>14}  {'med α_b':>10}  {'std α_b':>10}")

for bucket in [0.5, 1.5, 2.5]:
    states = np.where(dist_from_mid == bucket)[0]
    a_b = B6[-1, i_z6, states, :]              # (n_states_in_bucket, n_w)
    a_s = S6[-1, i_z6, states, :]
    a_bill = 1.0 - a_s - a_b
    p3 = float(np.mean(np.abs(a_bill) > 3))
    p5 = float(np.mean(np.abs(a_bill) > 5))
    print(f"    {bucket:>4}        {len(states):>8}  {100*p3:>13.1f}%  {100*p5:>13.1f}%  "
          f"{np.median(a_b):>+10.3f}  {a_b.std():>10.3f}")

# Also at age 95
print()
print("  Same buckets at age 95 (idx 73, z=mid):")
print(f"    {'bucket':<10}  {'#states':>8}  {'P(|α_bill|>3)':>14}  "
      f"{'P(|α_bill|>5)':>14}  {'med α_b':>10}  {'std α_b':>10}")
for bucket in [0.5, 1.5, 2.5]:
    states = np.where(dist_from_mid == bucket)[0]
    a_b = B6[73, i_z6, states, :]
    a_s = S6[73, i_z6, states, :]
    a_bill = 1.0 - a_s - a_b
    p3 = float(np.mean(np.abs(a_bill) > 3))
    p5 = float(np.mean(np.abs(a_bill) > 5))
    print(f"    {bucket:>4}        {len(states):>8}  {100*p3:>13.1f}%  {100*p5:>13.1f}%  "
          f"{np.median(a_b):>+10.3f}  {a_b.std():>10.3f}")

# -----------------------------------------------------------------------------
# 3. Are the 16 innermost states clean?  (Most economically relevant cells.)
# -----------------------------------------------------------------------------
print()
print("=" * 78)
print("3. The 16 innermost-state cells at terminal (axes all in {2,3})")
print("=" * 78)
inner = np.where(dist_from_mid == 0.5)[0]
print(f"  States: {len(inner)} cells with all axes in {{2,3}}.")
print(f"  α_b at terminal age, z=mid, mid-W (iw=90):")
for s in inner:
    coord = decode(int(s), sizes6)
    print(f"    i_s={int(s):>4} {coord}:  α_s={S6[-1, i_z6, int(s), 90]:+.4f}  "
          f"α_b={B6[-1, i_z6, int(s), 90]:+.4f}")

# -----------------------------------------------------------------------------
# 4. Same mid-of-mid view at age 95 to confirm pattern
# -----------------------------------------------------------------------------
print()
print("=" * 78)
print("4. Same 16 innermost-state cells at age 95")
print("=" * 78)
print(f"  α_b at age 95, z=mid, mid-W:")
for s in inner:
    coord = decode(int(s), sizes6)
    print(f"    i_s={int(s):>4} {coord}:  α_s={S6[73, i_z6, int(s), 90]:+.4f}  "
          f"α_b={B6[73, i_z6, int(s), 90]:+.4f}  c={C6[73, i_z6, int(s), 90]:.4f}")
