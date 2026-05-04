"""Scan a policy bundle for extreme c(x) curvature — kinks, wiggles, sign flips.

For each (age, iz, j_s) slice of C_mat we compute:
  - d2c / d(log x)^2 (geomspace wealth → uniform log spacing)
  - relative curvature: |d2c| / (mean |dc| over slice)
  - non-monotonic c(x) flags (dc/dx < 0 anywhere)
  - non-monotonic c/x flags (d(c/x)/dx > 0 above some threshold — should be falling)

Reports the top-K worst slices by each metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lifecycle"))

from lifecycle.policy_io import load_policy_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    C, S, B, diagnostics, metadata = load_policy_bundle(args.bundle_dir)
    rc = metadata.get("run_config", {})
    disc = rc.get("discretization", {})
    n_age, n_z, N_state, n_w = C.shape
    wealth_min = disc.get("wealth_min", 0.05)
    wealth_max = disc.get("wealth_max", 200.0)
    wealth_grid = np.geomspace(wealth_min, wealth_max, n_w)
    log_w = np.log(wealth_grid)
    dlog = log_w[1] - log_w[0]  # uniform

    print(f"Bundle: {args.bundle_dir}")
    print(f"Shape (n_age, n_z, N_state, n_w) = {C.shape}")
    print(f"Wealth grid: [{wealth_min:.3f}, {wealth_max:.1f}], dlog={dlog:.4f}")
    print()

    # Compute slice-level metrics over (age, iz, j_s)
    nslices = n_age * n_z * N_state
    records = []  # (max_rel_curv, n_neg_dc, n_pos_dcx, age, iz, js, w_at_max_curv)

    for t in range(n_age):
        for iz in range(n_z):
            for js in range(N_state):
                c = C[t, iz, js, :]
                # First diff in log-wealth (uniform spacing)
                dc = np.diff(c)              # length n_w-1
                d2c = np.diff(dc)            # length n_w-2 → at interior nodes 1..n_w-2

                # Reference scale for curvature: mean |dc|, avoid divide-by-zero
                scale = np.mean(np.abs(dc)) + 1e-12
                rel_curv = np.abs(d2c) / scale  # length n_w-2

                # non-monotonic c (dc < 0)
                n_neg_dc = int(np.sum(dc < -1e-10))

                # non-monotonic c/x (should be weakly decreasing in x; flag rises)
                cx = c / wealth_grid
                dcx = np.diff(cx)
                n_pos_dcx = int(np.sum(dcx > 1e-8))

                # max relative curvature and where
                imax = int(np.argmax(rel_curv))
                w_at_max = float(wealth_grid[imax + 1])  # interior offset
                max_rel = float(rel_curv[imax])

                records.append((max_rel, n_neg_dc, n_pos_dcx, t, iz, js, w_at_max,
                                float(c[imax + 1]), float(c[imax]), float(c[imax + 2])))

    # ----- Top by relative curvature -----
    print(f"=== Top {args.top} slices by max relative second-diff (|d2c|/mean|dc|) ===")
    print(f"{'age':>4} {'iz':>3} {'js':>4} {'rel_curv':>10} {'w@max':>8} "
          f"{'c_lo':>8} {'c_mid':>8} {'c_hi':>8} {'neg_dc':>6} {'pos_dcx':>7}")
    for r in sorted(records, key=lambda r: -r[0])[:args.top]:
        max_rel, n_neg, n_pos, t, iz, js, w, c_mid, c_lo, c_hi = r
        print(f"{t:>4d} {iz:>3d} {js:>4d} {max_rel:>10.2f} {w:>8.2f} "
              f"{c_lo:>8.3f} {c_mid:>8.3f} {c_hi:>8.3f} {n_neg:>6d} {n_pos:>7d}")

    # ----- Slices with non-monotonic c (dc/dx < 0 anywhere) -----
    nm = [r for r in records if r[1] > 0]
    print()
    print(f"=== Non-monotone c(x): {len(nm)}/{nslices} slices have dc/dx < 0 somewhere ===")
    if nm:
        for r in sorted(nm, key=lambda r: -r[1])[:args.top]:
            max_rel, n_neg, n_pos, t, iz, js, w, *_ = r
            print(f"  age={t} iz={iz} js={js}: {n_neg} negative-dc points, max_curv={max_rel:.2f}")

    # ----- Slices with rising c/x (above modest threshold) -----
    rising = [r for r in records if r[2] > 2]
    print()
    print(f"=== Rising c/x (rate increasing in wealth): {len(rising)}/{nslices} slices ===")
    if rising:
        for r in sorted(rising, key=lambda r: -r[2])[:args.top]:
            max_rel, n_neg, n_pos, t, iz, js, w, *_ = r
            print(f"  age={t} iz={iz} js={js}: {n_pos} rising-c/x points, max_curv={max_rel:.2f}")

    # ----- Aggregate distribution of rel_curv -----
    rels = np.array([r[0] for r in records])
    print()
    print("=== Distribution of max relative curvature across slices ===")
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  q={q:.3f}: {np.quantile(rels, q):>8.2f}")
    print(f"  mean: {rels.mean():.2f}")

    # ----- Per-age aggregation -----
    print()
    print("=== Per-age max relative curvature (worst slice each age) ===")
    by_age = {}
    for r in records:
        by_age.setdefault(r[3], []).append(r[0])
    for t in sorted(by_age):
        worst = max(by_age[t])
        print(f"  age_idx={t:>3d}: worst rel_curv = {worst:.2f}")


if __name__ == "__main__":
    main()
