"""Review of the 6^4 partial bundle (ages 95-99 solved, delta_bequest=0.0).

Goal: characterise whether and where the bequest motive is producing degenerate
policies. The live probe in bench.log already shows alpha_b ~ -8, alpha_bill ~ +9
at the SCF-median state across ages 91-98. We want to know:

1.  Is the degeneracy tail-only (extreme W or state corners) or pervasive?
2.  How does it scale across z (mortality differs by income), age (bequest weight
    grows with mortality), and W (CRRA bequest spike at low W)?
3.  Is the terminal-age policy (pure bequest, z-invariant) showing the standard
    Markowitz pattern, or already degenerate?
4.  Does s/x (savings rate) look correct, or is the bequest term making
    households save too much / too little at the margin?

This is read-only: just load policy_arrays.npz and report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Force utf-8 stdout on Windows so we can print Greek letters
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

try:
    REPO = Path(__file__).resolve().parents[2]
except NameError:
    REPO = Path.cwd()
sys.path.insert(0, str(REPO))

BUNDLE = REPO / "saved_runs/system_iv_full_var_unconstrained_cholesky_grid6x6x6x6_nz11_y1lob_calib1_partial_age95"


def hr(s=""):
    print()
    print("=" * 78)
    if s:
        print(s)
        print("=" * 78)


def main():
    np.set_printoptions(precision=4, linewidth=140, suppress=False)

    # -------------------------------------------------------------------------
    # 1.  Load
    # -------------------------------------------------------------------------
    arr = np.load(BUNDLE / "policy_arrays.npz")
    print("npz keys:", list(arr.keys()))
    C = arr["C_mat"]
    S = arr["S_mat"]
    B = arr["B_mat"]
    print(f"  C/S/B shape  = {C.shape}  (n_age, n_z, N_state, n_w)")
    print(f"  C dtype      = {C.dtype}")

    # Solved age indices 73-77 (ages 95-99 with start_age=22)
    start_age = 22
    n_age = C.shape[0]
    ages = np.arange(start_age, start_age + n_age)
    solved_idx = np.arange(73, 78)
    solved_ages = ages[solved_idx]
    print(f"  ages span    = {ages[0]} .. {ages[-1]}, n_age = {n_age}")
    print(f"  solved idxs  = {solved_idx}, ages = {solved_ages}")

    # Wealth grid (rebuild from disc config; we don't have wealth_grid.npy)
    # n_w = 180, wealth_min = 0.05, wealth_max = 750, log-spaced ish — but we
    # don't need exact x-axis values for the degeneracy review. Use index.
    n_age, n_z, N_state, n_w = C.shape
    print(f"  n_z={n_z}, N_state={N_state}, n_w={n_w}")

    # alpha_s = S / x = (savings * weight_s) / wealth.  Bundle convention: B and S
    # are alpha_s * (x - c) and alpha_b * (x - c)  -- the dollar AMOUNTS in stock
    # and bond.  We need wealth grid to recover alpha; or we can just look at S/B
    # in dollars relative to (x - c).  Verify convention against simulator.
    #
    # From simulation.py / solver.py: the policy is (c, alpha_s, alpha_b) per
    # (age, z, state, w_idx).  Bundle stores:
    #   C = c
    #   S = a_s    (the JAX solver returns "alpha_s")  -- check.
    #   B = a_b
    #
    # bench.log live probe shows alpha values directly (e.g. alpha_b = -8.054),
    # so S and B in this bundle ARE the alpha shares (not dollar amounts).

    # -------------------------------------------------------------------------
    # 2.  Headline:  alpha_s, alpha_b, alpha_bill distribution at each solved age
    # -------------------------------------------------------------------------
    hr("2.  Headline alpha distribution per solved age")
    print(f"  {'age':>4}  {'min α_s':>9}  {'max α_s':>9}  {'min α_b':>9}  {'max α_b':>9}  "
          f"{'med α_bill':>10}  {'max α_bill':>10}  {'%cells |α_bill|>3':>16}")
    for t in solved_idx:
        a_s = S[t]                    # (n_z, N_state, n_w)
        a_b = B[t]
        a_bill = 1.0 - a_s - a_b
        n_total = a_s.size
        n_extreme = int(np.sum(np.abs(a_bill) > 3.0))
        print(f"  {ages[t]:>4}  {a_s.min():>+9.3f}  {a_s.max():>+9.3f}  "
              f"{a_b.min():>+9.3f}  {a_b.max():>+9.3f}  "
              f"{np.median(a_bill):>+10.3f}  {a_bill.max():>+10.3f}  "
              f"{100*n_extreme/n_total:>15.1f}%")

    # -------------------------------------------------------------------------
    # 3.  Terminal age (pure bequest, z-invariant):  is the policy Markowitz-shaped?
    # -------------------------------------------------------------------------
    # At terminal age, the FOC reduces to E[mu_bq * dR_p/dα] = 0.  With
    # delta_bequest=0 + b_bar=10, b̄·(W/A)^{-γ}/A is pure CRRA over end-of-period
    # wealth. The unconstrained Markowitz at γ=5 should give roughly
    # alpha_s ~ 0.39, alpha_b ~ 0.58, alpha_bill ~ 0.03 at the unconditional state.
    hr("3.  Terminal age (idx 77, age 99) — pure bequest, z-invariant")
    a_s_T = S[-1]    # (n_z, N_state, n_w) — should be z-invariant
    a_b_T = B[-1]
    a_bill_T = 1.0 - a_s_T - a_b_T

    # Confirm z-invariance (broadcast claim in solver.py:2590)
    z_var_s = a_s_T.std(axis=0).max()
    z_var_b = a_b_T.std(axis=0).max()
    print(f"  std across z (max over (state, w)):  α_s {z_var_s:.3e}  α_b {z_var_b:.3e}")
    if max(z_var_s, z_var_b) > 1e-10:
        print(f"  *** WARNING: terminal policy is NOT z-invariant; bundle convention assumption is wrong.")
    else:
        print(f"  ✓ terminal policy is z-invariant as expected (broadcast at solver.py:2590).")

    # Mid-state, mid-wealth slice
    i_z_med = n_z // 2
    i_s_med = N_state // 2
    i_w_med = n_w // 2
    print(f"\n  Slice at (z=mid={i_z_med}, state=mid={i_s_med}):")
    print(f"  {'iw':>4}  {'α_s':>9}  {'α_b':>9}  {'α_bill':>9}  {'c':>10}")
    for iw in [0, 5, 20, 60, 90, 120, 150, 179]:
        print(f"  {iw:>4}  {a_s_T[i_z_med, i_s_med, iw]:>+9.3f}  "
              f"{a_b_T[i_z_med, i_s_med, iw]:>+9.3f}  "
              f"{a_bill_T[i_z_med, i_s_med, iw]:>+9.3f}  "
              f"{C[-1, i_z_med, i_s_med, iw]:>10.4f}")

    # Worst cell
    flat = np.abs(a_bill_T).reshape(-1)
    worst = np.argmax(flat)
    iz_w, is_w, iw_w = np.unravel_index(worst, a_bill_T.shape)
    print(f"\n  Worst |α_bill| at terminal: |α_bill| = {a_bill_T[iz_w, is_w, iw_w]:+.3f}")
    print(f"    at (iz={iz_w}, i_s={is_w}, iw={iw_w}), α_s = {a_s_T[iz_w, is_w, iw_w]:+.3f}, "
          f"α_b = {a_b_T[iz_w, is_w, iw_w]:+.3f}, c = {C[-1, iz_w, is_w, iw_w]:.4f}")

    # -------------------------------------------------------------------------
    # 4.  Variation of alpha across W at terminal age (CRRA myopic should be
    #     INDEPENDENT of W -- Merton property)
    # -------------------------------------------------------------------------
    hr("4.  Wealth-dependence of α at terminal age (CRRA-myopic should be flat in W)")
    print(f"  Slice (z=mid, state=mid), α_s and α_b across the wealth grid:")
    a_s_slice = a_s_T[i_z_med, i_s_med, :]
    a_b_slice = a_b_T[i_z_med, i_s_med, :]
    print(f"    α_s: min={a_s_slice.min():+.4f}  max={a_s_slice.max():+.4f}  "
          f"std={a_s_slice.std():.4f}  range/|mean|={(a_s_slice.max()-a_s_slice.min())/max(abs(a_s_slice.mean()),1e-9):.3f}")
    print(f"    α_b: min={a_b_slice.min():+.4f}  max={a_b_slice.max():+.4f}  "
          f"std={a_b_slice.std():.4f}  range/|mean|={(a_b_slice.max()-a_b_slice.min())/max(abs(a_b_slice.mean()),1e-9):.3f}")
    print()
    print(f"  At delta_bequest=0 (pure CRRA), (W/A)^(1-γ) is homothetic in W; the")
    print(f"  optimal alpha is INDEPENDENT of W.  Any visible W-dependence at the")
    print(f"  terminal age signals either (i) numerical noise from W^{{-γ}} at small")
    print(f"  W, or (ii) discretization-induced artefacts.")

    # -------------------------------------------------------------------------
    # 5.  Pervasiveness: alpha_bill at z=mid, state=mid, across W and across ages
    # -------------------------------------------------------------------------
    hr("5.  Pervasiveness of α_bill > 3 across (state, w) at age 99 (terminal)")
    # Terminal age, z=mid: how many cells have |α_bill| > 3, > 5, > 8?
    a_bill_99 = 1.0 - S[-1, i_z_med] - B[-1, i_z_med]   # (N_state, n_w)
    for thr in [1.0, 3.0, 5.0, 8.0]:
        frac = float(np.mean(np.abs(a_bill_99) > thr))
        print(f"  age 99 (terminal):  P(|α_bill| > {thr:.0f}) = {frac:.1%}  "
              f"(of {a_bill_99.size} (state, w) cells at z=mid)")

    # By age, how does the pervasiveness evolve?
    print()
    print(f"  By age (z=mid, all (state, w)):")
    for t in solved_idx:
        a_bill_t = 1.0 - S[t, i_z_med] - B[t, i_z_med]
        f1 = float(np.mean(np.abs(a_bill_t) > 1))
        f3 = float(np.mean(np.abs(a_bill_t) > 3))
        f5 = float(np.mean(np.abs(a_bill_t) > 5))
        f8 = float(np.mean(np.abs(a_bill_t) > 8))
        med = float(np.median(a_bill_t))
        p99 = float(np.percentile(a_bill_t, 99))
        print(f"    age {ages[t]}:  P(>1)={f1:.0%}  P(>3)={f3:.0%}  "
              f"P(>5)={f5:.0%}  P(>8)={f8:.0%}  median={med:+.3f}  p99={p99:+.3f}")

    # -------------------------------------------------------------------------
    # 6.  z-dependence of α_bill at age 95 (working-bequest weight scales with mortality)
    # -------------------------------------------------------------------------
    hr("6.  z-dependence of α_bill at age 95 (mortality varies with z)")
    # mu_bq weight is (1 - psi_z), and psi_z is higher (longer life) for high z.
    # If bequest is what's driving alpha_bill up, low-z (higher mortality) should
    # have MORE extreme alpha_bill than high-z.
    t = 73   # age 95
    print(f"  Age {ages[t]} = idx {t}; bequest weight is (1 - psi_z), higher at low z.")
    print(f"  {'iz':>4}  {'med α_s':>9}  {'med α_b':>9}  {'med α_bill':>10}  "
          f"{'p95 α_bill':>10}  {'p99 α_bill':>10}")
    for iz in range(n_z):
        a_s_iz = S[t, iz]
        a_b_iz = B[t, iz]
        a_bill_iz = 1.0 - a_s_iz - a_b_iz
        print(f"  {iz:>4}  {np.median(a_s_iz):>+9.3f}  {np.median(a_b_iz):>+9.3f}  "
              f"{np.median(a_bill_iz):>+10.3f}  "
              f"{np.percentile(a_bill_iz, 95):>+10.3f}  "
              f"{np.percentile(a_bill_iz, 99):>+10.3f}")

    # -------------------------------------------------------------------------
    # 7.  Diagnostic conclusion: does the degeneracy pattern fit "bequest-driven"?
    # -------------------------------------------------------------------------
    hr("7.  Diagnostic conclusion")
    # Bequest hypothesis predicts:
    #  H1: alpha_bill grows with mortality => low-z (high-mortality) cells should
    #      have MORE extreme alpha_bill than high-z.
    #  H2: terminal age (pure bequest, no alive branch) should be MORE extreme
    #      than age 95 (mortality only ~13%).
    #  H3: alpha_bill should be roughly W-independent at terminal age (CRRA
    #      homothetic property).  Visible W-dependence => numerical artefact.
    h1_low_z = np.percentile(1.0 - S[73, 0] - B[73, 0], 95)
    h1_high_z = np.percentile(1.0 - S[73, n_z - 1] - B[73, n_z - 1], 95)
    print(f"  H1 (mortality scaling): p95 α_bill at z=lowest = {h1_low_z:+.3f}, "
          f"at z=highest = {h1_high_z:+.3f}.")
    print(f"     low-z mortality is HIGHER, so bequest hypothesis says low-z α_bill is MORE extreme.")
    print(f"     Test:  {'CONSISTENT' if abs(h1_low_z) > abs(h1_high_z) else '*** NOT CONSISTENT ***'} "
          f"with bequest-driven hypothesis.")

    h2_terminal = np.percentile(1.0 - S[-1, i_z_med] - B[-1, i_z_med], 95)
    h2_age95 = np.percentile(1.0 - S[73, i_z_med] - B[73, i_z_med], 95)
    print(f"  H2 (bequest weight): p95 α_bill at age 99 = {h2_terminal:+.3f}, "
          f"at age 95 = {h2_age95:+.3f}.")
    print(f"     age 99 has bequest weight=1; age 95 has weight ~mortality.")
    print(f"     Test:  {'CONSISTENT' if abs(h2_terminal) > abs(h2_age95) else 'NOT CONSISTENT'} "
          f"with bequest-driven hypothesis.")

    a_s_T_slice = S[-1, i_z_med, i_s_med, :]
    a_b_T_slice = B[-1, i_z_med, i_s_med, :]
    rng_s = a_s_T_slice.max() - a_s_T_slice.min()
    rng_b = a_b_T_slice.max() - a_b_T_slice.min()
    print(f"  H3 (W-homothetic at terminal): α_s range over W = {rng_s:.4f}, "
          f"α_b range over W = {rng_b:.4f}.")
    print(f"     CRRA pure bequest (delta=0) implies α should be FLAT in W exactly.")
    print(f"     Range > 0.01 indicates discretization noise; Range > 0.1 indicates failure.")


if __name__ == "__main__":
    main()
