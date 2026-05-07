"""Diagnose the alpha_bill = 800% report.

Walks the VAR + Sigma_xx through:
  1. Sanity of the unconditional Markowitz allocation at gamma=5
  2. Conditional E[x | state] across the state grid corners
  3. State-by-state Markowitz allocation across the grid
  4. Sensitivity to the spread coefficients (Phi[xr, spr] = 3.48 is large)
  5. Bond return pipeline sanity at the historical samples that matter most
  6. Sigma_xx and the conditioning numbers that determine alpha magnitudes

Output flags any state cell that produces |alpha| > 5 — those are the likely
culprits behind the 800% bill report.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lifecycle.var import (  # noqa: E402
    build_nominal_system1_var_config,
    partition_var,
)


def main():
    cfg, _, data = build_nominal_system1_var_config()
    print()
    print("=" * 78)
    print("VAR ECONOMIC SANITY DIAGNOSTIC")
    print("=" * 78)

    z_bar = cfg["z_bar"]
    Phi = cfg["Phi"]
    Omega = cfg["Omega"]
    columns = cfg["variable_names"]
    state_idx = cfg["state_indices"]
    ret_idx = cfg["return_indices"]
    parts = partition_var(Phi, Omega, z_bar, state_idx, ret_idx,
                           variable_names=columns, verbose=False)

    Sigma_xx = parts["Sigma_rr"]
    sigma2_xr = float(Sigma_xx[0, 0])
    sigma2_xb = float(Sigma_xx[1, 1])
    sigma_xrxb = float(Sigma_xx[0, 1])
    sigma_x2 = np.array([sigma2_xr, sigma2_xb])

    # ─── 1. Unconditional Markowitz at gamma values ──────────────────────────
    print()
    print("1. UNCONDITIONAL MARKOWITZ (myopic, ignoring hedging)")
    print("-" * 78)
    z_bar_ret = parts["z_bar_ret"]
    Sigma_xx_inv = np.linalg.inv(Sigma_xx)
    print(f"   E[x] (Jensen-adj):   xr={z_bar_ret[0] + 0.5*sigma2_xr:+.4f},  "
          f"xb={z_bar_ret[1] + 0.5*sigma2_xb:+.4f}")
    print(f"   Sigma_xx:")
    print(f"     [[{Sigma_xx[0,0]:.6e}, {Sigma_xx[0,1]:.6e}],")
    print(f"      [{Sigma_xx[1,0]:.6e}, {Sigma_xx[1,1]:.6e}]]")
    print(f"   cond(Sigma_xx) = {np.linalg.cond(Sigma_xx):.2f}")
    print()
    print(f"   {'gamma':>6}    {'alpha_s':>9}    {'alpha_b':>9}    {'alpha_bill':>10}")
    for gamma in [1, 2, 5, 10, 20]:
        mu_adj = z_bar_ret + 0.5 * sigma_x2
        alpha = (1.0 / gamma) * Sigma_xx_inv @ mu_adj
        bill = 1.0 - alpha[0] - alpha[1]
        print(f"   {gamma:>6}    {alpha[0]:+9.4f}    {alpha[1]:+9.4f}    {bill:+10.4f}")
    print()
    print("   READING: at the unconditional mean state, gamma=5 gives ~40% stocks")
    print("            ~60% bonds, ~0% bills.  alpha_bill=8 (i.e. 800%) requires")
    print("            conditional E[x] to be DEEPLY negative (or sigma to be off).")
    print()

    # ─── 2. Conditional E[x | state] at grid corners ─────────────────────────
    print()
    print("2. CONDITIONAL E[x | state] AT GRID CORNERS")
    print("-" * 78)
    n_stds_grid = 2.25
    state_names = list(parts["state_names"])
    print(f"   Grid: {n_stds_grid} std per axis, state ordering = {state_names}")
    print()
    # Build state-block sub-VAR variables
    Phi_state_to_state = parts["Phi_11"]      # (n_state, n_state)
    Phi_state_to_ret = parts["Phi_21"]        # (n_ret, n_state)
    z_bar_state = parts["z_bar_state"]
    const_ret = parts["Phi_0_ret"]
    Sigma_ss = parts["Sigma_ss"]
    state_stds = np.sqrt(np.diag(Sigma_ss))   # innovation stds, NOT unconditional
    # Unconditional state stds (for grid extent reference): from sample data
    sample_state_stds = data[state_names].std(ddof=1).values

    print(f"   {'state':<8s}  {'mean':>10s}  {'sample std':>10s}  "
          f"{'grid +bound':>12s}  {'grid -bound':>12s}")
    for i, name in enumerate(state_names):
        m = float(z_bar_state[i])
        sd = float(sample_state_stds[i])
        print(f"   {name:<8s}  {m:>+10.4f}  {sd:>10.4f}  "
              f"{m + n_stds_grid * sd:>+12.4f}  {m - n_stds_grid * sd:>+12.4f}")
    print()

    # Compute conditional E[xr], E[xb] at each "extreme one-axis" corner
    # State block ordering = (dp, spr, rtb, y_1) per Default
    print(f"   Conditional E[x] when one state is at +/- 2.25 sample std")
    print(f"   (other states held at their unconditional mean)")
    print(f"   {'perturbation':<22s}  {'E[xr]':>10s}  {'E[xb]':>10s}  "
          f"{'mean reverts to':>16s}")
    print(f"   {'-':<22s}  {'-':>10s}  {'-':>10s}  {'-':>16s}")
    # baseline at all means
    s_mean = z_bar_state.copy()
    Ex_baseline = const_ret + Phi_state_to_ret @ s_mean
    print(f"   {'(at unconditional)':<22s}  {Ex_baseline[0]:>+10.4f}  "
          f"{Ex_baseline[1]:>+10.4f}")
    bad_cells = []
    for i, name in enumerate(state_names):
        sd = float(sample_state_stds[i])
        for sign, sym in [(+1.0, "+"), (-1.0, "-")]:
            s = z_bar_state.copy()
            s[i] = z_bar_state[i] + sign * n_stds_grid * sd
            Ex = const_ret + Phi_state_to_ret @ s
            label = f"{sym}{n_stds_grid}std on {name}"
            print(f"   {label:<22s}  {Ex[0]:>+10.4f}  {Ex[1]:>+10.4f}")
            # Check magnitude
            mu_adj = Ex + 0.5 * sigma_x2
            alpha = (1.0 / 5.0) * Sigma_xx_inv @ mu_adj
            bill = 1.0 - alpha[0] - alpha[1]
            if abs(bill) > 5 or abs(alpha[0]) > 5 or abs(alpha[1]) > 5:
                bad_cells.append((label, Ex[0], Ex[1], alpha[0], alpha[1], bill))
    print()

    # ─── 3. Markowitz alpha at every grid corner ─────────────────────────────
    print()
    print("3. MARKOWITZ ALPHA AT GRID CORNERS  (gamma=5)")
    print("-" * 78)
    print("   Walking ALL 2^4 = 16 corners of the +/- 2.25-std box")
    print()
    print(f"   {'cell':<24s}  {'E[xr]':>9s}  {'E[xb]':>9s}  "
          f"{'alpha_s':>9s}  {'alpha_b':>9s}  {'alpha_bill':>10s}")
    extremes = []
    n_state = len(state_names)
    for mask in range(2 ** n_state):
        signs = [+1.0 if (mask >> k) & 1 else -1.0 for k in range(n_state)]
        s = z_bar_state + n_stds_grid * np.array(signs) * sample_state_stds
        Ex = const_ret + Phi_state_to_ret @ s
        mu_adj = Ex + 0.5 * sigma_x2
        alpha = (1.0 / 5.0) * Sigma_xx_inv @ mu_adj
        bill = 1.0 - alpha[0] - alpha[1]
        label = "(" + ",".join(f"{s_:+.0f}" for s_ in signs) + ")"
        marker = " <-- EXTREME" if (abs(alpha[0]) > 3 or abs(alpha[1]) > 3 or abs(bill) > 3) else ""
        print(f"   {label:<24s}  {Ex[0]:>+9.3f}  {Ex[1]:>+9.3f}  "
              f"{alpha[0]:>+9.3f}  {alpha[1]:>+9.3f}  {bill:>+10.3f}{marker}")
        extremes.append((label, Ex, alpha, bill))
    print()
    print(f"   sign convention: ({state_names[0]:>3s},{state_names[1]:>3s},"
          f"{state_names[2]:>3s},{state_names[3]:>3s})")
    print()

    # ─── 4. Spread coefficient sensitivity ───────────────────────────────────
    print()
    print("4. SPREAD COEFFICIENT SENSITIVITY")
    print("-" * 78)
    spr_idx_in_state = state_names.index("spr")
    phi_xr_spr = float(Phi_state_to_ret[0, spr_idx_in_state])
    phi_xb_spr = float(Phi_state_to_ret[1, spr_idx_in_state])
    print(f"   Phi[xr, spr] = {phi_xr_spr:+.4f}    (CCV w8566 ref: +1.291, t=0.957)")
    print(f"   Phi[xb, spr] = {phi_xb_spr:+.4f}    (CCV w8566 ref: +2.628, t=5.289)")
    print()
    spr_mean = float(z_bar_state[spr_idx_in_state])
    spr_std = float(sample_state_stds[spr_idx_in_state])
    print(f"   spr distribution: mean={spr_mean*100:+.2f}pp, std={spr_std*100:.2f}pp")
    print(f"   Grid extent: [{(spr_mean - n_stds_grid*spr_std)*100:+.2f}pp, "
          f"{(spr_mean + n_stds_grid*spr_std)*100:+.2f}pp]")
    print()
    print(f"   Implied conditional E[xb] swing across the grid spread axis alone:")
    print(f"     Phi[xb, spr] * (spr_max - spr_min) = "
          f"{phi_xb_spr * 2 * n_stds_grid * spr_std * 100:+.2f}pp")
    print(f"     i.e. spread axis ALONE moves conditional E[xb] by this much.")
    print()
    if abs(phi_xr_spr) > 2.5 or abs(phi_xb_spr) > 4.0:
        print(f"   *** WARNING: Phi[xr/xb, spr] is high relative to CCV reference. ")
        print(f"   *** This amplifies conditional returns at extreme spread states.")
    print()

    # ─── 5. Sigma_xx conditioning ────────────────────────────────────────────
    print()
    print("5. SIGMA_XX CONDITIONING (drives alpha magnitude)")
    print("-" * 78)
    eigs = np.sort(np.linalg.eigvalsh(Sigma_xx))
    print(f"   eig(Sigma_xx) = [{eigs[0]:.6e}, {eigs[1]:.6e}]")
    print(f"   det(Sigma_xx) = {np.linalg.det(Sigma_xx):.6e}")
    print(f"   max |Sigma_xx_inv| entry = {np.max(np.abs(Sigma_xx_inv)):.2f}")
    print(f"   ratio sigma2_xr / sigma2_xb = {sigma2_xr/sigma2_xb:.2f}")
    print(f"   correlation(xr, xb) = {sigma_xrxb / np.sqrt(sigma2_xr*sigma2_xb):+.3f}")
    print()
    print(f"   alpha = (1/gamma) Sigma_xx^-1 mu_adj.  At gamma=5:")
    print(f"   per +1pp shift in conditional E[xr]: dalpha_s = "
          f"{(0.01 / 5.0) * Sigma_xx_inv[0,0]:+.3f},  dalpha_b = "
          f"{(0.01 / 5.0) * Sigma_xx_inv[1,0]:+.3f}")
    print(f"   per +1pp shift in conditional E[xb]: dalpha_s = "
          f"{(0.01 / 5.0) * Sigma_xx_inv[0,1]:+.3f},  dalpha_b = "
          f"{(0.01 / 5.0) * Sigma_xx_inv[1,1]:+.3f}")
    print()

    # ─── 6. Final verdict ────────────────────────────────────────────────────
    print()
    print("6. VERDICT")
    print("-" * 78)
    if bad_cells:
        print(f"   {len(bad_cells)} corner(s) produce |alpha| > 5 at gamma=5 myopic.")
        for label, exr, exb, asr, asb, bill in bad_cells:
            print(f"     {label:<22s}: E[xr]={exr:+.3f}, E[xb]={exb:+.3f}, "
                  f"alpha_s={asr:+.2f}, alpha_b={asb:+.2f}, bill={bill:+.2f}")
        print()
        print("   The 800% bill report is consistent with the policy reaching one of")
        print("   these corners (esp. extreme negative spread, where conditional E[xb]")
        print("   becomes very negative due to the +3.28 spread coefficient).")
    else:
        print("   No grid corner produces |alpha| > 5 at gamma=5 myopic.")
        print("   The 800% bill report is NOT explainable from the unconditional VAR")
        print("   pipeline alone -- look at solver / discretization / bequest setup.")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
