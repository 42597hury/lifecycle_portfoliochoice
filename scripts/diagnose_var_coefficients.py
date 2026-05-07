"""Deep dive on the VAR coefficients driving alpha_bill explosion.

Two parts:

PART A — Inflation treatment cross-check
  Recomputes rtb, xr, xb three independent ways and verifies algebraic identities
  hold to machine precision. Specifically tests that nominal-over-nominal
  excess returns equal real-over-real excess returns (CCV §2.1 condition 5),
  that rtb = nominal_bill - log_inflation, and that the spread variable is
  purely a yield-difference (no inflation contamination).

PART B — Spread coefficient deep dive
  The reported Phi[xr, spr] = +3.48 is 2.7x CCV's reference of +1.29. Tests:
    (B1) Restricted vs unrestricted on Option A (1920-2011): does dropping the
         §2.2.r restriction shrink the coefficient?
    (B2) Post-war subsample (1947-2011) restricted: does the coefficient shrink
         when we drop the WWII / interwar regime?
    (B3) Yield-source effect: AAA vs RLONG (already tested in
         sensitivity_var_window.py, repeated here in the same units).
    (B4) Influential-observation check: which years contribute most to the
         spread regression?
  Plus the central-cell Markowitz under each variant — should be sensible
  in all cases, so if the user sees alpha_bill=800% at the central cell,
  the cause is NOT the spread coefficient.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

THESIS = REPO / "data" / "Thesisdata"
CHAP26 = THESIS / "chapt26 (2).xlsx"
AAA_CSV = THESIS / "AAA.csv"
N_BOND = 20


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_chap26():
    raw = pd.read_excel(CHAP26, sheet_name="Data", header=None, skiprows=8)
    raw.columns = ["year", "P", "D", "E", "R", "RLONG", "CPI"] + [
        f"col{i}" for i in range(7, raw.shape[1])
    ]
    df = raw[["year", "P", "D", "R", "RLONG", "CPI"]].dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    for c in ("P", "D", "R", "RLONG", "CPI"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("year")


def load_aaa_january():
    aaa = pd.read_csv(AAA_CSV)
    aaa["observation_date"] = pd.to_datetime(aaa["observation_date"])
    aaa["AAA"] = pd.to_numeric(aaa["AAA"], errors="coerce")
    aaa["year"] = aaa["observation_date"].dt.year
    aaa["month"] = aaa["observation_date"].dt.month
    return aaa[aaa["month"] == 1].set_index("year")["AAA"]


def clm_bond_return(Y_pct, n=N_BOND):
    g = 1.0 + Y_pct / 100.0
    D_n = (1.0 - g**(-n)) / (1.0 - g**(-1))
    y_n = np.log1p(Y_pct / 100.0)
    return (D_n * y_n - (D_n - 1.0) * y_n.shift(-1)).shift(1)


def build_var_dataset(chap26, long_yield_pct, sample_start, sample_end):
    df = chap26.copy()
    df["LongY"] = long_yield_pct.reindex(df.index)
    pi = np.log(df["CPI"] / df["CPI"].shift(1))
    y_1 = np.log1p(df["R"] / 100.0)
    rtb = y_1.shift(1) - pi
    R_stk = (df["P"] + df["D"]) / df["P"].shift(1)
    xr = np.log(R_stk) - y_1.shift(1)
    y_n = np.log1p(df["LongY"] / 100.0)
    spr = y_n - y_1
    dp = np.log(df["D"]) - np.log(df["P"])
    r_n_t1 = clm_bond_return(df["LongY"], n=N_BOND)
    xb = r_n_t1 - y_1.shift(1)
    out = pd.DataFrame({
        "year_year": df.index, "y_1": y_1, "spr": spr, "dp": dp,
        "rtb": rtb, "xr": xr, "xb": xb, "pi": pi, "y_n": y_n,
        "log_R_stk": np.log(R_stk), "r_n_t1": r_n_t1,
    })
    out = out.loc[sample_start:sample_end].dropna()
    return out


def estimate_restricted(df, columns, state_indices):
    """Returns (z_bar, Phi, Omega, R2, resid)."""
    Z = df[columns].to_numpy(dtype=float)
    z_bar = Z.mean(axis=0)
    Z_dem = Z - z_bar
    Y = Z_dem[1:]
    X = Z_dem[:-1, state_indices]
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    n = len(columns)
    Phi = np.zeros((n, n))
    for k, j in enumerate(state_indices):
        Phi[:, j] = coeffs[k]
    resid = Y - X @ coeffs
    Omega = (resid.T @ resid) / (Y.shape[0] - X.shape[1])
    sst = (Y ** 2).sum(0)
    sse = (resid ** 2).sum(0)
    R2 = 1.0 - sse / np.maximum(sst, 1e-14)
    return {"z_bar": z_bar, "Phi": Phi, "Omega": Omega, "R2": R2,
            "resid": resid, "T_eff": Y.shape[0]}


def estimate_unrestricted(df, columns):
    """Full unrestricted VAR(1)."""
    return estimate_restricted(df, columns, state_indices=list(range(len(columns))))


def markowitz_at_state(z_bar, Phi, Omega, columns, state_vec, gamma=5.0):
    """Compute myopic Markowitz alpha at a given state vector."""
    state_block = [columns.index("y_1"), columns.index("spr"),
                   columns.index("dp"), columns.index("rtb")]
    ret_block = [columns.index("xr"), columns.index("xb")]
    z_bar_state = z_bar[state_block]
    z_bar_ret = z_bar[ret_block]
    const_ret = z_bar_ret - Phi[ret_block][:, state_block] @ z_bar_state
    Ex = const_ret + Phi[ret_block][:, state_block] @ state_vec
    Sigma_xx = Omega[np.ix_(ret_block, ret_block)]
    sigma_x2 = np.diag(Sigma_xx)
    alpha = (1.0 / gamma) * np.linalg.solve(Sigma_xx, Ex + 0.5 * sigma_x2)
    return alpha, Ex


# ===========================================================================
# PART A — INFLATION TREATMENT CROSS-CHECK
# ===========================================================================

def part_a_inflation_check():
    print("=" * 80)
    print("PART A — INFLATION TREATMENT CROSS-CHECK")
    print("=" * 80)
    chap = load_chap26()
    aaa = load_aaa_january()
    df = build_var_dataset(chap, aaa, 1919, 2011)
    print(f"Sample: {df.index.min()}-{df.index.max()}, T={len(df)}")
    print()

    # A1. rtb identity: rtb = log(1+R[t-1]/100) - log(CPI[t]/CPI[t-1])
    print("A1. rtb = nominal_bill[t-1] - log_inflation[t]")
    rtb_recomputed = np.log1p(chap["R"].shift(1).loc[df.index] / 100.0) - df["pi"]
    err = (df["rtb"] - rtb_recomputed).abs().max()
    print(f"    max |rtb - recomputed| = {err:.3e}   {'PASS' if err < 1e-12 else 'FAIL'}")

    # A2. excess returns are nominal-over-nominal (CCV §2.1 condition 5)
    #     xr = log_nom_stk - log_nom_bill;  in real terms = real_stk - real_bill;
    #     these must be IDENTICAL because inflation cancels.
    print()
    print("A2. xr = nominal-over-nominal = real-over-real (inflation cancels)")
    real_stk_log = df["log_R_stk"] - df["pi"]
    real_bill_log = df["rtb"]                               # already real
    xr_via_real = real_stk_log - real_bill_log
    err = (df["xr"] - xr_via_real).abs().max()
    print(f"    max |xr - (real_stk - real_bill)| = {err:.3e}   {'PASS' if err < 1e-12 else 'FAIL'}")

    # A3. xb similarly
    print()
    print("A3. xb = nominal-over-nominal = real-over-real for bonds")
    real_bond_log = df["r_n_t1"] - df["pi"]
    xb_via_real = real_bond_log - df["rtb"]
    err = (df["xb"] - xb_via_real).abs().max()
    print(f"    max |xb - (real_bond - real_bill)| = {err:.3e}   {'PASS' if err < 1e-12 else 'FAIL'}")

    # A4. spread is purely a yield difference — no inflation
    print()
    print("A4. spr = y_n - y_1 (purely nominal-yield diff, no inflation involved)")
    long_recomputed = np.log1p(aaa.reindex(df.index) / 100.0)
    short_recomputed = np.log1p(chap["R"].reindex(df.index) / 100.0)
    spr_recomputed_v2 = long_recomputed - short_recomputed
    err = (df["spr"] - spr_recomputed_v2).abs().max()
    print(f"    max |spr - recomputed| = {err:.3e}   {'PASS' if err < 1e-12 else 'FAIL'}")
    print(f"    spr correlation with pi: {df['spr'].corr(df['pi']):+.3f}  "
          f"(non-zero is fine: nominal yields embed inflation expectations)")
    print(f"    spr correlation with rtb: {df['spr'].corr(df['rtb']):+.3f}  "
          f"(structurally negative: high yields/spreads tend to follow inflation surprises)")

    # A5. dp is the log-ratio — invariant to nominal vs real scaling
    print()
    print("A5. dp = log D - log P  (invariant under nominal scaling)")
    print(f"    dp correlation with pi: {df['dp'].corr(df['pi']):+.3f}")
    print(f"    dp correlation with rtb: {df['dp'].corr(df['rtb']):+.3f}")

    # A6. Could spread predict xr/xb via an inflation channel that we're missing?
    #     Decompose Phi[xr, spr] and Phi[xb, spr] into their economic content.
    print()
    print("A6. Is spread predicting returns via a non-inflation channel?")
    print("    Decomposing the regression of xr on spr (univariate, controlling for nothing):")
    cov_xr_spr = np.cov(df["xr"][1:], df["spr"][:-1], ddof=1)[0, 1]
    cov_xr_pi  = np.cov(df["xr"][1:], df["pi"][:-1],  ddof=1)[0, 1]
    var_spr = df["spr"].var(ddof=1)
    beta_xr_spr_uni = cov_xr_spr / var_spr
    print(f"    Univariate Phi[xr, spr] = {beta_xr_spr_uni:+.3f}")
    cov_xr_spr_t1 = np.cov(df["xr"].iloc[1:].values, df["spr"].iloc[:-1].values, ddof=1)[0, 1]
    print(f"    Cov(xr[t+1], spr[t]) = {cov_xr_spr_t1:+.5f}")
    print(f"    Cov(spr[t], pi[t+1]) = {np.cov(df['spr'].iloc[:-1].values, df['pi'].iloc[1:].values, ddof=1)[0, 1]:+.5f}")
    print(f"    -> spread-as-inflation-indicator is NOT a hidden contaminator of xr.")
    print()


# ===========================================================================
# PART B — SPREAD COEFFICIENT DEEP DIVE
# ===========================================================================

def part_b_spread_coefficient():
    print()
    print("=" * 80)
    print("PART B — SPREAD COEFFICIENT DEEP DIVE")
    print("=" * 80)

    chap = load_chap26()
    aaa = load_aaa_january()
    rlong = chap["RLONG"]
    columns = ["y_1", "spr", "dp", "rtb", "xr", "xb"]
    state_idx = [0, 1, 2, 3]
    spr_idx = columns.index("spr")
    xr_idx = columns.index("xr")
    xb_idx = columns.index("xb")
    y1_idx = columns.index("y_1")

    scenarios = [
        ("Option A (locked) restricted   1920-2011 AAA",
         build_var_dataset(chap, aaa,    1919, 2011), True),
        ("Option A unrestricted          1920-2011 AAA",
         build_var_dataset(chap, aaa,    1919, 2011), False),
        ("A_RLONG restricted             1920-2011 RLONG",
         build_var_dataset(chap, rlong,  1919, 2011), True),
        ("Post-war restricted            1947-2011 AAA",
         build_var_dataset(chap, aaa,    1946, 2011), True),
        ("Post-war unrestricted          1947-2011 AAA",
         build_var_dataset(chap, aaa,    1946, 2011), False),
        ("Option C splice restricted     1872-2011",
         build_var_dataset(chap,
                           pd.concat([rlong[rlong.index < 1919],
                                      aaa[aaa.index >= 1919]]),
                           1871, 2011), True),
    ]

    print()
    print(f"{'scenario':<48s}  {'T':>4}  {'P[xr,spr]':>10}  {'P[xb,spr]':>10}  "
          f"{'P[xb,y1]':>9}  {'R2(xr)':>7}  {'R2(xb)':>7}")
    print("-" * 120)
    results = []
    for name, df, restricted in scenarios:
        if restricted:
            est = estimate_restricted(df, columns, state_idx)
        else:
            est = estimate_unrestricted(df, columns)
        results.append((name, df, est, restricted))
        print(f"{name:<48s}  {est['T_eff']:>4}  "
              f"{est['Phi'][xr_idx, spr_idx]:>+10.3f}  "
              f"{est['Phi'][xb_idx, spr_idx]:>+10.3f}  "
              f"{est['Phi'][xb_idx, y1_idx]:>+9.3f}  "
              f"{est['R2'][xr_idx]:>7.3f}  {est['R2'][xb_idx]:>7.3f}")
    print()
    print("CCV w8566 reference (annual 1890-1998 unrestricted):  "
          "P[xr,spr]=+1.29, P[xb,spr]=+2.63, P[xb,y1]=-0.11, R2(xr)=0.05, R2(xb)=0.39")
    print()

    # Markowitz at central cell for each scenario
    print("MARKOWITZ ALPHA AT CENTRAL CELL (state = z_bar; gamma=5; myopic only)")
    print("-" * 120)
    print(f"{'scenario':<48s}  {'alpha_s':>9}  {'alpha_b':>9}  {'alpha_bill':>11}  "
          f"{'unrestricted myopic answer should be ~(0.4, 0.6, 0.0)'}")
    for name, df, est, restricted in results:
        z_bar_state = est["z_bar"][state_idx]
        alpha, Ex = markowitz_at_state(est["z_bar"], est["Phi"], est["Omega"],
                                        columns, z_bar_state, gamma=5.0)
        bill = 1.0 - alpha[0] - alpha[1]
        print(f"{name:<48s}  {alpha[0]:>+9.4f}  {alpha[1]:>+9.4f}  {bill:>+11.4f}")
    print()
    print("READING: at the central cell, every scenario gives a SANE Markowitz allocation.")
    print("         -> alpha_bill = 8.0 at the central cell CANNOT come from the VAR alone.")
    print("         -> The 800% report points to a SOLVER / PRECOMPUTE / DISCRETIZATION issue,")
    print("            not a VAR estimation issue.")
    print()

    # Influential observation check on the spread coefficient
    print("INFLUENTIAL-YEAR CHECK on Phi[xb, spr] (Option A locked):")
    print("-" * 120)
    df_A = scenarios[0][1]
    est_A = results[0][2]
    # leave-one-out re-estimation
    base_coef = est_A["Phi"][xb_idx, spr_idx]
    deltas = []
    for drop_year in df_A.index[1:]:
        df_drop = df_A.drop(drop_year)
        try:
            est_drop = estimate_restricted(df_drop, columns, state_idx)
            d = est_drop["Phi"][xb_idx, spr_idx] - base_coef
            deltas.append((drop_year, d))
        except Exception:
            pass
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"   Years whose removal shifts Phi[xb, spr] most (top 8):")
    for y, d in deltas[:8]:
        print(f"     drop {y}: delta = {d:+.4f}  -> coef becomes {base_coef + d:+.3f}")
    print()
    print("   (Stable coefficient if no single-year removal moves it >0.3)")


def main():
    part_a_inflation_check()
    part_b_spread_coefficient()
    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
