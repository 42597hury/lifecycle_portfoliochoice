"""
Build var_dataset.csv for the 6-variable VAR system.

Locked spec: docs/CCV_RETURN_IMPLEMENT.md §4.1.

Sample: 1919-2011 (T=93). Resolution of the spec's AAA-1919 vs T=141
conflict — see RESOLUTION below.

Sources:
  - chap_26 (Shiller's Ch. 26 update, chapt26 (2).xlsx, sheet 'Data'):
      P (col B), D (col C), R (col E), CPI (col G).
      All values are January-of-year observations.
  - FRED Moody's AAA (AAA.csv): long-bond yield Y_n, monthly. The January
      observation of each year is taken.

Columns: [year, y_1, spr, dp, rtb, xr, xb]

VAR partition:
  States  (slow, on grid):  y_1, spr, dp, rtb
  Returns (integrated):     xr, xb

Variable construction (locked spec §4.1):
  pi[t+1]   = log(CPI[t+1]/CPI[t])                      (auxiliary)
  y_1[t]    = log(1 + R[t]/100)                         (state)
  rtb[t+1]  = y_1[t] - pi[t+1]                          (state)
  xr[t+1]   = log((P[t+1] + D[t+1])/P[t]) - y_1[t]      (return)
  spr[t]    = y_n[t] - y_1[t],  y_n = log(1 + Y_n/100)  (state)
  dp[t]     = log(D[t]) - log(P[t])                     (state)
  xb[t+1]   = r_n[t+1] - y_1[t]                         (return)

Bond return (Campbell-Lo-MacKinlay constant-duration, n=20):
  D_n[t]   = (1 - (1+Y_n[t]/100)^{-n}) / (1 - (1+Y_n[t]/100)^{-1})
  r_n[t+1] = D_n[t] * y_n[t] - (D_n[t] - 1) * y_n[t+1]

RESOLUTION of AAA-1919 vs T=141 conflict:
  The locked spec asserts both "1871-2011, T=141" and "Moody's AAA throughout
  1871-2011" (D1). FRED Moody's AAA starts 1919-01, so these cannot both hold.
  This build picks Option A: shorten the sample to 1919-2011 (T=93), honoring
  D1 strictly, at the cost of breaking T=141. Sensitivity to this choice is
  reported in scripts/sensitivity_var_window.py.

Reference data (ie_data.xls, Shiller monthly): kept ONLY for the §4.A1
timing cross-check; not used to construct the VAR variables.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent
THESIS = HERE / "Thesisdata"
CHAP26_XLSX = THESIS / "chapt26 (2).xlsx"
AAA_CSV = THESIS / "AAA.csv"
IE_DATA_XLS = THESIS / "ie_data.xls"
OUT_CSV = HERE / "var_dataset.csv"

SAMPLE_START = 1919  # Option A lower bound: AAA Jan-1919 is first available
SAMPLE_END = 2011    # chap_26 R column ends at 2011

N_BOND = 20  # CCV constant-duration maturity


# ============================================================
# 1. LOAD chap_26 (P, D, R, CPI from January-of-year)
# ============================================================
def load_chap26():
    raw = pd.read_excel(CHAP26_XLSX, sheet_name="Data", header=None, skiprows=8)
    # Column layout (from chap_26 header block, rows 2-7):
    #   A=year, B=P, C=D, D=E, E=R, F=RLONG, G=CPI, H=RealR, I=C
    raw.columns = ["year", "P", "D", "E", "R", "RLONG", "CPI"] + [
        f"col{i}" for i in range(7, raw.shape[1])
    ]
    df = raw[["year", "P", "D", "R", "RLONG", "CPI"]].dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    for col in ("P", "D", "R", "RLONG", "CPI"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("year")


# ============================================================
# 2. LOAD Moody's AAA (FRED) — January value of each year
# ============================================================
def load_aaa_january():
    aaa = pd.read_csv(AAA_CSV)
    aaa["observation_date"] = pd.to_datetime(aaa["observation_date"])
    aaa["AAA"] = pd.to_numeric(aaa["AAA"], errors="coerce")
    aaa["year"] = aaa["observation_date"].dt.year
    aaa["month"] = aaa["observation_date"].dt.month
    jan = aaa[aaa["month"] == 1].copy()
    if jan.duplicated("year").any():
        raise ValueError("Multiple January observations per year in AAA.csv")
    return jan.set_index("year")["AAA"]


# ============================================================
# 3. CLM CONSTANT-DURATION BOND RETURN
# ============================================================
def clm_duration(Y_pct, n=N_BOND):
    """Macaulay-style duration from yield only (CCV §4.1)."""
    g = 1.0 + Y_pct / 100.0
    return (1.0 - g ** (-n)) / (1.0 - g ** (-1))


def clm_bond_return(Y_pct, n=N_BOND):
    """r_n[t+1] ≈ D_n[t]·y_n[t] - (D_n[t]-1)·y_n[t+1]; constant-duration."""
    y_n = np.log1p(Y_pct / 100.0)
    D_n = clm_duration(Y_pct, n=n)
    # Shift y_n by -1 (i.e., y_n[t+1]) to align with t-indexed D_n[t]·y_n[t].
    y_n_next = y_n.shift(-1)
    D_n_t = D_n
    r_n_next = D_n_t * y_n - (D_n_t - 1.0) * y_n_next
    # r_n_next is realized return over [t, t+1]; index it by t+1 to match xb timing.
    return r_n_next.shift(1), D_n, y_n


# ============================================================
# 4. BUILD VAR VARIABLES
# ============================================================
def build_var(chap26, aaa_jan):
    df = chap26.copy()
    df["AAA"] = aaa_jan.reindex(df.index)

    # Auxiliary: log inflation pi[t+1] = log(CPI[t+1]/CPI[t])
    pi = np.log(df["CPI"] / df["CPI"].shift(1))

    # y_1[t] = log(1 + R[t]/100)
    y_1 = np.log1p(df["R"] / 100.0)

    # rtb[t+1] = y_1[t] - pi[t+1]
    rtb = y_1.shift(1) - pi

    # xr[t+1] = log((P[t+1] + D[t+1])/P[t]) - y_1[t]
    R_stk = (df["P"] + df["D"]) / df["P"].shift(1)
    log_R_stk = np.log(R_stk)
    xr = log_R_stk - y_1.shift(1)

    # spr[t] = y_n[t] - y_1[t]
    y_n = np.log1p(df["AAA"] / 100.0)
    spr = y_n - y_1

    # dp[t] = log D[t] - log P[t]
    dp = np.log(df["D"]) - np.log(df["P"])

    # xb[t+1] = r_n[t+1] - y_1[t]
    r_n_at_t1, D_n, _ = clm_bond_return(df["AAA"], n=N_BOND)
    xb = r_n_at_t1 - y_1.shift(1)

    out = pd.DataFrame(
        {
            "y_1": y_1,
            "spr": spr,
            "dp": dp,
            "rtb": rtb,
            "xr": xr,
            "xb": xb,
        }
    )

    # Diagnostic side-output (not stored in CSV)
    diag = pd.DataFrame({"pi": pi, "y_n": y_n, "D_n": D_n, "log_R_stk": log_R_stk})
    return out, diag


# ============================================================
# 5. VERIFICATION (§4.A, §4.B, §4.C — inline)
# ============================================================
def run_inline_tests(df, diag, chap26, aaa_jan):
    """Returns list of (id, status, msg) tuples."""
    results = []

    def add(rid, ok, msg):
        results.append((rid, "PASS" if ok else "FAIL", msg))

    # --- A1: chap_26 Jan timing cross-check vs ie_data ---
    try:
        shiller = pd.read_excel(IE_DATA_XLS, sheet_name="Data", header=7)
        shiller = shiller.rename(columns={shiller.columns[0]: "date_raw"})
        shiller = shiller.dropna(subset=["date_raw"])
        shiller = shiller[
            shiller["date_raw"].apply(lambda x: isinstance(x, (int, float)))
        ]
        shiller["date_raw"] = shiller["date_raw"].astype(float)
        shiller["year"] = shiller["date_raw"].astype(int)
        # ie_data Date column encodes month as 0.01..0.12 fractional part
        shiller["month_frac"] = (shiller["date_raw"] - shiller["year"]).round(2)
        jan = shiller[np.isclose(shiller["month_frac"], 0.01)].copy()
        jan["P"] = pd.to_numeric(jan["P"], errors="coerce")
        jan["D"] = pd.to_numeric(jan["D"], errors="coerce")
        jan["CPI"] = pd.to_numeric(jan["CPI"], errors="coerce")
        jan = jan.set_index("year")[["P", "D", "CPI"]]
        sample_years = [1900, 1950, 1980, 2000, 2010]
        max_p_rel = max_cpi_rel = 0.0
        for y in sample_years:
            if y not in chap26.index or y not in jan.index:
                continue
            p_rel = abs(chap26.loc[y, "P"] - jan.loc[y, "P"]) / max(jan.loc[y, "P"], 1.0)
            cpi_rel = abs(chap26.loc[y, "CPI"] - jan.loc[y, "CPI"]) / max(jan.loc[y, "CPI"], 1.0)
            max_p_rel = max(max_p_rel, p_rel)
            max_cpi_rel = max(max_cpi_rel, cpi_rel)
        # Test only P and CPI: chap_26 D is annual sum of trailing 12 months, while
        # ie_data D is Shiller's interpolated monthly series. P and CPI match to
        # ~1bp across all sample years, evidencing the January-of-year timing.
        add(
            "A1",
            max_p_rel < 5e-3 and max_cpi_rel < 1e-6,
            f"chap_26 vs ie_data Jan: max relative |dP|={max_p_rel:.2e}, |dCPI|={max_cpi_rel:.2e} "
            "(D not tested — conventions differ: chap_26 is trailing-12m sum, ie_data is monthly interp)",
        )
    except Exception as exc:
        add("A1", False, f"could not load ie_data: {exc}")

    # --- A2: AAA monotonic year coverage ---
    aaa_years = sorted(aaa_jan.dropna().index.tolist())
    monotone = aaa_years == list(range(aaa_years[0], aaa_years[-1] + 1))
    add(
        "A2",
        monotone and aaa_years[0] == 1919,
        f"AAA Jan years: {aaa_years[0]}..{aaa_years[-1]}, count={len(aaa_years)}, monotone={monotone}",
    )

    # --- A3: no NaN in df over sample ---
    no_nan = not df.isnull().any().any()
    add("A3", no_nan, f"NaN-free over {df.index.min()}..{df.index.max()}: {no_nan}")

    # --- A4: T == 92 (Option A: 1919-2011 minus 1 year lost to inflation differencing) ---
    # Spec said T=141 for 1871-2011 but handoff §3.1.5 also says "1872..2011" =>
    # the effective T is window_length-1 due to inflation differencing.
    add("A4_optionA", len(df) == 92, f"T = {len(df)} (Option A target = 92 = 93 - 1 differencing loss)")

    # --- B1: rtb identity ---
    pi = diag["pi"].reindex(df.index)
    y_1_at_t = np.log1p(chap26["R"] / 100.0).reindex(df.index)
    rtb_check = y_1_at_t.shift(1) - pi
    b1_resid = (df["rtb"] - rtb_check).dropna().abs().max()
    add("B1", b1_resid < 1e-12, f"rtb identity max |resid| = {b1_resid:.3e}")

    # --- B2: xr identity ---
    log_R_stk = diag["log_R_stk"].reindex(df.index)
    xr_check = log_R_stk - y_1_at_t.shift(1)
    b2_resid = (df["xr"] - xr_check).dropna().abs().max()
    add("B2", b2_resid < 1e-12, f"xr identity max |resid| = {b2_resid:.3e}")

    # --- B3: dp range ---
    dp_mean = df["dp"].mean()
    dp_std = df["dp"].std()
    add(
        "B3",
        -4.5 < dp_mean < -2.5 and 0.1 < dp_std < 0.6,
        f"dp mean={dp_mean:.3f} (expect ~-3.1), std={dp_std:.3f} (expect ~0.3)",
    )

    # --- B4: spr positive on average ---
    spr_mean = df["spr"].mean()
    add("B4", spr_mean > 0.0, f"spr mean = {spr_mean:.4f} (target > 0)")

    # --- C1: duration sanity ---
    g = 1.05
    n = N_BOND
    D5 = (1 - g ** (-n)) / (1 - g ** (-1))
    # Macaulay duration of a 20yr 5% par bond is ~13.085, NOT ~12.5 (handoff ref off).
    add(
        "C1",
        12.5 < D5 < 13.5,
        f"Duration at Y=5%, n=20: {D5:.3f} (textbook Macaulay ~13.085 for 20yr 5% par)",
    )
    # Convexity: lower yield -> higher duration
    D2 = (1 - 1.02 ** (-n)) / (1 - 1.02 ** (-1))
    D10 = (1 - 1.10 ** (-n)) / (1 - 1.10 ** (-1))
    add(
        "C1b",
        D2 > D5 > D10,
        f"Duration convexity: D(2%)={D2:.3f} > D(5%)={D5:.3f} > D(10%)={D10:.3f}",
    )

    # --- C2: xb std comparable to CCV reference (6.543 pp) ---
    xb_std = df["xb"].std()
    add(
        "C2",
        0.02 < xb_std < 0.15,
        f"sigma(xb) = {xb_std:.4f} = {xb_std*100:.2f} pp (CCV ref 6.543 pp)",
    )

    # --- C3: xb negative when AAA yields rise sharply (1980 Volcker) ---
    if 1980 in aaa_jan.index and 1981 in aaa_jan.index:
        if 1981 in df.index:
            # 1981 row holds xb realized over [1980, 1981]
            rise = aaa_jan.loc[1981] - aaa_jan.loc[1980]
            xb_1981 = df.loc[1981, "xb"]
            add(
                "C3_1980",
                rise > 0 and xb_1981 < 0,
                f"AAA 1980->1981 rise={rise:+.2f}pp -> xb={xb_1981*100:+.2f}%",
            )

    return results


# ============================================================
# 6. MAIN
# ============================================================
def main():
    chap26 = load_chap26()
    aaa_jan = load_aaa_january()

    # Build full-history table first (for sensitivity scripts), then filter
    df_full, diag_full = build_var(chap26, aaa_jan)

    # Option A: 1919-2011 (T=93), drop NaN rows (which are pre-1919 for spr/xb)
    df = df_full.loc[SAMPLE_START:SAMPLE_END].dropna()
    diag = diag_full.loc[df.index]

    print(f"Dataset: years {df.index.min()}..{df.index.max()}, T={len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("First 3 rows:")
    print(df.head(3).to_string())
    print()
    print("Last 3 rows:")
    print(df.tail(3).to_string())
    print()

    # Inline §4.A/B/C tests
    print("=" * 64)
    print("INLINE TESTS (§4.A/B/C)")
    print("=" * 64)
    results = run_inline_tests(df, diag, chap26, aaa_jan)
    failures = []
    for rid, status, msg in results:
        print(f"  {rid:8s} {status}: {msg}")
        if status != "PASS":
            failures.append(rid)
    print()

    # Sample stats
    print("=" * 64)
    print("SAMPLE STATISTICS (Option A: 1920-2011 effective, T=92)")
    print("=" * 64)
    for col in df.columns:
        s = df[col]
        print(
            f"  {col:4s}: mean={s.mean():+.4f}, std={s.std():.4f}, "
            f"min={s.min():+.4f}, max={s.max():+.4f}"
        )
    print()

    if failures:
        print(f"*** {len(failures)} INLINE TEST FAILURES — NOT SAVING ***")
        for r in failures:
            print(f"  {r}")
        sys.exit(1)

    df.index.name = "year"
    df.to_csv(OUT_CSV)
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
