"""
Build var_dataset.csv — REAL-YIELD HEADLINE pipeline.

Replaces the old nominal pipeline (preserved at build_var_dataset_nominal.py).
Yields are deflated by Fisher decomposition: y_real = y_nom - E[pi].

Bill leg E[pi]:    Homer-Sylla (2005) / Schmelzing (2022) / RRS (2022).
                   7-year backward progressively-weighted realized inflation,
                   excluding current year t. Weights from
                   data.inflation_expectations.WEIGHTS_HOMER_SYLLA
                   (w_1 = 0.33 on t-1, exponential decay, sum to 1).

Bond leg E[pi]:    Static AR(1) cumulative-n forecast (analytical closed form).
                   Smooth and approximately maturity-matched. The Homer-Sylla
                   backward average has no natural forward extension; using
                   AR(1) here is the standard compromise.

Inflation risk premium:  zero (lambda_n = 0). Standard simplification when
                   TIPS data is unavailable; see CV (2002), HHHW (2016),
                   Schmelzing (2022).

Equity predictor:  cape = -log(Shiller CAPE), January-of-year, from ie_data.xls.
                   CAPE = real price / 10-year average real earnings (Shiller's
                   cyclically-adjusted P/E ratio). -log(CAPE) is the log earnings
                   yield, the canonical equity-predictability variable in
                   Campbell-Shiller (1988) and CCV w8566 (under their dp).

Sample:           1920-2011 (Option A: AAA from 1919 + 1 year lost to
                  inflation differencing).  T = 92.

Data sources:
  - chap_26 (Shiller's Ch.26 update, chapt26 (2).xlsx, sheet 'Data'):
      P (col B), D (col C), R (col E), CPI (col G).  January-of-year obs.
  - FRED Moody's AAA (AAA.csv): January-of-year long-bond yield.
  - Shiller ie_data.xls: monthly CAPE; January-of-year extracted.

Output columns (data/var_dataset.csv):
  year, cape, spr, y_1, xr, xb

  cape  = -log(Shiller CAPE)              equity predictor
  spr   = real term spread                = y_n_real - y_1_real
  y_1   = real short yield                = log(1+R/100) - E[pi_{t+1}]
  xr    = excess log stock return         = log((P+D)/P_lag) - y_1_nom_lag
                                            (inflation cancels in excess; CCV §2.1.5)
  xb    = excess log REAL bond return     = r_n_real - y_1_real_lag
                                            r_n_real = CLM constant-duration formula
                                            on real yields.

VAR partition under headline (FULL system):
  states  = (cape, spr, y_1)      on grid (3-state)
  returns = (xr, xb)              integrated

Ablation systems (reduced state):
  System 1: states = (y_1,)
  System 2: states = (spr, y_1)

The §2.2.r restriction (Phi_1[:, ret_cols] = 0) and §2.2.μ pinning
(Phi_0 = (I - Phi_1) mu_y_sample) are imposed by the estimator in
lifecycle.var, not by this script.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THESIS = HERE / "Thesisdata"
CHAP26_XLSX = THESIS / "chapt26 (2).xlsx"
AAA_CSV = THESIS / "AAA.csv"
IE_DATA_XLS = THESIS / "ie_data.xls"
OUT_CSV = HERE / "var_dataset.csv"

SAMPLE_START = 1919
SAMPLE_END = 2011
N_BOND = 20

# Inflation-risk premium (set to zero in the headline; sensitivity via
# scripts/sensitivity_var_window.py if needed)
LAMBDA_BP = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Raw data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_chap26():
    raw = pd.read_excel(CHAP26_XLSX, sheet_name="Data", header=None, skiprows=8)
    raw.columns = ["year", "P", "D", "E", "R", "RLONG", "CPI"] + [
        f"col{i}" for i in range(7, raw.shape[1])
    ]
    df = raw[["year", "P", "D", "R", "CPI"]].dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    for c in ("P", "D", "R", "CPI"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("year")


def load_aaa_january():
    aaa = pd.read_csv(AAA_CSV)
    aaa["observation_date"] = pd.to_datetime(aaa["observation_date"])
    aaa["AAA"] = pd.to_numeric(aaa["AAA"], errors="coerce")
    aaa["year"] = aaa["observation_date"].dt.year
    aaa["month"] = aaa["observation_date"].dt.month
    return aaa[aaa["month"] == 1].set_index("year")["AAA"]


def load_cape_january():
    """Load Shiller CAPE (column 'CAPE' in ie_data.xls), January-of-year only.

    ie_data Date is encoded as YYYY.MM (e.g. 1871.01 = January 1871). Extract
    rows where the fractional part is 0.01 (January).
    """
    shiller = pd.read_excel(IE_DATA_XLS, sheet_name="Data", header=7)
    shiller = shiller.rename(columns={shiller.columns[0]: "date_raw"})
    shiller = shiller.dropna(subset=["date_raw"])
    shiller = shiller[
        shiller["date_raw"].apply(lambda x: isinstance(x, (int, float)))
    ]
    shiller["date_raw"] = shiller["date_raw"].astype(float)
    shiller["year"] = shiller["date_raw"].astype(int)
    shiller["month_frac"] = (shiller["date_raw"] - shiller["year"]).round(2)
    jan = shiller[np.isclose(shiller["month_frac"], 0.01)].copy()
    jan["CAPE"] = pd.to_numeric(jan["CAPE"], errors="coerce")
    return jan.set_index("year")["CAPE"]


# ─────────────────────────────────────────────────────────────────────────────
# CLM constant-duration bond return on REAL yields
# ─────────────────────────────────────────────────────────────────────────────

def clm_bond_return_real(y_n_real_pct: pd.Series, n: int = N_BOND):
    """CLM constant-duration formula applied to real yields. Returns
    (r_n_at_t1, D_n, y_n_log) where r_n_at_t1[T] is the realized real
    bond return over [T-1, T]. D_n is the per-period Macaulay duration."""
    g = 1.0 + y_n_real_pct / 100.0
    D_n = (1.0 - g**(-n)) / (1.0 - g**(-1))
    y_n = np.log1p(y_n_real_pct / 100.0)
    r_n_next = D_n * y_n - (D_n - 1.0) * y_n.shift(-1)
    return r_n_next.shift(1), D_n, y_n


# ─────────────────────────────────────────────────────────────────────────────
# Build the dataset
# ─────────────────────────────────────────────────────────────────────────────

def build():
    # Inflation expectations module
    sys.path.insert(0, str(HERE))
    from inflation_expectations import (
        homer_sylla as f_hs,
        cumulative_ar1_static as f_cum_ar1,
    )

    chap = load_chap26()
    aaa_jan = load_aaa_january()
    cape_jan = load_cape_january()

    df = chap.copy()
    df["AAA"] = aaa_jan.reindex(df.index)
    df["CAPE"] = cape_jan.reindex(df.index)

    # Auxiliary / nominal yields
    df["pi"] = np.log(df["CPI"] / df["CPI"].shift(1))
    df["y_1_nom"] = np.log1p(df["R"] / 100.0)
    df["y_n_nom"] = np.log1p(df["AAA"] / 100.0)

    # Inflation expectations
    pi_full = df["pi"].dropna()
    E_pi_1 = f_hs(pi_full)                              # bill: 7-yr Homer-Sylla weighted avg
    E_pi_n = f_cum_ar1(pi_full, n=N_BOND)               # bond: AR(1) cumulative-n
    df["E_pi_1"] = E_pi_1.reindex(df.index)
    df["E_pi_n"] = E_pi_n.reindex(df.index)

    lambda_n = LAMBDA_BP / 1e4
    df["y_1_real"] = df["y_1_nom"] - df["E_pi_1"] - lambda_n
    df["y_n_real"] = df["y_n_nom"] - df["E_pi_n"] - lambda_n

    # State variables
    df["cape"] = -np.log(df["CAPE"])                    # equity predictor: -log(Shiller CAPE)
    df["spr"] = df["y_n_real"] - df["y_1_real"]

    # Excess returns. Inflation cancels in excess constructions (CCV §2.1.5)
    R_stk = (df["P"] + df["D"]) / df["P"].shift(1)
    df["xr"] = np.log(R_stk) - df["y_1_nom"].shift(1)

    # Real bond return via CLM on REAL long yield. xb = excess real bond
    # return over real bill (the bill is real-risk-free with rate y_1_real_lag)
    Y_n_real_pct = (np.exp(df["y_n_real"]) - 1.0) * 100.0
    r_n_real_at_t1, _, _ = clm_bond_return_real(Y_n_real_pct, n=N_BOND)
    df["xb"] = r_n_real_at_t1 - df["y_1_real"].shift(1)

    # Output columns ordered as agreed: (cape, spr, y_1, xr, xb)
    out_cols = ["cape", "spr", "y_1_real", "xr", "xb"]
    out = df.loc[SAMPLE_START:SAMPLE_END, out_cols].rename(columns={"y_1_real": "y_1"})
    diag_cols = ["pi", "y_1_nom", "y_n_nom", "y_n_real", "E_pi_1", "E_pi_n", "CAPE"]
    diag = df.loc[SAMPLE_START:SAMPLE_END, diag_cols]

    out = out.dropna()
    diag = diag.loc[out.index]

    return out, diag


# ─────────────────────────────────────────────────────────────────────────────
# Inline verification (light §4.A/B/C subset)
# ─────────────────────────────────────────────────────────────────────────────

def verify(out: pd.DataFrame, diag: pd.DataFrame) -> list:
    """Return list of (test_id, passed, message). Hard-fails the build if any
    fail. Sample-stat sanity only — full §4 suite is in scripts/."""
    results = []
    def add(tid, ok, msg):
        results.append((tid, ok, msg))

    # T sanity
    add("T", len(out) == 92, f"T = {len(out)} (expected 92 for 1920-2011)")

    # No NaN
    add("no_nan", not out.isnull().any().any(), f"no NaN: {not out.isnull().any().any()}")

    # Columns
    expected_cols = ["cape", "spr", "y_1", "xr", "xb"]
    add("columns", list(out.columns) == expected_cols,
        f"columns = {list(out.columns)}, expected {expected_cols}")

    # Sample-stat sanity
    add("y_1_real_mean", -0.005 < out["y_1"].mean() < 0.05,
        f"E[y_1_real] = {out['y_1'].mean()*100:+.2f}pp (expect ~+1-2pp per literature)")
    add("spr_real_mean", out["spr"].mean() > 0,
        f"E[spr_real] = {out['spr'].mean()*100:+.2f}pp (expect > 0)")
    # cape = -log(Shiller CAPE). Shiller CAPE typically in 10-30 range; log(CAPE) in 2.3-3.4;
    # so -log(CAPE) typically in -3.4 to -2.3. Old cy variable mean was ~-2.99 on 1963-2025.
    add("cape_mean", -3.5 < out["cape"].mean() < -2.0,
        f"E[cape] = {out['cape'].mean():+.3f} (expect roughly -3.0 to -2.5; -log of Shiller CAPE)")
    add("xb_std_sane",  out["xb"].std() < 0.20,
        f"sigma(xb) = {out['xb'].std()*100:.2f}pp (expect < 20pp; >35pp would indicate inflation-noise blow-up)")

    return results


def main():
    print("=" * 70)
    print("REAL-YIELD HEADLINE BUILD  (build_var_dataset.py)")
    print("=" * 70)
    out, diag = build()
    print(f"Years: {out.index.min()}..{out.index.max()},  T = {len(out)}")
    print()
    print("Sample statistics:")
    for col in out.columns:
        s = out[col]
        print(f"  {col:5s}: mean={s.mean():+.5f} ({s.mean()*100:+.3f}%), "
              f"std={s.std():.5f}")
    print()

    print("Inline tests:")
    results = verify(out, diag)
    n_fail = 0
    for tid, ok, msg in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  {tid:18s} {mark}: {msg}")
        if not ok:
            n_fail += 1
    print()

    if n_fail:
        print(f"*** {n_fail} INLINE TEST FAILURES — NOT SAVING ***")
        sys.exit(1)

    out.index.name = "year"
    out.to_csv(OUT_CSV)
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
