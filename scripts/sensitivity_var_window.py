"""
Sensitivity diagnostic: how do VAR estimates shift across plausible
resolutions of the AAA-1919-start vs T=141 conflict in CCV_RETURN_IMPLEMENT.md?

Compares four datasets:
  A         (locked baseline) — 1920-2011, AAA throughout (T=92)
  A_RLONG   1920-2011, chap_26 RLONG instead of AAA (isolates yield-source effect)
  C         1872-2011, splice: chap_26 RLONG pre-1919 + AAA from 1919
  D         1872-2011, chap_26 RLONG throughout (isolates sample-extension effect)

Reports for each: T, sample window, Phi diagonals, Phi[i,i] persistence,
max |eig(Phi)|, equation R²s, key Sigma_v scalars.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lifecycle.var import estimate_var1_from_csv  # noqa: E402

THESIS = REPO / "data" / "Thesisdata"
CHAP26 = THESIS / "chapt26 (2).xlsx"
AAA_CSV = THESIS / "AAA.csv"
N_BOND = 20
COLUMNS = ["y_1", "spr", "dp", "rtb", "xr", "xb"]
STATE_INDICES = [0, 1, 2, 3]


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
    return (D_n * y_n - (D_n - 1.0) * y_n.shift(-1)).shift(1), y_n


def build_dataset(chap26, long_yield_pct, sample_start, sample_end):
    """long_yield_pct: pd.Series indexed by year, in percent."""
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
    r_n_t1, _ = clm_bond_return(df["LongY"], n=N_BOND)
    xb = r_n_t1 - y_1.shift(1)

    out = pd.DataFrame({"y_1": y_1, "spr": spr, "dp": dp, "rtb": rtb, "xr": xr, "xb": xb})
    out = out.loc[sample_start:sample_end].dropna()
    return out


def estimate_from_df(df_data):
    """Estimate restricted VAR(1) from an in-memory DataFrame."""
    # Mirror estimate_var1_from_csv but skip the CSV path; see var.py:191-268.
    columns = COLUMNS
    z_bar = df_data[columns].mean(axis=0).to_numpy()
    Z = df_data[columns].to_numpy() - z_bar
    Y = Z[1:, :]
    state_idx = np.asarray(STATE_INDICES, dtype=int)
    X = Z[:-1, state_idx]
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    n = len(columns)
    Phi = np.zeros((n, n), dtype=float)
    for k, j in enumerate(state_idx):
        Phi[:, j] = coeffs[k, :]
    Y_hat = X @ coeffs
    resid = Y - Y_hat
    dof = Y.shape[0] - X.shape[1]
    Omega = (resid.T @ resid) / dof
    r2 = {}
    for i, col in enumerate(columns):
        sse = float(np.sum((Y[:, i] - Y_hat[:, i]) ** 2))
        sst = float(np.sum(Y[:, i] ** 2))
        r2[col] = 1.0 - sse / max(sst, 1e-14)
    return {"z_bar": z_bar, "Phi": Phi, "Omega": Omega, "r2": r2, "T": len(df_data)}


def main():
    chap = load_chap26()
    aaa_jan = load_aaa_january()

    rlong = chap["RLONG"]  # already in percent

    # Build splice for Option C: pre-1919 RLONG, 1919+ AAA
    splice = rlong.copy()
    for y in aaa_jan.index:
        if y >= 1919:
            splice.loc[y] = aaa_jan.loc[y]

    cases = [
        ("A         (locked)", build_dataset(chap, aaa_jan, 1919, 2011),
         "1920-2011, AAA throughout"),
        ("A_RLONG   (yield)", build_dataset(chap, rlong, 1919, 2011),
         "1920-2011, chap_26 RLONG (isolates yield-source effect)"),
        ("C         (splice)", build_dataset(chap, splice, 1871, 2011),
         "1872-2011, RLONG pre-1919 + AAA from 1919 (isolates sample-extension under D1)"),
        ("D         (RLONG only)", build_dataset(chap, rlong, 1871, 2011),
         "1872-2011, chap_26 RLONG throughout (overrules D1)"),
    ]

    results = []
    for label, df, descr in cases:
        est = estimate_from_df(df)
        results.append((label, descr, est, df))

    # ============================================================
    # Report
    # ============================================================
    print("=" * 100)
    print("SENSITIVITY DIAGNOSTIC — VAR estimates across spec resolutions")
    print("=" * 100)
    print()
    for label, descr, est, df in results:
        print(f"  {label}: {descr}")
        print(f"    T={est['T']:3d}, window={df.index.min()}-{df.index.max()}")
    print()

    # Headline moments
    print("HEADLINE MOMENTS (means and std devs)")
    print("-" * 100)
    header = f"  {'variable':10s} | " + " | ".join(f"{lab[:18]:18s}" for lab, _, _, _ in results)
    print(header)
    print("-" * 100)
    for col_idx, col in enumerate(COLUMNS):
        line = f"  E[{col}]      "
        for _, _, est, df in results:
            line += f" | {est['z_bar'][col_idx]:+.4f}            "
        print(line[:len(header)])
        line = f"  std({col})   "
        for _, _, est, df in results:
            line += f" | {df[col].std():+.4f}            "
        print(line[:len(header)])
    print()

    # VAR diagonals
    print("AR(1) DIAGONALS  Phi[i, i] (persistence)")
    print("-" * 100)
    for col_idx, col in enumerate(COLUMNS):
        line = f"  {col:10s}"
        for _, _, est, _ in results:
            v = est["Phi"][col_idx, col_idx]
            line += f" | {v:+.4f}            "
        print(line[:len(header)])
    print()

    # Max |eig|
    print("STATIONARITY  max |eig(Phi)|")
    print("-" * 100)
    line = "  max|eig|   "
    for _, _, est, _ in results:
        eigs = np.sort(np.abs(np.linalg.eigvals(est["Phi"])))[::-1]
        line += f" | {eigs[0]:.4f}              "
    print(line[:len(header)])
    print()

    # Equation R²
    print("EQUATION R^2")
    print("-" * 100)
    for col in COLUMNS:
        line = f"  R2({col})    "
        for _, _, est, _ in results:
            line += f" | {est['r2'][col]:.4f}              "
        print(line[:len(header)])
    print()

    # Sigma_xx scalars (the eq.10 inputs)
    xr_idx = COLUMNS.index("xr")
    xb_idx = COLUMNS.index("xb")
    print("SIGMA_v RETURN-BLOCK SCALARS  (sigma2_xr, sigma2_xb, sigma_xrxb -- these enter eq.10)")
    print("-" * 100)
    for label_short, key in [("sigma2_xr  ", (xr_idx, xr_idx)),
                              ("sigma2_xb  ", (xb_idx, xb_idx)),
                              ("sigma_xrxb ", (xr_idx, xb_idx))]:
        line = f"  {label_short}"
        for _, _, est, _ in results:
            v = est["Omega"][key[0], key[1]]
            line += f" | {v:+.6e}        "
        print(line[:len(header)])
    print()

    # Markowitz at gamma=1 (computed from each estimate's Sigma_xx and z_bar_ret)
    print("MARKOWITZ alpha at gamma=1  (myopic optimum from this estimate)")
    print("-" * 100)
    line_s = "  alpha_s    "
    line_b = "  alpha_b    "
    for _, _, est, _ in results:
        Sigma_xx = est["Omega"][np.ix_([xr_idx, xb_idx], [xr_idx, xb_idx])]
        mu_x = est["z_bar"][[xr_idx, xb_idx]]
        sigma_x2 = np.diag(Sigma_xx)
        try:
            alpha = np.linalg.solve(Sigma_xx, mu_x + 0.5 * sigma_x2)
            line_s += f" | alpha_s={alpha[0]:+.3f}    "
            line_b += f" | alpha_b={alpha[1]:+.3f}    "
        except np.linalg.LinAlgError:
            line_s += " | singular           "
            line_b += " | singular           "
    print(line_s[:len(header)])
    print(line_b[:len(header)])
    print()

    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("  Compare A vs A_RLONG: yield-source effect (AAA corporate vs RLONG government)")
    print("  Compare A vs D:       both effects together (sample window + yield source)")
    print("  Compare A vs C:       sample extension only (under spliced-yield D1 stance)")
    print("  Compare C vs D:       yield-source effect at full sample length")


if __name__ == "__main__":
    main()
