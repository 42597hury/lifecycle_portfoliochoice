"""
Build var_dataset_real.csv — real-yield variant of the CCV pipeline.

Five inflation-expectation methods available via --method:

  fisher_ar1       Campbell-Viceira (2002, Ch.3): AR(1) on inflation
                   estimated on the FULL sample, then closed-form cumulative
                   n-period forecast for the long bond and one-step for the
                   short bill. Maturity-matched.

  fisher_var       VAR(1) on (pi, y_1, y_n) jointly, used to forecast
                   cumulative expected inflation at the appropriate horizon
                   for each maturity. Stand-in for CSV (2017) affine.
                   Maturity-matched.

  homer_sylla      Schmelzing (2022) / RRS (2022) headline. 7-year backward
                   progressively-weighted average, EXCLUDING current year t.
                   Weights from data.inflation_expectations.WEIGHTS_HOMER_SYLLA
                   (w_1 = 0.33 on t-1, exponential decay, sum = 1).
                   NOT maturity-matched: same single E[pi] applied to all yields.

  hamilton         Hamilton-Hatzius-Harris-West (2016). AR(1) refit each year
                   on rolling 30-year window, OOS one-step-ahead forecast.
                   NOT maturity-matched: same single E[pi] applied to all yields.

  eichengreen      Eichengreen (2015). 7-year equal-weighted backward average,
                   INCLUDING current year t. NOT maturity-matched.

Both methods set the inflation-risk premium lambda_n = 0 in the headline.
A constant lambda_n offset can be applied via --lambda-bp.

Sample: 1919-2011 (Option A window; same as data/build_var_dataset.py).
Effective T = 92 after losing year 1 to inflation differencing.

Output:
  data/var_dataset_real.csv  with columns:
    year, y_1, spr, dp, xr, xb, pi, y_1_nom, y_n_nom, lambda_n, method

  Note: y_1, spr are REAL yields here. xr, xb are unchanged from nominal
  build (inflation-invariant per CCV §2.1.5). Diagnostic columns kept for
  audit.

Usage:
  python data/build_var_dataset_real.py --method fisher_ar1
  python data/build_var_dataset_real.py --method fisher_var
  python data/build_var_dataset_real.py --method fisher_ar1 --lambda-bp 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THESIS = HERE / "Thesisdata"
CHAP26_XLSX = THESIS / "chapt26 (2).xlsx"
AAA_CSV = THESIS / "AAA.csv"

SAMPLE_START = 1919   # AAA.csv first January obs
SAMPLE_END = 2011     # chap_26 R last obs
N_BOND = 20           # CCV constant-duration maturity


# ---------------------------------------------------------------------------
# Data loading (mirrors build_var_dataset.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inflation forecast — AR(1)
# ---------------------------------------------------------------------------

def estimate_ar1_inflation(pi_series: pd.Series):
    """Fit pi_{t+1} = a + b*pi_t + eps. Returns (a, b, sigma_eps, mu_pi)."""
    pi = pi_series.dropna()
    pi_t = pi.values[:-1]
    pi_t1 = pi.values[1:]
    b = np.cov(pi_t1, pi_t)[0, 1] / np.var(pi_t, ddof=1)
    a = pi_t1.mean() - b * pi_t.mean()
    mu = a / (1.0 - b)
    sigma_eps = float((pi_t1 - a - b * pi_t).std(ddof=1))
    return float(a), float(b), sigma_eps, float(mu)


def expected_cum_inflation_ar1(pi_t: np.ndarray, n: int, a: float, b: float):
    """Expected n-period CUMULATIVE-AVERAGE inflation pi_bar_n given pi_t.

    pi_{t+j} = a + b*pi_{t+j-1} + eps; iterating:
      E_t[pi_{t+j}] = (1 - b^j) * mu + b^j * pi_t,  mu = a/(1-b)
      pi_bar_n     = (1/n) * sum_{j=1..n} E_t[pi_{t+j}]
                   = mu + ((pi_t - mu) / n) * b*(1 - b^n)/(1 - b)
    """
    mu = a / (1.0 - b)
    pi_t = np.asarray(pi_t, dtype=float)
    if abs(1.0 - b) < 1e-12:
        # Degenerate: pi is a random walk with drift. Cumulative average
        # explodes; fall back to the last observed pi.
        return pi_t.copy()
    geom_sum = b * (1.0 - b ** n) / (1.0 - b)
    return mu + ((pi_t - mu) / n) * geom_sum


# ---------------------------------------------------------------------------
# Inflation forecast — VAR(1) on (pi, y_1, y_n)
# ---------------------------------------------------------------------------

def estimate_var1_inflation(pi: pd.Series, y_1: pd.Series, y_n: pd.Series):
    """VAR(1): z_{t+1} = c + A z_t + v.  z = (pi, y_1, y_n).

    Returns (c, A, Sigma_v, mu_z) where mu_z = (I-A)^{-1} c is the
    unconditional mean (used for forecast horizon analytics).
    """
    df = pd.concat([pi, y_1, y_n], axis=1).dropna()
    Z = df.to_numpy()
    z_bar = Z.mean(axis=0)
    Zd = Z - z_bar
    Y = Zd[1:, :]
    X = Zd[:-1, :]
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = A.T  # convention: rows = equations, columns = lagged regressors
    c = (np.eye(3) - A) @ z_bar    # mean-pinned per CCV §2.2.μ
    resid = Y - X @ A.T
    Sigma_v = (resid.T @ resid) / (Y.shape[0] - X.shape[1])
    return c, A, Sigma_v, z_bar


def expected_cum_inflation_var(pi_t: np.ndarray, y_1_t: np.ndarray, y_n_t: np.ndarray,
                                n: int, c: np.ndarray, A: np.ndarray):
    """Cumulative E_t[pi_bar_n] under the VAR(1) on (pi, y_1, y_n).

    E_t[z_{t+j}] = sum_{k=0..j-1} A^k @ c  +  A^j @ z_t
    """
    z_t = np.column_stack([pi_t, y_1_t, y_n_t])
    T = z_t.shape[0]
    cum_sum_pi = np.zeros(T)
    A_pow = np.eye(3)         # A^0
    cum_const = np.zeros(3)   # sum_{k=0..j-1} A^k @ c
    for j in range(1, n + 1):
        cum_const = cum_const + A_pow @ c   # add A^{j-1} @ c
        A_pow = A_pow @ A                    # advance to A^j
        z_forecast_j = (cum_const + (z_t @ A_pow.T))   # shape (T, 3)
        cum_sum_pi += z_forecast_j[:, 0]
    return cum_sum_pi / n


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def clm_bond_return_real(y_n_real_pct, n=N_BOND):
    """CLM constant-duration bond return on REAL yields.

    Uses the same formula as the nominal version, but with real yields,
    so the resulting return is real.
    """
    g = 1.0 + y_n_real_pct / 100.0
    # Guard against negative real yields making the duration formula degenerate
    # (which doesn't happen empirically but worth a sanity check)
    D_n = (1.0 - g**(-n)) / (1.0 - g**(-1))
    y_n = np.log1p(y_n_real_pct / 100.0)
    r_n_next = D_n * y_n - (D_n - 1.0) * y_n.shift(-1)
    return r_n_next.shift(1), D_n, y_n


def build(method: str, lambda_bp: float = 0.0, verbose: bool = True):
    chap = load_chap26()
    aaa_jan = load_aaa_january()

    # Build the auxiliary aligned table
    df = chap.copy()
    df["AAA"] = aaa_jan.reindex(df.index)
    df["pi"] = np.log(df["CPI"] / df["CPI"].shift(1))

    # Nominal yields (log)
    df["y_1_nom"] = np.log1p(df["R"] / 100.0)
    df["y_n_nom"] = np.log1p(df["AAA"] / 100.0)

    # ─── Inflation forecast ──────────────────────────────────────────────
    # Estimate on the maximum sample available (1872-2015 for chap_26 CPI),
    # but use the forecasts only for the SAMPLE_START..SAMPLE_END subset.
    pi_full = df["pi"].dropna()

    if method == "fisher_ar1":
        a, b, sigma_eps, mu = estimate_ar1_inflation(pi_full)
        if verbose:
            print(f"AR(1) inflation: pi_t+1 = {a:+.5f} + {b:+.4f}*pi_t + eps")
            print(f"  implied mu_pi = {mu*100:+.2f}pp, sigma_eps = {sigma_eps*100:.2f}pp")
            print(f"  fitted sample: {pi_full.index.min()}-{pi_full.index.max()}, T={len(pi_full)}")

        # 1-period expected inflation: pi_bar_1 = a + b*pi_t (= AR(1) forecast)
        pi_bar_1 = a + b * df["pi"].values   # at row T using pi[T]; forecasts pi[T+1]
        # n=20 cumulative expected: average over horizons 1..20
        pi_bar_n = expected_cum_inflation_ar1(df["pi"].values, n=N_BOND, a=a, b=b)

        df["E_pi_1"] = pi_bar_1     # expected inflation over [t, t+1] (for short)
        df["E_pi_n"] = pi_bar_n     # cumulative expected inflation [t, t+n] (for long bond)
        df["forecast_method"] = "fisher_ar1"

    elif method in ("homer_sylla", "hamilton", "eichengreen"):
        # Import works whether script is run as module or directly
        try:
            from data.inflation_expectations import (
                homer_sylla as _f_hs, hamilton as _f_hh, eichengreen as _f_eich,
                cumulative_ar1_static as _f_cum, cumulative_ar1_rolling as _f_cum_roll,
            )
        except ModuleNotFoundError:
            import sys as _sys
            _sys.path.insert(0, str(HERE))
            from inflation_expectations import (
                homer_sylla as _f_hs, hamilton as _f_hh, eichengreen as _f_eich,
                cumulative_ar1_static as _f_cum, cumulative_ar1_rolling as _f_cum_roll,
            )
        # Each published method gives a 1-period inflation expectation for the
        # BILL. For the long BOND, we use a cumulative-n expectation that's
        # method-consistent where possible:
        #   hamilton     -> rolling AR(1) cumulative (method-consistent)
        #   homer_sylla  -> static AR(1) cumulative (no natural forward
        #                   extension of a backward weighted average; using
        #                   AR(1) is a defensible compromise)
        #   eichengreen  -> static AR(1) cumulative (same reason)
        #
        # Applying the 1-period expectation uniformly to BOTH maturities
        # (papers' literal usage) gives sigma(xb) ~ 35pp because CLM
        # duration amplifies forecast noise -- the papers don't expose this
        # because they study one maturity at a time.
        if method == "homer_sylla":
            E_pi_1 = _f_hs(pi_full)
            E_pi_n = _f_cum(pi_full, n=N_BOND)
            bond_method_note = "static AR(1) cumulative (backward avg has no forward extension)"
        elif method == "hamilton":
            E_pi_1 = _f_hh(pi_full, window=30)
            E_pi_n = _f_cum_roll(pi_full, n=N_BOND, window=30)   # method-consistent
            bond_method_note = "rolling AR(1) cumulative (method-consistent)"
        else:   # eichengreen
            E_pi_1 = _f_eich(pi_full)
            E_pi_n = _f_cum(pi_full, n=N_BOND)
            bond_method_note = "static AR(1) cumulative (backward avg has no forward extension)"

        df["E_pi_1"] = E_pi_1.reindex(df.index)
        df["E_pi_n"] = E_pi_n.reindex(df.index)
        df["forecast_method"] = method
        if verbose:
            print(f"{method}: bill leg uses paper's method; bond leg uses {bond_method_note}")
            n_valid = int(df["E_pi_1"].notna().sum())
            print(f"  E_pi_1 valid observations: {n_valid}")
            print(f"  Sample (E_pi_1, E_pi_n) values:")
            head_yrs = df.dropna(subset=["E_pi_1"]).index[:3].tolist()
            tail_yrs = df.dropna(subset=["E_pi_1"]).index[-3:].tolist()
            for y in head_yrs + tail_yrs:
                print(f"    [{y}] E_pi_1 = {df.loc[y, 'E_pi_1']*100:+.3f}pp,  "
                      f"E_pi_n = {df.loc[y, 'E_pi_n']*100:+.3f}pp")

    elif method == "fisher_var":
        # Joint VAR(1) on (pi, y_1_nom, y_n_nom) — uses the yield-curve
        # cross-section to inform the inflation forecast
        c, A, Sigma_v, mu_z = estimate_var1_inflation(
            pi_full,
            df["y_1_nom"].dropna(),
            df["y_n_nom"].dropna(),
        )
        if verbose:
            print(f"VAR(1) on (pi, y_1, y_n):")
            print(f"  c = {c}")
            print(f"  A =\n{A}")
            print(f"  diag Sigma_v = {np.diag(Sigma_v)}")
            print(f"  mu_z = {mu_z}")
            eigs = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
            print(f"  max |eig(A)| = {eigs[0]:.4f}  ({'STATIONARY' if eigs[0] < 1.0 else 'NON-STATIONARY'})")

        pi_t = df["pi"].values
        y_1_t = df["y_1_nom"].values
        y_n_t = df["y_n_nom"].values
        pi_bar_1 = expected_cum_inflation_var(pi_t, y_1_t, y_n_t, n=1, c=c, A=A)
        pi_bar_n = expected_cum_inflation_var(pi_t, y_1_t, y_n_t, n=N_BOND, c=c, A=A)

        df["E_pi_1"] = pi_bar_1
        df["E_pi_n"] = pi_bar_n
        df["forecast_method"] = "fisher_var"

    else:
        raise ValueError(f"unknown method: {method!r}")

    lambda_n = lambda_bp / 1e4   # bp to decimal log
    df["lambda_n"] = lambda_n

    # ─── Real yields via Fisher decomposition ────────────────────────────
    df["y_1_real"] = df["y_1_nom"] - df["E_pi_1"] - lambda_n
    df["y_n_real"] = df["y_n_nom"] - df["E_pi_n"] - lambda_n

    # ─── VAR variables on the new real-yield basis ───────────────────────
    # Note: xr, xb are inflation-invariant (CCV §2.1.5), so they can still be
    # constructed from observed nominal data. Equivalently, xr_real = xr_nom.
    df["dp"] = np.log(df["D"]) - np.log(df["P"])
    df["spr"] = df["y_n_real"] - df["y_1_real"]   # NOTE: real spread

    # rtb under real-yield setup is mechanically y_1_real lagged (deterministic
    # at t in real terms), but we keep a column for diagnostic purposes.
    df["rtb_real_implied"] = df["y_1_real"].shift(1)

    # xr unchanged: nominal stock - nominal bill, equals real-real by §2.1.5
    R_stk = (df["P"] + df["D"]) / df["P"].shift(1)
    df["xr"] = np.log(R_stk) - df["y_1_nom"].shift(1)

    # xb under REAL yields: bond return constructed via CLM on real long yield
    # We pass in y_n_real_pct = (exp(y_n_real)-1)*100 to use the same formula
    Y_n_real_pct = (np.exp(df["y_n_real"]) - 1.0) * 100.0
    r_n_real_t1, D_n, _ = clm_bond_return_real(Y_n_real_pct, n=N_BOND)
    # xb = excess REAL bond return over real bill = r_n_real - y_1_real
    df["xb"] = r_n_real_t1 - df["y_1_real"].shift(1)

    # ─── Build output ────────────────────────────────────────────────────
    out_cols = ["y_1_real", "spr", "dp", "xr", "xb"]
    out = df.loc[SAMPLE_START:SAMPLE_END, out_cols].rename(columns={"y_1_real": "y_1"})
    diag_cols = ["pi", "y_1_nom", "y_n_nom", "y_n_real", "rtb_real_implied", "lambda_n", "E_pi_1", "E_pi_n"]
    diag = df.loc[SAMPLE_START:SAMPLE_END, diag_cols]

    out = out.dropna()
    diag = diag.loc[out.index]
    out["lambda_n"] = lambda_n
    out["method"] = method

    return out, diag


def main():
    p = argparse.ArgumentParser(description="Build var_dataset_real.csv")
    p.add_argument("--method",
                   choices=["fisher_ar1", "fisher_var",
                            "homer_sylla", "hamilton", "eichengreen"],
                   default="fisher_ar1")
    p.add_argument("--lambda-bp", type=float, default=0.0,
                   help="constant inflation risk premium in basis points (decimal log)")
    p.add_argument("--out", default=str(HERE / "var_dataset_real.csv"))
    args = p.parse_args()

    out, diag = build(method=args.method, lambda_bp=args.lambda_bp, verbose=True)

    print()
    print("=" * 72)
    print(f"REAL-YIELD VAR DATASET   (method={args.method}, lambda={args.lambda_bp}bp)")
    print("=" * 72)
    print(f"Years: {out.index.min()}-{out.index.max()},  T={len(out)}")
    print()
    print("Sample stats:")
    for col in ["y_1", "spr", "dp", "xr", "xb"]:
        s = out[col]
        print(f"  {col:5s} mean={s.mean():+.4f} ({s.mean()*100:+.2f}%), "
              f"std={s.std():.4f}")
    print()
    print("Real-yield realism check:")
    print(f"  E[y_1_real] = {out['y_1'].mean()*100:+.2f}pp  (Cocco-Gomes-Maenhout: ~+1-2pp)")
    print(f"  std(y_1_real) = {out['y_1'].std()*100:.2f}pp")
    print(f"  E[spr_real] = {out['spr'].mean()*100:+.2f}pp")
    print(f"  E[xb_real]+0.5 sigma^2 = {(out['xb'].mean() + 0.5*out['xb'].var())*100:+.2f}pp")
    print(f"  sigma(xb_real) = {out['xb'].std()*100:.2f}pp")
    print()

    out.to_csv(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
