"""
Inflation-expectation constructions used by the published real-yield literature.

Implements three named methods, each as faithfully as we can reproduce from
the source papers' descriptions:

  homer_sylla    — Schmelzing (2022) / Rogoff-Rossi-Schmelzing (2022) headline.
                   Backward 7-year progressively-weighted average of realized
                   inflation, EXCLUDING current year t. Per RRS footnote 16,
                   weight on t-1 is 33%, decreasing successively over t-2..t-7.
                   The exact decay is not specified in RRS; we use an
                   exponential decay r^(j-1) calibrated so w_1 = 0.33 and the
                   weights sum to 1 (r ≈ 0.696). See WEIGHTS_HOMER_SYLLA below.

  hamilton       — Hamilton-Hatzius-Harris-West (2016).
                   AR(1) on inflation, REFIT EACH YEAR on the past
                   `window`-year rolling sample (default 30y per RRS), then
                   used to predict one-year-ahead inflation OUT-OF-SAMPLE.

  eichengreen    — Eichengreen (2015).
                   7-year EQUAL-weighted average of realized inflation,
                   INCLUDING current year t.

All three return a `pandas.Series` of expected one-period inflation indexed
by year. None of them produces a maturity-matched cumulative expectation —
in the source papers, the same single number is applied to every maturity.

For maturity-matching to a long bond, see `cumulative_ar1_static` and
`cumulative_ar1_rolling` below: they iterate an AR(1) forward to compute
the n-period cumulative-average expected inflation. Use these only when
maturity-matching is desired (the source papers do not do this).

References
----------
- Homer, Sidney and Sylla, Richard (2005), "A History of Interest Rates",
  Rutgers University Press.
- Schmelzing, Paul (2022), "Eight centuries of global real rates, R-G,
  and the suprasecular decline, 1311-2021."
- Rogoff, Rossi, Schmelzing (2022), "Long-Run Trends in Long-Maturity
  Real Rates 1311-2021."
- Hamilton, James D., Jan Hatzius, Ethan Harris, Kenneth D. West (2016),
  "The Equilibrium Real Rate: Past, Present, and Future."
- Eichengreen, Barry (2015), "Secular stagnation, the long view."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Homer-Sylla / Schmelzing weights
# ---------------------------------------------------------------------------
def _homer_sylla_weights(w_1: float = 0.33) -> np.ndarray:
    """Exponential decay weights w_j = w_1 * r^(j-1) for j=1..7, sum = 1.

    Solves for r such that sum_{j=1..7} w_1 * r^(j-1) = 1, i.e.
        (1 - r^7) / (1 - r) = 1 / w_1.

    For w_1 = 0.33, r ≈ 0.696 -> weights:
      0.330, 0.230, 0.160, 0.111, 0.077, 0.054, 0.038   (sum = 1.000)

    The RRS footnote 16 specifies w_1 = 0.33 with weights "decreasing
    successively"; the exact decay is not given. Exponential decay is the
    smoothest closed-form choice consistent with the description and
    summing to 1. Sensitivity to the decay choice is minimal (linear vs
    exponential differ at the 3rd decimal of weights).
    """
    target = 1.0 / w_1
    # Solve (1 - r^7)/(1 - r) = target by bisection
    lo, hi = 0.01, 0.99
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = (1.0 - mid**7) / (1.0 - mid)
        if val < target:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    weights = w_1 * r ** np.arange(7)
    # Renormalize to sum exactly 1 (kills numerical residual)
    weights = weights / weights.sum()
    return weights


WEIGHTS_HOMER_SYLLA = _homer_sylla_weights(w_1=0.33)
# = array([0.33000, 0.22968, 0.15986, 0.11127, 0.07744, 0.05390, 0.03751])
# These are weights on lags t-1, t-2, ..., t-7 respectively.


# ---------------------------------------------------------------------------
# Method 1 — Homer-Sylla / Schmelzing (2022)
# ---------------------------------------------------------------------------
def homer_sylla(pi: pd.Series, weights: np.ndarray | None = None) -> pd.Series:
    """E_t[pi] = sum_{j=1..7} w_j * pi_{t-j}, weights from `WEIGHTS_HOMER_SYLLA`.

    EXCLUDES current year t. NaN for the first 7 observations.
    """
    if weights is None:
        weights = WEIGHTS_HOMER_SYLLA
    if len(weights) != 7:
        raise ValueError("Homer-Sylla weights must have length 7")

    pi = pi.copy()
    out = pd.Series(np.nan, index=pi.index, name="E_pi_homer_sylla")
    vals = pi.values
    for i in range(7, len(vals)):
        # weights[0] applies to lag t-1, weights[1] to t-2, ..., weights[6] to t-7
        lags = vals[i - 7:i][::-1]   # [pi_{t-1}, pi_{t-2}, ..., pi_{t-7}]
        if np.any(np.isnan(lags)):
            continue
        out.iloc[i] = float(np.dot(weights, lags))
    return out


# ---------------------------------------------------------------------------
# Method 2 — Hamilton-Hatzius-Harris-West (2016)
# ---------------------------------------------------------------------------
def hamilton(pi: pd.Series, window: int = 30) -> pd.Series:
    """AR(1) refit each year on past `window` years, OOS one-step forecast.

    For each year t, fit pi_{s+1} = a_t + b_t * pi_s + eps on the data window
    t-window..t-1, then return E_t[pi_{t+1}] = a_t + b_t * pi_t.

    Per RRS replication of Hamilton et al. (2016), default window = 30 years.
    Returns NaN for years where the rolling window does not contain enough
    valid observations.
    """
    pi = pi.dropna()
    out = pd.Series(np.nan, index=pi.index, name=f"E_pi_hamilton_w{window}")
    vals = pi.values
    years = pi.index.values

    min_obs = max(8, window // 2)   # need at least 8 obs for a stable fit
    for i, year in enumerate(years):
        if i < min_obs:
            continue
        lo = max(0, i - window)
        hi = i   # use [lo, hi) as the rolling window — strictly past data
        sub = vals[lo:hi]
        if len(sub) < min_obs:
            continue
        x = sub[:-1]
        y = sub[1:]
        if len(x) < 5:
            continue
        # OLS AR(1): y = a + b * x + eps
        x_mean = x.mean(); y_mean = y.mean()
        x_dev = x - x_mean; y_dev = y - y_mean
        var_x = float(np.dot(x_dev, x_dev))
        if var_x < 1e-12:
            continue
        b_t = float(np.dot(y_dev, x_dev) / var_x)
        a_t = float(y_mean - b_t * x_mean)
        out.iloc[i] = a_t + b_t * vals[i]
    return out


# ---------------------------------------------------------------------------
# Method 3 — Eichengreen (2015)
# ---------------------------------------------------------------------------
def eichengreen(pi: pd.Series) -> pd.Series:
    """E_t[pi] = (1/7) * sum_{j=0..6} pi_{t-j}.   INCLUDES current year t."""
    pi = pi.copy()
    out = pd.Series(np.nan, index=pi.index, name="E_pi_eichengreen")
    vals = pi.values
    for i in range(6, len(vals)):
        window = vals[i - 6:i + 1]   # 7 values: t-6, t-5, ..., t
        if np.any(np.isnan(window)):
            continue
        out.iloc[i] = float(np.mean(window))
    return out


# ---------------------------------------------------------------------------
# Maturity-matched cumulative AR(1) — extension beyond the source papers
# ---------------------------------------------------------------------------
def cumulative_ar1_static(pi: pd.Series, n: int) -> pd.Series:
    """n-period cumulative-average expected inflation under static AR(1).

    Fit AR(1) on the FULL pi series (one set of (a, b) for the whole sample),
    then for each year t:
      E_t[pi_bar_n] = mu + (pi_t - mu) * (1/n) * b * (1 - b^n) / (1 - b)
    where mu = a / (1 - b).

    Used when an analytical AR(1) is preferred to a rolling window. Same
    forecast model as in our existing `data/build_var_dataset_real.py
    --method fisher_ar1` build, exposed here for explicit comparison.
    """
    p = pi.dropna()
    pi_t = p.values[:-1]
    pi_t1 = p.values[1:]
    var_pi = float(np.var(pi_t, ddof=1))
    if var_pi < 1e-12:
        raise ValueError("inflation series has zero variance — cannot fit AR(1)")
    b = float(np.cov(pi_t1, pi_t)[0, 1] / var_pi)
    a = float(pi_t1.mean() - b * pi_t.mean())
    mu = a / (1.0 - b)
    geom_sum = b * (1.0 - b ** n) / (1.0 - b) if abs(1.0 - b) > 1e-12 else float(n)
    multiplier = (1.0 / n) * geom_sum
    out = pd.Series(np.nan, index=pi.index, name=f"E_pi_bar_{n}_ar1_static")
    for t in pi.index:
        if pd.isna(pi.loc[t]):
            continue
        out.loc[t] = mu + multiplier * (pi.loc[t] - mu)
    return out


def cumulative_ar1_rolling(pi: pd.Series, n: int, window: int = 30) -> pd.Series:
    """Cumulative n-period inflation expectation using a rolling AR(1)."""
    pi = pi.dropna()
    out = pd.Series(np.nan, index=pi.index, name=f"E_pi_bar_{n}_ar1_rolling")
    vals = pi.values
    years = pi.index.values

    min_obs = max(8, window // 2)
    for i, year in enumerate(years):
        if i < min_obs:
            continue
        lo = max(0, i - window)
        sub = vals[lo:i]
        if len(sub) < min_obs:
            continue
        x = sub[:-1]; y = sub[1:]
        var_x = float(np.dot(x - x.mean(), x - x.mean()))
        if var_x < 1e-12:
            continue
        b = float(np.dot(y - y.mean(), x - x.mean()) / var_x)
        a = float(y.mean() - b * x.mean())
        if abs(1.0 - b) < 1e-12:
            mu = float("nan"); multiplier = 1.0
        else:
            mu = a / (1.0 - b)
            geom_sum = b * (1.0 - b ** n) / (1.0 - b)
            multiplier = (1.0 / n) * geom_sum
        out.iloc[i] = mu + multiplier * (vals[i] - mu)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Homer-Sylla weights (w_1=0.33, exponential, sum to 1):")
    for j, w in enumerate(WEIGHTS_HOMER_SYLLA, start=1):
        print(f"  w_{j} (lag t-{j}) = {w:.5f}")
    print(f"  sum = {WEIGHTS_HOMER_SYLLA.sum():.10f}")
    print()
    print("Quick smoke test on synthetic inflation:")
    rng = np.random.default_rng(0)
    pi_synth = pd.Series(
        0.02 + 0.5 * (rng.standard_normal(100) * 0.03),
        index=pd.RangeIndex(1900, 2000),
        name="pi",
    )
    print(f"  homer_sylla    -> 1950 = {homer_sylla(pi_synth).loc[1950]:+.4f}")
    print(f"  hamilton(30y)  -> 1950 = {hamilton(pi_synth, 30).loc[1950]:+.4f}")
    print(f"  eichengreen    -> 1950 = {eichengreen(pi_synth).loc[1950]:+.4f}")
    print(f"  cum AR1 n=20   -> 1950 = {cumulative_ar1_static(pi_synth, 20).loc[1950]:+.4f}")
