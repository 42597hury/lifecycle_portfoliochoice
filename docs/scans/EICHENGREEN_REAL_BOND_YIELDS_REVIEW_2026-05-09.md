# Eichengreen real bond-yield calculation — methodology audit

**Date:** 2026-05-09
**Branch:** `jax-rewrite`
**Scope:** Verify the `eichengreen` (and `eichengreen_literal`) variants in
`data/build_var_dataset_real.py` are implemented correctly. Audit only — no
methodology change.

---

## 1. TL;DR

**PASS.** The `eichengreen` and `eichengreen_literal` variants in
`data/build_var_dataset_real.py` correctly implement the 7-year equal-weighted
backward inflation expectation from Eichengreen (2015), and the resulting real
yields display the canonical 1970s "negative real bill" pattern. The
maturity-mismatch caveat (long-bond leg uses static AR(1) cumulative for
`eichengreen`, vs. same 7-yr MA for `eichengreen_literal`) is documented in
the build script. **`var_dataset_real.csv` (active canonical) uses
`fisher_ar1`, NOT `eichengreen`** — the production VAR is therefore not
sensitive to this scan.

---

## 2. Methodology identification

The build script and helper module cite the source explicitly:

- `data/inflation_expectations.py:43`: `Eichengreen, Barry (2015), "Secular
  stagnation, the long view." [AEA P&P 105(5)]`
- `data/build_var_dataset_real.py:32-44` reproduces the verbatim Eichengreen
  quote: *"the yield on ten-year constant-maturity government bonds with and
  without adjustment for realized consumer price inflation, where the
  adjustment involves subtracting a seven-year moving average of CPI inflation
  (the current year and past six)."*

Implementation in `data/inflation_expectations.py:162-172`:

```python
def eichengreen(pi: pd.Series) -> pd.Series:
    """E_t[pi] = (1/7) * sum_{j=0..6} pi_{t-j}.   INCLUDES current year t."""
    ...
    for i in range(6, len(vals)):
        window = vals[i - 6:i + 1]   # 7 values: t-6, t-5, ..., t
        out.iloc[i] = float(np.mean(window))
```

This matches Eichengreen (2015) AEA P&P verbatim (current year + past six,
equal weights). The 1-period inflation expectation is then plugged into
`y_real = y_nom - E[pi] - lambda_n` at `build_var_dataset_real.py:334-335`.

Two variants are produced:

- `eichengreen` (`var_dataset_real_eichengreen.csv`) — bill leg uses the
  Eichengreen 7-yr MA, bond leg uses static AR(1) cumulative-20-period
  forecast. Maturity-matched compromise; Eichengreen himself doesn't
  maturity-match.
- `eichengreen_literal` (`var_dataset_real_eichengreen_literal.csv`) — same
  7-yr MA applied to **both** bill and 20-yr bond, exactly as Eichengreen
  states (he applies the 7-yr MA to a single 10-yr-constant-maturity yield in
  the cited paper).

---

## 3. Math verification

### 3.1 Inflation expectation (line-by-line)

Manual recompute at year 1980 (peak inflation episode):

| year | pi (log CPI growth) |
|-----:|-------------------:|
| 1974 | 8.97% |
| 1975 | 11.16% |
| 1976 | 6.50% |
| 1977 | 5.08% |
| 1978 | 6.61% |
| 1979 | 8.87% |
| 1980 | 13.02% |

Manual mean = **8.6041%**. Function output `eichengreen()[1980]` =
**8.6041%**. Match to 4 decimals.

### 3.2 Fisher decomposition

`y_1_real = y_1_nom - E_pi_1 - lambda_n` (build_var_dataset_real.py:334).

Convention check:
- `y_1_nom = log(1 + R/100)` (continuous compounding, line 215)
- `pi = log(CPI_t / CPI_{t-1})` (continuous compounding, line 212)
- Both legs are in **log/continuous units**, so the subtraction is dimensionally
  correct. (If one were APR and the other log, this would be a bug — it isn't.)
- `lambda_n` is in decimal log (basis points / 1e4), zero in headline.

### 3.3 Alignment

`E_pi_1[t]` is constructed using current-year `pi[t]` and past 6 years
(inclusive of t per Eichengreen). It is then subtracted from `y_1_nom[t]`. So
the "real yield at t" is `nominal[t] - backward-7-yr-MA[t]`. This is a
*backward-looking adaptive expectation* — fine per Eichengreen but it is NOT
truly forward expected inflation. The script does not pretend otherwise.

### 3.4 Bond leg

For the maturity-matched `eichengreen` variant, the long bond uses
`cumulative_ar1_static(pi, n=20)` (build line 276; helper at
`inflation_expectations.py:178-206`). Closed-form formula:

```
E_t[pi_bar_n] = mu + (pi_t - mu) * (1/n) * b * (1 - b^n) / (1 - b)
```

where (a, b) come from a single AR(1) fit on the full sample and mu = a/(1-b).
This is the same closed form as `expected_cum_inflation_ar1` at
`build_var_dataset_real.py:125-140` — verified to match by inspection.

### 3.5 CLM bond return

`clm_bond_return_real` (line 190) reuses CLM constant-duration formula on
the real long yield (taking `Y_n_real_pct = (exp(y_n_real) - 1) * 100` to feed
the formula in APR form, then converting back). This is consistent with the
nominal build's CLM convention.

---

## 4. Output sanity checks

`data/var_dataset_real_eichengreen.csv`:

| stat | y_1 | spr | xb |
|------|----:|----:|---:|
| mean | +1.27% | +2.34% | +2.12% |
| std | 2.90% | 3.00% | 8.56% |
| min | -6.08% | -3.51% | -17.05% |
| max | +7.32% | +8.79% | +27.14% |
| sample | 1920-2011 (T=92) | | |

Spot checks on the famous high-inflation episode 1975-1981 (real bill):

| year | y_1 (eichengreen) |
|-----:|-----------------:|
| 1975 | +0.93% |
| 1976 | -0.83% |
| 1977 | -1.09% |
| 1978 | +1.05% |
| 1979 | +3.07% |

Negative real bill yields in 1976-1977 — exactly the canonical pattern
(nominal Treasury yields lagged the inflation surge; backward-looking
expectations had caught up).

No NaN, no Inf, sample length 92 = SAMPLE_END(2011) - SAMPLE_START(1919) - 1
loss to inflation differencing. Mean real bill yield +1.27% is within
historical bounds. Mean real spread +2.34% is on the high side (this is the
maturity-mismatch artifact — bond uses anchored AR(1) ~2% expected π_bar_20,
bill uses adaptive 7-yr MA which can drift far from the AR(1) anchor, so the
spread inherits part of the inflation forecast disagreement).

For `eichengreen_literal` (same 7-yr MA on both legs), the spread compresses
to mean +1.26% / std 1.61%, but `xb` blows up to std 17.7% — duration
amplification of the forecast residual, exactly as the docstring warns.
Confirmed not a bug; documented.

Plot: `docs/scans/figures/eichengreen_real_yields_compare.png`.

---

## 5. Cross-check vs alternatives

`y_1` series at 1976 (peak negative real bill year):

| method | y_1 |
|--------|----:|
| eichengreen | -0.83% |
| eichengreen_literal | -0.83% (same bill leg) |
| hamilton | +0.65% |
| homer_sylla | -2.02% |
| fisher_ar1 (active) | +2.04% |

All non-AR(1) methods (which use realized-inflation history) give
mid-1970s real bill yields in [-2%, +1%]. `fisher_ar1` (mean-reverting AR(1))
is anchored to the long-run mean ~3% and never goes negative. Order of
magnitudes consistent across methods, as expected — methodologies disagree
about whether agents adjust expectations at the speed of realized inflation,
not about the algebra. **Internal consistency: PASS.**

---

## 6. Active-dataset linkage

```
data/var_dataset_real.csv  -->  method = fisher_ar1   (line 1 of CSV)
```

Confirmed by `df_active['method'].unique() == ['fisher_ar1']`. The canonical
VAR uses `fisher_ar1`, not `eichengreen`. The eichengreen CSVs are produced
by the same `data/build_var_dataset_real.py` builder for comparison /
robustness only; nothing in `lifecycle/` references them.

`scripts/compare_real_yield_methods.py` and `scripts/sharpe_inflation_methods.py`
consume the eichengreen CSVs but those are diagnostic, not production.

**Conclusion:** A bug in the Eichengreen leg would NOT affect the production
canonical run. Risk is contained to robustness comparisons.

---

## 7. Verdict + recommendation

**PASS** — methodology is correctly implemented, math is sound, output sanity
checks pass, internal cross-check vs alternative methodologies is reasonable,
and the headline `eichengreen` and `eichengreen_literal` variants are clearly
documented in the build script's docstring.

**No code changes required.**

Recommendation: keep the `eichengreen_literal` variant for thesis robustness
table (faithful reproduction of Eichengreen's published method), and use
`eichengreen` only as a maturity-matching exploration. Production canonical
should remain `fisher_ar1` per current configuration.

---

## Files referenced

- `data/build_var_dataset_real.py` (lines 26-44 docstring; 239-298 method
  branches; 274-284 eichengreen logic; 334-335 Fisher decomposition)
- `data/inflation_expectations.py` (lines 162-172 eichengreen helper; 178-206
  cumulative_ar1_static)
- `data/var_dataset_real_eichengreen.csv` (output, T=92, 1920-2011)
- `data/var_dataset_real_eichengreen_literal.csv` (output, T=92, 1920-2011)
- `data/var_dataset_real.csv` (active, method=fisher_ar1)
- `docs/scans/figures/eichengreen_real_yields_compare.png`
