# US Net Worth by Age — Survey of Consumer Finances (2022)

**Source.** Aladangady, Aditya, et al. *Changes in U.S. Family Finances from 2019 to 2022:
Evidence from the Survey of Consumer Finances*, Federal Reserve Bulletin Vol. 109,
No. 4 (October 2023), Table 2 (p. 12). PDF:
https://www.federalreserve.gov/publications/files/scf23.pdf

**Unit of observation.** Family (PEU = primary economic unit), classified by age of the
reference person. The SCF reference person is the male in mixed-sex couples and the
older partner in same-sex couples; otherwise, the sole adult.

**Wealth concept.** Net worth = total family assets − total family liabilities.
Includes financial assets (transaction accounts, retirement accounts, stocks, bonds,
pooled funds, CDs, savings bonds, cash-value life insurance, other managed assets) and
nonfinancial assets (primary residence, other real estate, vehicles, business equity,
other), net of all debts. Excludes Social Security wealth and DB pension wealth (asset
side) — a non-trivial omission for older households.

**Units.** Thousands of 2022 dollars (CPI-U-RS deflated by Fed). Numbers below are
copied verbatim from Table 2 of the cited bulletin.

## Mean and median family net worth by age of reference person

| Age of reference person | Median 2019 | Median 2022 | Mean 2019 | Mean 2022 |
|---|---:|---:|---:|---:|
| < 35  |   16.1 |   39.0 |    88.5 |   183.5 |
| 35–44 |  105.9 |  135.6 |   505.6 |   549.6 |
| 45–54 |  195.4 |  247.2 |   965.9 |   975.8 |
| 55–64 |  246.3 |  364.5 | 1,363.1 | 1,566.9 |
| 65–74 |  308.8 |  409.9 | 1,411.6 | 1,794.6 |
| 75+   |  295.4 |  335.6 | 1,133.2 | 1,624.1 |
| All families | 141.1 | 192.9 | 868.0 | 1,063.7 |

## Notes for use as a calibration target

- **Mean ≫ median** at every age — the wealth distribution is heavily right-skewed.
  For matching a representative-agent lifecycle model (especially under CRRA with
  γ ≥ 3), the **median** is usually a better target than the mean, since the mean is
  driven by the top few percent. The mean is closer to what an aggregate (per-capita)
  series would give.
- **Age groupings are 10-year bins** (5-year for the youngest open-ended bin). To
  compare with a model indexed by single-year age, plot the empirical numbers at the
  bin midpoints (30, 40, 50, 60, 70, 80) — keeping in mind that within-bin profiles
  are not flat.
- **Missing concepts.** SCF net worth excludes the present value of Social Security
  and DB pension claims. Catherine (2025) and most lifecycle papers reporting wealth
  alongside SCF do the same; if your model treats Social Security as a separate
  income stream, the comparison is consistent.
- **Income normalization.** If your model expresses wealth in units of mean labor
  earnings (Catherine 2025 convention), divide by an appropriate denominator —
  e.g. 2022 mean household labor income from SCF Table 1 (= $141.4k) or median
  household income from CPS (= $74.6k in 2022). The choice changes W substantially.

## Cross-check: 75+ age group

The bulletin highlights (p. 13) that for families 75+, mean net worth grew 43% (vs.
14% median) between 2019 and 2022 — i.e., wealth concentration *within* the oldest
age group widened. Use the median–mean gap there with care.

## CSV form

A machine-readable copy lives in `scf_net_worth_by_age_2022.csv` (this folder).
