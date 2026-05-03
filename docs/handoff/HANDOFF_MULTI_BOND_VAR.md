# Handoff: Multi-Bond VAR Support

## Goal

Make the VAR estimation pipeline support different bond systems that can be
slotted into the lifecycle model interchangeably. Currently everything is
hardcoded around a 10-year Treasury zero-coupon bond (GSW yield curve).
The first alternative to support is a 20-year system using Moody's AAA yields.

The downstream code (precompute, solver, simulation) is already fully generic —
it reads `var_config` as a dict and never references bond maturity. The work is
entirely in **data construction** and **var.py entry points**.

---

## What Exists Today

### Data Construction (`data/data_construction.ipynb`)
- Loads GSW NSS parameters, computes y(10) and y(9.75) via the NSS formula
- Quarterly bond return: `r_bond = -9.75 * y(9.75)_t/100 + 10 * y(10)_{t-1}/100`
- Excess return: `xb = r_bond - r_bill`
- Yield state variable: `y_nom = y(10)/100`
- Aggregates to annual: returns summed over 4 quarters, levels = Q4 value
- Writes `data/var_dataset.csv` with columns `[rtb, xr, xb, y_nom, dp]`

### VAR Estimation (`var.py`)
- `build_nominal_system1_var_config()` (line 436): reads `var_dataset.csv`,
  hardcodes `columns = ["rtb", "xr", "xb", "y_nom", "dp"]`,
  `state_indices = [0,3,4]`, `return_indices = [1,2]`
- `build_nominal_system1_var_config_hardcoded()` (line 550): fallback with
  baked-in parameter arrays
- `build_var_config_from_dataset()` (line 333): generic factory that both
  wrappers call — this is already system-agnostic

### Downstream (all generic, no changes needed)
- `precompute.py:409` `build_model()` — reads `var_config` dict fields
- `precompute.py:177` — uses `bill_rate_index_in_state` to extract r_bill grid
- `precompute.py:182` — uses `annuity_yield_index_in_state` for annuity pricing
- `solver.py` — receives precomputed arrays only, no VAR knowledge
- `simulation.py` — uses `Sigma_r_cond`, `ret_nodes` generically
- `diagnostics.py` — reads model fields generically

---

## What Needs to Change

### 1. New Data Construction Script

Create a **Python script** (not notebook) that can build `var_dataset.csv` for
any supported bond system. This replaces the hardcoded notebook pipeline.

**File:** `data/build_var_dataset.py`

It should support at minimum two bond configurations:

#### Config A: 10-Year Treasury (current baseline)
- Yield source: GSW NSS curve (`feds200628 (1).csv`)
- Maturities: y(10), y(9.75) computed from NSS parameters
- Quarterly bond return: `r_bond = 10 * y10_{t-1}/100 - 9.75 * y975_t/100`
- Yield state variable: `y_nom = y(10)/100`
- Sample: 1962-2025 (bound by GSW start date 1961-Q2)

#### Config B: 20-Year Moody's AAA
- Yield source: `AAA.csv` (monthly Moody's AAA corporate bond yield, % p.a.)
- Single yield series (no term structure), so use same yield for buy and sell
- Annual bond return (direct, no quarterly intermediate):
  `r_bond = 20 * y_AAA_{t-1}/100 - 19 * y_AAA_t/100`
- Yield state variable: `y_nom = y_AAA/100`
- Sample: 1950-2025 (AAA data starts 1919, TB3MS starts 1934)

**Key design decisions:**
- The AAA system works at **annual frequency directly** (end-of-year yields,
  annual return = one-period holding return). No quarterly intermediate needed
  since we only have a single yield series, not a term structure.
- The 10-year Treasury system goes quarterly first (because the NSS curve gives
  us y(9.75) for the quarter-shorter maturity), then aggregates to annual.
- Both systems produce the **same output format**: a CSV with columns
  `[rtb, xr, xb, y_nom, dp]` at annual frequency.
- The other three columns (rtb, xr, dp) come from the same sources in both
  systems (TB3MS, CPIAUCSL, Shiller). Only xb and y_nom differ.

**Suggested interface:**
```python
def build_var_dataset(bond_system="treasury_10y", start_year=None):
    """
    Build var_dataset.csv for the specified bond system.

    bond_system: "treasury_10y" or "aaa_20y"
    start_year: first year to include (default: 1962 for treasury, 1950 for AAA)

    Returns: DataFrame with columns [rtb, xr, xb, y_nom, dp], annual frequency
    """
```

The common pipeline (rtb, xr, dp construction from TB3MS, CPI, Shiller) should
be shared code, not duplicated.

### 2. New VAR Entry Point in `var.py`

Add a convenience wrapper analogous to `build_nominal_system1_var_config()`:

```python
def build_aaa20_var_config(
    csv_path="data/var_dataset_aaa20.csv",
    state_indices=(0, 3, 4),
    return_indices=(1, 2),
    bill_rate_index_in_state=0,
    annuity_yield_index_in_state=1,
    trend="c",
    estimation="restricted",
):
    """
    20-year Moody's AAA bond system.
    columns = [rtb, xr, xb, y_nom, dp]
    States: rtb(0), y_nom(3), dp(4)   Returns: xr(1), xb(2)

    y_nom is now Moody's AAA yield (annual decimal).
    xb is excess return on a 20-year AAA bond (duration approx).
    """
    columns = ["rtb", "xr", "xb", "y_nom", "dp"]
    return build_var_config_from_dataset(
        csv_path=csv_path, columns=columns,
        state_indices=state_indices, return_indices=return_indices,
        bill_rate_index_in_state=bill_rate_index_in_state,
        annuity_yield_index_in_state=annuity_yield_index_in_state,
        trend=trend, estimation=estimation,
    )
```

The column names stay identical (`xb`, `y_nom`) so downstream code doesn't
need to change. The economic interpretation changes (20yr AAA vs 10yr Tsy)
but the VAR structure is the same.

### 3. Separate Dataset Files

Use distinct CSV files to avoid confusion:
- `data/var_dataset.csv` — 10yr Treasury (current, unchanged)
- `data/var_dataset_aaa20.csv` — 20yr AAA

### 4. Annuity Pricing / Bequest Utility Must Adapt

**This is critical.** The bequest utility uses an annuity factor that is coupled
to the bond maturity. Changing the bond system without updating this breaks
economic coherence.

#### Current setup (10yr Treasury)
- `b_bar = 10` in `base_config` (set in `main.ipynb` line 158)
- `annuity_factor(y_nom, b_bar)` in `model.py:194` discounts 10 annual
  payments at the 10-year nominal yield
- The docstring states: "bequest horizon b_bar equals the bond maturity"
- `b_bar` also serves as the **bequest weight** in utility:
  `b(W) = b_bar * (W/A)^(1-gamma) / (1-gamma)` (model.py:215-231)
- Used in `precompute.py:187-188` to build `annuity_factors` grid
- Used pervasively in `solver.py` for bequest marginal utility

#### What needs to change for 20yr AAA
- `b_bar` should become **20** to match the bond maturity, so the annuity
  prices a 20-year consumption stream discounted at the 20-year AAA yield.
- **BUT** `b_bar` also scales the bequest motive strength. Changing 10→20
  doubles the bequest weight. This is a real economic change, not just a
  pricing adjustment.
- The coding agent should make `b_bar` a parameter that can be set per bond
  system, and the user should be aware that switching bond systems changes
  both the annuity pricing AND the bequest motive calibration.
- Consider whether the bequest weight and the annuity horizon should be
  decoupled (separate `b_bar_weight` and `b_bar_horizon` parameters) to
  allow switching bond maturity without changing the bequest motive strength.
  This is a modeling decision for the user.

#### Downstream impact
- `precompute.py:187-188` — annuity_factor call uses `model.b_bar`; no code
  change needed if `b_bar` is set correctly in `base_config`
- `solver.py` — uses `annuity_factor_is` and `b_bar` throughout; generic,
  no code change needed
- `model.py:194-260` — annuity_factor, bequest_utility, bequest_marginal,
  bequest_marginal_inv are all generic functions; only docstrings/comments
  reference "10" and would need updating

#### Open question: interpolated annuity yield

Instead of using a single yield (either the short rate or the 20-year yield)
for annuity pricing, interpolate between rtb and y_AAA to match the actual
annuity duration (b_bar). For example, if b_bar=10 but the bond is 20-year,
use a blended yield that prices a 10-year annuity more accurately than either
endpoint alone.

This is **computationally free**. The annuity factor is precomputed once on
the state grid (`precompute.py:187-188`) as an `(N_state,)` array. The solver
only does a scalar lookup (`annuity_factor_is = annuity_factors[i_s]`).
Whether the precompute step reads one column or interpolates two columns,
the solver sees the same precomputed number. Cost: a few microseconds of
extra vectorized arithmetic on ~125-343 grid points.

Implementation sketch (in `precompute.py`, replacing lines 187-188):
```python
_y_short = self.state_grid[:, model.bill_rate_index_in_state]
_y_long  = self.state_grid[:, model.annuity_yield_index_in_state]
_y_ann   = interpolate(_y_short, _y_long, target_duration=model.b_bar)
self.annuity_factors = annuity_factor(_y_ann, model.b_bar)
```

The interpolation method (linear in maturity, Nelson-Siegel-style, etc.) is
a modeling choice. This would allow decoupling the bond maturity (20yr) from
the bequest horizon (b_bar=10) without discounting at a mismatched yield.

### 5. Do NOT Change (beyond above)

- `precompute.py` — already generic (only comments reference 10yr)
- `solver.py` — already generic
- `simulation.py` — already generic
- `diagnostics.py` — already generic
- `discretization.py` — already generic
- The existing `data/data_construction.ipynb` — keep as reference/documentation
- The hardcoded fallback `build_nominal_system1_var_config_hardcoded()` — keep as-is

---

## Empirical Results (from `return_estimation/bond20_var_comparison.ipynb`)

Key differences found between the two systems (same 1962-2025 sample):

| Quantity | 10yr Treasury | 20yr AAA |
|----------|--------------|----------|
| Mean xb | +1.95% | +2.10% |
| Std xb | 10.49% | 17.61% |
| Sharpe (xb) | 0.185 | 0.119 |
| M[xb, y_nom] | -10.23 | -20.15 |
| y_nom persistence | 0.871 | 0.921 |
| corr(xr, xb) | 0.160 | 0.336 |
| corr(xb, dp) | -0.101 | -0.269 |
| Cond. std xb | 1.55% | 1.23% |
| Var explained (xb) | 97.8% | 99.5% |

The higher stock-bond correlation in AAA comes from the credit spread
(flight-to-quality effect), not from the longer duration — a 20yr Treasury
(GS20) has corr(xb, dp) = -0.049, similar to the 10yr Treasury.

---

## Data Files Available

All in `data/Thesisdata/`:
- `AAA.csv` — Moody's AAA yield, monthly, 1919-2026 (observation_date, AAA in %)
- `GS20.csv` — 20yr Treasury constant maturity, monthly, 1953-2026 (7yr gap 1987-1993)
- `DGS30.csv` — 30yr Treasury daily, 1977-2026
- `feds200628 (1).csv` — GSW yield curve, daily, 1961-2026
- `TB3MS.csv` — 3-month T-bill, monthly
- `CPIAUCSL.csv` — CPI-U seasonally adjusted, monthly
- `ie_data.xls` — Shiller S&P 500 data (P, D, RTRP)

---

## Testing

After implementation, verify by running:
1. `build_var_dataset("treasury_10y")` produces output matching existing
   `var_dataset.csv` to machine precision
2. `build_var_dataset("aaa_20y")` produces output matching the alternative
   dataset constructed in `return_estimation/bond20_var_comparison.ipynb`
3. Both `var_config` dicts can be passed to `build_model()` without error
4. `diagnose_var_pre(model, pc)` passes for both systems
