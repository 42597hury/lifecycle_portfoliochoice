# Returns And Financial State Variables

This is the current financial VAR and return-construction spec. Older notes in
`docs/archive/` and dated `docs/scans/` may describe previous nominal, AAA, CP,
or `rtb`-based experiments; the baseline below is the one wired into the
lifecycle model.

## Baseline

The baseline dataset is annual, January-to-January, with sample `1920-2011`
(`T=92`). It is built by `data/build_var_dataset_ar1_10y.py` and written to:

- `data/var_dataset.csv` - active lifecycle baseline
- `data/var_dataset_ar1_10y.csv` - explicit AR(1)-matched 10-year copy

Inflation expectations use a static full-sample AR(1) on Shiller
December-over-December log CPI inflation:

```text
pi_{t+1} = a + phi * pi_t + eps_{t+1}
a        = +1.289pp
phi      = +0.3884
mu       = +2.107pp
```

January year `t` states use information through December `t-1`. The one-year
bill subtracts the AR(1) one-year forecast. The 10-year Shiller `RLONG` yield
subtracts the same AR(1)'s average 10-year forecast.

## Variables

The active VAR has five variables:

| Variable | Role | Definition | Units |
|---|---|---|---|
| `cape` | state | `-log(Shiller CAPE_t)` | log level |
| `spr` | state | `y_10_real,t - y_1,t` | log-yield spread |
| `y_1` | state | real one-year log bill yield | annual log return |
| `xr` | return | stock log excess return | annual log excess |
| `xb` | return | 10-year real bond log excess return | annual log excess |

State order is `('cape', 'spr', 'y_1')`. Return order is `('xr', 'xb')`.
There is no separate `rtb` return variable. The bill is real-risk-free by
construction:

```text
log R_bill,t+1 = y_1,t
```

The solver forms:

```text
log R_stock,t+1 = y_1,t + xr,t+1
log R_bond,t+1  = y_1,t + xb,t+1
```

## State Construction

Let `E1_t` be the AR(1) one-year inflation expectation and `E10_t` the AR(1)
average 10-year inflation expectation, both based on December `t-1`
information.

```text
y_1_nom,t    = log(1 + R_t / 100)
y_10_nom,t   = log(1 + RLONG_t / 100)

y_1,t        = y_1_nom,t  - E1_t
y_10_real,t  = y_10_nom,t - E10_t
spr_t        = y_10_real,t - y_1,t
cape_t       = -log(CAPE_t)
```

`R_t`, `RLONG_t`, `P_t`, and `D_t` are from Shiller Chapter 26 annual data.
`CAPE_t` and CPI are from Shiller `ie_data.xls`.

## Return Construction

Rows are labelled by the realization year. Row `t` contains states observed in
January `t` and returns realized from January `t-1` to January `t`.

Stock excess return:

```text
xr_t = log((P_t + D_t) / P_{t-1}) - y_1_nom,t-1
```

The 10-year real bond yield is converted back to a simple yield for the
Campbell-Lo-MacKinlay constant-duration approximation:

```text
Y_10,t = exp(y_10_real,t) - 1
D_t    = (1 - (1 + Y_10,t)^(-10)) / (1 - (1 + Y_10,t)^(-1))
```

The one-year real log holding-period return is:

```text
r_10,t = D_{t-1} * y_10_real,t-1 - (D_{t-1} - 1) * y_10_real,t
```

The excess real bond return is:

```text
xb_t = r_10,t - y_1,t-1
```

## Timing

```text
January t:
  observe cape_t, spr_t, y_1,t
  y_1,t uses Shiller R_t and Dec t-1 inflation information
  y_10_real,t uses Shiller RLONG_t and Dec t-1 inflation information

During [t, t+1]:
  bill earns y_1,t
  stocks realize xr_{t+1}
  long bond realizes xb_{t+1}

VAR estimation:
  row t states predict row t+1 returns
```

The builder checks the key identities:

```text
spr_t = y_10_real,t - y_1,t
y_1,t = y_1_nom,t - E1_t
xb_t  = r_10,t - y_1,t-1
```

## VAR Estimation

The VAR is a restricted, mean-pinned VAR(1):

```text
z_{t+1} - z_bar = Phi * (z_t - z_bar) + eps_{t+1}
```

Only lagged state variables enter each equation. Lagged `xr` and `xb` columns
of `Phi` are zero by construction. The intercept is recovered as:

```text
const = (I - Phi) * z_bar
```

Baseline moments:

```text
sample:       1920-2011, T=92
state means:  cape=-2.7274, spr=+0.718pp, y_1=+2.023pp
return means: xr=+5.184pp, xb=+0.614pp
Sharpe:       xr=+0.367, xb=+0.129
max |eig|:    0.9296
R2:           cape=0.7999, spr=0.3736, y_1=0.6490, xr=0.0960, xb=0.1294
```

The full-system hardcoded fallback in `lifecycle/var.py` matches the CSV
estimate exactly.

## Code References

- `data/build_var_dataset_ar1_10y.py` - authoritative active data builder
- `data/build_var_dataset.py` - compatibility wrapper that delegates to the builder above
- `data/build_var_dataset_cp_shiller.py` - preserved CP alternative
- `lifecycle/var.py` - VAR estimation, system builders, hardcoded fallback
- `lifecycle/precompute.py` - state grid, conditional return means, return quadrature
- `lifecycle/solver.py` - uses `log_R_bill = state[y_1]` and integrates `xr`, `xb`
- `lifecycle/simulation.py` - same real-yield return convention in simulation
