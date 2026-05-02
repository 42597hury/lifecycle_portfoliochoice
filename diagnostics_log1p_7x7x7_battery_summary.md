# Log1p Grid Battery Summary

## Bundle

- Bundle:
  `saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1`
- Solved ages: `65-99`
- State grid: `(7,7,7)`
- State support: `(2.0,2.25,2.25)`
- Return quadrature: `(3,7,5)`
- State quadrature: `(2,2,5)`
- Saved metadata still reports `n_wealth=150`, `n_savings=150`

Interpretation: this run changed wealth / savings grid spacing, not the point
count.

## Solver Health

The bundle is numerically clean:

- `solve_status="stopped_early"` as intended for the partial solve
- `total_calls=15,743,700`
- `total_newton_failures=0`
- `total_mono_violations=0`
- `worst_foc_resid ~ 1e-7`

## Battery Results

### 1. Centered simulated-path Euler errors

Report: `diagnostics_euler_log1p_centered_report.md`

- working mean `log10|EE| = -3.014`
- working max `log10|EE| = -2.365`
- retirement mean `log10|EE| = -2.467`
- retirement max `log10|EE| = -0.791`

Relative to the earlier narrow-support `7x7x7` centered report
`diagnostics_euler_7x7x7_centered_report.md`:

- working mean improved from `-2.941 -> -3.014`
- retirement mean improved from `-2.167 -> -2.467`
- retirement max improved from `0.352 -> -0.791`

So the new spacing/support combination materially improves the simulated
off-grid policy quality, especially in retirement.

### 2. State-support clipping

Report: `diagnostics_state_clipping_log1p_7x7x7.md`

- overall joint outside share: `6.00%`
- working joint outside share: `0.20%`
- retirement joint outside share: `6.59%`

Relative to the earlier narrow-support centered `7x7x7` stratification
`diagnostics_sim_ee_stratification_7x7x7.md`:

- overall outside share improved from about `45.4% -> 6.0%`
- working outside share improved from about `3.9% -> 0.2%`
- retirement outside share improved from about `49.6% -> 6.6%`

This is the clearest reason the centered simulation EE improved.

### 3. Exact-grid Euler sweep

Report: `diagnostics_gridpoint_ee_log1p_same.md`

- overall mean `log10|EE| = -5.7639`
- p95 `log10|EE| = -0.5352`
- max `log10|EE| = 0.2996`

At first glance this looks much worse than the earlier exact-grid reports.
But the pathology is extremely localized:

- every worst-case row is at the very first wealth node `iw=0`, `x=1e-4`
- worst states are corner retirement / age-65 states such as `(6,0,6)` and
  `(3,0,6)`
- portfolio positions there are tiny, and consumption is around `1e-5`

### 4. Low-wealth investigation

Reports:

- `diagnostics_log1p_low_wealth_investigation.md`
- `diagnostics_log1p_low_wealth_sweep.md`

Focused exact-grid results by wealth index:

| iw | x | mean `log10|EE|` | p95 | max |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.000100` | `-0.8753` | `0.0147` | `0.2996` |
| 1 | `0.036337` | `-2.2681` | `-1.4021` | `-1.1776` |
| 2 | `0.073886` | `-4.0695` | `-2.6302` | `-2.3773` |
| 3 | `0.112796` | `-4.0472` | `-3.0479` | `-2.7586` |
| 4 | `0.153116` | `-4.2114` | `-3.3297` | `-3.0095` |
| 5 | `0.194897` | `-4.2606` | `-3.2336` | `-2.8052` |
| 10 | `0.427636` | `-4.5339` | `-3.2693` | `-2.9212` |
| 15 | `0.705707` | `-4.7578` | `-3.4665` | `-3.1821` |
| 30 | `1.909145` | `-5.4016` | `-3.8839` | `-3.7173` |
| 75 | `13.432729` | `-6.6749` | `-4.7259` | `-4.1414` |
| 134 | `116.851484` | `-8.0178` | `-5.4461` | `-4.7679` |
| 149 | `200.000000` | `-8.4940` | `-5.7498` | `-4.9426` |

Most important summary:

- excluding only `iw=0`, the default grid-point sweep improves to:
  - mean `log10|EE| = -6.9861`
  - p95 `log10|EE| = -3.7189`
  - max `log10|EE| = -3.1821`

So the remaining exact-grid problem is basically a one-node floor issue.

## Interpretation

The new `log1p` spacing looks like a real improvement overall:

- much less state-support clipping
- better centered simulated-path EE
- especially better retirement behavior

But it also introduces a new localized weakness:

- the first wealth node `x=1e-4` is now numerically problematic on the exact
  grid
- the problem decays very quickly by `iw=1` and is mostly gone by `iw=2-4`

So the current read is:

- the redesign helps where households actually live
- the remaining issue is concentrated at the wealth floor, not spread across
  the policy surface

## Caveat on Cross-Bundle Comparisons

The current diagnostics rebuild wealth and savings grids from metadata using the
current code path. After the `log1p` spacing change, old pre-`log1p` bundles no
longer carry enough metadata to reconstruct their original wealth/savings grids
exactly.

Implication:

- diagnostics run on the new `log1p` bundle are valid
- direct old-vs-new comparisons that reconstruct both grids are **not yet
  trustworthy** unless the comparison script knows which bundle used legacy
  spacing and which used `log1p`

So this run can be evaluated on its own battery cleanly, but historical
cross-bundle policy drift needs a saved grid-spacing flag before it becomes
fully reliable again.
