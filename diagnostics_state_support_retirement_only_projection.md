# Retirement-Only State Support Projection

Bundle context: `saved_runs/checkpoints/unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2`

Method:
- No solve required.
- Simulate only the exogenous financial-state VAR.
- Start at age `67` from the median state.
- Propagate through ages `67-98`.
- Transform states into principal-grid bracket coordinates `b`.
- Measure how often simulated states fall outside candidate `state_n_stds`.

Current support:
- `(0.6, 1.75, 2.0)`

## Required Half-Width by Marginal Quantile of `|b_j|`

| axis | p90 | p95 | p97.5 | p99 | p99.5 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `axis0` | `1.417` | `1.718` | `1.988` | `2.308` | `2.538` | `4.549` |
| `axis1` | `1.587` | `1.899` | `2.177` | `2.505` | `2.732` | `5.202` |
| `axis2` | `1.562` | `1.875` | `2.154` | `2.485` | `2.715` | `4.656` |

## Joint Outside Share Under Candidate `state_n_stds`

| candidate | retirement outside share |
| --- | ---: |
| `(1.0, 1.75, 2.0)` | `30.69%` |
| `(1.25, 1.75, 2.0)` | `22.96%` |
| `(1.5, 1.75, 2.0)` | `17.77%` |
| `(1.5, 2.0, 2.0)` | `15.06%` |
| `(1.75, 2.0, 2.0)` | `11.67%` |
| `(1.75, 2.0, 2.25)` | `10.08%` |
| `(2.0, 2.0, 2.25)` | `8.04%` |
| `(2.0, 2.25, 2.25)` | `6.26%` |
| `(2.25, 2.5, 2.5)` | `3.12%` |

## Age Profile

Selected ages, joint outside share:

| age | current `(0.6,1.75,2.0)` | `(1.0,1.75,2.0)` | `(1.75,2.0,2.25)` |
| ---: | ---: | ---: | ---: |
| `67` | `0.0%` | `0.0%` | `0.0%` |
| `68` | `7.0%` | `1.5%` | `0.5%` |
| `69` | `19.8%` | `6.5%` | `2.1%` |
| `70` | `29.1%` | `11.4%` | `3.5%` |
| `71` | `35.7%` | `16.2%` | `4.6%` |
| `94` | `60.1%` | `39.6%` | `13.5%` |
| `95` | `60.0%` | `39.6%` | `13.9%` |
| `96` | `59.9%` | `39.5%` | `13.9%` |
| `97` | `59.8%` | `39.7%` | `13.9%` |
| `98` | `59.9%` | `39.6%` | `13.9%` |

## Practical Read

- `(1.0, 1.75, 2.0)` is not enough for retirement-only work.
- If the target is roughly `90%` of retirement states inside support, `(1.75, 2.0, 2.25)` is the first reasonable candidate.
- If the target is closer to `95%` inside support, `(2.0, 2.25, 2.25)` is a better target.
