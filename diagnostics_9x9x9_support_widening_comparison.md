# 9x9x9 Retirement Support-Widening Comparison

## Bundles

- Narrower support reference:
  `saved_runs/checkpoints/unconstrained_principal_grid9x9x9_nz9_retirement_only`
  - solved ages `67-99`
  - `state_n_stds=2.0`
  - `n_state_quad_nodes=3`
- Wider-support test:
  `saved_runs/checkpoints/unconstrained_principal_grid9x9x9_nz9_from_age70_kret3x7x5_ns2p0x2p25x2p25_v2`
  - solved ages `70-99`
  - `state_n_stds=(2.0,2.25,2.25)`
  - `n_state_quad_nodes=(2,2,5)`

This is not a perfect one-axis comparison because the solve windows and state
quadrature differ. But both bundles are retirement-only `9x9x9` runs with the
same return quadrature `n_ret_nodes_1d=(3,7,5)`, so they are still a useful
rough anchor for the support-widening question.

## Solver Health

Both bundles are numerically clean:

- zero Newton failures
- zero monotonicity violations
- `worst_foc_resid ~ 1e-7`

## State-Clipping Diagnostic

Reports:

- `diagnostics_state_clipping_9x9x9_age67_narrow.md`
- `diagnostics_state_clipping_9x9x9_age70_wide.md`

Headline comparison:

| bundle | joint outside share | axis0 | axis1 | axis2 |
| --- | ---: | ---: | ---: | ---: |
| narrow support | `9.37%` | `2.29%` | `3.79%` | `3.52%` |
| wide support | `5.37%` | `2.08%` | `1.91%` | `1.56%` |

Main read:

- widening support cuts the joint clipping rate by about `4.0 pp`
- the gain is concentrated on axes `u1` and `u2`
- late-retirement clipping remains, but is materially smaller

## Centered Retirement Euler Diagnostic

Reports:

- `diagnostics_euler_9x9x9_age67_narrow_report.md`
- `diagnostics_euler_9x9x9_age70_wide_report.md`

Headline comparison:

| bundle | retirement mean `log10|EE|` | retirement max `log10|EE|` |
| --- | ---: | ---: |
| narrow support | `-1.567` | `-0.103` |
| wide support | `-2.591` | `-0.836` |

Main read:

- the wider support materially improves the local retirement Euler check
- the mean improves by about `1.02` log10 units
- the worst evaluated age also improves substantially
- the bundle is still not close to publication-grade Euler gates

## Bottom Line

For retirement-only `9x9x9` work, widening the principal-grid support to
`state_n_stds=(2.0,2.25,2.25)` looks directionally right and economically
meaningful:

- less boundary clipping
- better retirement Euler accuracy
- no new solver-health problems

So this change looks like a genuine improvement, not just a cosmetic one.
