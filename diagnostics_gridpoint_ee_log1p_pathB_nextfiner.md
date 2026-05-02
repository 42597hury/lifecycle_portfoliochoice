# Grid-Point Euler Sweep: `system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1`

- Bundle: `saved_runs\checkpoints\system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1`
- Eval mode: `next_finer`
- Policy quadrature: ret=`(3, 7, 5)`, state=`(2, 2, 5)`
- Eval quadrature:   ret=`(5, 9, 7)`, state=`(3, 3, 6)`
- Ages tested: `87-98` (12 ages)
- z indices: `[0, 4, 8]`
- State cube points: `27` (low/mid/high per axis on a 7x7x7 grid)
- Wealth indices: `[0, 15, 75, 134, 149]`
- Total grid points: `4860`  (valid: `3966`)

## Summary

- Mean `log10|EE|`: `-1.6862`
- P95 `log10|EE|`: `-0.0192`
- Max `log10|EE|`: `-0.0002`
- Min `log10|EE|`: `-5.5846`
- Share with `log10|EE| < -4`: `48/3966`
- Share with `log10|EE| < -5`: `6/3966`
- Share with `log10|EE| < -6`: `0/3966`

## By Phase

| phase | count | mean | p95 | max | min |
| --- | ---: | ---: | ---: | ---: | ---: |
| retirement | 3966 | -1.6862 | -0.0192 | -0.0002 | -5.5846 |

## Worst Grid Points

| age | phase | iz | state | iw | x | c | stock | bond | log10|EE| | abs EE |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 92 | retirement | 4 | (6, 6, 0) | 15 | 0.7057 | 0.133097 | 8.4084 | 1.2920 | -0.0002 | 9.9952e-01 |
| 92 | retirement | 8 | (3, 6, 0) | 75 | 13.4327 | 1.730464 | 5.8751 | 1.5504 | -0.0002 | 9.9951e-01 |
| 94 | retirement | 0 | (6, 6, 0) | 15 | 0.7057 | 0.104408 | 8.4407 | 1.2151 | -0.0004 | 9.9909e-01 |
| 98 | retirement | 4 | (6, 6, 0) | 15 | 0.7057 | 0.118582 | 8.5189 | 1.2659 | -0.0006 | 9.9863e-01 |
| 87 | retirement | 0 | (3, 6, 0) | 15 | 0.7057 | 0.078391 | 5.4093 | 1.3883 | -0.0006 | 9.9861e-01 |
| 95 | retirement | 8 | (3, 6, 0) | 15 | 0.7057 | 0.104510 | 5.2882 | 1.5288 | -0.0007 | 9.9838e-01 |
| 98 | retirement | 8 | (3, 6, 0) | 75 | 13.4327 | 1.501973 | 5.3962 | 1.6598 | -0.0008 | 9.9813e-01 |
| 97 | retirement | 0 | (3, 6, 6) | 75 | 13.4327 | 2.307227 | -0.4432 | 6.1430 | -0.0010 | 9.9761e-01 |
| 89 | retirement | 4 | (6, 6, 0) | 75 | 13.4327 | 2.177315 | 8.9804 | 1.0500 | -0.0012 | 9.9734e-01 |
| 88 | retirement | 4 | (0, 0, 0) | 15 | 0.7057 | 0.120878 | 1.1126 | -5.3543 | -0.0013 | 9.9705e-01 |
| 96 | retirement | 8 | (6, 6, 0) | 75 | 13.4327 | 2.217643 | 8.8675 | 1.1259 | -0.0014 | 9.9684e-01 |
| 94 | retirement | 4 | (6, 6, 0) | 75 | 13.4327 | 2.118398 | 8.7566 | 1.1367 | -0.0014 | 9.9677e-01 |
| 98 | retirement | 0 | (6, 6, 0) | 15 | 0.7057 | 0.101686 | 8.2925 | 1.3101 | -0.0014 | 9.9673e-01 |
| 89 | retirement | 4 | (6, 6, 0) | 15 | 0.7057 | 0.139643 | 8.4339 | 1.2835 | -0.0014 | 9.9667e-01 |
| 97 | retirement | 0 | (3, 6, 6) | 15 | 0.7057 | 0.121357 | -0.4441 | 6.1506 | -0.0016 | 9.9631e-01 |

