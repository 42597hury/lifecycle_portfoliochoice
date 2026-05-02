# Grid-Point Euler Sweep: `system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1`

- Bundle: `saved_runs\checkpoints\system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1`
- Eval mode: `same`
- Policy quadrature: ret=`(3, 7, 5)`, state=`(2, 2, 5)`
- Eval quadrature:   ret=`(3, 7, 5)`, state=`(2, 2, 5)`
- Ages tested: `87-98` (12 ages)
- z indices: `[0, 4, 8]`
- State cube points: `27` (low/mid/high per axis on a 7x7x7 grid)
- Wealth indices: `[0, 15, 75, 134, 149]`
- Total grid points: `4860`  (valid: `4860`)

## Summary

- Mean `log10|EE|`: `-6.3040`
- P95 `log10|EE|`: `-0.5787`
- Max `log10|EE|`: `-0.0945`
- Min `log10|EE|`: `-14.5396`
- Share with `log10|EE| < -4`: `3579/4860`
- Share with `log10|EE| < -5`: `3183/4860`
- Share with `log10|EE| < -6`: `2726/4860`

## By Phase

| phase | count | mean | p95 | max | min |
| --- | ---: | ---: | ---: | ---: | ---: |
| retirement | 4860 | -6.3040 | -0.5787 | -0.0945 | -14.5396 |

## Worst Grid Points

| age | phase | iz | state | iw | x | c | stock | bond | log10|EE| | abs EE |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 87 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000014 | -0.0005 | -0.0020 | -0.0945 | 8.0449e-01 |
| 88 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000014 | -0.0005 | -0.0020 | -0.1255 | 7.4902e-01 |
| 87 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0011 | -0.0016 | -0.1488 | 7.0988e-01 |
| 89 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000014 | -0.0005 | -0.0020 | -0.1570 | 6.9666e-01 |
| 88 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0011 | -0.0016 | -0.1807 | 6.5962e-01 |
| 90 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000014 | -0.0005 | -0.0020 | -0.1914 | 6.4356e-01 |
| 87 | retirement | 0 | (0, 0, 6) | 0 | 0.0001 | 0.000011 | -0.0021 | -0.0011 | -0.2024 | 6.2746e-01 |
| 89 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0011 | -0.0016 | -0.2130 | 6.1235e-01 |
| 91 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000014 | -0.0005 | -0.0020 | -0.2268 | 5.9319e-01 |
| 87 | retirement | 0 | (6, 0, 3) | 0 | 0.0001 | 0.000014 | 0.0017 | -0.0075 | -0.2315 | 5.8681e-01 |
| 87 | retirement | 0 | (6, 3, 3) | 0 | 0.0001 | 0.000012 | 0.0019 | 0.0005 | -0.2316 | 5.8665e-01 |
| 88 | retirement | 0 | (0, 0, 6) | 0 | 0.0001 | 0.000011 | -0.0021 | -0.0011 | -0.2353 | 5.8167e-01 |
| 90 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0011 | -0.0016 | -0.2484 | 5.6446e-01 |
| 92 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000015 | -0.0005 | -0.0020 | -0.2646 | 5.4370e-01 |
| 88 | retirement | 0 | (6, 3, 3) | 0 | 0.0001 | 0.000012 | 0.0019 | 0.0006 | -0.2657 | 5.4238e-01 |

