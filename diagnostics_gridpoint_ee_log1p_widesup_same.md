# Grid-Point Euler Sweep: `system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1`

- Bundle: `saved_runs\checkpoints\system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1`
- Eval mode: `same`
- Policy quadrature: ret=`(3, 7, 5)`, state=`(2, 2, 5)`
- Eval quadrature:   ret=`(3, 7, 5)`, state=`(2, 2, 5)`
- Ages tested: `65-98` (34 ages)
- z indices: `[0, 4, 8]`
- State cube points: `27` (low/mid/high per axis on a 7x7x7 grid)
- Wealth indices: `[0, 15, 75, 134, 149]`
- Total grid points: `13770`  (valid: `13770`)

## Summary

- Mean `log10|EE|`: `-5.7639`
- P95 `log10|EE|`: `-0.5352`
- Max `log10|EE|`: `0.2996`
- Min `log10|EE|`: `-14.5396`
- Share with `log10|EE| < -4`: `9886/13770`
- Share with `log10|EE| < -5`: `8668/13770`
- Share with `log10|EE| < -6`: `6190/13770`

## By Phase

| phase | count | mean | p95 | max | min |
| --- | ---: | ---: | ---: | ---: | ---: |
| working | 810 | -5.2755 | -0.4929 | 0.2794 | -12.5532 |
| retirement | 12960 | -5.7945 | -0.5353 | 0.2996 | -14.5396 |

## Worst Grid Points

| age | phase | iz | state | iw | x | c | stock | bond | log10|EE| | abs EE |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 67 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000012 | -0.0006 | -0.0022 | 0.2996 | 1.9934e+00 |
| 68 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2883 | 1.9423e+00 |
| 66 | working | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2794 | 1.9030e+00 |
| 69 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2761 | 1.8886e+00 |
| 65 | working | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2739 | 1.8789e+00 |
| 70 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2630 | 1.8324e+00 |
| 67 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000011 | -0.0011 | -0.0017 | 0.2565 | 1.8051e+00 |
| 71 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2497 | 1.7769e+00 |
| 68 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000011 | -0.0011 | -0.0017 | 0.2449 | 1.7576e+00 |
| 72 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2354 | 1.7194e+00 |
| 66 | working | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000012 | -0.0011 | -0.0018 | 0.2342 | 1.7148e+00 |
| 69 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000011 | -0.0011 | -0.0017 | 0.2324 | 1.7076e+00 |
| 65 | working | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000012 | -0.0011 | -0.0018 | 0.2273 | 1.6878e+00 |
| 73 | retirement | 0 | (6, 0, 6) | 0 | 0.0001 | 0.000013 | -0.0006 | -0.0022 | 0.2200 | 1.6597e+00 |
| 70 | retirement | 0 | (3, 0, 6) | 0 | 0.0001 | 0.000012 | -0.0011 | -0.0017 | 0.2189 | 1.6555e+00 |

