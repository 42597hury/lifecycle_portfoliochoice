# Euler Error Report: `system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1`

## Setup

- Bundle: `saved_runs\checkpoints\system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1`
- Solved window: ages `87-99`
- Simulation households: `5000`
- Evaluated households per age cap: `256`
- Seed: `42`
- Return draw mode: `monte_carlo`
- Initial z: `stationary`
- Initial state: `median`
- Partial init mode: `centered`
- Eval mode: `next_finer`
- Policy quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`
- Eval quadrature: `ret=(5, 9, 7)`, `state=(3, 3, 6)`, `eta=5`, `eps=5`

## Gates

| phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| working | publication | no | nan | < nan | nan | < nan |
| retirement | publication | no | -2.074 | < -4.5 | -0.005 | < -3.0 |
| working | welfare | no | nan | < nan | nan | < nan |
| retirement | welfare | no | -2.074 | < -5.5 | -0.005 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 87 | retirement | 5000 | 256 | 256 | 0 | -2.829 | -2.805 | -2.805 | 0.1486 | 0.1566 | 0.1566 |
| 88 | retirement | 4610 | 256 | 256 | 0 | -2.058 | -0.227 | -0.065 | 2.2150 | 1.4318 | 86.0327 |
| 89 | retirement | 4187 | 256 | 256 | 0 | -2.008 | -0.088 | -0.005 | 3.1957 | 3.1273 | 98.9158 |
| 90 | retirement | 3787 | 256 | 254 | 2 | -2.004 | -0.106 | -0.008 | 3.3394 | 4.6041 | 98.2290 |
| 91 | retirement | 3404 | 256 | 253 | 3 | -1.959 | -0.067 | -0.018 | 4.8780 | 30.0266 | 96.0053 |
| 92 | retirement | 2964 | 256 | 255 | 1 | -2.011 | -0.080 | -0.013 | 4.4375 | 20.5042 | 96.9786 |
| 93 | retirement | 2599 | 256 | 254 | 2 | -1.916 | -0.028 | -0.010 | 7.1113 | 74.6342 | 97.7573 |
| 94 | retirement | 2213 | 256 | 256 | 0 | -1.985 | -0.033 | -0.014 | 4.2228 | 15.0834 | 96.7258 |
| 95 | retirement | 1856 | 256 | 256 | 0 | -2.007 | -0.184 | -0.118 | 3.4970 | 16.3674 | 76.1810 |
| 96 | retirement | 1554 | 256 | 255 | 1 | -2.045 | -0.208 | -0.065 | 2.7314 | 7.8516 | 86.1297 |
| 97 | retirement | 1289 | 256 | 256 | 0 | -2.007 | -0.022 | -0.013 | 4.8617 | 28.1560 | 96.9453 |
| 98 | retirement | 1052 | 256 | 255 | 1 | -2.059 | -0.104 | -0.012 | 3.5786 | 10.1882 | 97.2759 |

## Notes

- This is a partial-window simulation starting at age `87` because the bundle only solves ages `87-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
