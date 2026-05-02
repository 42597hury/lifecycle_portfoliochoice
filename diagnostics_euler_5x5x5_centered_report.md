# Euler Error Report: `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`

## Setup

- Bundle: `saved_runs\checkpoints\unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`
- Solved window: ages `65-99`
- Simulation households: `64`
- Evaluated households per age cap: `16`
- Seed: `42`
- Return draw mode: `monte_carlo`
- Initial z: `median`
- Initial state: `median`
- Partial init mode: `centered`
- Eval mode: `same`
- Policy quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`
- Eval quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`

## Gates

| phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| working | publication | no | -2.771 | < -4.0 | -1.727 | < -3.0 |
| retirement | publication | no | -1.945 | < -4.5 | 0.290 | < -3.0 |
| working | welfare | no | -2.771 | < -5.0 | -1.727 | < -4.0 |
| retirement | welfare | no | -1.945 | < -5.5 | 0.290 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | working | 64 | 16 | 16 | 0 | -3.448 | -3.448 | -3.448 | 0.0356 | 0.0356 | 0.0356 |
| 66 | working | 62 | 16 | 16 | 0 | -2.093 | -1.730 | -1.727 | 0.9813 | 1.8104 | 1.8769 |
| 67 | retirement | 62 | 16 | 16 | 0 | -2.078 | -1.379 | -1.312 | 1.1337 | 2.5228 | 4.8730 |
| 68 | retirement | 62 | 16 | 16 | 0 | -2.048 | -1.223 | -1.123 | 1.2973 | 3.0994 | 7.5354 |
| 69 | retirement | 61 | 16 | 16 | 0 | -2.099 | -1.707 | -1.696 | 0.9551 | 1.7721 | 2.0131 |
| 70 | retirement | 61 | 16 | 16 | 0 | -1.966 | -1.533 | -1.497 | 1.2460 | 2.1592 | 3.1876 |
| 71 | retirement | 61 | 16 | 16 | 0 | -2.134 | -1.451 | -1.425 | 1.0413 | 2.8415 | 3.7561 |
| 72 | retirement | 59 | 16 | 16 | 0 | -1.980 | -1.227 | -1.177 | 1.5367 | 3.9765 | 6.6509 |
| 73 | retirement | 59 | 16 | 16 | 0 | -1.970 | -1.189 | -1.181 | 1.8175 | 6.0372 | 6.5846 |
| 74 | retirement | 57 | 16 | 16 | 0 | -1.995 | -1.198 | -1.137 | 1.5242 | 3.9560 | 7.3012 |
| 75 | retirement | 57 | 16 | 16 | 0 | -1.911 | -1.455 | -1.432 | 1.3706 | 2.8839 | 3.6966 |
| 76 | retirement | 56 | 16 | 16 | 0 | -2.081 | -1.360 | -1.359 | 1.4462 | 4.3253 | 4.3769 |
| 77 | retirement | 52 | 16 | 16 | 0 | -1.887 | -1.310 | -1.309 | 1.9501 | 4.8530 | 4.9121 |
| 78 | retirement | 50 | 16 | 16 | 0 | -1.992 | -1.366 | -1.352 | 1.4407 | 3.7984 | 4.4507 |
| 79 | retirement | 48 | 16 | 16 | 0 | -2.191 | -1.588 | -1.551 | 0.9129 | 1.8945 | 2.8144 |
| 80 | retirement | 48 | 16 | 16 | 0 | -1.908 | -1.353 | -1.349 | 1.8340 | 4.3116 | 4.4731 |
| 81 | retirement | 46 | 16 | 16 | 0 | -1.952 | -1.386 | -1.382 | 1.4813 | 3.9953 | 4.1452 |
| 82 | retirement | 44 | 16 | 16 | 0 | -2.039 | -1.500 | -1.489 | 1.1979 | 2.8806 | 3.2427 |
| 83 | retirement | 44 | 16 | 16 | 0 | -1.934 | -1.253 | -1.227 | 1.7742 | 4.4444 | 5.9330 |
| 84 | retirement | 42 | 16 | 16 | 0 | -2.037 | -1.384 | -1.377 | 1.3721 | 3.8568 | 4.2008 |
| 85 | retirement | 38 | 16 | 16 | 0 | -2.101 | -1.241 | -1.234 | 1.7956 | 5.3943 | 5.8330 |
| 86 | retirement | 36 | 16 | 16 | 0 | -1.942 | -1.300 | -1.278 | 1.6624 | 4.1668 | 5.2674 |
| 87 | retirement | 31 | 16 | 16 | 0 | -1.906 | -1.405 | -1.397 | 1.5990 | 3.6740 | 4.0059 |
| 88 | retirement | 30 | 16 | 16 | 0 | -2.102 | -1.491 | -1.479 | 1.2232 | 2.8812 | 3.3202 |
| 89 | retirement | 25 | 16 | 16 | 0 | -1.816 | 0.085 | 0.228 | 13.0149 | 56.3006 | 169.1636 |
| 90 | retirement | 22 | 16 | 16 | 0 | -1.865 | -0.004 | 0.249 | 12.2990 | 47.0633 | 177.2329 |
| 91 | retirement | 17 | 16 | 16 | 0 | -1.830 | 0.010 | 0.269 | 13.0265 | 49.0733 | 185.7891 |
| 92 | retirement | 16 | 16 | 16 | 0 | -1.911 | 0.050 | 0.290 | 13.5415 | 52.4100 | 195.1133 |
| 93 | retirement | 16 | 16 | 16 | 0 | -1.814 | 0.030 | 0.287 | 13.4514 | 51.2525 | 193.8581 |
| 94 | retirement | 14 | 14 | 14 | 0 | -1.843 | -0.025 | 0.184 | 12.0653 | 55.8813 | 152.6308 |
| 95 | retirement | 14 | 14 | 14 | 0 | -1.885 | -0.095 | 0.094 | 10.1162 | 46.2916 | 124.2498 |
| 96 | retirement | 10 | 10 | 10 | 0 | -1.750 | -0.040 | 0.093 | 13.8065 | 70.0219 | 123.9873 |
| 97 | retirement | 8 | 8 | 8 | 0 | -1.764 | -0.030 | 0.075 | 15.9633 | 78.4873 | 118.7304 |
| 98 | retirement | 7 | 7 | 7 | 0 | -1.521 | 0.175 | 0.273 | 28.3964 | 132.6234 | 187.6044 |

## Notes

- This is a partial-window simulation starting at age `65` because the bundle only solves ages `65-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
