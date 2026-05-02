# Euler Error Report: `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`

## Setup

- Bundle: `saved_runs\checkpoints\unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`
- Solved window: ages `65-99`
- Simulation households: `8`
- Evaluated households per age cap: `2`
- Seed: `42`
- Return draw mode: `monte_carlo`
- Initial z: `stationary`
- Initial state: `stationary`
- Warm-start source bundle: `unconstrained_principal_grid5x5x5_nz9`
- Eval mode: `next_finer`
- Policy quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`
- Eval quadrature: `ret=(5, 9, 7)`, `state=(3, 3, 6)`, `eta=5`, `eps=5`

## Gates

| phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| working | publication | no | 0.407 | < -4.0 | 3.270 | < -3.0 |
| retirement | publication | no | 1.714 | < -4.5 | 4.352 | < -3.0 |
| working | welfare | no | 0.407 | < -5.0 | 3.270 | < -4.0 |
| retirement | welfare | no | 1.714 | < -5.5 | 4.352 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | working | 8 | 2 | 2 | 0 | -1.686 | -1.377 | -1.370 | 2.6301 | 4.0986 | 4.2618 |
| 66 | working | 8 | 2 | 2 | 0 | 2.500 | 3.255 | 3.270 | 95839.7561 | 177271.4318 | 186319.3958 |
| 67 | retirement | 8 | 2 | 2 | 0 | 1.690 | 1.724 | 1.725 | 4917.4887 | 5268.6003 | 5307.6127 |
| 68 | retirement | 8 | 2 | 2 | 0 | 2.166 | 3.304 | 3.327 | 106688.1174 | 201797.4718 | 212365.1778 |
| 69 | retirement | 8 | 2 | 2 | 0 | -0.112 | 1.621 | 1.656 | 2266.7454 | 4305.6309 | 4532.1737 |
| 70 | retirement | 8 | 2 | 2 | 0 | 1.919 | 2.107 | 2.110 | 9113.0493 | 12514.8681 | 12892.8480 |
| 71 | retirement | 8 | 2 | 2 | 0 | 0.859 | 3.466 | 3.519 | 165221.2341 | 313918.9234 | 330440.8889 |
| 72 | retirement | 8 | 2 | 2 | 0 | -0.005 | 1.701 | 1.736 | 2723.9300 | 5173.8549 | 5446.0688 |
| 73 | retirement | 8 | 2 | 2 | 0 | 2.186 | 2.492 | 2.498 | 19481.9561 | 30283.4789 | 31483.6481 |
| 74 | retirement | 8 | 2 | 2 | 0 | 1.822 | 2.575 | 2.591 | 20051.8348 | 37083.1530 | 38975.5217 |
| 75 | retirement | 7 | 2 | 2 | 0 | 1.971 | 2.017 | 2.018 | 9407.7423 | 10328.7706 | 10431.1071 |
| 76 | retirement | 7 | 2 | 2 | 0 | 3.644 | 4.316 | 4.330 | 1114417.0413 | 2035784.7052 | 2138158.8901 |
| 77 | retirement | 7 | 2 | 2 | 0 | 2.065 | 2.070 | 2.070 | 11602.6510 | 11739.6044 | 11754.8215 |
| 78 | retirement | 7 | 2 | 2 | 0 | 2.663 | 4.308 | 4.342 | 1099414.4471 | 2088021.0336 | 2197866.2099 |
| 79 | retirement | 7 | 2 | 2 | 0 | -0.467 | 0.973 | 1.003 | 503.6781 | 955.9487 | 1006.2010 |
| 80 | retirement | 7 | 2 | 2 | 0 | 0.722 | 4.279 | 4.352 | 1123783.7770 | 2135189.0649 | 2247567.4302 |
| 81 | retirement | 7 | 2 | 2 | 0 | 1.659 | 2.252 | 2.264 | 9749.0487 | 17503.7769 | 18365.4134 |
| 82 | retirement | 6 | 2 | 2 | 0 | 0.037 | 2.236 | 2.281 | 9553.0678 | 18150.2713 | 19105.5162 |
| 83 | retirement | 6 | 2 | 2 | 0 | 1.862 | 2.322 | 2.332 | 11964.9340 | 20516.1654 | 21466.3022 |
| 84 | retirement | 6 | 2 | 2 | 0 | 1.879 | 2.290 | 2.298 | 11374.7684 | 19019.0022 | 19868.3615 |
| 85 | retirement | 6 | 2 | 2 | 0 | 3.034 | 3.468 | 3.477 | 169321.7771 | 286620.5106 | 299653.7032 |
| 86 | retirement | 5 | 2 | 2 | 0 | 0.318 | 2.495 | 2.539 | 17313.0166 | 32893.6071 | 34624.7838 |
| 87 | retirement | 4 | 2 | 2 | 0 | 3.031 | 3.544 | 3.554 | 195110.7266 | 341783.4888 | 358080.4624 |
| 88 | retirement | 4 | 2 | 2 | 0 | 3.061 | 3.662 | 3.674 | 250204.6703 | 450157.4433 | 472374.4181 |
| 89 | retirement | 3 | 2 | 2 | 0 | 2.582 | 2.716 | 2.718 | 40069.1618 | 51069.1010 | 52291.3164 |
| 90 | retirement | 1 | 1 | 1 | 0 | 2.559 | 2.559 | 2.559 | 36254.1398 | 36254.1398 | 36254.1398 |
| 91 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 92 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 93 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 94 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 95 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 96 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 97 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |
| 98 | retirement | 0 | 0 | 0 | 0 | nan | nan | nan | nan | nan | nan |

## Notes

- This is a partial-window simulation starting at age `65` because the bundle only solves ages `65-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
