# Euler Error Report: `unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2`

## Setup

- Bundle: `saved_runs\checkpoints\unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2`
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
| working | publication | no | -2.941 | < -4.0 | -1.922 | < -3.0 |
| retirement | publication | no | -2.167 | < -4.5 | 0.352 | < -3.0 |
| working | welfare | no | -2.941 | < -5.0 | -1.922 | < -4.0 |
| retirement | welfare | no | -2.167 | < -5.5 | 0.352 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | working | 64 | 16 | 16 | 0 | -3.432 | -3.432 | -3.432 | 0.0370 | 0.0370 | 0.0370 |
| 66 | working | 62 | 16 | 16 | 0 | -2.451 | -1.939 | -1.922 | 0.4699 | 0.9910 | 1.1954 |
| 67 | retirement | 62 | 16 | 16 | 0 | -2.421 | -1.395 | -1.303 | 0.7724 | 2.1499 | 4.9829 |
| 68 | retirement | 62 | 16 | 16 | 0 | -2.420 | -1.213 | -1.077 | 0.9231 | 2.8778 | 8.3742 |
| 69 | retirement | 61 | 16 | 16 | 0 | -2.586 | -1.890 | -1.870 | 0.4400 | 1.0825 | 1.3494 |
| 70 | retirement | 61 | 16 | 16 | 0 | -2.440 | -1.438 | -1.389 | 0.7293 | 2.4604 | 4.0833 |
| 71 | retirement | 61 | 16 | 16 | 0 | -2.425 | -1.358 | -1.302 | 0.8714 | 2.8322 | 4.9887 |
| 72 | retirement | 59 | 16 | 16 | 0 | -2.194 | -1.271 | -1.213 | 1.1095 | 3.4077 | 6.1285 |
| 73 | retirement | 59 | 16 | 16 | 0 | -2.113 | -1.164 | -1.162 | 1.6276 | 6.7570 | 6.8845 |
| 74 | retirement | 57 | 16 | 16 | 0 | -2.281 | -1.203 | -1.130 | 1.1599 | 3.6785 | 7.4144 |
| 75 | retirement | 57 | 16 | 16 | 0 | -2.344 | -1.458 | -1.413 | 0.7558 | 2.4310 | 3.8605 |
| 76 | retirement | 56 | 16 | 16 | 0 | -2.319 | -1.355 | -1.321 | 1.0731 | 3.3442 | 4.7713 |
| 77 | retirement | 52 | 16 | 16 | 0 | -2.120 | -1.303 | -1.280 | 1.4994 | 4.1051 | 5.2443 |
| 78 | retirement | 50 | 16 | 16 | 0 | -2.271 | -1.464 | -1.448 | 1.0107 | 2.9889 | 3.5631 |
| 79 | retirement | 48 | 16 | 16 | 0 | -2.318 | -1.547 | -1.482 | 0.7235 | 1.7378 | 3.2982 |
| 80 | retirement | 48 | 16 | 16 | 0 | -1.989 | -1.324 | -1.316 | 1.5863 | 4.3863 | 4.8326 |
| 81 | retirement | 46 | 16 | 16 | 0 | -2.045 | -1.466 | -1.461 | 1.3676 | 3.2788 | 3.4572 |
| 82 | retirement | 44 | 16 | 16 | 0 | -2.072 | -1.671 | -1.670 | 1.0287 | 2.1071 | 2.1371 |
| 83 | retirement | 44 | 16 | 16 | 0 | -2.091 | -1.330 | -1.318 | 1.5556 | 4.2087 | 4.8090 |
| 84 | retirement | 42 | 16 | 16 | 0 | -2.192 | -1.382 | -1.368 | 1.1232 | 3.6862 | 4.2825 |
| 85 | retirement | 38 | 16 | 16 | 0 | -2.076 | -1.250 | -1.225 | 1.8288 | 4.5463 | 5.9509 |
| 86 | retirement | 36 | 16 | 16 | 0 | -2.105 | -1.399 | -1.399 | 1.2715 | 3.9765 | 3.9944 |
| 87 | retirement | 31 | 16 | 16 | 0 | -2.120 | -1.376 | -1.363 | 1.2609 | 3.7605 | 4.3332 |
| 88 | retirement | 30 | 16 | 16 | 0 | -2.250 | -1.563 | -1.543 | 0.8154 | 2.3045 | 2.8653 |
| 89 | retirement | 25 | 16 | 16 | 0 | -1.972 | 0.129 | 0.281 | 14.2343 | 61.7456 | 190.8088 |
| 90 | retirement | 22 | 16 | 16 | 0 | -1.931 | 0.047 | 0.309 | 13.8077 | 53.6432 | 203.6282 |
| 91 | retirement | 17 | 16 | 16 | 0 | -2.114 | 0.050 | 0.325 | 14.1066 | 55.1354 | 211.1843 |
| 92 | retirement | 16 | 16 | 16 | 0 | -2.126 | 0.102 | 0.346 | 14.9156 | 59.4278 | 222.0093 |
| 93 | retirement | 16 | 16 | 16 | 0 | -2.030 | 0.082 | 0.349 | 15.0700 | 58.6313 | 223.3515 |
| 94 | retirement | 14 | 14 | 14 | 0 | -2.094 | 0.046 | 0.251 | 13.6567 | 65.4278 | 178.1066 |
| 95 | retirement | 14 | 14 | 14 | 0 | -2.094 | -0.018 | 0.174 | 11.7070 | 55.5161 | 149.3192 |
| 96 | retirement | 10 | 10 | 10 | 0 | -2.235 | 0.038 | 0.178 | 16.1108 | 84.8029 | 150.7975 |
| 97 | retirement | 8 | 8 | 8 | 0 | -1.869 | 0.052 | 0.165 | 19.3234 | 96.3436 | 146.3179 |
| 98 | retirement | 7 | 7 | 7 | 0 | -1.701 | 0.246 | 0.352 | 33.3884 | 158.6309 | 224.9978 |

## Notes

- This is a partial-window simulation starting at age `65` because the bundle only solves ages `65-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
