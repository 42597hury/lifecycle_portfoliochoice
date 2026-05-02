# Euler Error Report: `unconstrained_principal_grid9x9x9_nz9_from_age70_kret3x7x5_ns2p0x2p25x2p25_v2`

## Setup

- Bundle: `saved_runs\checkpoints\unconstrained_principal_grid9x9x9_nz9_from_age70_kret3x7x5_ns2p0x2p25x2p25_v2`
- Solved window: ages `70-99`
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
| working | publication | no | nan | < nan | nan | < nan |
| retirement | publication | no | -2.591 | < -4.5 | -0.836 | < -3.0 |
| working | welfare | no | nan | < nan | nan | < nan |
| retirement | welfare | no | -2.591 | < -5.5 | -0.836 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70 | retirement | 64 | 16 | 16 | 0 | -3.157 | -3.157 | -3.157 | 0.0696 | 0.0696 | 0.0696 |
| 71 | retirement | 61 | 16 | 16 | 0 | -3.090 | -2.598 | -2.597 | 0.1203 | 0.2507 | 0.2530 |
| 72 | retirement | 60 | 16 | 16 | 0 | -2.959 | -2.569 | -2.558 | 0.1373 | 0.2428 | 0.2769 |
| 73 | retirement | 60 | 16 | 16 | 0 | -3.118 | -1.687 | -1.540 | 0.2818 | 0.9462 | 2.8856 |
| 74 | retirement | 60 | 16 | 16 | 0 | -2.784 | -2.363 | -2.357 | 0.1875 | 0.4124 | 0.4395 |
| 75 | retirement | 59 | 16 | 16 | 0 | -2.722 | -2.328 | -2.317 | 0.2189 | 0.4283 | 0.4818 |
| 76 | retirement | 56 | 16 | 16 | 0 | -2.601 | -1.497 | -1.361 | 0.5089 | 1.4922 | 4.3600 |
| 77 | retirement | 54 | 16 | 16 | 0 | -2.497 | -1.748 | -1.647 | 0.4250 | 0.9231 | 2.2530 |
| 78 | retirement | 50 | 16 | 16 | 0 | -2.518 | -1.511 | -1.490 | 0.6100 | 2.5610 | 3.2369 |
| 79 | retirement | 46 | 16 | 16 | 0 | -2.673 | -1.807 | -1.713 | 0.3402 | 0.8243 | 1.9369 |
| 80 | retirement | 45 | 16 | 16 | 0 | -2.542 | -1.811 | -1.739 | 0.3943 | 0.9068 | 1.8255 |
| 81 | retirement | 38 | 16 | 16 | 0 | -2.468 | -1.876 | -1.816 | 0.4007 | 0.8389 | 1.5259 |
| 82 | retirement | 35 | 16 | 16 | 0 | -2.538 | -1.476 | -1.399 | 0.5809 | 1.9233 | 3.9867 |
| 83 | retirement | 35 | 16 | 16 | 0 | -2.573 | -1.580 | -1.493 | 0.4869 | 1.4359 | 3.2170 |
| 84 | retirement | 35 | 16 | 16 | 0 | -2.587 | -1.735 | -1.666 | 0.4246 | 1.1033 | 2.1583 |
| 85 | retirement | 33 | 16 | 16 | 0 | -2.532 | -1.484 | -1.359 | 0.5710 | 1.5784 | 4.3755 |
| 86 | retirement | 29 | 16 | 16 | 0 | -2.383 | -1.101 | -0.991 | 1.0430 | 3.9551 | 10.2142 |
| 87 | retirement | 28 | 16 | 16 | 0 | -2.588 | -2.189 | -2.178 | 0.3186 | 0.5848 | 0.6639 |
| 88 | retirement | 26 | 16 | 16 | 0 | -2.571 | -1.678 | -1.615 | 0.4739 | 1.3012 | 2.4269 |
| 89 | retirement | 21 | 16 | 16 | 0 | -2.702 | -2.202 | -2.185 | 0.2975 | 0.5412 | 0.6532 |
| 90 | retirement | 21 | 16 | 16 | 0 | -2.473 | -2.164 | -2.157 | 0.3796 | 0.6434 | 0.6959 |
| 91 | retirement | 21 | 16 | 16 | 0 | -2.696 | -2.130 | -2.110 | 0.2840 | 0.6233 | 0.7767 |
| 92 | retirement | 19 | 16 | 16 | 0 | -2.477 | -1.965 | -1.928 | 0.4143 | 0.8000 | 1.1799 |
| 93 | retirement | 17 | 16 | 16 | 0 | -2.630 | -2.350 | -2.349 | 0.2613 | 0.4411 | 0.4475 |
| 94 | retirement | 16 | 16 | 16 | 0 | -2.455 | -2.152 | -2.138 | 0.3849 | 0.6243 | 0.7278 |
| 95 | retirement | 14 | 14 | 14 | 0 | -2.176 | -1.267 | -1.209 | 1.0945 | 3.6090 | 6.1772 |
| 96 | retirement | 8 | 8 | 8 | 0 | -2.141 | -0.863 | -0.836 | 2.8311 | 11.6255 | 14.5715 |
| 97 | retirement | 8 | 8 | 8 | 0 | -2.247 | -1.210 | -1.155 | 1.2802 | 4.9560 | 7.0019 |
| 98 | retirement | 7 | 7 | 7 | 0 | -2.243 | -1.208 | -1.153 | 1.3653 | 5.1776 | 7.0243 |

## Notes

- This is a partial-window simulation starting at age `70` because the bundle only solves ages `70-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
