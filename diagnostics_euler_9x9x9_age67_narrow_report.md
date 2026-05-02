# Euler Error Report: `unconstrained_principal_grid9x9x9_nz9_retirement_only`

## Setup

- Bundle: `saved_runs\checkpoints\unconstrained_principal_grid9x9x9_nz9_retirement_only`
- Solved window: ages `67-99`
- Simulation households: `64`
- Evaluated households per age cap: `16`
- Seed: `42`
- Return draw mode: `monte_carlo`
- Initial z: `median`
- Initial state: `median`
- Partial init mode: `centered`
- Eval mode: `same`
- Policy quadrature: `ret=(3, 7, 5)`, `state=3`, `eta=3`, `eps=3`
- Eval quadrature: `ret=(3, 7, 5)`, `state=(3, 3, 3)`, `eta=3`, `eps=3`

## Gates

| phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| working | publication | no | nan | < nan | nan | < nan |
| retirement | publication | no | -1.567 | < -4.5 | -0.103 | < -3.0 |
| working | welfare | no | nan | < nan | nan | < nan |
| retirement | welfare | no | -1.567 | < -5.5 | -0.103 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 67 | retirement | 64 | 16 | 16 | 0 | -2.948 | -2.948 | -2.948 | 0.1128 | 0.1128 | 0.1128 |
| 68 | retirement | 64 | 16 | 16 | 0 | -1.606 | -0.924 | -0.866 | 3.5159 | 7.5716 | 13.6204 |
| 69 | retirement | 62 | 16 | 16 | 0 | -1.916 | -0.655 | -0.563 | 3.6191 | 11.8478 | 27.3717 |
| 70 | retirement | 61 | 16 | 16 | 0 | -1.717 | -0.450 | -0.413 | 7.8095 | 25.9015 | 38.6756 |
| 71 | retirement | 58 | 16 | 15 | 1 | -1.778 | -0.897 | -0.858 | 2.9481 | 9.2448 | 13.8650 |
| 72 | retirement | 55 | 16 | 14 | 2 | -1.526 | -0.680 | -0.668 | 7.8164 | 18.8234 | 21.4787 |
| 73 | retirement | 53 | 16 | 14 | 2 | -1.629 | -0.648 | -0.639 | 5.4126 | 20.7458 | 22.9760 |
| 74 | retirement | 49 | 16 | 14 | 2 | -1.695 | -0.761 | -0.721 | 4.1660 | 12.7641 | 18.9982 |
| 75 | retirement | 47 | 16 | 15 | 1 | -1.574 | -0.510 | -0.501 | 7.8067 | 28.6132 | 31.5223 |
| 76 | retirement | 44 | 16 | 15 | 1 | -1.402 | -0.600 | -0.548 | 6.7825 | 16.9841 | 28.2867 |
| 77 | retirement | 42 | 16 | 15 | 1 | -1.617 | -0.758 | -0.752 | 4.8949 | 16.5157 | 17.6976 |
| 78 | retirement | 42 | 16 | 15 | 1 | -1.428 | -0.515 | -0.510 | 8.9150 | 29.1890 | 30.9128 |
| 79 | retirement | 41 | 16 | 14 | 2 | -1.204 | -0.587 | -0.577 | 9.7448 | 23.6945 | 26.4866 |
| 80 | retirement | 41 | 16 | 14 | 2 | -1.470 | -0.778 | -0.759 | 5.7376 | 14.2473 | 17.4071 |
| 81 | retirement | 39 | 16 | 16 | 0 | -1.573 | -0.643 | -0.616 | 6.2201 | 18.0337 | 24.1829 |
| 82 | retirement | 37 | 16 | 14 | 2 | -1.762 | -0.884 | -0.884 | 3.2364 | 13.0289 | 13.0695 |
| 83 | retirement | 37 | 16 | 16 | 0 | -1.670 | -1.027 | -1.025 | 3.2357 | 9.1836 | 9.4402 |
| 84 | retirement | 30 | 16 | 15 | 1 | -1.539 | -0.665 | -0.660 | 7.5614 | 20.6852 | 21.8989 |
| 85 | retirement | 29 | 16 | 15 | 1 | -1.630 | -0.575 | -0.559 | 6.4874 | 23.1156 | 27.6003 |
| 86 | retirement | 27 | 16 | 13 | 3 | -1.323 | -0.743 | -0.733 | 7.5334 | 16.5184 | 18.5050 |
| 87 | retirement | 26 | 16 | 15 | 1 | -1.491 | -0.629 | -0.619 | 7.5018 | 21.4104 | 24.0552 |
| 88 | retirement | 24 | 16 | 15 | 1 | -1.412 | -0.779 | -0.771 | 6.1198 | 15.4563 | 16.9523 |
| 89 | retirement | 23 | 16 | 15 | 1 | -1.376 | -0.840 | -0.827 | 5.4697 | 12.8938 | 14.8886 |
| 90 | retirement | 21 | 16 | 16 | 0 | -1.435 | -0.542 | -0.536 | 7.0139 | 27.2392 | 29.0799 |
| 91 | retirement | 19 | 16 | 16 | 0 | -1.524 | -0.452 | -0.391 | 7.7184 | 22.1509 | 40.6749 |
| 92 | retirement | 17 | 16 | 16 | 0 | -1.314 | -0.581 | -0.574 | 9.4999 | 24.7564 | 26.6445 |
| 93 | retirement | 15 | 15 | 15 | 0 | -1.421 | -0.823 | -0.806 | 6.2664 | 12.9892 | 15.6270 |
| 94 | retirement | 13 | 13 | 13 | 0 | -1.570 | -0.485 | -0.443 | 6.9070 | 24.0603 | 36.0670 |
| 95 | retirement | 11 | 11 | 11 | 0 | -1.446 | -0.759 | -0.755 | 6.9445 | 16.8359 | 17.5664 |
| 96 | retirement | 7 | 7 | 6 | 1 | -1.746 | -0.799 | -0.775 | 5.0797 | 14.0045 | 16.7910 |
| 97 | retirement | 5 | 5 | 5 | 0 | -1.086 | -0.126 | -0.103 | 21.9678 | 67.2140 | 78.8853 |
| 98 | retirement | 4 | 4 | 4 | 0 | -1.322 | -0.338 | -0.313 | 15.3424 | 42.3954 | 48.6542 |

## Notes

- This is a partial-window simulation starting at age `67` because the bundle only solves ages `67-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
