# Euler Error Report: `system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1`

## Setup

- Bundle: `saved_runs\checkpoints\system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1`
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
| working | publication | no | -3.014 | < -4.0 | -2.365 | < -3.0 |
| retirement | publication | no | -2.467 | < -4.5 | -0.791 | < -3.0 |
| working | welfare | no | -3.014 | < -5.0 | -2.365 | < -4.0 |
| retirement | welfare | no | -2.467 | < -5.5 | -0.791 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | working | 64 | 16 | 16 | 0 | -3.152 | -3.058 | -3.058 | 0.0723 | 0.0874 | 0.0875 |
| 66 | working | 62 | 16 | 16 | 0 | -2.876 | -2.368 | -2.365 | 0.1879 | 0.4150 | 0.4315 |
| 67 | retirement | 62 | 16 | 16 | 0 | -2.754 | -2.194 | -2.186 | 0.2866 | 0.5959 | 0.6521 |
| 68 | retirement | 62 | 16 | 16 | 0 | -2.643 | -1.306 | -1.114 | 0.6858 | 2.2260 | 7.6838 |
| 69 | retirement | 61 | 16 | 16 | 0 | -2.963 | -2.268 | -2.256 | 0.1807 | 0.4833 | 0.5548 |
| 70 | retirement | 61 | 16 | 16 | 0 | -2.769 | -1.477 | -1.395 | 0.5577 | 1.8603 | 4.0298 |
| 71 | retirement | 61 | 16 | 16 | 0 | -2.908 | -2.312 | -2.293 | 0.2080 | 0.4122 | 0.5099 |
| 72 | retirement | 59 | 16 | 16 | 0 | -2.496 | -1.781 | -1.703 | 0.4287 | 0.9469 | 1.9815 |
| 73 | retirement | 59 | 16 | 16 | 0 | -2.608 | -1.536 | -1.504 | 0.5558 | 2.2239 | 3.1333 |
| 74 | retirement | 57 | 16 | 16 | 0 | -2.345 | -1.512 | -1.428 | 0.7400 | 1.7003 | 3.7312 |
| 75 | retirement | 57 | 16 | 16 | 0 | -2.538 | -2.085 | -2.070 | 0.4073 | 0.7229 | 0.8509 |
| 76 | retirement | 56 | 16 | 16 | 0 | -2.495 | -1.961 | -1.951 | 0.4842 | 0.9941 | 1.1201 |
| 77 | retirement | 52 | 16 | 16 | 0 | -2.417 | -1.376 | -1.332 | 0.8494 | 2.9223 | 4.6608 |
| 78 | retirement | 50 | 16 | 16 | 0 | -2.501 | -1.789 | -1.754 | 0.5239 | 1.2114 | 1.7629 |
| 79 | retirement | 48 | 16 | 16 | 0 | -2.458 | -1.762 | -1.738 | 0.5345 | 1.4047 | 1.8290 |
| 80 | retirement | 48 | 16 | 16 | 0 | -2.310 | -1.441 | -1.410 | 0.8396 | 2.7701 | 3.8932 |
| 81 | retirement | 46 | 16 | 16 | 0 | -2.643 | -1.869 | -1.863 | 0.4509 | 1.2788 | 1.3724 |
| 82 | retirement | 44 | 16 | 16 | 0 | -2.266 | -1.808 | -1.787 | 0.6757 | 1.3032 | 1.6323 |
| 83 | retirement | 44 | 16 | 16 | 0 | -2.420 | -1.505 | -1.489 | 0.7803 | 2.7285 | 3.2418 |
| 84 | retirement | 42 | 16 | 16 | 0 | -2.630 | -2.128 | -2.116 | 0.3415 | 0.6670 | 0.7665 |
| 85 | retirement | 38 | 16 | 16 | 0 | -2.540 | -1.458 | -1.401 | 0.7410 | 2.2457 | 3.9699 |
| 86 | retirement | 36 | 16 | 16 | 0 | -2.289 | -1.536 | -1.473 | 0.7184 | 1.7954 | 3.3686 |
| 87 | retirement | 31 | 16 | 16 | 0 | -2.335 | -1.848 | -1.819 | 0.5666 | 1.1011 | 1.5180 |
| 88 | retirement | 30 | 16 | 16 | 0 | -2.186 | -1.874 | -1.865 | 0.7657 | 1.2232 | 1.3657 |
| 89 | retirement | 25 | 16 | 16 | 0 | -2.247 | -0.960 | -0.791 | 1.5334 | 4.9542 | 16.1667 |
| 90 | retirement | 22 | 16 | 16 | 0 | -2.325 | -1.996 | -1.995 | 0.5349 | 1.0080 | 1.0106 |
| 91 | retirement | 17 | 16 | 16 | 0 | -2.351 | -1.960 | -1.959 | 0.6101 | 1.0885 | 1.0983 |
| 92 | retirement | 16 | 16 | 16 | 0 | -2.259 | -1.938 | -1.935 | 0.6757 | 1.1172 | 1.1612 |
| 93 | retirement | 16 | 16 | 16 | 0 | -2.338 | -1.964 | -1.950 | 0.5366 | 0.9609 | 1.1213 |
| 94 | retirement | 14 | 14 | 14 | 0 | -2.515 | -2.070 | -2.061 | 0.3730 | 0.7851 | 0.8690 |
| 95 | retirement | 14 | 14 | 14 | 0 | -2.336 | -1.976 | -1.975 | 0.5561 | 1.0562 | 1.0581 |
| 96 | retirement | 10 | 10 | 10 | 0 | -2.443 | -2.108 | -2.105 | 0.4470 | 0.7593 | 0.7851 |
| 97 | retirement | 8 | 8 | 8 | 0 | -2.347 | -1.962 | -1.952 | 0.5450 | 1.0120 | 1.1168 |
| 98 | retirement | 7 | 7 | 7 | 0 | -2.280 | -2.064 | -2.060 | 0.5559 | 0.8338 | 0.8714 |

## Notes

- This is a partial-window simulation starting at age `65` because the bundle only solves ages `65-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
