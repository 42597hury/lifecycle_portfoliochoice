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
- Eval mode: `same`
- Policy quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`
- Eval quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`

## Gates

| phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| working | publication | no | nan | < nan | nan | < nan |
| retirement | publication | no | -2.439 | < -4.5 | -0.703 | < -3.0 |
| working | welfare | no | nan | < nan | nan | < nan |
| retirement | welfare | no | -2.439 | < -5.5 | -0.703 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 87 | retirement | 5000 | 256 | 256 | 0 | -4.975 | -4.281 | -4.278 | 0.0022 | 0.0051 | 0.0053 |
| 88 | retirement | 4610 | 256 | 256 | 0 | -2.184 | -1.867 | -1.784 | 0.7228 | 1.1465 | 1.6461 |
| 89 | retirement | 4187 | 256 | 256 | 0 | -2.174 | -1.868 | -1.842 | 0.7371 | 1.2386 | 1.4403 |
| 90 | retirement | 3787 | 256 | 256 | 0 | -2.181 | -1.753 | -1.174 | 0.7723 | 1.2671 | 6.7001 |
| 91 | retirement | 3404 | 256 | 256 | 0 | -2.184 | -1.144 | -0.973 | 0.8391 | 1.2366 | 10.6450 |
| 92 | retirement | 2964 | 256 | 256 | 0 | -2.225 | -1.903 | -1.042 | 0.7010 | 1.1804 | 9.0808 |
| 93 | retirement | 2599 | 256 | 256 | 0 | -2.201 | -1.349 | -1.061 | 0.7690 | 1.1906 | 8.6820 |
| 94 | retirement | 2213 | 256 | 256 | 0 | -2.201 | -1.432 | -0.703 | 0.8214 | 1.1426 | 19.7963 |
| 95 | retirement | 1856 | 256 | 256 | 0 | -2.247 | -1.471 | -1.331 | 0.6938 | 1.1570 | 4.6662 |
| 96 | retirement | 1554 | 256 | 256 | 0 | -2.219 | -1.314 | -0.751 | 0.8222 | 1.3207 | 17.7397 |
| 97 | retirement | 1289 | 256 | 256 | 0 | -2.233 | -1.198 | -0.743 | 0.8724 | 1.5055 | 18.0840 |
| 98 | retirement | 1052 | 256 | 256 | 0 | -2.244 | -1.312 | -0.820 | 0.7704 | 1.3192 | 15.1522 |

## Notes

- This is a partial-window simulation starting at age `87` because the bundle only solves ages `87-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
