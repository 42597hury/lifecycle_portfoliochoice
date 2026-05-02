# Euler Error Report: `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`

## Setup

- Bundle: `saved_runs\checkpoints\unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`
- Solved window: ages `65-99`
- Simulation households: `32`
- Evaluated households per age cap: `8`
- Seed: `42`
- Return draw mode: `monte_carlo`
- Initial z: `stationary`
- Initial state: `stationary`
- Warm-start source bundle: `unconstrained_principal_grid5x5x5_nz9`
- Eval mode: `same`
- Policy quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`
- Eval quadrature: `ret=(3, 7, 5)`, `state=(2, 2, 5)`, `eta=3`, `eps=3`

## Gates

| phase | grade | pass | mean log10|EE| | gate | max log10|EE| | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| working | publication | no | -1.497 | < -4.0 | 1.660 | < -3.0 |
| retirement | publication | no | 0.018 | < -4.5 | 3.359 | < -3.0 |
| working | welfare | no | -1.497 | < -5.0 | 1.660 | < -4.0 |
| retirement | welfare | no | 0.018 | < -5.5 | 3.359 | < -4.0 |

## By Age

| age | phase | alive | eval | valid | invalid | mean log10|EE| | p99 log10|EE| | max log10|EE| | mean |EE| % | p95 |EE| % | max |EE| % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | working | 32 | 8 | 8 | 0 | -0.962 | 1.651 | 1.660 | 1013.5243 | 4186.8660 | 4568.2727 |
| 66 | working | 31 | 8 | 8 | 0 | -2.032 | -0.427 | -0.425 | 9.9392 | 36.7351 | 37.6187 |
| 67 | retirement | 31 | 8 | 8 | 0 | -1.278 | 1.865 | 1.946 | 1191.5071 | 5956.2540 | 8834.9605 |
| 68 | retirement | 31 | 8 | 8 | 0 | 0.025 | 2.128 | 2.144 | 3752.5907 | 11953.1354 | 13935.5649 |
| 69 | retirement | 30 | 8 | 8 | 0 | 0.054 | 2.179 | 2.191 | 5223.7748 | 13774.2416 | 15522.9963 |
| 70 | retirement | 30 | 8 | 8 | 0 | -0.809 | 1.863 | 1.901 | 1306.5114 | 5979.4164 | 7957.7325 |
| 71 | retirement | 29 | 8 | 8 | 0 | 0.302 | 2.273 | 2.293 | 5137.8464 | 16256.8707 | 19645.1632 |
| 72 | retirement | 29 | 8 | 8 | 0 | -0.330 | 2.057 | 2.095 | 2067.8456 | 9323.1723 | 12456.1224 |
| 73 | retirement | 28 | 8 | 8 | 0 | -0.421 | 2.251 | 2.277 | 3505.5749 | 15167.7611 | 18917.4987 |
| 74 | retirement | 27 | 8 | 8 | 0 | -1.055 | 2.259 | 2.307 | 3174.4972 | 14662.4823 | 20290.6517 |
| 75 | retirement | 25 | 8 | 8 | 0 | -0.769 | 2.295 | 2.310 | 4107.8313 | 17620.6503 | 20415.3612 |
| 76 | retirement | 25 | 8 | 8 | 0 | 0.664 | 2.376 | 2.387 | 9735.9434 | 21824.1491 | 24358.0829 |
| 77 | retirement | 25 | 8 | 8 | 0 | -0.336 | 2.457 | 2.487 | 5269.8527 | 23904.3806 | 30724.6027 |
| 78 | retirement | 25 | 8 | 8 | 0 | -1.160 | 1.030 | 1.103 | 185.2838 | 863.8007 | 1267.5303 |
| 79 | retirement | 24 | 8 | 8 | 0 | 0.401 | 2.655 | 2.672 | 10572.8442 | 39910.2061 | 47038.8518 |
| 80 | retirement | 24 | 8 | 8 | 0 | 0.131 | 2.362 | 2.378 | 4993.2697 | 20437.8322 | 23852.9710 |
| 81 | retirement | 23 | 8 | 8 | 0 | -0.021 | 2.446 | 2.457 | 6321.1031 | 25781.5931 | 28612.7056 |
| 82 | retirement | 21 | 8 | 8 | 0 | 0.821 | 2.858 | 2.876 | 25973.8278 | 63597.9999 | 75122.0171 |
| 83 | retirement | 21 | 8 | 8 | 0 | 0.037 | 2.697 | 2.721 | 9817.3330 | 42754.0249 | 52546.6391 |
| 84 | retirement | 19 | 8 | 8 | 0 | 0.550 | 2.717 | 2.735 | 14538.6919 | 45628.6444 | 54376.7734 |
| 85 | retirement | 18 | 8 | 8 | 0 | 0.376 | 2.861 | 2.866 | 18431.9747 | 69362.1466 | 73535.1856 |
| 86 | retirement | 15 | 8 | 8 | 0 | 0.009 | 3.218 | 3.271 | 31284.2517 | 132908.4254 | 186526.1678 |
| 87 | retirement | 12 | 8 | 8 | 0 | 0.758 | 3.103 | 3.137 | 27276.3487 | 104747.5851 | 137000.9575 |
| 88 | retirement | 11 | 8 | 8 | 0 | 0.291 | 3.220 | 3.268 | 28325.0490 | 133788.7078 | 185301.6050 |
| 89 | retirement | 10 | 8 | 8 | 0 | 0.425 | 2.678 | 2.686 | 12549.9584 | 44663.4845 | 48565.9988 |
| 90 | retirement | 9 | 8 | 8 | 0 | -0.250 | 2.693 | 2.748 | 8702.4947 | 39605.2445 | 55914.0588 |
| 91 | retirement | 9 | 8 | 8 | 0 | 0.060 | 2.711 | 2.722 | 13024.0782 | 47160.4752 | 52664.1103 |
| 92 | retirement | 9 | 8 | 8 | 0 | 0.198 | 2.824 | 2.833 | 16375.8224 | 62072.8466 | 68129.0619 |
| 93 | retirement | 8 | 8 | 8 | 0 | 0.489 | 2.980 | 2.995 | 21460.9829 | 85147.9459 | 98917.5101 |
| 94 | retirement | 6 | 6 | 6 | 0 | 0.284 | 3.007 | 3.017 | 28980.7889 | 94256.3427 | 104035.3118 |
| 95 | retirement | 6 | 6 | 6 | 0 | 0.455 | 3.147 | 3.167 | 34875.6614 | 124633.4493 | 146937.8680 |
| 96 | retirement | 5 | 5 | 5 | 0 | -0.179 | 2.990 | 3.042 | 23138.4493 | 89176.1030 | 110065.6116 |
| 97 | retirement | 5 | 5 | 5 | 0 | 0.245 | 3.206 | 3.259 | 38006.2090 | 146832.6543 | 181388.1634 |
| 98 | retirement | 4 | 4 | 4 | 0 | 0.607 | 3.317 | 3.359 | 59425.8131 | 195537.3825 | 228404.3223 |

## Notes

- This is a partial-window simulation starting at age `65` because the bundle only solves ages `65-99`.
- `EE` is reported in the unit-free consumption-error form `1 - c_implied / c_policy`, where `c_implied` is backed out from an independent expectation rule.
- The working/retirement publication-grade gates follow `contextfiles/GRID_CONVERGENCE_CRITERIA.md`.
