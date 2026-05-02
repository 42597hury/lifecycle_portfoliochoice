# Policy Convergence Diagnostic Report

Model bundle: `saved_runs\unconstrained_principal_grid5x5x5_nz9`
Reference bundle: `saved_runs\checkpoints\unconstrained_principal_grid5x5x5_nz9_from_age65_kret5x9x5_v2`
Threshold mode: `publication`

Implemented subset from `contextfiles/GRID_CONVERGENCE_CRITERIA.md`:

- bundle integrity / loadability
- solver-health gates (Newton convergence, worst FOC residual, monotonicity)
- common-probe policy drift versus a reference bundle

Not yet implemented here:

- simulation-path Euler errors
- Den Haan-Marcet style residual orthogonality
- boundary-mass simulation diagnostics

## Bundle Health

| Bundle | Status | Ages | Calls | Failures | Conv rate | Worst FOC | Avg iter | Max iter | Mono viol | Gates |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `unconstrained_principal_grid5x5x5_nz9_from_age65_kret5x9x5_v2` | `stopped_early` | `65-99` | 5,737,500 | 0 | 100.0000% | 1.000e-07 | 0.641 | 7 | 0 | conv, foc, mono, pass |
| `unconstrained_principal_grid5x5x5_nz9_from_age65_v2` | `stopped_early` | `65-99` | 5,737,500 | 0 | 100.0000% | 1.000e-07 | 0.636 | 7 | 0 | conv, foc, mono, pass |
| `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2` | `stopped_early` | `65-99` | 5,737,500 | 0 | 100.0000% | 1.000e-07 | 0.641 | 7 | 0 | conv, foc, mono, pass |

## Policy Drift vs Reference

Reference bundle: `saved_runs\checkpoints\unconstrained_principal_grid5x5x5_nz9_from_age65_kret5x9x5_v2`

### `unconstrained_principal_grid5x5x5_nz9_from_age65_v2`

Common age probes: `[65, 66, 67, 75, 85, 95]`
Probe count per metric: `882`

| Segment | Metric | Median | P95 | Max |
| --- | --- | ---: | ---: | ---: |
| `all` | `C rel %` | 0.0008 | 0.0475 | 0.3042 |
| `all` | `C/W rel %` | 0.0008 | 0.0475 | 0.3042 |
| `all` | `Stock share pp` | 0.0042 | 0.0745 | 16.6360 |
| `all` | `Bond share pp` | 0.0334 | 0.2779 | 29.1027 |
| `all` | `Bill share pp` | 0.0199 | 0.2183 | 29.8588 |
| `working` | `C rel %` | 0.0007 | 0.0544 | 0.3042 |
| `working` | `C/W rel %` | 0.0007 | 0.0544 | 0.3042 |
| `working` | `Stock share pp` | 0.0072 | 0.1266 | 16.6360 |
| `working` | `Bond share pp` | 0.0350 | 0.4483 | 29.1027 |
| `working` | `Bill share pp` | 0.0304 | 0.3264 | 29.8588 |
| `retirement` | `C rel %` | 0.0008 | 0.0385 | 0.1168 |
| `retirement` | `C/W rel %` | 0.0008 | 0.0385 | 0.1168 |
| `retirement` | `Stock share pp` | 0.0042 | 0.0603 | 0.5676 |
| `retirement` | `Bond share pp` | 0.0220 | 0.2030 | 2.1452 |
| `retirement` | `Bill share pp` | 0.0139 | 0.1699 | 1.7593 |

### `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`

Common age probes: `[65, 66, 67, 75, 85, 95]`
Probe count per metric: `882`

| Segment | Metric | Median | P95 | Max |
| --- | --- | ---: | ---: | ---: |
| `all` | `C rel %` | 0.0000 | 0.0003 | 0.1317 |
| `all` | `C/W rel %` | 0.0000 | 0.0003 | 0.1317 |
| `all` | `Stock share pp` | 0.0000 | 0.0031 | 10.0957 |
| `all` | `Bond share pp` | 0.0000 | 0.0042 | 15.0209 |
| `all` | `Bill share pp` | 0.0000 | 0.0022 | 16.7804 |
| `working` | `C rel %` | 0.0000 | 0.0017 | 0.1317 |
| `working` | `C/W rel %` | 0.0000 | 0.0017 | 0.1317 |
| `working` | `Stock share pp` | 0.0000 | 0.0080 | 10.0957 |
| `working` | `Bond share pp` | 0.0000 | 0.0118 | 15.0209 |
| `working` | `Bill share pp` | 0.0000 | 0.0070 | 16.7804 |
| `retirement` | `C rel %` | 0.0000 | 0.0001 | 0.0065 |
| `retirement` | `C/W rel %` | 0.0000 | 0.0001 | 0.0065 |
| `retirement` | `Stock share pp` | 0.0000 | 0.0007 | 0.0300 |
| `retirement` | `Bond share pp` | 0.0000 | 0.0010 | 0.0380 |
| `retirement` | `Bill share pp` | 0.0000 | 0.0007 | 0.0578 |
