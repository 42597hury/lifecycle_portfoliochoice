# Policy Convergence Diagnostic Report

Model bundle: `saved_runs\unconstrained_principal_grid5x5x5_nz9`
Reference bundle: `saved_runs\checkpoints\unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2`
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
| `unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2` | `stopped_early` | `65-99` | 15,743,700 | 0 | 100.0000% | 1.000e-07 | 0.643 | 7 | 0 | conv, foc, mono, pass |
| `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2` | `stopped_early` | `65-99` | 5,737,500 | 0 | 100.0000% | 1.000e-07 | 0.641 | 7 | 0 | conv, foc, mono, pass |

## Policy Drift vs Reference

Reference bundle: `saved_runs\checkpoints\unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2`

### `unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`

Common age probes: `[65, 66, 67, 75, 85, 95]`
Probe count per metric: `882`

| Segment | Metric | Median | P95 | Max |
| --- | --- | ---: | ---: | ---: |
| `all` | `C rel %` | 0.0031 | 4.9946 | 5.4948 |
| `all` | `C/W rel %` | 0.0031 | 4.9946 | 5.4948 |
| `all` | `Stock share pp` | 0.0113 | 2.3393 | 9.5467 |
| `all` | `Bond share pp` | 0.0210 | 3.7134 | 8.9131 |
| `all` | `Bill share pp` | 0.0229 | 3.1770 | 8.0977 |
| `working` | `C rel %` | 0.0063 | 5.2172 | 5.4948 |
| `working` | `C/W rel %` | 0.0063 | 5.2172 | 5.4948 |
| `working` | `Stock share pp` | 0.0195 | 2.5635 | 9.5467 |
| `working` | `Bond share pp` | 0.0231 | 4.3299 | 8.9131 |
| `working` | `Bill share pp` | 0.0318 | 3.6623 | 7.9758 |
| `retirement` | `C rel %` | 0.0027 | 4.6372 | 5.3644 |
| `retirement` | `C/W rel %` | 0.0027 | 4.6372 | 5.3644 |
| `retirement` | `Stock share pp` | 0.0084 | 1.9870 | 4.7935 |
| `retirement` | `Bond share pp` | 0.0173 | 3.1521 | 8.0091 |
| `retirement` | `Bill share pp` | 0.0183 | 2.4109 | 8.0977 |
