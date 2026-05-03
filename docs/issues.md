# Known Issues

Potential issues encountered when investigating the model.
Each section groups issues by the model component they affect.

### Priority labels

- **[HIGH]** — Meaningful accuracy concerns, fix before final results
- **[MEDIUM]** — Known limitations, address if time permits
- **[LOW]** — Minor or cosmetic issues

### Tags

- **[GRID]** — Related to discretization grid coarseness

---

## Labour Income

### ~~[MEDIUM] [GRID] Solver income interpolation error ~17%~~ — RESOLVED

The solver now computes income on-the-fly via `scalar_disposable_income`
at [solver.py:868](solver.py#L868). No z-interpolation of income remains.
Solver and simulation use the identical njit function — error is 0% by
construction. See LABOUR.md Section 4.6 and Section 5 for validation.

## Returns

## State Variables

## Simulation

### [HIGH] Wealth-grid extrapolation blowup

`fast_interp_1d` in [simulation.py:246](simulation.py#L246) linearly
extrapolates portfolio shares beyond `wealth_max = 200`. When agents
accumulate wealth past the grid boundary, extrapolated stock/bond shares
can exceed 1 (leverage), producing explosive returns that feed back into
more wealth growth. With proper z initialization (mean=0, std=sigma_z),
~0.1% of agents exceed the grid by age 67 and trigger the blowup.

Previously hidden because `initial_z="stationary"` was broken (see below),
keeping all agents at z_min with very low income and slow wealth accumulation.

Fix options: (a) clamp portfolio shares to [0, 1] in the simulation,
(b) flat-extrapolate policies beyond the grid instead of linear,
(c) increase `wealth_max`.

### [LOW] Pi_z transition matrix is degenerate — FIXED (workaround)

`Pi_z` from `discretize_income_ar1_mixture` is lower-bidiagonal (can only
stay or move down in z) with an absorbing state at iz=0. Its stationary
distribution puts 100% mass at z_grid[0]. Root cause is in the Tauchen
mixture-generalized discretization — the upper-triangular transitions are
missing.

Impact was limited: `Pi_z` is not used by the solver (which integrates via
GH quadrature on eta) or by the simulation's z dynamics (which draws eta
from the mixture directly). Only `initial_z="stationary"` consumed it.

Fix: `initial_z="stationary"` now draws from N(0, sigma_z^2) directly
([simulation.py:676](simulation.py#L676)), bypassing Pi_z entirely. The
underlying Pi_z bug remains but is dead code.

## Pension

### ~~[MEDIUM] [GRID] Solver pension interpolation error > 5%~~ — INVALID

Pension is not interpolated in z. The retirement FOC uses direct grid-index
lookup: `pension_next = pension_1d[z_i]` at [solver.py:1628](solver.py#L1628).
No z-interpolation occurs because z does not evolve during retirement (no eta
shocks). This issue was mislabeled — the mechanism described does not exist
in the code.
