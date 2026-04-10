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

### [MEDIUM] [GRID] Solver income interpolation error ~17%

The solver linearly interpolates after-tax income between z-grid points:
`income = (1-frac) * table[iz_lo, ie] + frac * table[iz_lo+1, ie]`.
The simulation computes income exactly: `tax(exp(f(age) + z + eps))`.
Because `exp(z)` is convex, linear interpolation over `dz = 1.12` (`n_z = 11`,
`dz/sigma_eta = 4.48`) introduces up to ~17% relative error at midpoints
between grid cells. The error is worst at high z where `exp(z)` curves most
steeply.

The simulation is more accurate here — the issue is on the solver side.
Increasing `n_z` (reducing `dz`) would shrink the error mechanically.

## Returns

## State Variables

## Simulation

## Pension

### [MEDIUM] [GRID] Solver pension interpolation error > 5%

Same mechanism as the income interpolation issue. The pension formula involves
`exp(z)` through AIME (average lifetime earnings). The solver interpolates
pension linearly between z-grid points, while the simulation computes it
directly from continuous z. With `dz = 1.12`, linear interpolation of the
nonlinear pension function exceeds 5% relative error at cell midpoints.

Fix is the same: increase `n_z` to reduce `dz`.
