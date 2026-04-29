# State Grid Implementation Plan

## Goal

Implement the new financial-state grid design from `state_grid_design_note.md` with correctness as the top priority.

Recommended production target:

- `state_grid_mode = "principal"`
- `state_n_stds = 3.0`
- `state_grid_sizes = (7, 7, 7)`

But the rollout should be staged so we do not silently break old saved bundles, diagnostics, or tests.

## What Is Already True In This Repo

The current codebase is not at the same baseline assumed by the older handoff docs.

These parts are already implemented:

- State innovation quadrature is already live in the solver.
- Continuous-state simulation is already live in `simulation.py`.
- The bill rate is already treated as a return dimension (`rtb`, `xr`, `xb`), not as a fixed current-state scalar in the working/retirement FOCs.
- The VAR is already estimated with the CCV constrained estimator, so `z_bar` is pinned to the sample mean. The older “grid centered on the wrong mean” issue is already fixed upstream in `var.py`.

Implication:

- This task is **not** “implement state quadrature”.
- This task **is** “change the geometry of the interpolation grid and thread the new coordinate system through the existing quadrature-based solver/simulation stack”.

## Dry-Run Results Against The Actual Calibration

I validated the design note numerically against the current annual VAR from `data/var_dataset.csv`.

### Stationary covariance of the slow states

- `std(y_1, spr, cy) = (0.03554, 0.01593, 0.52959)`
- Correlation matrix:

```text
[[ 1.    -0.597  0.712]
 [-0.597  1.    -0.103]
 [ 0.712 -0.103  1.   ]]
```

- Eigenvalues of `Sigma_z`:

```text
[8.70e-05, 7.86e-04, 2.81e-01]
```

- Variance shares:

```text
[0.03%, 0.28%, 99.69%]
```

This confirms the design note’s main claim: the stationary distribution is highly tilted and nearly one-dimensional.

### Coverage at `N = 7^3`

- Current grid: historical coverage `92.1%`, stationary Monte Carlo coverage `83.5%`
- Option A (`lyapunov-axis`, `n_stds=3.0`): historical coverage `100.0%`, stationary Monte Carlo coverage `99.24%`
- Option B (`principal`, `n_stds=3.0`): historical coverage `100.0%`, stationary Monte Carlo coverage `99.17%`

### Volume gain

- Option B / Option A hull-volume ratio: `0.4626`

This matches the design note closely and is strong evidence that the principal-axis grid is the right production target.

### Geometry sanity checks

- Flat index rule `i = i0*N1*N2 + i1*N2 + i2` remains valid on the transformed grid.
- Trilinear interpolation in principal coordinates reproduces linear functions of the economic state to machine precision (`max error = 4.44e-16`).
- Boundary-hit rate over `(state grid point, K=3 quadrature node)` combinations drops from about `33.3%` on the current grid to about `26.6%` under Option B.

## Main Design Decision

Use a Cartesian grid in transformed coordinates and keep policy storage in economic coordinates.

Define:

- `s` = economic slow state
- `mu_s` = unconditional mean of the slow state
- `Sigma_z` = stationary covariance of the slow state from the discrete Lyapunov equation
- `L @ L.T = Sigma_z`
- `u = L_inv @ (s - mu_s)`

Then:

- The interpolation grid is Cartesian in `u`
- The stored `state_grid` remains in `s`
- Solver and simulation convert `s_next -> u_next` just before bracketing / nearest-axis lookup

This preserves the current flat indexing, 8-corner interpolation, and quadrature architecture.

## Critical Adjustments Relative To The Design Note

The design note is directionally right, but several code snippets there are stale relative to the current repo.

### 1. Do not reintroduce the old return convention

Current solver logic uses three return dimensions:

- `rtb`
- `xr`
- `xb`

The implementation plan must preserve:

- `const_r = Phi_0_ret`
- `A_r = Phi_21`
- `mu_r_k = const_r + A_r @ s_i + M @ v_k`

Do **not** revert to an older “fixed current-state `R_bill` + 2 excess-return” formulation.

### 2. Do not assume `state_grids` means “economic axis values”

Right now `pc.state_grids[d]` are axis-aligned economic coordinates.
Under principal mode, the interpolation axes become `u`-coordinates.

That semantic change affects:

- solver bracketing
- simulation nearest-state lookup
- diagnostics
- any notebook or helper code that interprets `pc.state_grids[d]` as `y_1`, `spr`, or `cy`

For clarity, introduce a new canonical name for the interpolation axes:

- `state_bracket_grids`

and use that in solver/simulation.

`state_grid` should remain the economic-state lattice in `s`-space.

## Recommended Rollout

### Phase 0: Compatibility-first staging

Do **not** immediately flip the default discretization mode to `"principal"`.

Reason:

- existing saved bundle metadata does not contain the new grid-mode field
- several tests reconstruct `DiscretizationConfig` from old metadata
- `load_policy_bundle()` already fails on archived `diagnostics.pkl` because old pickled config objects no longer match the current `NamedTuple` layout

Safe recommendation:

- add the new feature with `state_grid_mode="naive"` as the code default
- explicitly request `"principal"` in new runs / notebooks / verification scripts
- only consider flipping the default after bundle metadata and loaders are refreshed

This is the single most important migration guardrail.

### Phase 1: Add a dedicated state-grid builder

Files:

- `model.py`
- `discretization.py`

Changes:

1. Extend `DiscretizationConfig` with:

- `state_grid_mode: str = "naive"` for the first rollout
- `state_n_stds: float = 3.0`

2. In `discretization.py`, add:

- `stationary_covariance(Phi, Sigma_innov)`
- `build_state_grid(...)`

3. `build_state_grid(...)` should return a dict containing at least:

- `mode`
- `mu_s`
- `Sigma_z`
- `sigma_z`
- `L`
- `L_inv`
- `state_bracket_grids`
- `state_grid`
- `state_indices`
- `Pi_state`

Recommended modes:

- `"naive"`: current behavior
- `"lyapunov-axis"`: axis-aligned grid with half-width `state_n_stds * sigma_z[d]`
- `"principal"`: Cholesky-rotated grid in `u`

Notes:

- Keep `rouwenhorst_multivariate()` unchanged for legacy compatibility.
- Use `build_state_grid()` only from new code paths.

### Phase 2: Switch `Precompute` to the new builder

Files:

- `precompute.py`

Changes:

1. Replace the current `rouwenhorst_multivariate(...)` state-grid block with `build_state_grid(...)`.
2. Store:

- `self.state_grid`
- `self.state_indices`
- `self.Pi_state`
- `self.state_bracket_grids`
- `self.state_grid_mode`
- `self.state_grid_mu_s`
- `self.state_grid_L`
- `self.state_grid_L_inv`

3. Keep `self.state_grids` only as a compatibility alias if needed, but do not let new runtime code depend on its old economic-axis meaning.
4. Delete or retire `_build_state_grid()` once the new builder is in place.

Important:

- `self.state_grid` must always be in economic `s`-space.
- `annuity_factors`, `const_r`, `A_r`, `mu_r`, and all return logic should continue to use `self.state_grid` exactly as they do now.

### Phase 3: Thread transformed coordinates through the solver

Files:

- `solver.py`

Changes:

1. Everywhere the solver currently does:

- propagate `s_next`
- call `bracket_state_3d(s_next_0, s_next_1, s_next_2, grids_0, grids_1, grids_2)`

replace that with:

- propagate `s_next` in economic space
- map to bracketing coordinates
  - principal mode: `u_next = L_inv @ (s_next - mu_s)`
  - other modes: same call path with identity / zero transform
- call `bracket_state_3d()` on `state_bracket_grids`

2. Thread these arrays through the quadrature FOCs and period solvers:

- `state_grid_L_inv`
- `state_grid_mu_s`
- `state_bracket_grids[0:3]`

3. Keep the terminal solver unchanged.

Why unchanged:

- it already works directly off `state_grid`, `const_r`, `A_r`, and `M_v_nodes`
- it does not depend on axis-aligned state bracketing

### Phase 4: Update simulation consistently

Files:

- `simulation.py`

Changes:

1. In the continuous-state branch, transform `s_next` before the nearest-grid lookup:

- principal mode: nearest axis point in `u`
- other modes: identity transform

2. Use `state_bracket_grids`, not economic-axis assumptions.

3. For `initial_state="stationary"`:

- do **not** use the stationary distribution of `Pi_state` under principal mode

Recommended behavior:

- principal mode: build mode-aware initialization probabilities from the `u`-axis bins, or draw `u ~ N(0, I)` and quantize each axis
- naive / lyapunov-axis: current `Pi_state` logic is acceptable

4. If a discrete-state simulation path is exposed later, guard against using it with `state_grid_mode="principal"` unless a proper discrete transition is explicitly defined for that mode.

### Phase 5: Fix diagnostics and loader assumptions

Files:

- `diagnostics.py`
- `policy_io.py`
- tests that reconstruct `DiscretizationConfig` from metadata

Changes:

1. Diagnostics must stop reading `pc.state_grids[d]` as economic state axes.
2. Economic reporting should use:

- `pc.state_grid[:, d].min()`
- `pc.state_grid[:, d].max()`
- or direct mode-aware reporting

3. Add a compatibility path for old bundles:

- if `state_grid_mode` is absent in metadata, treat it as `"naive"`
- if `state_n_stds` is absent, fall back to legacy behavior

4. Consider making `load_policy_bundle()` robust to stale `diagnostics.pkl` files:

- either add `load_diagnostics=False`
- or catch pickle schema errors and continue with arrays + metadata only

This compatibility problem already exists today and should not be conflated with the grid change.

## Test Plan

The new tests should not depend on old saved bundles or pickled diagnostics.

### Tier 1: Exact geometry / math

New test module, independent of saved runs:

1. `stationary_covariance()` solves the Lyapunov equation.
2. Principal grid center point equals `mu_s`.
3. Principal grid maps back to exact `u` extrema `[-state_n_stds, +state_n_stds]`.
4. Flat index ordering is preserved.
5. Trilinear interpolation in transformed coordinates reproduces linear functions exactly.

### Tier 2: Coverage / support

Using the actual VAR calibration:

1. Historical coverage at `N=7^3`, `state_n_stds=3.0`:

- principal mode at least `99%`
- lyapunov-axis at least `99%`

2. Stationary Monte Carlo coverage at `N=7^3`, `state_n_stds=3.0`:

- principal mode at least `99%`
- lyapunov-axis at least `99%`

3. Principal / lyapunov volume ratio roughly between `0.3` and `0.6`

### Tier 3: Integration with existing runtime

1. `Precompute(...)` builds in all three modes.
2. `run_lifecycle_solver(...)` completes on a small grid in:

- `naive`
- `lyapunov-axis`
- `principal`

3. No NaN/Inf in policy arrays.
4. Constrained shares remain within bounds.

### Tier 4: Simulation integration

1. `simulate_lifecycle(...)` runs in principal mode with `initial_state="median"`.
2. Continuous-state nearest lookup in principal mode returns valid indices.
3. If `initial_state="stationary"` is supported for principal mode, verify:

- probabilities sum to 1
- initialization is centered at `mu_s`

### Tier 5: Economic regression checks

On a common calibration and fixed seed:

1. Current-grid vs principal-grid policies are close in interior states.
2. Differences grow in tail `y_1` states, where the current grid undercovers.
3. Bond-share response to `y_1` is smoother in principal mode.

## Files That Should Not Change

Unless a bug is found during implementation, these should remain untouched:

- `var.py`
- income quadrature logic
- return quadrature logic
- terminal portfolio formulas

## Recommended Verification Order For Claude Code

1. Implement `build_state_grid()` and its exact tests first.
2. Wire `Precompute` second and keep the solver untouched until the new arrays are proven correct.
3. Update solver bracketing next.
4. Update simulation after solver changes pass.
5. Only then enable `"principal"` in the notebook/run configuration used for production verification.

## Bottom Line

The principal-axis design is numerically well supported by the current calibration and should improve grid efficiency materially.

The hard part is not the math. The hard part is avoiding silent semantic breakage in code that currently assumes:

- axis-aligned state grids
- `state_grids[d]` are economic coordinates
- old metadata can be reconstructed with current config defaults

If those migration hazards are handled explicitly, the implementation should be straightforward and low-risk.
