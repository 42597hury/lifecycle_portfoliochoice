# State Grid Change Log

This note documents the state-grid redesign that was implemented for the current codebase. It is written as a reviewer handoff so Claude can see exactly what changed, where it changed, and why each change matters.

## Reviewer Mental Model

The important conceptual change is that the code now distinguishes between:

- `pc.state_grid`: the flat list of economic slow-state points `(y_1, spr, cy)` used to evaluate the VAR and conditional returns.
- `pc.state_bracket_grids`: the 1D axes used for interpolation bracketing.
- `pc.state_bracket_shift` and `pc.state_bracket_L_inv`: the transform used to map an economic state `s` into interpolation coordinates before bracketing.

In `naive` and `lyapunov-axis` mode, the interpolation coordinates are the same as economic-state coordinates, so:

- `state_bracket_shift = 0`
- `state_bracket_L_inv = I`

In `principal` mode, the interpolation grid lives in transformed coordinates `u` with:

- `s = mu_s + L u`
- `u = L^{-1} (s - mu_s)`

That means downstream code must no longer assume that `pc.state_grids[d]` is always an economic-state axis. To preserve compatibility, `pc.state_grids` is still present, but it is now a legacy alias for `pc.state_bracket_grids`.

## High-Level Compatibility Goal

The rollout was made opt-in and compatibility-first:

- The default config stays `state_grid_mode="naive"`.
- Existing policy array shapes and flat state indexing are unchanged.
- Solver and simulation were updated so they work correctly in all three modes.
- Diagnostics and bundle loading were updated so they do not misinterpret the new grid contract.

## File-By-File Map

### `model.py`

Where: `model.py:92-107`

What changed:

- Added `state_grid_mode: str = "naive"`.
- Added `state_n_stds: float = 3.0`.

Purpose:

- These fields make the new grid design configurable from `DiscretizationConfig`.
- Keeping the default at `"naive"` avoids silently changing behavior for old code paths, saved metadata, or old tests.

### `discretization.py`

Where: `discretization.py:16`, `discretization.py:127-145`, `discretization.py:150-172`, `discretization.py:175-188`, `discretization.py:191-220`, `discretization.py:223-340`

What changed:

- Imported `solve_discrete_lyapunov` at `:16`.
- Added `stationary_covariance(...)` at `:127-145`.
- Added `_normal_bin_probs(...)` at `:150-172`.
- Added `_stationary_probs_from_transition(...)` at `:175-188`.
- Added `_independence_rouwenhorst_pi(...)` at `:191-220`.
- Added the new mode-aware `build_state_grid(...)` at `:223-340`.

Purpose:

- `stationary_covariance(...)` computes the unconditional covariance of the slow-state VAR. This is the basis for both the Lyapunov-axis grid and the principal-axis grid.
- `_normal_bin_probs(...)` provides marginal Gaussian bin masses for the transformed `u` axes in principal mode.
- `_stationary_probs_from_transition(...)` provides a stationary distribution fallback for axis-based modes using `Pi_state`.
- `_independence_rouwenhorst_pi(...)` preserves the old flat-indexed joint transition structure so state indexing remains compatible with existing solver/storage code.
- `build_state_grid(...)` is the new central abstraction. It constructs:
  - `state_grid` in economic-state coordinates.
  - `state_bracket_grids` in interpolation coordinates.
  - `L` and `L_inv` for the principal transform.
  - `bracket_shift` and `bracket_L_inv` so downstream code can bracket in the correct coordinate system.
  - `state_indices`, `Pi_state`, and `stationary_probs` so existing downstream code still has the same state-count and transition metadata.

Mode-specific behavior inside `build_state_grid(...)`:

- `naive`: preserves the old per-dimension Rouwenhorst-style axes.
- `lyapunov-axis`: uses unconditional standard deviations but keeps axis-aligned interpolation.
- `principal`: builds a uniform cube in transformed `u` coordinates and maps it into economic space with `mu_s + L u`.

Why this file is the root of the redesign:

- All other changes are downstream of this new contract.
- If a reviewer wants to understand the new geometry, this is the first file to inspect.

### `precompute.py`

Where: `precompute.py:118-143`, `precompute.py:380`

What changed:

- Replaced the old state-grid construction block with a call to `build_state_grid(...)`.
- Stored the returned geometry and compatibility metadata on `Precompute`.
- Added a summary print that exposes grid mode and half-width.

New attributes stored on `pc`:

- `state_grid_mode`
- `state_grid_mu_s`
- `state_grid_Sigma_z`
- `state_grid_sigma_z`
- `state_grid_L`
- `state_grid_L_inv`
- `state_bracket_shift`
- `state_bracket_L_inv`
- `state_bracket_grids`
- `state_stationary_probs`

Compatibility detail:

- `self.state_grids = self.state_bracket_grids` at `precompute.py:135` is intentional.
- This keeps legacy callers alive while changing the meaning of `state_grids` to "interpolation axes" rather than "economic-state axes".

Purpose:

- `Precompute` is now the single place where downstream modules learn whether they are in `naive`, `lyapunov-axis`, or `principal` mode.
- It exposes both the economic state lattice and the interpolation coordinate system without changing the external state count or flat indexing.

### `solver.py`

Where: `solver.py:281-290`, `solver.py:342-390`, `solver.py:487-552`, `solver.py:1093-1100`, `solver.py:1770-1779`, `solver.py:1926-1955`, `solver.py:2126-2155`, `solver.py:2236-2245`

What changed:

- Added `transform_state_for_bracketing_3d(...)` at `:281-290`.
- Updated both quadrature FOC/Jacobian kernels to transform `s_next` into bracketing coordinates before calling `bracket_state_3d(...)`.
- Threaded `state_bracket_shift` and `state_bracket_L_inv` through the retirement and working-age portfolio solvers.
- Threaded the same transform arrays through the JIT period solvers and public period-solver wrappers.
- Updated `run_lifecycle_solver(...)` to read `pc.state_bracket_grids`, `pc.state_bracket_shift`, and `pc.state_bracket_L_inv` instead of assuming the interpolation axes are stored directly as economic-state grids.

Purpose:

- This is the most important downstream compatibility change.
- The solver still evolves the true economic state `s_next` using the VAR in economic coordinates.
- But when it needs policy interpolation, it first maps `s_next` into the correct interpolation coordinates.
- Without this change, principal-mode interpolation would have looked up policies on the wrong grid.

What did not change:

- The economic-state transition itself still uses `Phi_0_state`, `Phi_11`, and the economic `state_grid_i`.
- Flat state indexing and policy tensor shapes were preserved.

### `simulation.py`

Where: `simulation.py:127-139`, `simulation.py:363-405`, `simulation.py:529-536`, `simulation.py:766`, `simulation.py:865-932`

What changed:

- Added `_resolve_initial_state_indices(...)` at `:127-139`.
- Expanded `simulate_lifecycle_core(...)` to accept `state_bracket_shift` and `state_bracket_L_inv`.
- In the continuous-state transition path, transformed `s_next` into bracketing coordinates before finding the nearest state-grid point.
- Updated `simulate_lifecycle(...)` to:
  - initialize `initial_state="stationary"` from `pc.state_stationary_probs` when available.
  - pass `pc.state_bracket_grids`, `pc.state_bracket_shift`, and `pc.state_bracket_L_inv` into the core simulator.

Purpose:

- Solver and simulation now use the same coordinate system for policy lookup.
- This keeps simulation behavior aligned with the new principal-mode interpolation logic.
- Using `pc.state_stationary_probs` for initial states avoids treating principal-mode stationary weights as if they necessarily came from the old `Pi_state` interpretation.

Why this matters for downstream users:

- Any code that simulates or projects forward using continuous slow states must now respect the bracketing transform, not just the economic state lattice.

### `diagnostics.py`

Where: `diagnostics.py:668-692`, `diagnostics.py:710-721`

What changed:

- The state-grid quality report now prints the active grid mode.
- Economic-state min and max are now computed from `pc.state_grid[:, d]`.
- Unconditional means and sigmas are now read from `pc.state_grid_mu_s` and `pc.state_grid_sigma_z` when available.
- Principal-mode diagnostics now print the transformed `u`-axis ranges from `pc.state_bracket_grids`.
- Return-distribution diagnostics now prefer `pc.state_stationary_probs` before falling back to an eigenvector-based stationary distribution of `Pi_state`.

Purpose:

- This prevents diagnostics from misreading transformed interpolation axes as economic-state axes.
- It also makes principal-mode geometry legible in the printed diagnostics output.

Key reviewer takeaway:

- Before this change, it was safe to inspect `pc.state_grids[d]` as if it were an economic-state axis.
- After this change, diagnostics had to be updated because that assumption is no longer always true.

### `policy_io.py`

Where: `policy_io.py:17`, `policy_io.py:144-177`

What changed:

- Added `import warnings` at `:17`.
- Wrapped `pickle.load(...)` for `diagnostics.pkl` in a `try/except` inside `load_policy_bundle(...)`.

Purpose:

- Some older saved bundles have diagnostics pickles that are schema-incompatible with the current code.
- The new loader behavior allows arrays and metadata to remain usable even if diagnostics cannot be deserialized cleanly.
- This was added as a compatibility guard during the grid rollout so old saved runs do not fail harder than necessary.

What this does not do:

- It does not fully migrate old bundle metadata schemas.
- It only makes bundle loading more robust when diagnostics are stale.

### `tests/test_state_grid_modes.py`

Where: `tests/test_state_grid_modes.py:1-269`

What was added:

- A plain Python regression script that does not depend on `pytest`.
- `run_geometry_checks(...)` at `:96-177`.
- `run_precompute_mode_checks(...)` at `:181-210`.
- `run_solver_simulation_smoke(...)` at `:212-253`.

Purpose:

- `run_geometry_checks(...)` validates the mathematical contract of the new grid:
  - Lyapunov equation residual.
  - principal-grid center point.
  - exact transformed-axis bounds.
  - preserved flat index ordering.
  - historical coverage.
  - principal-volume ratio.
  - trilinear exactness for linear functions.
  - normalized stationary probabilities.
- `run_precompute_mode_checks(...)` validates that `Precompute` stores the correct mode-aware metadata in all three modes.
- `run_solver_simulation_smoke(...)` validates that a small principal-mode solve and a principal-mode simulation both execute successfully and produce finite, feasible outputs.

Why this file matters:

- This is the staged validation harness for the rollout.
- It checks both the geometry itself and the downstream solver/simulation integration.

### `STATE_GRID_IMPLEMENTATION_PLAN.md`

Where: `STATE_GRID_IMPLEMENTATION_PLAN.md:1-...`

Purpose:

- This file is not runtime code.
- It captures the design rationale, dry-run numerical checks, rollout strategy, and the compatibility-first implementation plan that guided the actual code changes.

## Most Important Downstream Rule

If downstream code needs the economic slow-state values, use:

- `pc.state_grid`

If downstream code needs interpolation axes or needs to bracket a continuous state, use:

- `pc.state_bracket_grids`
- `pc.state_bracket_shift`
- `pc.state_bracket_L_inv`

Do not assume `pc.state_grids[d]` is an economic-state axis in principal mode.

## Validation That Was Run

The implementation was tested with:

- `python -m py_compile model.py discretization.py precompute.py solver.py simulation.py diagnostics.py policy_io.py tests/test_state_grid_modes.py`
- `python tests/test_state_grid_modes.py`

Observed result:

- `tests/test_state_grid_modes.py` finished with `37 passed, 0 failed`.

Additional manual checks were also run during implementation:

- A small legacy-mode solve with `state_grid_mode="naive"` completed successfully.
- A principal-mode diagnostics run completed successfully.
- A saved-bundle loader check confirmed that arrays and metadata still load even when diagnostics compatibility is uncertain.

## Suggested Claude Review Order

If Claude wants to review the change quickly with the highest signal first, the recommended order is:

1. `discretization.py` to understand the new grid contract.
2. `precompute.py` to see how that contract is exposed.
3. `solver.py` and `simulation.py` to verify transformed-coordinate lookup.
4. `diagnostics.py` and `policy_io.py` for compatibility hardening.
5. `tests/test_state_grid_modes.py` for the validation strategy.
