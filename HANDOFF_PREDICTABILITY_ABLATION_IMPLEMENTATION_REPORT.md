# Predictability Ablation Implementation Report

## Purpose

The purpose of this ablation is to run the lifecycle portfolio model under progressively weaker financial predictability while preserving the same overall modeling pipeline.

The target product is a four-system setup:

1. **System I**: iid returns with no predictive financial state. A dummy one-point state axis is retained only so the existing model pipeline still has a state object.
2. **System II**: `y_1` is the only predictive state.
3. **System III**: `spr` and `y_1` are predictive states and `cy` is shut down.
4. **System IV**: the existing full baseline with `(cy, spr, y_1)` on the state grid.

The implementation goal was not just to add new VAR estimators. It also had to make the model stack work when the financial state dimension shrinks from 3D to 2D or 1D, and to let the bequest annuity calculation continue working when `y_1` and/or `spr` are no longer on the state grid.

## What Was Implemented

The work landed in four logical stages.

### Stage 1: Allow annuity inputs to come from either the state grid or scalar fallbacks

The original code assumed both `y_1` and `spr` were always present on the financial state grid. That assumption is incompatible with Systems I and II.

Changes:

- In [model.py](model.py), `LifecyclePortfolioModel` now allows nullable grid indices plus scalar fallbacks:
  - `y_1_index_in_state` at line 82
  - `spr_index_in_state` at line 83
  - `y_1_scalar_fallback` at line 84
  - `spr_scalar_fallback` at line 85
- In [precompute.py](precompute.py), `Precompute` now builds `annuity_factors` from either:
  - state-grid values when `y_1` / `spr` are on the grid, or
  - scalar fallbacks when one or both are absent.
  - The fallback logic is in lines 216-222.
- In [precompute.py](precompute.py), `build_model(...)` now validates these annuity inputs explicitly:
  - `y_1` handling at lines 474-490
  - `spr` handling at lines 493-509
  - distinct-index check at lines 511-516
- In [diagnostics.py](diagnostics.py), scalar-fallback reporting was added so diagnostics do not break when `y_1` or `spr` are no longer state dimensions.
  - Example anchor: line 707.

Effect:

- System IV remains compatible with the original grid-based annuity path.
- Systems I-III can omit one or more state variables without breaking bequest-annuity construction.

### Stage 2: Add ablation-specific VAR builders

The baseline code already had generic VAR estimation and partitioning machinery. The ablation implementation added builders that assemble the right `var_config` for each reduced-predictability system.

Changes in [var.py](var.py):

- `_read_y1_spread_sample_means(...)` at line 432
  - Reads sample means of `y_1` and `spr` from the VAR dataset for scalar fallback use.
- `build_no_cy_var_config(...)` at line 438
  - System III
  - Uses state vector `(spr, y_1)` and drops `cy`.
- `build_y1_only_var_config(...)` at line 467
  - System II
  - Uses state vector `(y_1)` and injects `spr` as a scalar fallback equal to the sample mean.
- `build_iid_var_config(...)` at line 493
  - System I
  - Creates a one-state dummy VAR with zero persistence and return covariance equal to the sample covariance of `(rtb, xr, xb)`.
  - Uses sample means of `y_1` and `spr` as scalar fallbacks for the annuity factor.

Effect:

- All four systems can be constructed through the same downstream `build_model(...)` and `Precompute(...)` interfaces.

### Stage 3: Generalize the financial-state runtime from 3D-only to 1D/2D/3D

This was the largest code change. The original runtime assumed exactly three financial state dimensions in discretization, interpolation, and simulation.

Changes in [discretization.py](discretization.py):

- `_normal_bin_probs(...)` now accepts singleton grids:
  - line 87 allows `n >= 1`
  - line 92 returns a unit mass for `n == 1`
- `_independence_rouwenhorst_pi(...)` now supports singleton axes:
  - lines 138-144 special-case `Nd == 1`
- `build_state_grid(...)` now accepts 1, 2, or 3 state dimensions:
  - line 215 validates `1 <= k <= 3`
  - line 222 allows `N_vec >= 1`
- Principal-mode singleton axes are centered at zero:
  - lines 243-252
- Non-principal singleton handling was also added for the legacy branch:
  - lines 281-286

Changes in [solver.py](solver.py):

- The solver still uses hot 3-coordinate JIT kernels, but lower-dimensional states are padded into that layout rather than rewriting the whole solver kernel family.
- `_pad_state_solver_inputs_to_3d(...)` at line 102 performs that embedding.
- `run_lifecycle_solver(...)` at line 2706 now feeds padded state objects to the lower-level solver path.

Changes in [simulation.py](simulation.py):

- `_pad_simulation_state_inputs_to_3d(...)` at line 219 embeds 1D/2D states into the 3-coordinate simulation core.
- `simulate_lifecycle_core(...)` at line 512 now stores continuous state coordinates in `sim_state_coords`:
  - allocation at line 609
  - storage at lines 630-632
- Lower-dimensional grids are handled with singleton-safe bracketing and corner indexing:
  - bracketing changes begin at line 648
  - corner-index padding begins at line 702
- `simulate_lifecycle(...)` at line 959 pads state inputs for the simulation core and returns trimmed `state_coords`:
  - state padding call at line 1191
  - public output at line 1307

Effect:

- Systems I, II, III, and IV can all solve and simulate through the same top-level API.
- The implementation chose padding/dispatch over a fully generic new JIT kernel, which preserved the existing 3D kernel structure.

### Stage 4: Notebook orchestration for system selection

The notebook now has explicit predictability-system selection and projects a 3D discretization template onto the active state vector.

Changes in [predictability_ablation.py](predictability_ablation.py):

- `project_predictability_disc_config(...)` at line 166
  - Projects the baseline `DiscretizationConfig` template onto the active state names.
  - Special-cases System I with a singleton dummy axis.
- `prepare_predictability_system(...)` at line 219
  - Normalizes the requested system code / alias
  - Dispatches to the corresponding VAR builder
  - Returns the `var_config`, projected `disc_config`, and metadata such as system labels and state names

Changes in [main.ipynb](main.ipynb):

- Predictability-system selector added at line 97:
  - `PREDICTABILITY_SYSTEM = "IV"`
- Notebook imports `prepare_predictability_system(...)` and `project_predictability_disc_config(...)` at line 56.
- Notebook-level system setup starts at line 275.
- Bundle metadata now records the active ablation system:
  - constrained solve metadata around line 867
  - unconstrained solve metadata around line 933
- Partial-solve attribution config now projects from a 3D template using state names:
  - line 997
- The notebook helper `refresh_model_if_stale(...)` at line 401 was simplified to rebuild the model from the live `base_config` and `var_config` each time, instead of trying to detect stale objects heuristically.

Effect:

- A notebook user can switch systems by changing a single selector.
- The active system determines both the VAR builder and the projected discretization.
- Saved bundles now carry ablation metadata.

## Post-Implementation Fixes

Two review-driven fixes were applied after the initial rollout.

### Fix 1: System I principal-mode discretization

Problem:

- System I originally projected to `state_n_stds=(0.0,)`, which broke `build_state_grid(...)` because the discretization layer requires strictly positive `n_stds`.

Fix:

- [predictability_ablation.py](predictability_ablation.py) now uses `state_n_stds=(1.0,)` for the dummy axis in line 176.
- [discretization.py](discretization.py) principal mode now places singleton axes at exactly zero in lines 243-252.

### Fix 2: System I dummy state drifting in simulation

Problem:

- The System I dummy axis initially still received Gaussian state shocks in simulation, which made returned state paths meaningless even though policy lookup was effectively clamped to one state.

Fix:

- [simulation.py](simulation.py) now detects the true dummy-state case and zeroes its state-transition shock loading:
  - lines 256-260

## Tests and Validation Added

The ablation implementation added a dedicated test file:

- [tests/test_predictability_ablation.py](tests/test_predictability_ablation.py)

Key coverage:

- Notebook discretization projection:
  - `test_prepare_predictability_system_projects_notebook_disc_template(...)`
  - line 171
- System I `Precompute(...)` smoke test:
  - `test_prepare_predictability_system_i_precompute_succeeds()`
  - line 191
- Model build acceptance for Systems I-III:
  - line 207
- Small end-to-end solves for Systems I-IV:
  - line 324
- Low-dimensional simulation smoke tests:
  - line 351
  - includes a System I assertion that simulated `state_coords` stay at zero
- Validation that malformed annuity-index configs are rejected:
  - line 418

Validation commands run during implementation:

```powershell
python -m pytest tests/test_predictability_ablation.py -q
python -m pytest tests/test_partial_solve.py::test_partial_solve_checkpoint_bundle tests/test_inf_horizon_solver.py::test_markowitz_cold_start_properties -q
```

Observed results:

- `tests/test_predictability_ablation.py`: `27 passed`
- targeted existing regressions: `2 passed`

Additional notebook integrity check:

```powershell
@'
import json
from pathlib import Path
with Path("main.ipynb").open("r", encoding="utf-8") as f:
    json.load(f)
print("main.ipynb parsed successfully")
'@ | python -
```

Result:

- `main.ipynb parsed successfully`

## Current Known Review Findings

These issues were identified in a later read-only review and were **not** fixed yet.

### 1. Singleton axes are still wrong in `lyapunov-axis` mode

Location:

- [discretization.py](discretization.py), lines 272-277

Issue:

- The `lyapunov-axis` branch still uses `np.linspace(mu - a*sigma, mu + a*sigma, Nd)`.
- When `Nd == 1`, the single point becomes the left endpoint instead of the mean.

Direct reproduction during review:

- `prepare_predictability_system("I", ... state_grid_mode="lyapunov-axis")`
- `Precompute(...)`
- produced `pc.state_grid == [[-1.0]]` and `pc.state_bracket_grids == [[-1.0]]`

Impact:

- System I is only correctly centered under the current principal-mode path.
- The claim that 1D/2D/3D support is fully mode-agnostic is not yet true.

### 2. Notebook `run_sim(...)` guard is still too weak

Location:

- [main.ipynb](main.ipynb), lines 351-372

Issue:

- `run_sim(...)` loads policy arrays from a bundle but simulates them with the current notebook-global `pc` and `model`.
- It only checks:
  - `system_label`
  - `state_grid_sizes`
- It does **not** check:
  - `state_n_stds`
  - `n_state_quad_nodes`
  - `n_z`
  - return quadrature
  - or broader calibration consistency

Impact:

- A bundle can still be simulated against the wrong runtime objects if the grid shape matches but other discretization or VAR details differ.

## Suggested Review Focus For The Next Agent

If another coding agent is reviewing this implementation, the highest-value review targets are:

1. Confirm that the System I-IV product definition in the notebook matches the intended empirical experiment.
2. Re-review the low-dimensional state padding path in:
   - [solver.py](solver.py)
   - [simulation.py](simulation.py)
3. Fix and test the remaining singleton bug in `lyapunov-axis` mode.
4. Strengthen notebook bundle compatibility checks in `run_sim(...)`, or rebuild `model` and `pc` directly from bundle metadata before simulation.
5. Decide whether `predictability_ablation.py` and `tests/test_predictability_ablation.py` should be committed as new tracked files. They are currently present in the working tree as new files.

## Relevant Files

Core implementation files:

- `predictability_ablation.py`
- `var.py`
- `model.py`
- `precompute.py`
- `discretization.py`
- `solver.py`
- `simulation.py`
- `diagnostics.py`
- `main.ipynb`
- `tests/test_predictability_ablation.py`

Repo-state note:

- The repository is currently a dirty worktree with many unrelated modified and untracked files. Any reviewer should scope their review to the files above rather than assuming `git status` reflects only the ablation work.
