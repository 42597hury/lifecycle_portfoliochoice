# Handoff: Port Euler-Equation Residual Diagnostic from Numba `main`

**Branch:** `jax-rewrite`
**Status when this doc was written:** the post-solve diagnostic in `verify_benchmark_bundle.py` only runs `diagnose_terminal_portfolio_states` (terminal age FOC sanity check). **No all-ages Euler-equation residual diagnostic exists in the JAX branch.** The Numba `main` branch had `scripts/diagnostics/_diag_euler_errors.py` and friends — those got deleted in handoff 1 along with the rest of `scripts/`.

**Target deployment:** agent ports the EE-residual logic from `main`'s `_diag_euler_errors.py` to the JAX branch, **adapting the economic features for rtb-as-state + CCV log returns + 4-D state**. Output: a `verify_ee_residuals.py` script that loads a saved policy bundle and reports per-age residual statistics.

**Effort:** 1-2 days. The math/structure is portable from `main`; the JAX branch's specific FOC functions (`retirement_foc_jac_ccv`, `working_foc_jac_ccv`) are already written and can be reused for the residual computation.

**Time-critical:** tonight's 5⁴ retirement-only bundle lands in ~30 min (target 19:00 user time). **If this script is ready when the bundle lands, we can characterize the bundle immediately.** Otherwise we wait for the next iteration.

---

## 1. Goal

Produce `verify_ee_residuals.py` that:

1. **Loads** a saved policy bundle from `./saved_runs/<bundle-name>/`.
2. **Re-evaluates the FOC** at every solved (or a sampled subset of) cell, using the same precompute machinery the solver used.
3. **Reports residual statistics** per age: median, p95, p99, max, fraction-above-tolerance.
4. **Flags catastrophic failures**: NaN cells, cells where residual > 1e-2 (way above tol).
5. **Saves a JSON summary** alongside the bundle: `./saved_runs/<bundle-name>/ee_residuals.json`.

Output should answer: "is this bundle a valid solver output, or did Newton fail to converge in some/many cells?"

---

## 2. Scope and non-goals

### In scope

- Grid-based EE residual computation: at every cell `(t, z_idx, state_idx, w_idx)`, evaluate the FOC at the solved policy and report residual norm.
- Per-age aggregation: histogram, percentile stats, fail count above threshold.
- Save JSON output to bundle directory.
- Run from command line: `python verify_ee_residuals.py <bundle-name>` or with `--bundle-path`.

### Out of scope

- **Simulation-path-based EE check.** Depends on the simulator, which is being fixed for CCV correctness in another handoff. Add as follow-up after the simulator fix lands.
- **Higher-order quadrature accuracy check.** That's a "is the solver's quadrature dense enough" question — separate handoff (the quadrature sensitivity study).
- **Plotting / visualisation.** JSON output only; visualisation is downstream.
- **Auto-running from `verify_benchmark_bundle.py`.** Add later; for now run manually after the bundle lands.

### Hard constraints

- **Use the existing FOC functions.** Don't re-implement `retirement_foc_jac_ccv` etc. Import and call them.
- **No solver-side code changes.** Pure read-only diagnostic.
- **Adapted for 4-D state + rtb-as-state.** The `main` branch's diagnostic was 3-D state; this version must use 4-D state and read rtb from the next-period state vector.
- **CCV log returns**, not arithmetic returns.

---

## 3. What to port from `main`

The Numba `main` branch had:

- `scripts/diagnostics/_diag_euler_errors.py` — main EE battery
- `scripts/diagnostics/_diag_gridpoint_ee.py` — per-cell residual computation
- `scripts/diagnostics/_diag_invalid_cells.py` — NaN / extreme-alpha sanity

You need to port the **structure** of `_diag_gridpoint_ee.py` (per-cell residual computation) and `_diag_invalid_cells.py` (sanity check). You do NOT need the simulator-path or quadrature-sweep diagnostics for this initial version.

If `main` is checked out somewhere (or accessible via `git show main:scripts/diagnostics/_diag_euler_errors.py`), read those files first. They contain the basic structure: load bundle → loop over cells → compute residual → aggregate.

The math is the same; what changes is:
- 3-D state → 4-D state (extra axis)
- 8 corners → 16 corners (multilinear interp)
- rtb in return draws → rtb in state vector (read from `state_{t+1}[rtb_idx]`)
- Arithmetic R_p → CCV log R_p with variance correction

---

## 4. Implementation outline

### 4.1 Loading the bundle

Use the existing `lifecycle/policy_io.py`:

```python
from lifecycle.policy_io import load_policy_bundle
C, S, B, diagnostics, run_config = load_policy_bundle(bundle_path)
# C, S, B shapes: (n_age, n_z, N_state, n_w)
```

### 4.2 Rebuilding precompute from bundle metadata

The bundle contains `run_config["discretization_config"]` and `run_config["base_config"]`. Use these to rebuild the same `model` and `pc` that the solver used:

```python
from lifecycle.model import DiscretizationConfig, BASE_CONFIG_KEYS
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute

disc_config = DiscretizationConfig(**run_config["discretization_config"])
base_config = run_config["base_config"]   # already a dict
var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(base_config, var_config, verbose=False)
pc = build_precompute(model, disc_config, verbose=False)
```

Now `model`, `pc` match what the solver used. `pc.disc_config`, `pc.state_grid`, `pc.s_grid`, etc. are all available.

### 4.3 Per-cell residual computation

For each retirement age `t` (or all ages if running full lifecycle), and each cell `(z_idx, state_idx, w_idx)`:

1. **Read solved policy:**
   ```python
   c_t = C[t, z_idx, state_idx, w_idx]
   alpha_s = S[t, z_idx, state_idx, w_idx]
   alpha_b = B[t, z_idx, state_idx, w_idx]
   ```

2. **Skip if obviously bad:**
   ```python
   if not jnp.isfinite(c_t) or c_t <= sc.min_consumption:
       residual = jnp.nan; continue
   ```

3. **Compute savings:**
   ```python
   wealth = pc.wealth_grid[w_idx]
   savings = wealth - c_t
   if savings <= sc.tiny_savings:
       continue   # tiny-savings fallback cell, skip
   ```

4. **Build the FOC inputs for this cell:**
   - Use `_build_step_log_returns` and `_build_step_state_brackets` (both already in `solver.py`) to get `log_R_bill, log_x_s, log_x_b, j_corners, w_corners` for this state grid index.
   - Get `c_next = C[t+1]` (consumption policy at next age).
   - Gather `c_corners_at_z = c_next[z_idx, j_corners, :]` for retirement (or the working-age 4-axis variant).

5. **Evaluate the FOC at the solved policy:**
   ```python
   from lifecycle.solver import retirement_foc_jac_ccv
   foc_s, foc_b, J_ss, J_bb, J_sb, V_dot = retirement_foc_jac_ccv(
       alpha_s, alpha_b, savings, ...,
   )
   ```

6. **Compute residual norm:**
   ```python
   residual = jnp.sqrt(foc_s**2 + foc_b**2)
   # Optionally: scale by the Newton scale (the e0 from foc_fn(0,0))
   # to get relative residual instead of absolute
   ```

7. **Store in array** of shape `(n_retire_ages, n_z, N_state, n_w)`.

### 4.4 Vectorise via vmap

Don't loop in Python — vmap over (z_idx, state_idx, w_idx) for each age. Reuses the kernel-builder pattern from `solver.py`:

```python
@jit
def per_age_ee_residual(t, C, S, B, ...):
    # vmap over (z_idx, state_idx, w_idx)
    def per_cell(z_idx, state_idx, w_idx):
        ...
        return residual
    return vmap(...)(z_idx_arr, state_idx_arr, w_idx_arr).reshape(n_z, N_state, n_w)
```

This makes the whole diagnostic run in seconds even at canonical 5⁴ × 33 ages.

### 4.5 Aggregate + save JSON

```python
import json
import numpy as np

residuals = ...   # shape (n_retire_ages, n_z, N_state, n_w)

per_age_stats = []
for t_idx, age in enumerate(retire_ages):
    r = residuals[t_idx]
    finite = r[np.isfinite(r)]
    per_age_stats.append({
        "age": int(age),
        "n_cells": int(r.size),
        "n_nan": int(np.sum(~np.isfinite(r))),
        "median": float(np.median(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "p99": float(np.percentile(finite, 99)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "frac_above_1e-5": float(np.sum(finite > 1e-5) / max(finite.size, 1)),
        "frac_above_1e-2": float(np.sum(finite > 1e-2) / max(finite.size, 1)),
    })

summary = {
    "bundle_path": str(bundle_path),
    "tolerance_used_at_solve": run_config["solver_config"]["tol"],
    "max_iter_used_at_solve": run_config["solver_config"]["max_iter"],
    "per_age": per_age_stats,
    "global_max_residual": float(np.max(residuals[np.isfinite(residuals)])),
    "global_nan_count": int(np.sum(~np.isfinite(residuals))),
}

with open(bundle_path / "ee_residuals.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print summary to stdout too
print("=" * 70)
print("EE residuals summary")
print("=" * 70)
print(f"Global max residual: {summary['global_max_residual']:.2e}")
print(f"Global NaN count: {summary['global_nan_count']}")
for s in per_age_stats[:10] + per_age_stats[-3:]:
    print(f"Age {s['age']:3d}: median={s['median']:.2e}  p99={s['p99']:.2e}  "
          f"max={s['max']:.2e}  >1e-5: {s['frac_above_1e-5']*100:.1f}%")
```

### 4.6 Working ages (if/when full lifecycle bundle lands)

For working-age cells, the FOC needs the eta/eps inner quadrature. Use `working_foc_jac_ccv` instead of `retirement_foc_jac_ccv`. The pattern is the same; just an extra inner-product over income shocks.

For tonight's bundle (retirement-only `youngest_age_to_solve=67`), only retirement and the work-to-retirement boundary age need to be checked. If `youngest_age_to_solve` is set in `solve_control`, only check ages from there to `terminal_age - 1`.

---

## 5. Adapting for rtb-as-state + CCV

Three concrete adaptations from the `main` branch's diagnostic:

### 5.1 rtb sourcing

In `main`, rtb came from the return-quadrature output. Now rtb is read from the next-period state vector. Use the same logic that `_build_step_log_returns` uses in `solver.py:740-754`:

```python
log_R_bill_kv = s_next[:, rtb_idx]  # (n_state_quad,)
```

### 5.2 CCV log return formula

`main` may have used arithmetic `R_p = α_s·R_s + α_b·R_b + α_bill·R_bill`. **Use CCV log returns** (the formula in `_ccv_log_return_and_grad`). The `retirement_foc_jac_ccv` function already does this — just call it.

### 5.3 4-D state grid corners

8-corner trilinear → 16-corner quadrilinear. The `_build_step_state_brackets` function already produces the right corners; just use its output directly.

---

## 6. Output spec

`./saved_runs/<bundle-name>/ee_residuals.json`:

```json
{
  "bundle_path": "saved_runs/system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark",
  "tolerance_used_at_solve": 1e-7,
  "max_iter_used_at_solve": 100,
  "per_age": [
    {"age": 99, "n_cells": 61875, "n_nan": 0, "median": 1.2e-9, "p95": 4.5e-8, "p99": 1.1e-7, "max": 6.3e-6, "frac_above_1e-5": 0.0001, "frac_above_1e-2": 0.0},
    {"age": 98, ...},
    ...
  ],
  "global_max_residual": 6.3e-6,
  "global_nan_count": 0
}
```

Plus stdout summary as in §4.5.

---

## 7. Pass / fail criteria

- **Pass**: `frac_above_1e-2 == 0.0` for all ages. Max residual < 1e-2. NaN count == 0.
- **Concerning**: `frac_above_1e-5 > 1%` for some age, or max > 1e-3. Indicates Newton struggled at boundary cells.
- **Fail**: any age has `frac_above_1e-2 > 0`, or NaN count > 0. Means the bundle has cells where FOC residual is large at the solved policy — Newton genuinely failed to converge.

For tonight's bundle (`max_iter=100`, backward-age warm-start), expectation: median ~1e-9, p99 ~1e-6, max ~1e-4. **Anything substantially worse means the solver is producing unsuitable policies.**

---

## 8. Implementation checklist

- [ ] Read `main` branch's `_diag_euler_errors.py` and `_diag_gridpoint_ee.py` for structure (via `git show main:scripts/diagnostics/_diag_euler_errors.py` if needed).
- [ ] Create `verify_ee_residuals.py` at repo root.
- [ ] Argparse: accept `bundle-path` as positional or `--bundle-name` (default looks up `./saved_runs/<name>/`).
- [ ] Load bundle via `lifecycle/policy_io.py`.
- [ ] Rebuild precompute from bundle metadata.
- [ ] Implement `per_age_ee_residual(t)` using `retirement_foc_jac_ccv` (and `working_foc_jac_ccv` for working ages).
- [ ] vmap over cells, jit the per-age function.
- [ ] Aggregate per-age stats + global stats.
- [ ] Save JSON to bundle dir.
- [ ] Print stdout summary.
- [ ] Test on a tiny smoke bundle first (run `verify_smoke.py`, then `verify_ee_residuals.py system_iv_..._smoke`).
- [ ] Run on tonight's bundle when it lands. Report findings.

---

## 9. Why ship "first cut" tonight, not "perfect" later

We need ANY characterisation of tonight's bundle to know whether the rtb-as-state migration produced sensible policies. The user expects **"the bundle will probably fail miserably"** — meaning they want to see big residuals confirmed (or refuted) **before** investing more compute in 7⁴.

Even a partial diagnostic (retirement ages only, residuals only, no histogram) is enormously more informative than "we have a bundle but no idea if it's right."

Ship a working-but-minimal version. Add bells and whistles in a follow-up.
