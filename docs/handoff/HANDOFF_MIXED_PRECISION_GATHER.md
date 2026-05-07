# Handoff: Mixed-Precision Gather Path (fp32 gather/interp, fp64 FOC arithmetic)

**Branch:** `jax-rewrite`
**Status when this doc was written:** the entire solver hot path runs in fp64. Tonight's GH200 run measured **70% of theoretical fp64 peak** (~6.8 TFLOPS effective from 9.7 TFLOPS) at 273 s/age on the 5⁴ benchmark. The c_corners gather is the largest memory-bandwidth-bound section of the per-FOC eval. **Casting only the gather + multilinear interpolation to fp32 — keeping all CRRA / FOC / Newton arithmetic in fp64 — should reduce wall by 30-50%** without compromising convergence or policy quality.

**Important: the agent should NOT just implement the proposal below blindly. Step 0 is to critically review whether each proposed cast site is correct and to flag any I missed in either direction.** Then implement the agreed-upon set.

**Effort:** 2-3 days, including the review pass and bit-stability testing.

---

## 1. Goal

Reduce numerical precision to fp32 **only** at sites where:
1. The arithmetic is dominated by memory bandwidth, not compute precision.
2. The values are bounded and well-conditioned (no near-zero singularities, no large-cancellation subtractions).
3. Downstream code casts back to fp64 before any precision-critical computation.

**Speed target:** 30-50% wall reduction on canonical configs at GH200/H200/H100. Bigger gains where memory bandwidth dominates (higher `n_state_quad × n_w`).

**Precision target:** alpha values agree with the all-fp64 baseline within `1e-4` relative across all cells; no NaN/Inf appearance; no changes to convergence behaviour at `tol=1e-7`.

---

## 2. Scope

### In scope

- Modify `lifecycle/solver.py` to introduce explicit fp32 cast at the gather + multilinear-interp boundary, then cast back to fp64 before CRRA/FOC arithmetic.
- Add `SolverConfig.gather_precision: str = "f64"` field with values `{"f64", "f32"}`. Default `"f64"` = current behaviour, no regression.
- Modify `lifecycle/_interp_c_and_mpc_at_cell` and the `c_corners` gather sites in `_solve_retirement_at_cell` and `_solve_working_at_cell` to honour the precision flag.
- Add `verify/mixed_precision.py` smoke + alpha-range comparison test.

### Out of scope (do not implement)

- **Going to all-fp32 anywhere in CRRA / Newton / FOC arithmetic.** That's a different risk profile we explicitly reject (see §4 below).
- **Bfloat16 / TF32 explicit usage.** XLA-CUDA may use TF32 for matmul under the hood when fp32 is requested; that's fine, but don't manually cast to bf16/TF32.
- **Touching the precompute path.** `lifecycle/precompute.py` produces fp64 grids; keep them fp64. Cast happens in solver, on entry to gather.
- **Touching the simulator.** `lifecycle/simulation.py` is being fixed for CCV correctness in a separate handoff. Do not bundle precision changes with that fix.
- **Touching `_newton_fori`, `_backtracking_fori`, `_ccv_log_return_and_grad`, the FOC functions.** These stay fp64 throughout.
- **Mixed precision at any level finer than "fp32 for the gather, fp64 for everything else."** No per-axis tuning, no fp32-for-some-quad-points-fp64-for-others. One toggle, one boundary.

### Hard constraints

- **No NaN/Inf appearance** at any tail wealth state in the smoke or canonical configs.
- **Alpha-range agreement** between fp64 and fp32 paths within `1e-4` relative on the smoke.
- **Default behaviour unchanged.** `gather_precision="f64"` (the default) must produce bit-identical output to today.
- **No precision contamination across the boundary.** The cast back to fp64 must happen *before* any CRRA / log / exp / division operation.

---

## 3. Proposed cast sites — REVIEW BEFORE IMPLEMENTING

The following list reflects my analysis of where fp32 is safe and where fp64 is mandatory. **Step 0 of implementation is to critically review this list.** For each item, the agent should answer:

- "I agree, this is safe to cast." → keep
- "I disagree, this needs fp64 because [specific reason]." → flag and explain in the report
- "I'd add this site to the cast list because [reason]." → propose addition
- "I'd remove this site from the cast list because [reason]." → propose removal

The agent's review report goes into the report file (§7). **Do not implement until the review is done and the user confirms the final cast list.**

### 3.1 Sites where fp32 should be SAFE (cast TO fp32)

| Site | Why safe |
|---|---|
| **`c_corners` gather** in `_solve_retirement_at_cell:c_corners_at_z = c_next[z_idx, j_corners_i, :]` | Just moving values around. Values are policy consumption, well-bounded. Bandwidth-bound. |
| **`c_corners_T` gather + transpose** in `_solve_working_at_cell` | Same as above, plus extra `n_z` axis. Same logic. |
| **Multilinear-interp corner values** (`c_per_corner`, `slope_per_corner`) inside `_interp_c_and_mpc_at_cell` | Bounded, ~1e-3 to ~10 magnitude. fp32 has 7 digits; we use 4-5. |
| **Multilinear-interp corner weights** (`w_kv` × `c_per_corner`) | Weights in [0,1] from fractional bracket. fp32 ample precision. |
| **`c_next` itself** if cast at build time of the kernel closure | Smaller HBM footprint at the broadcast/in_axes level. |
| **Wealth-grid `searchsorted` index arithmetic** | Already int32. No change. |
| **`x_next` = `s × R_p + pension_next_z`** if computed in fp32 | Magnitudes 0.1-1000, well-conditioned. **Wait — see §3.2 caveat below.** |

### 3.2 Sites where fp64 is MANDATORY (keep in fp64)

| Site | Why fp64 mandatory |
|---|---|
| **`c_at_xn ** (-gamma)`** in `terminal_foc_jac_ccv` and friends | At γ=10 and c=0.01, this is 1e20. fp32 fits but precision is degraded; γ=15 is risky. |
| **`bequest_mu_and_mup`** | At W=750 and small δ_bequest, mu can be ~1e-15. Below fp32's effective range. |
| **`mu_alive = c_at_xn ** (-gamma)`** | Same as CRRA. |
| **`mup_alive = -gamma * mu_alive / c_at_xn * mpc_at_xn`** | Combines two fp32-risky operations (CRRA and division). Cumulative error. |
| **Newton residual computation** (`fs`, `fb`, `err = sqrt(fs² + fb²)`) | Tolerance check `err < tol * scale` at `tol=1e-7` requires ~6 digits of relative precision in residuals; fp32 has 7. Tight. |
| **Jacobian computation** (J_ss, J_bb, J_sb) | Sums of products of similar magnitude. Cancellation risk. |
| **CCV variance correction** (`0.5 × α'(diag(Σ)-Σα)`) | Subtraction of similar-magnitude terms. Catastrophic cancellation risk in fp32. |
| **`exp(r_p)`** to compute `R_p` | Exponential blow-up: small fp32 error in r_p → larger fp64 error in R_p. Better to do exp in fp64. |
| **EGM inverse mapping `x = c + s`** | Both can be small (1e-3); fp32 limits sum precision. |
| **Newton step calculation** (`step_s = -(Jbb*fs - Jsb*fb) / det`) | Division by determinant — det can be small near singularity. Critical fp64 zone. |
| **Backward-age warm-start init lookup** (`init_a_s_arr[z_idx, i_s, w_ref_idx]`) | Could be fp32 internally, but stored array is fp64. Keep fp64 to avoid casts at every age. |
| **The `static` tuple in kernel builders** (tol, max_iter, etc.) | Small Python ints/floats. No reason to touch. |

### 3.3 Sites needing the agent's judgment

These are unclear to me — I want the agent's reasoning before deciding:

| Site | Question |
|---|---|
| **`log_R_bill, log_x_s, log_x_b`** scenario tensors | Magnitudes 0.01-0.1. fp32 has 8 digits there. Possibly safe. **But** they feed into `r_p = log_R_bill + α·log_x + 0.5·variance_correction` which has the cancellation issue. Probably keep fp64. |
| **`weight_kv_kr`** (state quad × ret quad weights) | Bounded (0,1). Multiplied with mu in fp64 sum. Probably safe to cast. |
| **`s_grid` and `wealth_grid`** | Bounded (0, 750). Used in interp brackets. Probably safe to cast for the gather, keep fp64 for arithmetic. |
| **The eta/eps inner-quadrature path in `_solve_working_at_cell`** | This is an additional 4×4=16 tensor reduction inside the FOC. Same precision concerns as the main FOC sum. Probably keep fp64. |
| **Per-cell `psi_z` (survival probability)** | Bounded (0,1). Multiplied with mu_alive in fp64. Could go either way — probably safe. |
| **`A_is` (annuity factor at i_s)** | Bounded (3.7, 22). Used in bequest formula. **Bequest is fp64 mandatory, so A_is should match.** Keep fp64. |

### 3.4 Things I'm worried about that should also be reviewed

- **Compound error across 33 ages with backward-age warm-start.** Each age's policies are stored in fp64 but computed via the (partially-fp32) gather. If gather error compounds wrong, alphas drift over ages. **Mitigation:** the cast-back-to-fp64 happens before the FOC sum, so errors don't compound through `init_a_s_arr` reads (those are stored in fp64). Should be fine, but verify by checking whether age-99-to-age-67 alpha trajectory matches the all-fp64 trajectory within tolerance.
- **`c_at_xn` lower bound.** The `min_consumption=1e-10` floor protects against c near zero. After fp32 gather → cast back to fp64 → max with min_consumption: should give the same floor as all-fp64. Verify the order of operations preserves the floor.
- **Numerical stability of the c_corners gather under XLA fp32 fusion.** XLA may aggressively fuse the gather + interp + `**(-gamma)` chain. **The agent must verify the cast-to-fp64 boundary survives XLA fusion** by inspecting the HLO output (`jax.jit(...).lower().as_text()`) on a small case.

---

## 4. Why not all-fp32?

Briefly, for the reviewer's reference (and for documentation):

1. **CRRA marginal utility at γ=10 and small consumption** approaches fp32's noise floor. At γ=15 it dips below.
2. **Bequest motive at extreme wealth** produces values below fp32's representable range (1e-15 ish). Going through a `c^(-γ)` operation in fp32 loses these contributions to zero — wrong, but accidentally tolerable for high-wealth states. NOT tolerable in general.
3. **CCV variance correction** has catastrophic cancellation potential when α terms are similar magnitudes.
4. **Newton convergence at `tol=1e-7`** — fp32 has 7 digits, residual sums need ~6 below the term magnitude. Genuinely tight.
5. **Compound across 33 ages** of backward induction: fp32 noise of 1e-7 × 33 ≈ 3e-6 — approaches policy tolerance.

The mixed-precision boundary moves the bandwidth-bound arithmetic to fp32 (where precision doesn't matter much) while keeping the precision-critical arithmetic in fp64. **It captures the bandwidth win without any of the all-fp32 risks.**

---

## 5. Implementation

### 5.1 SolverConfig field

Add to `lifecycle/model.py` `SolverConfig`:

```python
# --- Mixed precision toggle ---
# When "f64" (default): all solver arithmetic in fp64. Identical to today.
# When "f32": the c_corners gather + multilinear interpolation runs in fp32;
#             results are cast back to fp64 BEFORE any CRRA / FOC / Newton
#             arithmetic. Captures memory-bandwidth savings without precision
#             loss in convergence-critical paths.
# WARNING: f32 path produces alphas that differ from f64 by ~1e-5 relative
# (real arithmetic noise, not just bit-shuffle). Bit-identity test does not
# apply; agreement test at 1e-4 relative does.
gather_precision: str = "f64"
```

### 5.2 Cast helper

Add a small helper near the top of `lifecycle/solver.py`:

```python
def _cast_for_gather(arr, gather_dtype):
    """Cast arr to gather_dtype if different from arr's current dtype."""
    return arr if arr.dtype == gather_dtype else arr.astype(gather_dtype)
```

### 5.3 The gather + interp boundary

Modify `_interp_c_and_mpc_at_cell` (lines ~565+ in solver.py):

```python
def _interp_c_and_mpc_at_cell(c_corners_kv, w_corners_kv,
                                x_next, wealth_grid, min_consumption,
                                gather_dtype=jnp.float64):
    """Multilinear interp at the trilinear/quadlinear corners.

    c_corners_kv, w_corners_kv: cast to gather_dtype for the gather + interp.
    Output c_at_xn, mpc_at_xn: cast back to fp64 before return.
    """
    # Cast inputs to gather precision.
    c_corners_g = _cast_for_gather(c_corners_kv, gather_dtype)
    w_corners_g = _cast_for_gather(w_corners_kv, gather_dtype)
    x_next_g = _cast_for_gather(x_next, gather_dtype)
    wealth_grid_g = _cast_for_gather(wealth_grid, gather_dtype)

    # ... rest of multilinear interp logic, using the _g variants ...
    # Returns c_at_xn_g, mpc_at_xn_g in gather_dtype

    # Cast back to fp64 BEFORE caller does CRRA/FOC arithmetic.
    c_at_xn = c_at_xn_g.astype(jnp.float64)
    mpc_at_xn = mpc_at_xn_g.astype(jnp.float64)

    # min_consumption floor in fp64.
    c_at_xn = jnp.maximum(c_at_xn, min_consumption)
    mpc_at_xn = jnp.clip(mpc_at_xn, 0.0, 1.0)
    return c_at_xn, mpc_at_xn
```

### 5.4 Plumbing the flag through

The `gather_dtype` parameter needs to thread through:
- `_build_per_age_*_kernel_vmap_only` and `_pmap` builders — close over `gather_dtype` from `sc.gather_precision`
- `_solve_*_at_cell` functions — accept `gather_dtype` as static arg
- `_interp_c_and_mpc_at_cell` — accept and use as in §5.3
- The `static` tuple needs to include `gather_dtype` so it's baked into the JIT trace

`sc.gather_precision` is a string ("f64" or "f32"); convert to `jnp.float32`/`jnp.float64` once at builder time:

```python
gather_dtype = jnp.float32 if sc.gather_precision == "f32" else jnp.float64
static = (sc.tol, sc.max_iter, ..., bool(sc.use_fori_newton), gather_dtype)
```

`gather_dtype` is a Python value (the JAX dtype object), so it bakes into the trace cleanly without forcing recompile per call.

### 5.5 Validation

Add `verify/mixed_precision.py`:

```python
"""Verify fp32 gather + fp64 FOC produces alphas within 1e-4 of all-fp64."""
import numpy as np
from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

# Smoke config
disc = DiscretizationConfig(
    n_wealth=20, wealth_min=0.13, wealth_max=200.0,
    n_savings=20,
    state_grid_sizes=(3, 3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=5, n_eps_nodes=3, n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 2, 3),
)
base = dict(BASE_CONFIG)
base.update(start_age=60, retire_age=63, terminal_age=65)
var = build_nominal_system1_var_config_hardcoded()
model = build_model(base, var, verbose=False)
pc = build_precompute(model, disc, verbose=False)

# Baseline: all fp64
sc_f64 = CANONICAL_SOLVER._replace(max_iter=100, gather_precision="f64")
C64, S64, B64, _ = run_lifecycle_solver(model, pc, sc_f64, verbose=0)

# Test: fp32 gather
sc_f32 = CANONICAL_SOLVER._replace(max_iter=100, gather_precision="f32")
C32, S32, B32, _ = run_lifecycle_solver(model, pc, sc_f32, verbose=0)

# Compare
rel_err_C = np.max(np.abs(C32 - C64) / (np.abs(C64) + 1e-10))
rel_err_S = np.max(np.abs(S32 - S64) / (np.abs(S64) + 1e-10))
rel_err_B = np.max(np.abs(B32 - B64) / (np.abs(B64) + 1e-10))

print(f"Max relative error: C={rel_err_C:.2e}  S={rel_err_S:.2e}  B={rel_err_B:.2e}")
print(f"NaN check (f32):    C={int(np.isnan(C32).sum())}  S={int(np.isnan(S32).sum())}  B={int(np.isnan(B32).sum())}")
print(f"alpha_s f64 range: [{S64.min():.4f}, {S64.max():.4f}]")
print(f"alpha_s f32 range: [{S32.min():.4f}, {S32.max():.4f}]")
print(f"alpha_b f64 range: [{B64.min():.4f}, {B64.max():.4f}]")
print(f"alpha_b f32 range: [{B32.min():.4f}, {B32.max():.4f}]")

assert rel_err_C < 1e-4, "C deviates more than 1e-4"
assert rel_err_S < 1e-4, "S deviates more than 1e-4"
assert rel_err_B < 1e-4, "B deviates more than 1e-4"
assert int(np.isnan(C32).sum()) == 0, "NaN in C32"
assert int(np.isnan(S32).sum()) == 0, "NaN in S32"
assert int(np.isnan(B32).sum()) == 0, "NaN in B32"
print("✅ Mixed-precision smoke OK")
```

### 5.6 HLO inspection (one-time check)

After implementing, inspect the JIT'd HLO to verify the cast boundary survived fusion:

```python
from lifecycle.solver import _build_per_age_retirement_kernel_vmap_only
# ... build the kernel ...
hlo_text = jax.jit(kernel).lower(*example_args).as_text()
print(hlo_text[:5000])  # check first 5KB

# Look for: f32 ops in the gather/interp section, f64 ops in the CRRA/FOC section.
# Specifically: search for "convert" ops at the boundary — should show f32 → f64
# right before the c^(-γ) / `**` operation.
```

Document this in the report — XLA sometimes elides "useless" casts and unifies precision; if it does, the precision boundary is broken.

---

## 6. Verification gates (in order)

1. **Default unchanged.** `python verify/smoke.py` with `gather_precision="f64"` (default) produces identical alphas to the pre-change baseline. Bit-identical (1e-12 tolerance).
2. **Smoke fp32 agreement.** `python verify/mixed_precision.py` passes — relative error < 1e-4 on smoke config, no NaN.
3. **HLO inspection.** Cast boundary visible in the JIT'd HLO; fp32 in gather, fp64 in FOC.
4. **Larger-config check.** Run `verify/canonical_small.py` (or equivalent) at fp32; confirm alpha ranges sane and no NaN. (~10 min on local CPU; no GPU needed.)
5. **No tail-cell pathology.** Spot-check alphas at extreme states (top wealth, bottom z, edge state-grid points). Compare fp32 vs fp64 — relative error should still be < 1e-4 even at edges.

If any gate fails, the agent should report and stop. Don't proceed to canonical-scale runs without fixing the gate failure.

---

## 7. Output / reporting

Create `docs/scans/MIXED_PRECISION_REVIEW_2026-05-XX.md` with:

```markdown
# Mixed-Precision Cast Site Review

## §3.1 sites — agent agrees / disagrees / proposes changes

[Per-site review with reasoning. If agree, just "agreed". If disagree, explain.]

## §3.2 sites — agent agrees / disagrees / proposes changes

[Same.]

## §3.3 ambiguous sites — agent's recommendation

[For each, agent picks fp32 or fp64 with one-paragraph reasoning.]

## §3.4 worries — agent's mitigation

[Each worry: "addressed by [X]" or "still a concern, recommend [Y]".]

## Additional sites the agent identified

[Any sites I missed.]

## Final cast list (after review)

[The agreed-upon set of sites. This is what gets implemented.]
```

The agent commits this report **before** implementing. User reviews. User confirms. Then implement.

---

## 8. Implementation checklist (after review approval)

- [ ] Add `SolverConfig.gather_precision: str = "f64"` field per §5.1.
- [ ] Add `_cast_for_gather` helper per §5.2.
- [ ] Modify `_interp_c_and_mpc_at_cell` per §5.3.
- [ ] Plumb `gather_dtype` through kernel builders and `_solve_*_at_cell` per §5.4.
- [ ] Apply gather casts at the c_corners gather sites in `_solve_retirement_at_cell` and `_solve_working_at_cell` (and any others identified during review).
- [ ] Write `verify/mixed_precision.py` per §5.5.
- [ ] Run gate 1 (`verify/smoke.py` default unchanged) — must pass bit-identical.
- [ ] Run gate 2 (`verify/mixed_precision.py`) — must pass relative tolerance.
- [ ] Run gate 3 (HLO inspection) — document boundary visible in HLO.
- [ ] Run gate 4 (`verify/canonical_small.py`) — alphas sane, no NaN.
- [ ] Run gate 5 (tail-cell spot check) — relative error <1e-4 at edges.
- [ ] Commit:
  ```
  solver: mixed-precision gather (fp32 c_corners + interp, fp64 FOC arithmetic)

  Adds SolverConfig.gather_precision (default "f64", no-op).
  When "f32", the c_corners gather + multilinear interpolation runs in
  fp32; results cast back to fp64 BEFORE any CRRA / FOC / Newton
  arithmetic. Captures memory-bandwidth savings (~30-50% wall reduction
  expected at canonical sizes) without precision loss in
  convergence-critical paths.

  Verified:
  - Default behaviour unchanged (gather_precision="f64" bit-identical
    to pre-change baseline).
  - fp32 gather alphas within 1e-4 relative of fp64 on smoke.
  - No NaN/Inf at any tail state.
  - HLO inspection confirms f32→f64 cast at gather/FOC boundary
    survives XLA fusion.

  Cast site list: [from review report]
  ```
- [ ] Push to `jax-rewrite`. Report back with commit SHA + paths to the verify scripts. Stop.

---

## 9. Performance expectations

**At GH200 (97 GB HBM, fp64 9.7 / fp32 19.5 TFLOPS):**
- 5⁴ + reduced quad: 273 s/age → **170-220 s/age** (1.25-1.6× wall reduction)
- 7⁴ + reduced quad: extrapolated 1050 s/age → **600-800 s/age**
- 7⁴ + full quad (with chunking): extrapolated 4200 s/age → **2400-3000 s/age**

**At H100 SXM5 (80 GB HBM, fp64 30 / fp32 60 TFLOPS):** similar 1.5× ratio — bandwidth-bound and compute-bound both improve 2×, net wall improvement ~1.5×.

**At consumer cards (fp32 20-60× fp64):** mixed precision is the **only viable mode**. The gather path runs at fp32 throughput; FOC stays fp64 but the FOC throughput is much smaller fraction of the trace. Expected: 5-15× wall reduction over fp64-only.

**Memory savings:** c_corners batch peak HBM halved. At 7⁴ full-quad, ~1.06 TB worst-case becomes ~530 GB — still way over single-GPU HBM, but combined with chunking, headroom is doubled.

---

## 10. What we're NOT doing and why (for the record)

- **Not converting log returns to fp32.** They go into `r_p` which has the variance correction subtraction — cancellation risk.
- **Not converting Newton state to fp32.** Convergence at `tol=1e-7` requires the residual computation in fp64.
- **Not converting bequest mu/mup to fp32.** Tail values below fp32's representable range.
- **Not using bf16 anywhere.** Less precision than fp32 in the mantissa — for a precision-sensitive solver, no upside.
- **Not using TF32.** It's an XLA implementation detail; let XLA pick. Don't manually request it.
- **Not changing the simulator.** Out of scope; that's the CCV-correctness fix's responsibility.
- **Not changing the precompute path.** Stays fp64.

---

## 11. Risk assessment

- **Risk of precision loss at tail cells:** Medium. Mitigation: gates 4 and 5 (canonical-small + tail spot check). If those pass, tail behaviour is good.
- **Risk of XLA fusion eliding the cast boundary:** Low-medium. Mitigation: gate 3 (HLO inspection). If XLA elides the cast, the agent must add `lax.optimization_barrier` or similar to enforce the boundary.
- **Risk of breaking default fp64 behaviour:** Very low if §5 implementation is followed literally — `gather_precision="f64"` means `_cast_for_gather` is a no-op, all paths are bit-identical to today.
- **Risk of compound error across 33 ages:** Low. The cast back to fp64 happens before any CRRA op; backward-age warm-start reads from fp64 storage. Errors don't compound through ages.

If gates 1-5 all pass, the implementation is sound. If any fails, agent stops and reports.

---

## 12. Why this handoff is "review then implement"

The cast site list in §3 is my best analysis but I'm not 100% certain about every site. The agent should:

1. **Read the FOC functions** (`terminal_foc_jac_ccv`, `retirement_foc_jac_ccv`, `working_foc_jac_ccv`, `_ccv_log_return_and_grad`) line by line.
2. **Trace data flow** from gather to FOC residual.
3. **Identify any precision-critical operation** between the gather and the FOC sum.
4. **Critically evaluate each entry in §3.1, §3.2, §3.3, §3.4.**
5. **Produce the review report** before any code change.

The user reviews the report, confirms, then the agent implements.

This avoids the failure mode where someone implements a clean-looking design that subtly breaks at γ=15 or extreme wealth states, only discovered during a paid GPU benchmark.
