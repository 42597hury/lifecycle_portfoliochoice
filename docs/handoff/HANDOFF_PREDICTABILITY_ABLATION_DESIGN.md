# HANDOFF — Design a return-predictability ablation tool

**Status:** open. Design only — implementation in a separate task. The
goal of this handoff is to lock down the API and the math before code
gets written.

**Audience:** design agent. Decide on (1) which version of "shut down
predictability" to ship as the default, (2) the API surface, and (3) the
acceptance criteria. Implementation is straightforward (~50 LOC) and
follows mechanically from the design choices.

---

## 1. The need

We want an **ablation tool** that takes the estimated VAR and returns a
modified VAR where one or more state variables no longer predict
returns, while the marginal mean and variance of returns remain
unchanged. The use case is robustness-checking results: re-solve the
lifecycle model under "no cy predictability" or "no predictability at
all" and compare the simulated wealth distribution to the baseline. If
the unconstrained-leverage pathology disappears in the no-predictability
ablation, it's driven by predictability; if it persists, it's structural.

This is **not a discretization feature.** Setting `state_grid_sizes[d] = 1`
("freeze the grid axis") creates an irrational agent who ignores a
predictive variable in a world that still has predictability. That's a
different ablation. The tool we're designing here modifies the *world*,
producing a no-predictability baseline that an otherwise-unmodified
agent solves rationally.

## 2. Mathematical specification

The full VAR in CCV-constrained form:

```
z_t = const + Phi · z_{t-1} + ε_t,        ε_t ~ N(0, Omega)

z = [s; r],   Omega = [[Σ_ss, Σ_sr],
                       [Σ_rs, Σ_rr]]

Phi (restricted) = [[Phi_11, 0],
                    [Phi_21, 0]]
```

Marginal moments of `r`:
- `E[r] = z_bar_ret` (the bottom block of `z_bar`).
- `Var(r) = Phi_21 · Σ_z · Phi_21^T + Σ_rr`, where `Σ_z` is the stationary
  state covariance solving `Σ_z = Phi_11 · Σ_z · Phi_11^T + Σ_ss`.

To shut down predictability of state variable `d` (with `d` indexed by
position in `state_indices`):

1. **Direct channel:** `Phi_21'[:, d] = 0`. Lagged `s_d` no longer enters
   the conditional return mean.
2. **Innovation channel:** `Σ_rs'[:, d] = 0`. Equivalent to `M'[:, d] = 0`
   since `M = Σ_rs · Σ_ss^{-1}`. State innovation `v^s_d` no longer
   bleeds into returns.
3. **Marginal-mean preservation:** `const_ret' = z_bar_ret − Phi_21' · z_bar_state`.
   Verifies `(I − Phi') · z_bar = const'` so the implied stationary mean
   is bit-exactly `z_bar`.
4. **Marginal-variance preservation:** `Σ_rr' = Σ_rr + ΔV` where
   `ΔV = Phi_21 · Σ_z · Phi_21^T − Phi_21' · Σ_z · Phi_21'^T`. `ΔV` is
   PSD by construction, so the inflation is safe and `Σ_rr'` remains
   PD provided `Σ_rr` was. Verifies
   `Phi_21' · Σ_z · Phi_21'^T + Σ_rr' = Phi_21 · Σ_z · Phi_21^T + Σ_rr`.

`Phi_11`, `Σ_ss`, state-side `const`, and `z_bar` are **unchanged**, so
the state dynamics are identical to the baseline. Only return-side
parameters move.

### Special cases

- `axes_to_freeze = ()` (empty): identity operation, return a deep copy
  of the input config.
- `axes_to_freeze = (0, 1, 2)` (all state axes): full shutdown. Returns
  iid: `Phi_21' = 0`, `Σ_rs' = 0`, `M' = 0`, `Σ_rr' = Var(r)_marginal`,
  `const_ret' = z_bar_ret`. The agent perceives `r ~ N(z_bar_ret, Var(r)_marginal)`
  iid each period.

### Caveat on indirect predictability via `Phi_11`

If `Phi_11[j, d] ≠ 0` for some `j ≠ d`, then `s_d` indirectly predicts
returns through `s_{t+1, j}` even after the direct + innovation channels
are zeroed. To kill this completely we'd also need to zero `Phi_11[j ≠ d, d]`,
which changes `Σ_z` and complicates the marginal-variance bookkeeping.

For our calibration the indirect channel is non-trivial (cy → spr →
returns through `Phi_21[xb, spr] = +4.49`), so it matters. The design
agent should decide whether the default tool covers indirect
predictability or leaves it as opt-in.

## 3. Design decisions to resolve

### D1. Which version of "predictability shutdown" is the default?

Three escalating versions:

| version | what's zeroed | interpretation |
|---|---|---|
| **V1: Direct only** | `Phi_21[:, d]` | Lagged s_d doesn't enter the return equation. Indirect channel via Phi_11 still works. |
| **V2: Direct + innovation** | also `Σ_rs[:, d]` | Same plus contemporaneous v^s_d → r channel killed. M[:, d] = 0. |
| **V3: Full decoupling** | also `Phi_11[j ≠ d, d]` | s_d carries no information that propagates to any future variable. Truly orthogonal. Σ_z changes; marginal-variance bookkeeping more involved. |

**Recommendation:** Default to **V2**. It matches the natural reading of
"shut down predictability through state innovation d," lines up with
the partition-VAR structure (`Phi_21` and `Σ_rs` both indexed by state
axis d), and the math is clean. V3 is a more aggressive ablation and
should be available as an opt-in mode (e.g., a flag `full_decoupling=True`)
once V2 is shipped.

### D2. API surface

Three options:

| option | shape | trade-off |
|---|---|---|
| **A. Standalone function** | `shut_down_return_predictability(var_config, axes) -> var_config_new` | Most flexible; can be applied to any var_config from any source. Recommended. |
| **B. Flag in `build_nominal_system1_var_config`** | `build_nominal_system1_var_config(..., shut_down_axes=(0,))` | Discoverable; couples the ablation to the data-side build path. |
| **C. Method on a config object** | `var_config.shut_down(axes)` | Cleaner OO, but `var_config` is a plain dict in the codebase, not an object. |

**Recommendation:** **A**. Standalone function in `var.py`. Mirrors the
existing pattern of `partition_var(...)` and `estimate_var1_from_csv(...)`
as free functions. Doesn't disturb the build path. Can be applied to
hardcoded configs (e.g. `build_nominal_system1_var_config_hardcoded`) too.

### D3. Function name

Candidates:
- `shut_down_return_predictability(var_config, axes_to_freeze)`
- `ablate_return_predictability(var_config, axes_to_freeze)`
- `zero_return_predictability(var_config, axes_to_freeze)`
- `iid_returns_var_config(var_config, axes_to_freeze)`
- `decouple_returns_from_state(var_config, axes)`

**Recommendation:** `shut_down_return_predictability`. Verbose but
unambiguous; the verb signals it's a destructive transformation; "return"
specifies what's being decoupled (returns from state, not states from
returns).

### D4. Axis specification

Three options:

- **A. Indices into `state_indices`**: `axes_to_freeze=(0, 2)` means
  freeze state-vector positions 0 and 2 (under cy-first ordering, that's
  cy and y_1).
- **B. Names**: `axes_to_freeze=('cy', 'y_1')`. Reads naturally;
  resilient to ordering changes.
- **C. Both, with the same parameter accepting either int or string**:
  `axes_to_freeze=(0, 'y_1')`. Most flexible but ambiguous in edge cases.

**Recommendation:** **A** as the canonical form, with **B** as an
optional convenience. The function internally resolves names via
`var_config['state_predictor_columns']` (which already exists). If a
user passes names, the function looks them up and returns the indices.

### D5. Default value of `axes_to_freeze`

Options:
- `axes_to_freeze=None` → shut down ALL state axes (full iid). Convenient
  for the "all" use case; risky if someone calls with no args by mistake.
- `axes_to_freeze=()` → no-op identity. Safest default; explicit shutdown
  required.
- No default → require user to specify. Most explicit; mildly annoying.

**Recommendation:** Default `axes_to_freeze=None` meaning "all state
axes" (full iid). Document this prominently. Aligns with the user's
phrasing "shut down all state variables predictability" as the natural
calling convention. Empty tuple `()` produces an identity (useful as a
test case).

### D6. Marginal-moment preservation: full Var(r) or just diagonal?

- **Full covariance** preservation: `Phi_21' Σ_z Phi_21'^T + Σ_rr' =
  Phi_21 Σ_z Phi_21^T + Σ_rr` as a 3×3 matrix equality. Cross-asset
  correlations preserved.
- **Diagonal only**: only the per-asset variances are matched, allowing
  cross-asset correlations to shift.

**Recommendation:** **Full covariance**. It's mathematically cleaner
(`ΔV = Phi_21 Σ_z Phi_21^T − Phi_21' Σ_z Phi_21'^T` is a PSD matrix,
trivially added to `Σ_rr`), preserves return-correlation structure, and
the user's stated goal "maintain sample mean and stds" is naturally
read as "preserve the marginal joint distribution" rather than "match
per-asset volatilities only".

### D7. Validation behavior on PSD failure

If `Σ_rr'` or `Σ_r_cond' = Σ_rr' − M' · Σ_sr'` ends up non-PSD (e.g.
due to PSD-but-not-PD baseline `Σ_rr`, or numerical quirks):
- **Raise**: refuse to run. Strict.
- **Warn and clip**: project onto the PSD cone. Lenient but obscures
  failures.
- **Silent**: trust the inputs; downstream Cholesky calls will raise
  anyway. Minimal.

**Recommendation:** **Raise** with a clear error message that names
which matrix failed PSD. Predictability ablation is a controlled
parameter modification; if it produces an invalid VAR, the user should
know.

### D8. Should the modified `var_config` carry a provenance flag?

So that saved bundles can record "this bundle was solved with cy
predictability disabled":

- Add a key `'predictability_ablation'` to the returned config with
  value `{'frozen_axes': (...), 'preserved_moments': 'marginal_full_cov'}`.
- Or omit it (the user is responsible for tracking which config they
  used).

**Recommendation:** **Yes, add provenance.** Cheap to implement, makes
saved bundle metadata self-documenting, prevents confusion when comparing
ablation runs to baselines later. Use a key
`'predictability_ablation'` with structured contents.

### D9. Where do user-facing examples go?

The ablation is an investigative tool, not a production setting. Three
options:

- A new section in `main.ipynb` ("Step N: ablation runs").
- A separate notebook `ablation_studies.ipynb`.
- Just docstring examples; user copies into their own analysis.

**Recommendation:** Docstring examples plus a reference in
`contextfiles/RETURNS.md` §6 (Validation) noting the tool exists and
linking to the function. No new notebook scaffolding.

## 4. Acceptance criteria

The implementation must satisfy these tests (no solver run required;
all of these are pure VAR-algebra checks):

1. **Identity:** `shut_down_return_predictability(cfg, axes_to_freeze=())`
   returns a config that round-trips to the same `Phi`, `Omega`, `const`,
   `z_bar` to machine precision.
2. **Marginal mean preservation:** `(I − Phi') · z_bar = const'`,
   verified by computing `Phi_0_full = (I − Phi_full) · z_bar` and
   comparing the return-block to `const'_ret`. Residual `< 1e-13`.
3. **Marginal variance preservation:** Compute `Σ_z' = solve_discrete_lyapunov(Phi_11, Σ_ss)`
   (which equals `Σ_z` since state side is unchanged), then check
   `Phi_21' · Σ_z' · Phi_21'^T + Σ_rr' == Phi_21 · Σ_z · Phi_21^T + Σ_rr`
   to `< 1e-14`.
4. **Frozen-axis structural check:** for every `d in axes_to_freeze`,
   `Phi_21'[:, d]` and `Σ_rs'[:, d]` are exactly zero.
5. **PSD preservation:** all eigenvalues of `Omega'` and of
   `Σ_r_cond' = Σ_rr' − M' · Σ_sr'` are non-negative (numerical tolerance
   `> -1e-12`).
6. **Full-shutdown sanity:** with `axes_to_freeze=(0, 1, 2)`,
   `Phi_21' = 0`, `Σ_rs' = 0`, `M' = 0`, `Σ_r_cond' = Σ_rr' = Var(r)_marginal_orig`,
   `const_ret' = z_bar_ret`. Agent's perceived returns are iid.
7. **End-to-end model build:** `build_model(base_config, var_config_modified)`
   succeeds without errors; the resulting model has `Phi_21[:, d] = 0`
   for frozen `d` and the rest of the partition is consistent.

## 5. Constraints

- **Pure post-processor.** The function should not depend on
  `discretization`, `solver`, `simulation`, or `precompute`. It operates
  on the VAR config dict only. This makes it composable with any
  upstream estimation path.
- **Functional.** Returns a new config; does not mutate the input.
  Important for ablation comparisons where the user holds onto the
  baseline config alongside the modified one.
- **No solver involvement.** All correctness checks are linear-algebra
  identities; runs in milliseconds.
- **Compatible with both estimation paths.** Must work on output from
  `build_nominal_system1_var_config` (estimated from CSV) and from
  `build_nominal_system1_var_config_hardcoded` (fallback constants).

## 6. Out of scope

- **`Phi_11` modification (V3 in §D1).** Defer until V2 is shipped and
  the indirect-predictability question is empirically demonstrated to
  matter.
- **Grid freeze (`state_grid_sizes[d] = 1`).** A separate ablation
  answering a different question. Document the distinction in the
  function's docstring but don't bundle the implementation.
- **`Σ_ss` modification.** State innovation covariance is the agent's
  perceived state-shock distribution; we're not changing the world's
  state dynamics, only how returns load on state.
- **Bundle-format migration.** Existing bundles solved under the
  baseline VAR remain valid; ablation bundles are saved alongside with
  distinct paths (the user already adds suffixes like `_v2`).
- **Higher-order moment matching.** This tool preserves first and
  second marginal moments only. Skewness and tail thickness of returns
  may shift — the modified VAR is still Gaussian, just with redistributed
  variance.

## 7. Pointers

| What | Where |
|------|-------|
| `var.py:partition_var` (sister function, similar shape) | `var.py:49` |
| `var.py:build_nominal_system1_var_config` (one of two estimation entry points) | `var.py:381` |
| `var.py:_validate_partition_inputs` (helper for partition validation) | `var.py:_internal_` |
| `solve_discrete_lyapunov` (computes Σ_z) | `scipy.linalg`, used in `discretization.py:stationary_covariance` |
| Hardcoded fallback config (must work with this too) | `var.py:_NOM_COLS, _STATE_IDX, _Z_BAR, _PHI, _OMEGA` |
| Existing tests on partition algebra | `tests/test_state_grid_modes.py:run_geometry_checks` (sister) |

## 8. Recommended scope of next-task implementation

After the design agent locks the choices in §3:

1. ~30 LOC implementation in `var.py`.
2. ~80 LOC tests in `tests/test_var_ablation.py` (new file or appended
   to `tests/test_state_grid_modes.py`).
3. ~10 LOC docstring example showing both `axes_to_freeze=None` (full
   shutdown) and `axes_to_freeze=(0,)` (cy only) usage patterns.
4. One paragraph in `contextfiles/RETURNS.md` §6 referencing the new
   tool.

Total estimated effort: 1 hour. Risk surface is small because no
production code path consumes the modified config unless the user
explicitly opts in.
