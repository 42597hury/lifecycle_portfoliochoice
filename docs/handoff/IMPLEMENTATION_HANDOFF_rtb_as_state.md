# Implementation Handoff: rtb Variable Treatment Change

**Scope of this document:** specify *what* must change about the treatment of `rtb` (real bill return) in the VAR / state structure, and the numerical contracts the result must satisfy. Implementation strategy in the JAX branch is at the agent's discretion.

**Not in scope:** how to implement multi-axis grid construction, multilinear interpolation, or backward induction in JAX. These are framework choices — `lax.scan` vs explicit loops, `vmap` vs hand-written kernels, static vs dynamic shapes, etc. — and the agent should pick whatever is idiomatic for the target codebase.

---

## 1. The change in one paragraph

`rtb` (the real return on the one-year bill) currently lives in the return block of a partitioned VAR: it has its own residual innovation, its conditional mean is forecast from the state, and the solver consumes its realised value via the return quadrature. After this change, `rtb` lives in the state block: it goes on the state grid alongside `cy`, `spr`, `y_1`; its innovation is part of the state innovation block; and the solver reads its realised value directly from the next-period state vector. No new variables enter the model. The set of six VAR variables `(y_1, spr, cy, rtb, xr, xb)` is unchanged. The "state vs return" classification is the only thing that moves.

---

## 2. Variable treatment specification

### 2.1 Post-change state block (4 variables, gridded, conditioned on for forecasts)

| state position | variable | description | source of innovation |
|:---:|---|---|---|
| 0 | `cy` | log earnings yield (= −log CAPE) | state innovation block |
| 1 | `spr` | yield spread (AAA − y_1) | state innovation block |
| 2 | **`rtb`** | **real bill return** — *moved from return block* | **state innovation block** (carries the inflation surprise) |
| 3 | `y_1` | nominal 1-year Treasury yield | state innovation block |

### 2.2 Post-change return block (2 variables, integrated over residuals after state-conditioning)

| return position | variable | description | source of innovation |
|:---:|---|---|---|
| 0 | `xr` | excess stock return | return innovation block |
| 1 | `xb` | excess bond return | return innovation block |

### 2.3 Implicit (recoverable from state, never appears as a VAR variable)

- `π_{t+1} = log(1 + y_{1,t}) − rtb_{t+1}` — realised inflation. Compute on demand if needed for diagnostics or for an inflation-indexed extension; do not add as a VAR variable, do not include as a state.

### 2.4 Restriction (preserved, but tightened in scope)

The project's restriction "lagged returns do not predict anything" is preserved, but the set of variables it applies to shrinks from `{rtb, xr, xb}` to `{xr, xb}`:

- **Required:** `Phi[:, xr_column] == 0` and `Phi[:, xb_column] == 0` to machine precision.
- **Required:** `Phi[:, rtb_column]` is unrestricted and estimated freely. Lagged rtb is allowed to predict every variable in the system, including itself — this is the channel that captures inflation persistence.
- **Required:** `Phi[:, cy_column]`, `Phi[:, spr_column]`, `Phi[:, y_1_column]` remain unrestricted (already the case).

### 2.5 Budget constraint sourcing of rtb

This is the most important behavioural change in the solver and simulation paths.

**Before:** at each `(grid_node, state_innovation, return_innovation)` triple, `rtb_{t+1}` is reconstructed from the return-block output: state-conditional mean `+` state-projection of `state_innovation` `+` return-residual contribution from `return_innovation`.

**After:** at each `(grid_node, state_innovation)` pair, `rtb_{t+1}` is read directly as `state_{t+1}[rtb_index_in_state]`, where `state_{t+1} = c + Phi_ss · state_t + state_innovation`. There is no return-block residual contribution to `rtb` because rtb is no longer a return-block variable.

**The portfolio return formula is unchanged:** real portfolio gross log return = `α_xr · x_xr + α_xb · x_xb + rtb`. Only the *source* of `rtb` shifts, from the return block to the state block.

### 2.6 Index metadata required by the model

The model object must carry these indices so downstream code can read `rtb` from the state at simulation/budget time:

```
state_indices       = (2, 1, 3, 0)     # column indices into [y_1, spr, cy, rtb, xr, xb]
return_indices      = (4, 5)
cy_index_in_state   = 0                # if the model object carries one
spr_index_in_state  = 1
rtb_index_in_state  = 2                # NEW field — required
y_1_index_in_state  = 3                # was 2, now 3
```

The change adds `rtb_index_in_state` as a new piece of metadata. Code that previously pulled rtb from the return-quadrature output now needs an index into the state vector instead.

---

## 3. Cholesky / state-grid ordering

The four-element state ordering `(cy, spr, rtb, y_1)` is recommended, in that order. The rationale follows the project's existing two-prong design rule (the same logic that produced `(cy, spr, y_1)` in the current 3-state system):

- **Row 0 — clean leading knob.** Among the four state innovations, `cy` has the lowest mean |ρ| against the others (0.17). Putting it in row 0 gives axis 0 a 100%-pure cy interpretation, preserving the existing convention.
- **Row 3 — refinement lever.** `M[xb, y_1]` remains the dominant entry of the projection `M = Σ_rs Σ_ss⁻¹` by an order of magnitude (~8.85, vs the next-largest at ~1.0). Putting `y_1` in row 3 keeps `K[3]` as the targeted refinement axis for bond-return integration accuracy.
- **Middle (rows 1, 2) — minimise adjacent leakage.** The strongly correlated pair is `(spr, y_1)` with ρ ≈ −0.87. Avoid putting them in adjacent rows. This forces `rtb` between them: `(cy, spr, rtb, y_1)` has mean |adjacent ρ| = 0.28 vs the alternative `(cy, rtb, spr, y_1)` at 0.52.

The mathematical content of the model is invariant to permutations of state rows; this is purely an interpretability and refinement-targeting choice. If the JAX branch has its own ordering convention, deviating is fine, but the agent should record the choice and verify the resulting purity matrix before committing.

---

## 4. Numerical contract

Every implementation that reproduces this change must hit these numbers when the VAR is fit on `var_dataset.csv` (1963–2025, T=63) using the project's existing constrained CCV estimator (z̄ pinned to sample means, demeaned OLS without intercept, intercepts recovered as `(I − Φ) z̄`).

### 4.1 Reference values (CURRENT, PROPOSED, and the rejected OPTION-B for contrast)

| metric | CURRENT | PROPOSED | OPTION-B (do **not** produce this) |
|---|---:|---:|---:|
| state | (cy, spr, y_1) | **(cy, spr, rtb, y_1)** | (cy, spr, y_1, π) |
| return block | (rtb, xr, xb) | **(xr, xb)** | (rtb, xr, xb) |
| rtb forecast R² | 0.5179 | **0.6075** | 0.6218 |
| `Phi[rtb, rtb]` | 0 (by restriction) | **+0.3627** | 0 (by restriction) |
| `‖Phi[:, return_lag_cols]‖` | 0 | **0** | 0 |
| Φ₁₁ max ‖eigenvalue‖ | 0.9362 | **0.9256** | 0.8954 |
| `Σ_ss` shape | 3×3 | **4×4** | 4×4 |
| cond(`Σ_ss`) | 1.4 × 10³ | **1.5 × 10³** | 1.4 × 10³ |
| `Σ_r_cond` shape | 3×3 | **2×2** | 3×3 |
| smallest eigenvalue of `Σ_r_cond` | 1.4 × 10⁻⁴ | **5.0 × 10⁻⁴** | 3.2 × 10⁻⁷ |
| **cond(`Σ_r_cond`)** | 7.5 | **1.21** | 1.95 × 10³ |
| Cholesky `Σ_r_cond` diagonal | (0.0154, 0.0250, 0.0225) | **(0.0243, 0.0227)** | (**0.0006**, 0.0236, 0.0225) |
| Cholesky `Σ_ss` diagonal under recommended ordering | (0.167, 0.011, 0.007) | **(0.164, 0.011, 0.016, 0.007)** | (0.162, 0.011, 0.007, 0.014) |

The PROPOSED column is the target. OPTION-B is shown only so the agent can recognise an accidental implementation drift — the 0.0006 entry on the rtb axis of OPTION-B's Cholesky diagonal is the smoking gun for the rank-deficiency failure mode, and an automated test should detect and reject any configuration that produces it.

### 4.2 Pre-validation script (framework-agnostic)

The agent should run this script (or its JAX-branch equivalent) before and after the change. Before: confirms the CURRENT row reproduces. After: confirms the PROPOSED row reproduces and the assertions pass. The script depends only on `numpy`, `pandas`, and the project's `build_var_config_from_dataset` + `partition_var` (which are pure linear algebra — same in numba and JAX branches).

```python
import numpy as np, pandas as pd

# Adjust import to the JAX branch's module layout
from <project>.var import build_var_config_from_dataset, partition_var

CSV_BASE = "<path>/var_dataset.csv"
COLUMNS_BASE = ["y_1", "spr", "cy", "rtb", "xr", "xb"]


def fit_and_partition(state_idx, return_idx, label):
    cfg, _, _ = build_var_config_from_dataset(
        csv_path=CSV_BASE, columns=COLUMNS_BASE,
        state_indices=state_idx, return_indices=return_idx,
        y_1_index_in_state=[COLUMNS_BASE[i] for i in state_idx].index("y_1"),
        spr_index_in_state=[COLUMNS_BASE[i] for i in state_idx].index("spr"),
        estimation="restricted",
    )
    parts = partition_var(
        cfg["Phi"], cfg["Omega"], cfg["z_bar"],
        state_idx=cfg["state_indices"], ret_idx=cfg["return_indices"],
        variable_names=cfg["variable_names"], verbose=False,
    )
    Σss, Σrc = parts["Sigma_ss"], parts["Sigma_r_cond"]
    rtb_pos_full = COLUMNS_BASE.index("rtb")
    print(f"\n[{label}]")
    print(f"  rtb R^2          = {cfg['equation_r2']['rtb']:.4f}")
    print(f"  Phi[rtb, rtb]    = {cfg['Phi'][rtb_pos_full, rtb_pos_full]:+.4f}")
    print(f"  ||Phi_12||       = {parts['Phi_12_norm']:.3e}")
    print(f"  ||Phi_22||       = {parts['Phi_22_norm']:.3e}")
    print(f"  cond(Sigma_ss)   = {np.linalg.cond(Σss):.3e}")
    print(f"  cond(Sigma_rc)   = {np.linalg.cond(Σrc):.3e}")
    print(f"  smallest eig Σrc = {min(np.linalg.eigvalsh(Σrc)):.3e}")
    print(f"  L_rc diag        = {np.diag(np.linalg.cholesky(Σrc)).round(6).tolist()}")
    return cfg, parts


# Pre-change baseline (must reproduce after the change too — regression anchor)
cfg_cur, _ = fit_and_partition((2, 1, 0),    (3, 4, 5), "CURRENT (regression anchor)")

# Target configuration
cfg_pro, _ = fit_and_partition((2, 1, 3, 0), (4, 5),    "PROPOSED (target)")

# Hard assertions — implementation passes only if all of these hold
assert abs(cfg_pro["equation_r2"]["rtb"] - 0.6075) < 1e-3,  "rtb R² target missed"
assert cfg_pro["max_abs_return_lag_coeff"] < 1e-12,         "restriction violated"
rtb_idx = cfg_pro["variable_names"].index("rtb")
assert abs(cfg_pro["Phi"][rtb_idx, rtb_idx] - 0.3627) < 1e-3, "Phi[rtb, rtb] target missed"
print("\n✓ All numerical contracts satisfied.")
```

---

## 5. Deliverables

The implementation passes when all of D1–D5 hold simultaneously.

### D1. VAR configuration default uses the new indexing

The nominal six-variable VAR configuration in the JAX branch uses `state_indices=(2, 1, 3, 0)`, `return_indices=(4, 5)`, with `rtb_index_in_state=2`, `y_1_index_in_state=3`, `spr_index_in_state=1`. The validation script in §4.2 reproduces the PROPOSED column exactly.

### D2. Model object exposes `rtb_index_in_state`

The `LifecyclePortfolioModel` (or the JAX-branch equivalent) carries an `rtb_index_in_state` field. Required (non-`None`) when `rtb` is in the state block; consumers of `rtb` at simulation/budget time read it from `state_{t+1}[rtb_index_in_state]`.

### D3. State grid and innovation quadrature support 4D

The state grid generator and the state-innovation quadrature accept length-4 specifications for `state_grid_sizes`, `state_n_stds`, `n_state_quad_nodes`, and any per-axis Lobatto/tail tuples. Whatever caps or hardcoded `n_state == 3` branches exist in the JAX branch are lifted. The mathematical primitives (Cholesky factorisation, Gauss-Hermite tensor products, Rouwenhorst univariates) are dimension-generic and require no algorithmic change.

### D4. Solver consumes 4D state and reads rtb from state

The DP solver operates on a 4-D state grid. Wherever the existing solver reads `rtb_{t+1}` from the return-quadrature output, the post-change solver reads it from the next-period state vector at `rtb_index_in_state`. The portfolio return formula `r_p_real = α_xr · x_xr + α_xb · x_xb + rtb` is unchanged.

**Regression test:** the pre-change 3-state canonical run must reproduce bit-exact (or to machine precision in JAX-deterministic mode) under the new code. The agent picks the canonical baseline policy snapshot to compare against.

**Degeneracy test:** the 4-D solver run with a singleton fourth axis (e.g. `state_grid_sizes=(7, 7, 7, 1)`, `state_n_stds=(3, 3, 3, 0)`) must reproduce the 3-D solver run output to floating-point precision. This confirms the 4-D extension reduces correctly to the 3-D special case.

### D5. Diagnostic detects accidental drift to OPTION-B

The PD-Cholesky tests on `Σ_ss` and `Σ_r_cond` are augmented with a soft warning: if `min(eigvalsh(Sigma_r_cond)) < 1e-5`, emit a warning naming the configuration and the suspect axis. This catches accidental drift toward OPTION-B-style configurations (the rank-deficient case). The current PROPOSED configuration's smallest `Σ_r_cond` eigenvalue is 5.0×10⁻⁴, comfortably above this threshold.

---

## 6. End-to-end behavioural checks (post-implementation)

After the change is implemented, simulate ≥10,000 paths from the proposed model and confirm:

- Mean realised `rtb` ≈ 0.0091 (the unconditional sample mean) within Monte Carlo error.
- Standard deviation of realised `rtb` ≈ 0.027 (the marginal stationary std implied by the new VAR's `Sigma_z`) within Monte Carlo error.
- The realised inflation series `π = log(1+y_1_lag) − rtb`, computed inline on simulated paths, has autocorrelation ρ(1) ≈ 0.7 — matching the empirical inflation persistence in the data.
- Realised excess stock and bond returns retain their existing moments (the change should not affect xr or xb's marginal distributions to first order, only their conditional correlations with rtb-via-state).

---

## 7. Out of scope

- **Adding inflation π as a state.** Considered and rejected. Produces a rank-deficient innovation block (`Σ_r_cond` smallest eigenvalue collapses to ~3×10⁻⁷ along the rtb axis because `e_rtb = −e_π` mechanically). The OPTION-B column in §4.1 documents what this failure mode looks like.
- **Switching to the unrestricted VAR (CCV's published version).** Lets lagged xr, xb predict the state, which collapses state-block autonomy and breaks the grid-based DP architecture.
- **Modifying the labour-income process, tax/pension code, or working-income table.** State-dimension-independent; not affected by this change.
- **Adding TIPS or inflation-indexed bonds.** The proposed parameterisation has rtb explicit and π implicit. If TIPS becomes a future requirement, the cleaner extension is the dual parameterisation (π explicit, rtb computed inline), which is a separate design decision not part of this change.
- **Algorithmic redesign of the FOC solver, EGM logic, value-function iteration scheme, or quadrature rules.** All preserved as-is; only the state dimension flowing through them changes.

---

## 8. References

- `w8566.pdf` (Campbell, Chan & Viceira 2003): §2 (state-vector structure `z = [r_1; x; s]` with rtb as the first / benchmark element), §4.1 (explicit rationale for including ex-post real bill rate and nominal bill rate together to make inflation recoverable), §4.2 (VAR estimation results), §5 and Appendix C (TIPS construction relying on rtb being in the first state slot).
- Project file (numba branch, conceptually identical in JAX branch): `var.py`'s `build_nominal_system1_var_config()` docstring contains the full state-ordering rationale and the legacy `(0, 1, 2)` → current `(2, 1, 0)` migration note. The same logic extended to four states gives `(2, 1, 3, 0)`.
- Decision and analysis thread: this document is the deliverable from a design discussion that worked through (i) the inflation-VAR predictability gain (10.4 ppt rtb R² available), (ii) the rank-deficiency analysis ruling out OPTION-B, (iii) the Cholesky two-prong rule selecting `(cy, spr, rtb, y_1)`, and (iv) numerical pre-validation through the project's own VAR estimator confirming the PROPOSED column reproduces.
