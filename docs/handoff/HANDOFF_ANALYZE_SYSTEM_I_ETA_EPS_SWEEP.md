# Handoff: Analyze System I × (n_eta, n_eps) Sensitivity Sweep

**Branch:** `jax-rewrite`
**Effort:** ~half day. Pure analysis. Mirrors the structure of `HANDOFF_ANALYZE_SYSTEM_I_NZ_SWEEP.md`.
**Output:**
- `scripts/analysis/system_i_eta_eps_convergence.py` — main analysis script.
- `docs/scans/SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md` — short report with convergence numbers and a verdict on whether `(n_eta=3, n_eps=4)` is adequate.

---

## Purpose

Quantify how much the **working-age income-shock quadrature density `(n_eta, n_eps)`** matters for consumption and portfolio policies in System I (rtb-only state). The economic claim under test:

> **At (n_eta=3, n_eps=4), the policy is already converged.** Refining to (4,5) or (6,6) produces only negligible changes.

If true, all future ablation runs (Systems II, III, IV) can use the cheap (3,4) quadrature with no loss of policy accuracy. If false, the curve tells us where to stop.

---

## Inputs

Three bundles, fixed at `n_z=30`, varying `(n_eta, n_eps)`:

```
saved_runs/ablations/system_i_grid7_nz30_eta3eps4_calib1/
saved_runs/ablations/system_i_grid7_nz30_eta4eps5_calib1/
saved_runs/ablations/system_i_grid7_nz30_eta6eps6_calib1/
```

Each contains policy_arrays.npz, metadata.json, diagnostics.pkl. Same n_z and state grid across all three — so policy arrays have **identical shape** `(78, 30, 7, 180)`. **Direct element-wise comparison is valid; no interpolation needed.**

(Compare to the n_z sweep where bundles had different shapes and required z-axis interpolation.)

---

## Analysis plan

### §1 — Load and verify

```python
from lifecycle.policy_io import load_policy_bundle
bundles = {
    (3, 4): load_policy_bundle("saved_runs/ablations/system_i_grid7_nz30_eta3eps4_calib1/"),
    (4, 5): load_policy_bundle("saved_runs/ablations/system_i_grid7_nz30_eta4eps5_calib1/"),
    (6, 6): load_policy_bundle("saved_runs/ablations/system_i_grid7_nz30_eta6eps6_calib1/"),
}
```

Verify:
- All three return shape `(78, 30, 7, 180)` for C, S, B.
- No NaN/Inf in any.
- `total_newton_failures == 0` for all.
- `state_names == ('rtb',)` and `n_z == 30` consistent.

### §2 — Reference + direct comparison

Pick **(n_eta=6, n_eps=6) as the reference** (highest-density). For each coarser config, compute element-wise:

```python
delta_C = np.abs(C[(3,4)] - C[(6,6)])
delta_S = np.abs(S[(3,4)] - S[(6,6)])
delta_B = np.abs(B[(3,4)] - B[(6,6)])
# ... same for (4,5) vs (6,6)
```

**No interpolation needed.** Same shape across all bundles.

### §3 — Convergence metrics

For each (3,4) and (4,5):

1. **Sup-norm divergence**: `sup_C, sup_S, sup_B` — worst-cell absolute differences vs (6,6).
2. **Relative divergence**: `max(|delta_C / C_(6,6)|)` where C_(6,6) ≠ 0. Economically interpretable.
3. **Per-axis max**: where does the divergence concentrate?
   - Per-age: `delta.max(axis=(1,2,3))` → 78 numbers
   - Per-z: `delta.max(axis=(0,2,3))` → 30 numbers (which z-states see the most divergence?)
   - Per-wealth: `delta.max(axis=(0,1,2))` → 180 numbers (low/high wealth?)
4. **Working vs retirement split**: working ages exercise eta×eps; retirement doesn't. Confirm that:
   - **Retirement-age divergence ≈ 0** (policy depends only on (rtb, w), not income shocks)
   - **Working-age divergence is where the sensitivity lives**
   - This is a sanity check on the numbers — if retirement-age divergence is non-zero, something else is going on.

### §4 — Convergence-curve visualization

X-axis: `eta × eps` product (12, 20, 36) — log scale.
Y-axis: sup-norm divergence vs (6,6) reference. One line per metric (C, S, B).

If (3,4) ≈ (4,5) ≈ (6,6) in policy space → curve flat → STOPPED-AT-(3,4) confirmed.

Also plot: alpha_s, alpha_b vs age at one probe cell (z = mean, state = mid-rtb, wealth = SCF median, using the per-axis midpoint convention from the recent probe-index fix). Three overlaid lines (one per (eta, eps)). Visual overlap → converged.

### §5 — Verdict

GREEN (3,4 adequate) / YELLOW (3,4 close; 4,5 better) / RED (3,4 visibly under-converged).

For each verdict, state implications for downstream workflows:
- "Use (n_eta=3, n_eps=4) for all System II/III/IV runs."
- OR "Use (4,5) for canonical, (3,4) only for cheap exploration."
- OR "(6,6) is the asymptote; need at least (5,4) for thesis quality."

### §6 — Cross-link with n_z sweep finding

If both the n_z sweep and this sweep show "stops early," the user has two robust calibration findings:
- `n_z=10` adequate
- `(n_eta=3, n_eps=4)` adequate

Combined: every future ablation run uses these resolutions. Wall savings compound. **The verdict here is the second leg of that two-leg result.**

---

## Existing scripts to reuse

Same as the n_z handoff:
- `lifecycle/policy_io.py:load_policy_bundle` — bundle loader.
- (Skip `discretize_income_ar1_mixture` — not needed here since shapes match.)

---

## Implementation checklist

- [ ] §1 — Load 3 bundles, verify shapes + sanity.
- [ ] §2 — Compute element-wise differences vs (6,6) reference.
- [ ] §3 — Sup-norm, relative, per-axis metrics. Verify retirement-age divergence ≈ 0.
- [ ] §4 — Convergence-curve plot + per-cell line plots.
- [ ] §5 — Verdict.
- [ ] §6 — Cross-link with n_z findings.
- [ ] Commit:
  ```
  docs+analysis: System I (n_eta, n_eps) sensitivity convergence study
  
  Loads the 3 ablation bundles (system_i_grid7_nz30_eta{3eps4,4eps5,
  6eps6}_calib1) and computes policy convergence as quadrature density
  grows. Verdict: <GREEN: (3,4) adequate / YELLOW / RED>.
  
  Sup-norm divergences at (n_eta, n_eps)=(3,4) and (4,5) vs (6,6):
  C: ..., S: ..., B: ...
  
  Combined with n_z sweep (n_z=10 verdict): <one-line>.
  ```

---

## Pause points

- **Bundle loading fails** for any of the 3.
- **Shapes don't match** (would mean a config bug; surface for user review).
- **Retirement-age divergence is non-zero**: should be exactly 0 because eta/eps don't enter the retirement FOC. If non-zero, indicates a precision issue or a config inconsistency between bundles.
- **Sup-norm divergences are massive** (e.g., `sup_C > 0.5`): bundles aren't comparable; verify config.

---

## Out of scope

- **Solver-side changes.**
- **Re-running any of the 3 bundles.**
- **Cross-system comparison** (System I vs II/III/IV).
- **Sim-based EE comparison.** Stay grid-based.
- **Performance optimization** of the analysis code.

---

## Why this matters

If both this sweep and the n_z sweep show "stops at minimum tested value," the user has rigorous justification for using cheap discretization on every future ablation run. **Compounded savings: 3-7× on each of Systems II, III, IV.** Across the planned ablation set this is meaningful compute + dollars saved + faster iteration.
