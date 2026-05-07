# Handoff: Analyze System II Quadrature Sensitivity Sweep

**Branch:** `jax-rewrite`
**Effort:** ~half day. Pure analysis. Mirrors the structure of
`HANDOFF_ANALYZE_SYSTEM_I_NZ_SWEEP.md` and
`HANDOFF_ANALYZE_SYSTEM_I_ETA_EPS_SWEEP.md`.
**Output:**
- `scripts/analysis/system_ii_quad_convergence.py`
- `docs/scans/SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md`

---

## Purpose

Quantify how state-quadrature density and return-quadrature density affect
consumption and portfolio policies in System II (state vector = (rtb, y_1)).
The 4-run factorial isolates **four directional effects** cleanly.

---

## Inputs

Four bundles, all at fixed `n_z=15`, `(n_eta, n_eps)=(3, 4)`, full lifecycle
(78 ages):

```
saved_runs/ablations/system_ii_grid7x7_nz15_sq3x3_rq3x3_calib1/   # baseline
saved_runs/ablations/system_ii_grid7x7_nz15_sq4x4_rq3x3_calib1/   # uniform state refinement
saved_runs/ablations/system_ii_grid7x7_nz15_sq3x3_rq4x4_calib1/   # ret refinement
saved_runs/ablations/system_ii_grid7x7_nz15_sq3x5_rq3x3_calib1/   # y_1-axis refinement (no Lobatto)
```

Each contains `policy_arrays.npz`, `metadata.json`, `diagnostics.pkl`. **All
four bundles have identical shape** `(78, 15, 49, 180)` because `n_z`, `n_state`
(via state_grid_sizes), `n_w` are fixed across runs. **Direct element-wise
comparison is valid; no interpolation needed.**

(Same simplicity as the eta-eps handoff; different from the n_z handoff which
needed z-axis interpolation.)

---

## Four directional effects to isolate

| Effect | Compare | Tests |
|---|---|---|
| **State-quad uniform refinement** | (3,3)/(3,3) vs (4,4)/(3,3) | Does going from 9 → 16 nodes uniformly help? |
| **Ret-quad refinement** | (3,3)/(3,3) vs (3,3)/(4,4) | Does going from 9 → 16 ret nodes help? |
| **y_1-axis targeted refinement** | (3,3)/(3,3) vs (3,5)/(3,3) | Does spending K=5 on the y_1 axis help vs K=3? |
| **K-bump vs uniform refinement** | (4,4)/(3,3) vs (3,5)/(3,3) | Same total state nodes (16 vs 15) — biased toward y_1 vs uniform: which wins? |

The last comparison is the load-bearing economic question — if K-bump on y_1
matches uniform (4,4) at near-equal node count, it's a free Smolyak-style
optimization for future Systems III/IV.

---

## Analysis plan

### §1 — Load and verify

```python
from lifecycle.policy_io import load_policy_bundle
bundles = {
    "sq3x3_rq3x3": load_policy_bundle(".../sq3x3_rq3x3_calib1/"),
    "sq4x4_rq3x3": load_policy_bundle(".../sq4x4_rq3x3_calib1/"),
    "sq3x3_rq4x4": load_policy_bundle(".../sq3x3_rq4x4_calib1/"),
    "sq3x5_rq3x3": load_policy_bundle(".../sq3x5_rq3x3_calib1/"),
}
```

Verify:
- All four return shape `(78, 15, 49, 180)` for C, S, B.
- No NaN/Inf in any.
- `total_newton_failures == 0` for all.
- `state_names == ('rtb', 'y_1')` consistent.

### §2 — Pairwise divergence matrix

Pick **(3,3)/(3,3) as the baseline** (cheapest config; the user's question is
"is this enough?"). Compute element-wise sup-norm divergences for each of the
3 refinement variants vs baseline:

```python
delta_C[label] = np.abs(C[label] - C["sq3x3_rq3x3"])
# same for S, B
```

Build a **4×4 (or 6-pair) divergence table**:

| | Compare to | sup_C | sup_S | sup_B | rel_S_max | rel_B_max |
|---|---|---|---|---|---|---|
| sq4x4_rq3x3 | sq3x3_rq3x3 | ... | ... | ... | ... | ... |
| sq3x3_rq4x4 | sq3x3_rq3x3 | ... | ... | ... | ... | ... |
| sq3x5_rq3x3 | sq3x3_rq3x3 | ... | ... | ... | ... | ... |
| sq3x5_rq3x3 | sq4x4_rq3x3 | ... | ... | ... | ... | ... |

Plus the cross-comparison sq4x4_rq3x3 vs sq3x5_rq3x3 (the K-bump-vs-uniform
question). Optional: full pairwise 4×4 matrix for completeness.

### §3 — Per-axis decomposition

For each pairwise comparison, compute where the divergence concentrates:

- **Per-age**: `delta.max(axis=(1,2,3))` → 78 numbers. Working ages typically dominate; check if retirement-age divergence is non-trivial (would suggest the state-quadrature is affecting retirement, not income shocks).
- **Per-z**: 15 numbers. Identifies whether divergence is at extreme labour-income states.
- **Per-state**: 49 numbers (rtb × y_1 grid). **The most informative slice** — identifies whether divergence is at extreme rtb or y_1 states. For the y_1-axis K-bump comparison, expect divergence to concentrate at high/low y_1 cells if the K-bump is doing real work.
- **Per-wealth**: 180 numbers. Is divergence at low-wealth or high-wealth cells?

### §4 — Visualization

- **Convergence-curve plot**: x-axis = total quad nodes (state×ret), y-axis = sup-norm divergence vs baseline. One marker per refinement variant. Connect logically: state-only refinement; ret-only refinement; y_1-only refinement.
- **Per-cell line plots at probe**: at one probe cell (z=mean, state=mid-rtb-mid-y_1, wealth=SCF median), plot α_s, α_b, c/W vs age — four overlaid lines (one per bundle). Visual overlap = converged.
- **State-grid heatmap**: 7×7 grid of pairwise (rtb, y_1) cells; color = sup-norm divergence at that cell averaged over (z, age, wealth). Highlights whether divergence concentrates at corner cells.

### §5 — Verdict

Three sub-verdicts:

1. **State-quad refinement value**: GREEN/YELLOW/RED (does (4,4) materially differ from (3,3)?)
2. **Ret-quad refinement value**: GREEN/YELLOW/RED (does (4,4) ret materially differ?)
3. **K-bump on y_1 (Smolyak's claim)**: GREEN (better than uniform) / YELLOW (equal to uniform) / RED (worse than uniform).

Implication for canonical:
- If GREEN on K-bump (run 4 < run 2 in divergence vs reference TBD): adopt `n_state_quad=(3,5)` + `state_lobatto_Z=(None, 2.93)` or similar as canonical for any system with y_1 in state.
- If YELLOW (equal): default to whichever is cheaper; (3,5)=15 nodes < (4,4)=16 nodes, so (3,5) wins on cost.
- If RED (uniform better): adopt (4,4) uniform; the Smolyak rec doesn't apply at System II scale.

### §6 — Cross-link with System I findings

System I sweeps (n_z, n_eta×n_eps) showed "stops at minimum tested." If System II
quad sweep shows the same pattern → strong evidence the predictability-system
ablations can run at cheap discretization across the board.

If System II shows DIFFERENT sensitivity (e.g., ret-quad matters here but didn't
in System I) → the canonical choice depends on system; ablation ladders need
per-system calibration.

---

## Existing scripts to reuse

- `lifecycle/policy_io.py:load_policy_bundle` — bundle loader.
- `scripts/analysis/system_i_eta_eps_convergence.py` — closest precedent (also same-shape comparison; reuse the helper functions for sup-norm and per-axis aggregation).
- `scripts/analysis/system_i_nz_convergence.py` — different (interpolation-based) but the visualization scaffolding is reusable.

---

## Implementation checklist

- [ ] §1 — Load 4 bundles; verify shapes + sanity.
- [ ] §2 — Pairwise sup-norm divergences (focused on baseline-comparison + the K-bump-vs-uniform comparison).
- [ ] §3 — Per-axis decomposition; identify where divergence concentrates.
- [ ] §4 — Convergence-curve plot + per-cell line plots + state-grid heatmap.
- [ ] §5 — Three sub-verdicts + canonical recommendation.
- [ ] §6 — Cross-link with System I findings.
- [ ] Write `docs/scans/SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md`.
- [ ] Commit:
  ```
  docs+analysis: System II quadrature sensitivity convergence study
  
  Loads the 4 quad-sweep bundles (sq{3x3,4x4,3x3,3x5}/rq{3x3,3x3,4x4,3x3})
  and computes pairwise policy convergence. Verdicts on:
    - state-quad uniform refinement: <GREEN/YELLOW/RED>
    - ret-quad refinement: <GREEN/YELLOW/RED>
    - y_1-axis K-bump (vs uniform 4×4): <GREEN/YELLOW/RED>
  
  Sup-norm divergences vs baseline (sq3x3_rq3x3):
  C: ..., S: ..., B: ...
  
  Recommendation for downstream Systems II/III/IV runs: <one-line>.
  ```

---

## Pause points

- Bundle loading fails for any.
- Shapes don't match (config bug; surface).
- Sup-norm divergences are massive (>0.5 in absolute terms) — bundles aren't comparable; verify config first.
- Run 4 (sq3x5_rq3x3, the K-bump variant) shows LARGER divergence than (4,4) uniform — the Smolyak rec may not apply at System II scale, contrary to what we expected. Surface as a finding; don't editorialize.

---

## Out of scope

- **Solver-side changes.**
- **Re-running any of the 4 bundles.**
- **Full canonical (5,5)/(5,5) extension.** Out of scope per the user's "remove all 5,5" direction earlier today.
- **Cross-system comparison** (System II vs III/IV).
- **Sim-based EE comparison.** Stay grid-based.

---

## Why this matters

Combined with the System I findings (n_z stops at 10, η×ε stops at (3,4)), this
sweep tells us whether **state-quadrature density** is also "stop early" in
System II. If yes → cheap discretization defensible across the ablation set.
If no → the canonical state-quad choice depends on which system.

The K-bump-vs-uniform finding (run 2 vs run 4) is the most economically
interesting: it tests whether the Smolyak-style asymmetric K-bump on the y_1
axis (Smolyak audit's "big win" recommendation) actually provides value at
System II scale, where there are only 2 state axes (rtb, y_1).
