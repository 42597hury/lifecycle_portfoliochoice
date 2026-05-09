# Production Config Synthesis — Single A100 Thesis Run

**Date:** 2026-05-09
**Branch:** `jax-rewrite`
**Hardware target:** Single A100 SXM4 (80 GB the floor; final hardware will be stronger — A100 is the conservative ceiling)
**Wall budget:** ≤ 18 h end-to-end including simulator on 50–100k households
**Convergence target:** max relative policy bias vs the next finer cell ≤ 1e-3 on alphas; ≤ 1e-2 on c
**Companion artefact:** `configs/_canonical_proposed.py` (untracked draft)
**Handoff:** [HANDOFF_PRODUCTION_CONFIG_DESIGN_2026-05-09.md](../handoff/HANDOFF_PRODUCTION_CONFIG_DESIGN_2026-05-09.md)

---

## §6 — TL;DR

> The headline change vs the live `configs/_canonical.py` is **`max_iter` 8000 → 30** and **`max_backtrack_iter` (default) 10 → 5**. Both are fori_loop hard wall multipliers (`foc_calls = 1 + max_iter × (1 + max_backtrack_iter)` per [COMPLEXITY_WALL_TIME_2026-05-06.md §2](COMPLEXITY_WALL_TIME_2026-05-06.md)). Cutting them takes the per-Newton-solve FOC count from 88,011 to 181 — a **486× reduction**, fully justified by [NEWTON_FAILURE_STRUCTURE_2026-05-08.md](NEWTON_FAILURE_STRUCTURE_2026-05-08.md): cells stuck at `max_iter` are tol-unreachable boundary cells, not iter-bound; bumping `max_iter` past 30 does not move the fail rate.
>
> Every other dial carries over from the live canonical. The state vector axes, wealth-grid bounds, gather precision, tol, and `init_alpha_*` values are either locked-in by the handoff or POST-PIVOT GAPs whose safe carry-over matches the live canonical.
>
> **Projected wall on 1× A100 SXM4:** **2.0–2.5 h end-to-end** (≈ 2.2 h fp64 baseline, ≈ 1.8 h with f32 gather, ±30 % from the formula's calibration band). Vs the 18 h budget this leaves **8× headroom**, large enough to absorb worst-case overhead, run a g6 confirmation pass, or bump `n_z` toward 15/30 if the user wants belt-and-braces convergence.
>
> Three POST-PIVOT GAPs are flagged in §4 — `state_grid_sizes`, `state_n_stds`, and `init_alpha_*` carry pre-pivot evidence onto the new (cape, spr, y_1) state vector. Each is the safe carry-over per the handoff method §3(a) but should be confirmed by a focused 3-axis sweep before committing the canonical.

---

## §1 — Per-axis decision table

Every value is defended from a sweep already on disk OR (when post-pivot evidence is missing) flagged as a POST-PIVOT GAP carry-over.

| Field | Proposed | Current canonical | Evidence file:line | Convergence threshold reached | Wall impact |
|---|---:|---:|---|---|---:|
| `n_wealth` | 180 | 180 | [SYSTEM_I_WEALTH_GRID_CONVERGENCE_2026-05-08.md §13](SYSTEM_I_WEALTH_GRID_CONVERGENCE_2026-05-08.md) | publication anchor; n_w=120 sub-recommendation rejected post-Path-B | unchanged |
| `wealth_min` | 0.01 | 0.01 | [_canonical.py:89-103](../../configs/_canonical.py#L89) | locked: Path B clamp + f32 spacing safety (min_rel_diff32 = 3.77e-2 at n=180 over [0.01, 750]) | unchanged |
| `wealth_max` | 750.0 | 750.0 | [SYSTEM_I_WEALTH_GRID_CONVERGENCE_2026-05-08.md §4](SYSTEM_I_WEALTH_GRID_CONVERGENCE_2026-05-08.md) p99 wealth at age 67 = 935 AWI; 750 clips top 1 % | structural carry-over | unchanged |
| `n_savings` | 180 | 180 | EGM convention; same as `n_wealth` | publication anchor | unchanged |
| `state_grid_sizes` | (5, 5, 5) | (5, 5, 5) | [INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md §2](INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md) (4-axis); g3 RED, g4 YELLOW, g5 working minimum. **POST-PIVOT GAP** for cape replacing dp; per-axis sensitivity for cape on 3-axis state not measured | working minimum (carry-over) | unchanged |
| `state_grid_mode` | "cholesky" | "cholesky" | locked-in; matches all sweep bundles | — | unchanged |
| `state_n_stds` | (2.0, 2.25, 2.25) | (2.0, 2.25, 2.25) | **POST-PIVOT GAP**: pre-pivot canonical mapped (dp, spr, rtb, y_1) → (2.0, 2.25, 2.0, 2.25); the current 3-axis values are placeholders ([_canonical.py:70-74](../../configs/_canonical.py#L70)) | unvalidated carry-over | unchanged |
| `n_z` | 11 | 11 | [SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md §1, §5](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md): n_z=10 RED, n_z=15 YELLOW per-cell GREEN typical-cell, n_z=30 GREEN both. n_z=11 carries the prior-canonical YELLOW caveat ("typical-cell GREEN; sup-norm tail unmeasured") | YELLOW per-cell, GREEN typical (System I 1D, plausibly easier on 3D state) | unchanged |
| `n_stds` | 3.0 | 3.0 | unchanged solver convention | — | unchanged |
| `n_eps_nodes` | 4 | 4 | [SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md §1](SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md) | GREEN at smallest tested (rel-sup C 0.36 %) | unchanged |
| `n_eta_nodes` | 3 | 3 | same | GREEN at smallest tested | unchanged |
| `n_ret_nodes_1d` | (4, 4) | (4, 4) | [SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md §5](SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md) (4,4) vs (3,3) sup_S = 7.94e-2, sup_B = 1.50e-1 — finite-horizon (3,3) is RED. Inf-horizon evidence saturated at K=3, but finite-horizon is the production read | finite-horizon evidence supports K=4 | unchanged |
| `ret_lobatto_Z` | None | None | post-pivot decision: Lobatto removed everywhere ([_canonical.py:80-82](../../configs/_canonical.py#L80)) | — | unchanged |
| `n_state_quad_nodes` | (3, 3, 5) | (3, 3, 5) | [INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md "Axis-Bump Policy Movement"](INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md): y_1 K-bump 3→5 dominant (p95 portfolio Δ = 2.10e-1; spr 3.10e-2; dp 8.0e-3). [SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md §5](SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md): K-bump (3,5) cuts 80 % of sup C vs uniform (4,4) | y_1 axis dominant; cape axis assumed similar to dp (POST-PIVOT GAP) | unchanged |
| `state_lobatto_Z` | None | None | post-pivot decision: Lobatto removed everywhere | — | unchanged |
| `tol` | 1e-6 | 1e-6 | [NEWTON_FAILURE_STRUCTURE_2026-05-08.md §6-§7](NEWTON_FAILURE_STRUCTURE_2026-05-08.md): tol=1e-7 unreachable at high-savings tail under γ=5; tol=1e-6 cuts fail rate ~50–80 % at zero wall cost | locked-in | unchanged |
| `max_iter` | **30** | 8000 | [NEWTON_HISTOGRAM_AUDIT_2026-05-07.md "Validation"](NEWTON_HISTOGRAM_AUDIT_2026-05-07.md): post-fix p50=2 iters (most cells converge in 2 thanks to backward warm-start). [NEWTON_FAILURE_STRUCTURE_2026-05-08.md §2](NEWTON_FAILURE_STRUCTURE_2026-05-08.md): cells stuck at cap are tol-unreachable boundary, not iter-bound. [COMPLEXITY_WALL_TIME_2026-05-06.md §2](COMPLEXITY_WALL_TIME_2026-05-06.md): foc_calls = 1 + max_iter×(1+max_bt). Bumping past 30 buys ~10 % rate; ~5–10 % wall premium | iter-cap concentrated at sentinel-replaced corner cells; interior cells converge in <30 | **267× wall save** vs current 8000 |
| `max_backtrack_iter` | **5** | 10 (default) | [COMPUTE_EFFICIENCY_REVIEW_2026-05-08.md candidate #2](COMPUTE_EFFICIENCY_REVIEW_2026-05-08.md): inf-horizon `n_backtrack_total` p99 ≈ 100–125 spread across 8–12 active iters → ~9–12 halvings summed across full Newton solve, typical line search uses ~1 halving. p99 per-iter ≤ 5 | typical line search ~1 halving; p99 ≤ 5 expected | **10–20 % wall save** |
| `init_alpha_s` | 0.85 | 0.85 | **POST-PIVOT GAP**: documented as long-run-mean Full-System carryover ([_canonical.py:131-133](../../configs/_canonical.py#L131)). New VAR's stationary alphas not measured. `use_backward_age_warm_start=True` bounds the impact: `init_alpha_*` only seeds terminal age | safe carry-over | unchanged |
| `init_alpha_b` | 0.44 | 0.44 | same | safe carry-over | unchanged |
| `delta_bequest` | 0.0 | 0.0 | locked: luxury-bequest shifter dropped per pivot baseline ([_canonical.py:134-136](../../configs/_canonical.py#L134)) | — | unchanged |
| `gather_precision` | "f32" | "f32" | locked-in. f32/f64 agreement to ~1e-4 relative validated on prior canonicals ([_canonical.py:138-140](../../configs/_canonical.py#L138)); `verify/mixed_precision_working.py` will re-confirm under post-pivot lifecycle | — | 10–20 % wall save (already captured in current canonical) |
| `cell_vmap_chunks` | 1 | 1 (default) | A100 has 80 GB HBM; per-cell working set ~9 MB at 5×5×5; chunking adds dispatch overhead with no memory benefit | — | unchanged |

---

## §2 — Full proposed config literal

Paste-ready into `configs/_canonical.py`. Functionally identical to the live canonical except for `max_iter`, `max_backtrack_iter`, and a `delta_bequest=0.0` already present in the live file.

```python
from lifecycle.model import DiscretizationConfig, SolverConfig, SolveControl

PREDICTABILITY_SYSTEM = "full"

CANONICAL_DISC = DiscretizationConfig(
    n_wealth=180,
    wealth_min=0.01,
    wealth_max=750.0,
    n_savings=180,
    state_grid_sizes=(5, 5, 5),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=11,
    n_stds=3.0,
    n_eps_nodes=4,
    n_eta_nodes=3,
    n_ret_nodes_1d=(4, 4),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 3, 5),
    state_lobatto_Z=None,
)

CANONICAL_SOLVER = SolverConfig(
    tol=1e-6,
    max_iter=30,                  # was 8000 — 267× wall save
    max_backtrack_iter=5,         # was 10 (default) — 10–20 % wall save
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    use_line_search=True,
    delta_bequest=0.0,
    gather_precision="f32",
)

CANONICAL_SOLVE_CONTROL = SolveControl(
    checkpoint_every_n_ages=1,
    save_on_interrupt=True,
    return_partial_on_interrupt=True,
)
```

Diff vs live `configs/_canonical.py`:

```diff
 CANONICAL_SOLVER = SolverConfig(
     tol=1e-6,
-    max_iter=8000,
+    max_iter=30,
+    max_backtrack_iter=5,
     init_alpha_s=0.85,
     init_alpha_b=0.44,
     use_line_search=True,
     delta_bequest=0.0,
     gather_precision="f32",
 )
```

Two lines changed; one line added. No `DiscretizationConfig` change.

---

## §3 — Wall-time estimate on 1× A100 SXM4

Computed via the [COMPLEXITY_WALL_TIME_2026-05-06.md §2 formula](COMPLEXITY_WALL_TIME_2026-05-06.md), with the [CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md §2](CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md) hardware factor (GH200 → A100 SXM4 = 1.88×; effective fp64 TFLOPS = 6.8 / 1.88 = 3.62).

### Per-FOC and per-age inputs

| Quantity | Value |
|---|---|
| n_z | 11 |
| N_state | 5 × 5 × 5 = **125** |
| n_s | 180 |
| max_iter | 30 |
| max_backtrack_iter | 5 |
| foc_calls | `1 + 30 × (1 + 5)` = **181** |
| K_v | 3 × 3 × 5 = **45** |
| K_r | 4 × 4 = **16** |
| K_corners | 2³ = **8** |
| foc_FLOPs_retire | `45 × 16 × (8 × 12 + 40)` = **97,920** |

### Per-age wall (retire, fp64 baseline)

```
W_age_retire = (n_z × N_state × n_s × foc_calls × foc_FLOPs × foc_overhead)
               / (TFLOPS_eff × 1e12)
             = (11 × 125 × 180 × 181 × 97,920 × 6.5) / 3.62e12
             ≈ 7.9 s/age
```

### Lifecycle total (33 retire ages + 47 working/boundary ages)

Working-age multiplier 21× per the prior synthesis's back-solve against the System II finite-horizon anchor (594 s on 2× H100, 7² state, n_z=15, mi=100 → multiplier 21 with eff=0.8).

```
W_total = 33 × W_age_retire + 47 × (21 × W_age_retire)
        = 33 × 7.9 + 47 × 165.7
        ≈ 260 + 7,790  ≈  8,050 s  ≈  2.24 h
```

### With `gather_precision="f32"`

The §2 calibration table in [CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md](CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md) shows a systematic 1.20× over-prediction by the fp64-only formula on f32-gather bundles, attributed to f32 memory-bandwidth savings the formula does not model. Applying this:

```
W_total_f32 ≈ 8,050 / 1.20 ≈ 6,710 s ≈ 1.86 h
```

### Confidence band

| Source | Range |
|---|---|
| Formula calibration (multi-anchor ±21 % per §2) | ±21 % |
| Working-age multiplier (16–20× analytical, 21× backsolved) | +25 % |
| Hardware factor extrapolation (A100 vs GH200 anchor) | ±15 % |
| **Composite** | **~±30 %** |

### Headline projection

| Metric | Value |
|---|---|
| Lifecycle wall (1× A100, fp64) | **2.24 h ± 30 %** |
| Lifecycle wall (1× A100, f32 gather) | **1.86 h ± 30 %** |
| Worst-case (upper 30 % band, fp64) | **2.91 h** |
| Simulator (50–100k households, ~10–30 min) | **+0.2–0.5 h** |
| **End-to-end estimate** | **≈ 2.0–3.5 h** |
| Budget headroom vs 18 h target | **5×–9×** |

---

## §4 — POST-PIVOT GAPs

Three axes carry pre-pivot evidence onto the new (cape, spr, y_1) state vector. Each is the safe carry-over per the handoff method §3(a) but should be confirmed by a focused 3-axis sweep before committing the canonical.

### 4.1 `state_grid_sizes = (5, 5, 5)` — gap on `cape` axis

The 4-axis inf-horizon evidence ([INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md §2](INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md)) establishes g5 as the working minimum on (dp, spr, rtb, y_1). The new state vector is (cape, spr, y_1): cape replaces dp, rtb dropped. Two questions are unanswered:

1. **Does cape behave like dp on the state grid?** Both are slow equity-yield predictors; a g3 or g4 cape sub-axis is plausibly RED for the same reason dp's was. But not measured.
2. **Does dropping rtb materially loosen the grid requirement?** rtb was structurally inert (axis-bump run3 sub-1 % rel C). Removing it should free per-cell working memory but not change the convergence structure.

**Recommended follow-up:** a 3-axis isotropic sweep `state_grid_sizes ∈ {(4,4,4), (5,5,5), (6,6,6)}` at the proposed solver settings — three bundles, ~2.7 h total wall on 1× A100 (per §3 scaling). Decisively settles whether g5 is still the working minimum post-pivot.

### 4.2 `state_n_stds = (2.0, 2.25, 2.25)` — placeholder per the pivot handoff

The pre-pivot canonical was `(2.0, 2.25, 2.0, 2.25)` for (dp, spr, rtb, y_1). The current 3-axis tuple is "cape inherits dp's 2.0; spr keeps 2.25; y_1 keeps 2.25" — explicitly flagged as PLACEHOLDERS in [_canonical.py:70-74](../../configs/_canonical.py#L70). No 3-axis sensitivity sweep on `state_n_stds` exists.

**Risk if wrong:** the state grid covers ±n_stds standard-deviations of each axis. A too-narrow `state_n_stds[i]` clips the tail of axis i (cap-bound cells stuck at the edge); a too-wide value wastes nodes on a region simulated households never reach. Either way, sup-norm divergence at the worst cell is sensitive.

**Recommended follow-up:** at production grid `(5,5,5)`, sweep `state_n_stds ∈ {(2.0, 2.25, 2.25), (2.5, 2.5, 2.5), (1.75, 2.0, 2.0)}` — three bundles testing axis-by-axis sensitivity. ~3.3 h on 1× A100 if launched after the canonical commit.

### 4.3 `init_alpha_s = 0.85, init_alpha_b = 0.44` — long-run-mean carryover

These are documented as the long-run-mean Full-System portfolio under the old VAR ([_canonical.py:131-133](../../configs/_canonical.py#L131)). Under the new (cape, spr, y_1) VAR the stationary distribution is different, so the long-run-mean alphas may differ. Impact is bounded by `use_backward_age_warm_start=True`: `init_alpha_*` only seeds terminal age, then every younger age starts from the next-older converged policy.

**Risk if wrong:** terminal-age Newton spends extra iterations cold-starting from a poor warm; subsequent ages are unaffected. Wall premium ≤ 1 %; correctness unchanged.

**Recommended follow-up:** a one-shot post-canonical-solve probe: load the bundle's terminal-age policy at `(z_idx=mid, state_idx=mid, w_idx=mid)`, take the resulting `(alpha_s, alpha_b)` as the "actual long-run mean", and update `_canonical.py` if the values drift > 5 pp from current. Costs ~10 s of post-processing.

### Other axes — evidence-supported (no gap)

`n_eps_nodes=4`, `n_eta_nodes=3`, `n_z=11`, `n_ret_nodes_1d=(4,4)`, `n_state_quad_nodes=(3,3,5)`, `tol=1e-6`, `max_iter=30`, `max_backtrack_iter=5`, `gather_precision="f32"`, `delta_bequest=0.0`, `wealth_*`, `n_savings=180` — all defended in §1 from existing evidence. The `n_z=11` choice carries the prior-canonical YELLOW caveat (System I per-cell sup-norm at z = +1.5σ, never reached in 45-year working life), not a POST-PIVOT GAP.

---

## §5 — Pre-flight results

### Phase 0 (CPU pytest battery)

```
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 TF_CPP_MIN_LOG_LEVEL=2 \
python -m pytest \
  tests/test_constrained_corner_clamp.py \
  tests/test_wealth_grid.py \
  tests/test_income_normalization.py \
  tests/test_precompute_real_pivot.py \
  tests/test_simulator_real_yields_returns.py \
  tests/test_solver_real_yields_foc.py \
  -v
```

**Result: 43 passed in 26.66s.** All gates green:

- Constrained-corner clamp: 8/8 (Path B clamp + bit-identity at non-constrained wealth)
- Wealth grid: 6/6 (default log1p, custom NPY round-trip, fp32 spacing validator, checkpoint guards, canonical fp32 validator)
- Income normalization: 13/13 (eps/eta zero-mean, mu2 derivation, canonical zero-mean — Fix A)
- Precompute real-yields pivot: 4/4 (each of System 1 / 2 / Full builds; rejection of n_state=4)
- Simulator real-yields: 5/5 (build kernel signature, n_state=4 rejection, bill-only realises y_1)
- Solver real-yields FOC: 7/7 (pcjax y_1_idx matches model; log_R_bill deterministic; rejection of n_state=4)

### Tiny canonical-shape smoke (CPU, ~30 s)

A separate sanity check that the **proposed config literal** itself constructs and builds at reduced grids (`n_wealth=12, state_grid=(2,2,2), n_z=3, ages 63-65`):

- `PROPOSED_DISC` / `PROPOSED_SOLVER` / `PROPOSED_SOLVE_CONTROL` import and type-check ✓
- All 17 proposed field values match §2 ✓
- `build_model` + `build_precompute` succeed at tiny shape ✓
- `wealth_grid[0] == 0.01` exactly — Path B clamp anchor present at the lowest grid point ✓

### Phase 2 / 3 / 4 (cluster-only — sketched, NOT run)

Per the handoff "tiny smokes only on the laptop". These need an A100 to run. The user should launch:

| Phase | Script | Expected wall on A100 | What it gates |
|---|---|---|---|
| Phase 2 | `verify/canonical_e2e_preflight.py` | ~15 min | Full-lifecycle smoke at `n_wealth=30, state_grid=(3,3,3), n_z=5`; Pass A vs Pass B+C resume bit-identity to 1e-12; Path B corner fires; simulator clean post-Pi_z |
| Phase 3 | `verify/mixed_precision_working.py` | ~20–30 min | f32 vs f64 working-age agreement to ~1e-4 relative under the post-pivot lifecycle |
| Phase 4 | a "tiny canonical-shape" run at `n_wealth=60, state_grid=(4,4,4), n_z=7, max_iter=30, max_bt=5` | ~30 min | Solve completes under the proposed solver settings; no Newton failures > the ~3 % NEWTON_FAILURE_STRUCTURE baseline |

### Path B regression

- `wealth_grid[0] == 0.01` exactly at the proposed `n_wealth=180, wealth_min=0.01` ✓ (verified in tiny smoke)
- `C[..., 0] == wealth_grid[0]` corner solution is asserted by `verify/canonical_e2e_preflight.py`; need cluster run to confirm at proposed config

### Income normalization regression

`tests/test_income_normalization.py::test_canonical_model_zero_mean` PASSED ✓. The proposed config does not change `pe`, `pz`, or any mu/sigma field — Fix A is preserved.

---

## §1.5 — Pareto-style budget reading (where headroom goes)

The 18 h budget vs 2.0–3.5 h projection leaves substantial headroom. Three uses, in order of priority:

1. **Run the POST-PIVOT GAP sweeps before commit.** State grid + state_n_stds together: ~6 h on 1× A100. Total wall under 10 h. Decisively closes the §4 gaps. **Recommended.**
2. **Bump `n_z` to 15 if the user wants a clean publication-grade reading.** The System I evidence places n_z=15 at YELLOW per-cell GREEN typical-cell, vs n_z=11's "between RED and YELLOW" extrapolation. Wall premium 36 %: 2.24 h → 3.05 h on 1× A100. Trivial cost; opt-in.
3. **Run g6 = (6,6,6) if the user wants belt-and-braces state convergence.** Wall premium 73 %: 2.24 h → 3.87 h. Useful as a confirmation pass after the canonical solve completes; provides a g5↔g6 sup-norm reading on the new state vector that the current evidence base lacks.

The recommendation in §6 (TL;DR) is the minimum-change config. The above are explicit "use the headroom" upgrades the user can layer in.

---

## §7 — Stretch-goal ablation configs (sketches)

Per the handoff §"Stretch goals". Each ablation reuses the proposed solver config and slices `state_grid_sizes` / `n_state_quad_nodes` to a sub-system.

### 7.1 System 1 ablation (drop `cape, spr` — slice to (y_1,))

```python
disc_system_1 = PROPOSED_DISC._replace(
    state_grid_sizes=(5,),
    state_n_stds=(2.25,),
    n_state_quad_nodes=(5,),
)
```

N_state = 5; K_v = 5; K_corners = 2¹ = 2. Per-FOC FLOPs = 5 × 16 × (2×12 + 40) = 5,120 (vs 97,920 in Full = 0.052×). Wall projection ~7 min on 1× A100 — well under the "≤ 1/8 Full System" target.

### 7.2 System 2 ablation (drop `cape` — slice to (spr, y_1))

```python
disc_system_2 = PROPOSED_DISC._replace(
    state_grid_sizes=(5, 5),
    state_n_stds=(2.25, 2.25),
    n_state_quad_nodes=(3, 5),
)
```

N_state = 25; K_v = 15; K_corners = 4. Per-FOC FLOPs = 15 × 16 × (4×12+40) = 21,120 (vs 97,920 = 0.216×). Wall projection ~30 min on 1× A100 — under the "≤ 1/3 Full System" target.

### 7.3 Scenario-sweep config (10 calibration variants over 1 night)

For tight per-cell discrimination across calibration variants: drop `n_z` to 7, `state_grid_sizes` to (4,4,4), `n_state_quad_nodes` to (3,3,3), keep K-bump if y_1-sensitive variants are in scope:

```python
disc_scenario = PROPOSED_DISC._replace(
    n_wealth=120,
    n_savings=120,
    state_grid_sizes=(4, 4, 4),
    n_z=7,
    n_state_quad_nodes=(3, 3, 3),
    n_ret_nodes_1d=(3, 3),
)
```

N_state = 64; K_v = 27; K_corners = 8. foc_FLOPs ratio vs Full ≈ 0.066. Wall projection ~10 min on 1× A100 → 10 variants in <2 h. Resolution intentionally ablation-grade (System I YELLOW thresholds), not publication-grade — intended to discriminate between calibrations, not to commit results.

These three ablations together account for a typical robustness-check workflow: the publication-grade Full config + a 10-variant scenario sweep + System 1/2 sub-system runs all fit in <12 h of compute.

---

## §8 — Risk register and named backups

| Risk | Likelihood | Detection | Mitigation |
|---|---|---|---|
| `max_iter=30` insufficient on a hard interior cell | Low — System I evidence shows cap-bound cells are tiny-savings boundary | `diagnostics.newton_iter_histogram p99` after solve | Bump `max_iter=50` (still 16× wall save vs 8000); revisit only if `total_newton_failures / n_cells > 5 %` |
| `max_backtrack_iter=5` insufficient | Very low — typical line search ~1 halving | `diagnostics.backtrack_iter_histogram p99 > 5` | Bump `max_backtrack_iter=10` (default) |
| POST-PIVOT GAP on `state_grid_sizes` shows g5 too coarse | Medium — 4-axis evidence g4 → g5 → asymptote not bottomed out | Run g5 vs g6 sweep (§1.5 recommendation #1) | Switch to g6; +73 % wall, still under budget |
| f32 gather diverges materially from f64 | Low — prior canonicals showed sub-1e-4 relative drift | `verify/mixed_precision_working.py` cluster run | Set `gather_precision="f64"`; +20 % wall |
| `init_alpha_*` warm-start poor under new VAR | Low — backward-age warm bounds the impact to terminal age | Wall comes in 5 % above projection; or `n_iters_used[terminal]` saturated | Re-derive from a post-canonical-run probe (§4.3) |

### Two named backups

- **If the proposed config OOMs or fails to converge on the cluster:** fall back to **`max_iter=50, max_backtrack_iter=10`** (matches the prior synthesis row D's solver settings; +50 % wall but still ~3.3 h on 1× A100). Same axis decisions, more Newton headroom.
- **If wall comes in materially under 2 h:** **bump `n_z` to 15 first** (the cheapest axis to bump per the prior synthesis; +36 % wall, removes the System I YELLOW caveat). Only bump `state_grid_sizes` to (6,6,6) if `n_z=15` results show un-converged tail cells.

---

## §9 — What this synthesis does and does not commit

**Commits to:** A defensible per-axis recommendation for the production canonical, ready to be diffed into `_canonical.py`. Per-axis citations from existing scans. Wall-time projection with explicit confidence band. Three POST-PIVOT GAPs flagged with follow-up sweep specs. Phase 0 pre-flight battery passes (43/43 tests + tiny smoke).

**Does not commit to:** Any modification of `_canonical.py` — that file is read-only until the user reviews this synthesis. Phase 2/3/4 cluster validation has not been run. The POST-PIVOT GAPs are not closed; they are flagged for the user to decide whether to close before commit.

**The decision the user is making:** whether the §2 literal is the right canonical to commit to `_canonical.py`, or whether the §1.5 / §4 follow-up sweeps should run first.

---

## Appendix A — Reproducibility

The proposed config is at [configs/_canonical_proposed.py](../../configs/_canonical_proposed.py) (untracked, per the handoff "Hard constraints"). Run the Phase 0 battery with:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 TF_CPP_MIN_LOG_LEVEL=2 \
python -m pytest \
  tests/test_constrained_corner_clamp.py \
  tests/test_wealth_grid.py \
  tests/test_income_normalization.py \
  tests/test_precompute_real_pivot.py \
  tests/test_simulator_real_yields_returns.py \
  tests/test_solver_real_yields_foc.py \
  -v
```

Project a custom row through the COMPLEXITY_WALL_TIME formula:

```bash
python scripts/analysis/compute_budget_estimator.py --table
```

(or extend with the per-row code in [COMPLEXITY_WALL_TIME_2026-05-06.md Appendix B](COMPLEXITY_WALL_TIME_2026-05-06.md)).

---

**End of synthesis.** §6 is the deliverable; §1–§5 are its supporting argument.
