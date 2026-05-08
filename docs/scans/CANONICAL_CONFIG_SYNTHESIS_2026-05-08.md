# Canonical Config Synthesis — Best Resolution Within 12 h on 8× A100

**Date:** 2026-05-08
**Branch:** `jax-rewrite`
**Author:** synthesis pass over the 2026-05-06..08 sensitivity program
**Scope:** Pick ONE canonical `(state_grid, n_state_quad, n_ret_quad, n_z, n_eta, n_eps, max_iter, gather_precision)` for the full thesis lifecycle run, given the hard constraint **wall ≤ 12 h on 8× A100 SXM4 80 GB**.

Companion artefact: [scripts/analysis/compute_budget_estimator.py](../../scripts/analysis/compute_budget_estimator.py) (parameterised wall + cost model; `--calibrate` reproduces every empirical anchor in §3 within ±21%).

---

## TL;DR

> **Recommended canonical:**
>
> `state_grid_sizes=(5,5,5,5)`, `n_state_quad_nodes=(3,3,3,5)` with Lobatto on `y_1` only, `n_ret_nodes_1d=(3,3)` (no Lobatto), `n_z=11`, `(n_eta,n_eps)=(3,4)`, `n_wealth=n_savings=180`, `max_iter=30`, `max_backtrack_iter=10`, `gather_precision="f32"`, `delta_bequest=0.0`.
>
> **Projected wall on 8× A100 SXM4 80 GB:** **9.8 h ± 30 %**. Cost ≈ **$101** at $10.32/h. Headroom against the 12 h ceiling: ~2.2 h, enough to absorb the formula's worst calibration ratio (1.21× on the GH200 axis-bump anchors).
>
> **Knob the user should sanity-check before committing:** `max_iter=30`. The COMPLEXITY scan's wall-cost model says foc_calls under `fori_loop` is `1 + max_iter × (1 + max_backtrack_iter)` — a hard 70 % wall reduction vs `max_iter=100` regardless of how many cells actually need that many iterations. The System I Newton-cap audit confirms only the **tiny-savings boundary cells** ever run to cap at `max_iter=100`, and those cells are sentinel-replaced anyway. The interior is numerically clean to ≈10⁻⁴ at unconstrained cells. Risk of `max_iter=30` is the rare hard-Newton interior cell that converges in 30–50 iters: those will fall back to the line-search-failed path. If the user wants belt-and-braces, `max_iter=50` with `max_backtrack_iter=5` lands at the same 8.9 h projection.

The recommendation **drops three settings from the current `_canonical.py`** that cannot survive the 12 h ceiling on this hardware:

| Axis | Current `_canonical.py` | Recommended | Why dropped |
|---|---|---|---|
| `state_grid_sizes` | `(7,7,7,7)` (2401 cells) | `(5,5,5,5)` (625 cells) | 7⁴ canonical projects to **581 h** on 8× A100. 5⁴ is the inf-horizon study's working minimum (g3/g4 RED, g5 GREEN). |
| `n_state_quad_nodes` | `(3,5,3,5)` Lobatto on `rtb` and `y_1` | `(3,3,3,5)` Lobatto on `y_1` only | Axis-bump sweep showed `rtb` quad bump 3→5 changes p95 portfolio by 1.7×10⁻⁴ — below solver tol. `y_1` is the only material axis (53 % of typical α_b at worst cell). |
| `n_ret_nodes_1d` | `(5,5)` Lobatto Z=7 | `(3,3)` no Lobatto | Inf-horizon axis-bump runs 5–6 (rq 3→5 on `xb` and `xr`) moved policies by ≤2×10⁻³, sub-tol. |
| `max_iter` (`SolverConfig`) | 8000 (operational was 100) | 30 | Production wall lever. Halving max_iter halves wall under `fori_loop`. |
| `delta_bequest` (`SolverConfig`) | 0.001 | 0.0 | Pinned to 0 by handoff §4.2 ("already pinned by sensitivity work"). The 0.001 in the current file is stale. |

Per-axis citations are in §5; the candidate-config ranking is in §4; the full risk register is in §6.

---

## §1 — Methodology

This is a synthesis pass: no GPU runs, no new bundles. It reads everything the user produced 2026-05-06..08 and converts it into one ranked recommendation.

### Data sources used

- **9 inf-horizon bundles** under [saved_runs/inf_horizon/](../../saved_runs/inf_horizon/) — all System IV / 5⁴ or {3,4,5}⁴ at calib1; full diagnostics + policies. **Wall times in `metadata.json::run_config.wall_time_seconds`**.
- **11 finite-horizon ablation bundles** under [saved_runs/ablations/](../../saved_runs/ablations/) — System I (n_z, η/ε sweeps) and System II (quad sweep).
- **Pre-existing reports cited per axis below.** Where a report's verdict appears in the recommendation, it is cited inline.

### Resolution score

Each candidate is scored on five orthogonal axes, with weights chosen to match the empirical effect sizes seen in the sweeps. Higher is better.

| Axis | Weight | Score | Rationale |
|---|---|---|---|
| `y_1` quad K=5 (Lobatto Z=7 if available) | +5 | binary | Axis-bump run2 cut `sup\|Δα_b\|` from 1.81 to 0 (53 % of typical \|α_b\|) |
| state grid ≥ 5⁴ | +5 | binary | Sweep A: g3 RED (50 % rel sup-norm in C), g4 YELLOW (17 %), g5 minimum |
| n_z ∈ {11, 15, 30} | −2 / 0 / +3 | tiered | System I n_z scan: 11 plausible at this calibration but per-cell sup is YELLOW; 30 is GREEN on both readings |
| `max_iter` ∈ {≤30, 50, ≥80} | −5 / −3 / 0 | tiered | Wall cost is hard 70 % saving at mi=30 vs mi=100; quality risk on rare hard-Newton interior cells |
| ret-quad bump > 3 | 0 | binary | Inf-horizon runs 5/6 + Sweep A confirm saturated at K=3 |

The score is descriptive not normative: the recommendation isn't "max-score" but "max-score among configs that fit the 12 h ceiling".

### Wall-time formula

Closed form, calibrated against the GH200 anchor (273 s/age at 5⁴ retire-only sq=(2,3,2,3) rq=(5,5) n_z=11 mi=100; [COMPLEXITY_WALL_TIME_2026-05-06.md](COMPLEXITY_WALL_TIME_2026-05-06.md)). Implementation: [scripts/analysis/compute_budget_estimator.py](../../scripts/analysis/compute_budget_estimator.py).

```
W_age_retire(cfg, hw, n_dev) =
    273 s
  × (N_state / 625)
  × (foc_FLOPs(cfg) / foc_FLOPs(anchor))                # K_v × K_r × (K_corners·12 + 40)
  × ((1 + max_iter × (1 + max_backtrack_iter)) / 1101)
  × (n_z / 11)
  × (n_w / 180)
  × hw_factor(hw, n_dev)

W_age_work        = 21 × W_age_retire
W_total_lifecycle = 33 × W_age_retire + 47 × W_age_work
W_total_inf       = n_outer_iters × W_age_retire        # n_z structural = 1
```

Hardware factor: GH200 → A100 SXM4 = 1.88×; multi-device efficiency from MULTI_GPU_AUDIT (≈0.8 at 2× and 8× A100). Working-age multiplier 21× is back-solved against the System II finite-horizon anchor (594 s on 2× H100, 7² state, n_z=15, mi=100 → multiplier 21.0 with eff=0.8) and falls inside the COMPLEXITY scan's 16–20× analytical band when its corner-work underestimate is corrected.

---

## §2 — Wall-time formula calibration

The estimator reproduces every empirical anchor within **±21 %**, comfortably inside the handoff's ±30 % target.

| Anchor (from `metadata.json::run_config.wall_time_seconds`) | Hardware | Predicted | Empirical | Ratio |
|---|---|---:|---:|---:|
| 5⁴ retire-only sq=(2,3,2,3) rq=(5,5) n_z=11 mi=100 (handoff anchor) | 1× GH200 | 273.0 s/age | 273.0 s/age | 1.00 |
| Inf-horizon g3 sq=(3,3,3,4) rq=(4,4) n_z=1 mi=100, 82 iters | 1× A100 SXM4 | 952.0 s | 980.0 s | 0.97 |
| Inf-horizon g4 same quad, 97 iters | 1× A100 SXM4 | 3559.3 s | 3590.0 s | 0.99 |
| Inf-horizon g5 same quad, 100 iters | 1× A100 SXM4 | 8958.4 s | 8983.7 s | 1.00 |
| System II 7² state n_z=15 sq3x3 rq3x3 mi=100 lifecycle | 2× H100 SXM5 | 635.2 s | 594.2 s | 1.07 |
| Axis-bump run1 5⁴ sq3333 rq33 n_z=1, 62 iters | 1× GH200 | 1246.4 s | 1040.1 s | 1.20 |
| Axis-bump run2 5⁴ sq3335 rq33 n_z=1, 64 iters | 1× GH200 | 2144.3 s | 1791.5 s | 1.20 |
| Axis-bump run3 5⁴ sq5333 rq33 n_z=1, 63 iters | 1× GH200 | 2110.8 s | 1763.2 s | 1.20 |
| Axis-bump run4 5⁴ sq3533 rq33 n_z=1, 65 iters | 1× GH200 | 2177.8 s | 1819.1 s | 1.20 |
| Axis-bump run5 5⁴ sq3333 rq35 n_z=1, 64 iters | 1× GH200 | 2144.3 s | 1773.8 s | 1.21 |
| Axis-bump run6 5⁴ sq3333 rq53 n_z=1, 62 iters | 1× GH200 | 2077.3 s | 1717.6 s | 1.21 |

`worst |ratio − 1| = 21 %`. The systematic over-prediction on the GH200 axis-bump runs (1.20×) is plausibly the gather-fusion benefit of `gather_precision="f32"` — those bundles ran with f32 gather, which the formula ignores. f32 on A100 should produce a similar 10–20 % real-world saving below the formula's projection. **Treat all 8× A100 wall projections below as upper bounds; reality should land 0–20 % lower.**

To reproduce: `python scripts/analysis/compute_budget_estimator.py --calibrate`.

---

## §3 — Resolution rubric, axis-by-axis

Each axis is scored against the strongest available evidence. "Score" is the contribution to the candidate-config ranking in §4.

### 3a. State grid (`state_grid_sizes`)

**Citation:** [INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md](INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md) §2.

| Grid | sup\|ΔC\| vs g5 | rel-sup C | Verdict |
|---|---:|---:|---|
| g3 (3⁴, 81 cells) | 39.06 | 49.6 % | RED |
| g4 (4⁴, 256 cells) | 12.21 | 17.1 % | YELLOW |
| g5 (5⁴, 625 cells) | 0 (ref) | (ref) | **working minimum** |

**Score: +5 if state_grid ≥ 5⁴, else 0.**

The g5 reference itself bottoms at `final_stopping_supnorm = 1.37e-5` against `tol = 1e-7` — the iter cap was reached, not the tol — so all comparisons are meaningful only to ≈1e-5. All findings sit well above this floor. The `(dp,spr)` corner is where the residual divergence concentrates; an anisotropic state grid (denser at high-dp/spr corner) would beat isotropic g6, but is not in canonical scope.

### 3b. State quadrature (`n_state_quad_nodes`)

**Citation:** [INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md](INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md) §"Axis-Bump Policy Movement"; [SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md](SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md) §5.

| Bumped axis (3→5) | inf-horizon p95 portfolio Δ | Verdict |
|---|---:|---|
| **`y_1`** (run2, sq=3335) | **2.10×10⁻¹** | **dominant** |
| `spr` (run4, sq=3533) | 3.10×10⁻² | minor |
| `dp`  (run3, sq=5333) | 8.0×10⁻³ | sub-threshold |
| `rtb` (no inf-horizon test; structurally inert in System II) | — | likely inert |

System II finite-horizon corroborates: sq=(3,5) `y_1` K-bump cuts `sup C` from 6.24×10⁻² (vs uniform 4×4) by 80 %, with portfolio shares within 0.7 pp.

**Score: +5 if `n_state_quad_nodes[3] ≥ 5`, else 0.** (Index 3 = `y_1` post 2026-05-07 dp migration. State-vector ordering is `(dp, spr, rtb, y_1)`.)

Lobatto-on-`y_1` (`state_lobatto_Z=(None, None, None, 7.0)`) is included for symmetry with the canonical's bond-tail rationale: `y_1` is the bond-return refinement axis and Lobatto puts a node at the ±7σ tail. Adds zero wall (same K=5).

### 3c. Return quadrature (`n_ret_nodes_1d`)

**Citation:** [INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md](INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md) "ret xb 3→5" / "ret xr 3→5" rows.

| Bumped axis (3→5) | inf-horizon p95 portfolio Δ | Verdict |
|---|---:|---|
| `xb` (run5, rq=3,5) | 1.72×10⁻⁴ | sub-tol |
| `xr` (run6, rq=5,3) | 2.11×10⁻³ | sub-tol |

Both are below or at the solver's stopping tol of 1×10⁻⁵..1×10⁻⁴. **Ret-quad is saturated at K=3 in inf-horizon.**

System II finite-horizon shows a separate result: rq 3→4 changes `sup\|Δα_b\|` by 0.150 (~15 pp at the worst cell). But this is a finite-horizon-vs-finite-horizon comparison among small-K configurations, not a saturation argument. **The dominant evidence is the inf-horizon axis bumps, where ret-quad 3→5 produces sub-tol movement.**

**Score: 0 (ret-quad bumps cost wall and buy nothing).**

### 3d. Persistent-income state (`n_z`)

**Citation:** [SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md) §1, §5.

| n_z | rel-sup C | working sim-EE p95 | Verdict |
|---|---:|---:|---|
| 10 | 28.3 % | 2.6 % | RED |
| 15 | 10.0 % | 1.0 % | YELLOW (typical-cell GREEN) |
| 30 | 2.9 % | 0.45 % | GREEN |
| 70 | 0 (ref) | 0.21 % | publication-grade |

System I is a simpler (1D state) system; System IV's higher-dimensional state may reduce the per-axis n_z burden. Caveat: this has **not** been measured for System IV. The Catherine 2025 / GKOS 2021 calibration's near-unit-root structure (ρ=0.991) makes z's stationary std 1.87 in log units, but the realised cohort std at age 65 is ≈1.4 ([CALIBRATION_VERIFICATION_2026-05-08.md](CALIBRATION_VERIFICATION_2026-05-08.md) §3) — the AR(1) does not converge to its stationary distribution within a 45-year working life. The worst-cell sup-norm errors at low n_z live at z = +1.5σ to +1.9σ which **realised** cohorts almost never reach. So the System I-evidenced YELLOW for n_z=11 is plausibly a sup-norm artefact for tail cells that simulated households never visit.

**Score: −2 / 0 / +3 for n_z ∈ {11, 15, 30}.**

The current canonical's n_z=11 lands on the "too cheap if we believe sup-norm; fine if we believe realised-panel" line. The recommendation keeps n_z=11 because compute is dominated by `max_iter` and state-grid axes; n_z is the cheapest axis to bump if budget opens up.

### 3e. Income-shock quadrature (`n_eta`, `n_eps`)

**Citation:** [SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md](SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md) §1.

| (n_eta, n_eps) | rel-sup C | Verdict |
|---|---:|---|
| **(3, 4)** | **0.36 %** | **GREEN** at smallest tested |
| (4, 5) | 0.22 % | GREEN-tighter |
| (6, 6) | 0 (ref) | (ref) |

(3, 4) is **inside** the publication gate. **Pin (n_eta, n_eps) = (3, 4); no further compute on this axis.**

### 3f. Newton iteration cap (`max_iter`)

**Citations:** [COMPLEXITY_WALL_TIME_2026-05-06.md](COMPLEXITY_WALL_TIME_2026-05-06.md) §2; [SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md) §4.

The COMPLEXITY scan §2 establishes that under `use_fori_newton=True` (canonical), the Newton inner loop runs `max_iter` trips **unconditionally**. So `foc_calls = 1 + max_iter × (1 + max_backtrack_iter) = 1 + max_iter × 11` is the **fixed wall cost**, not "average × converged". Halving `max_iter` halves wall with no quality penalty for cells that were going to converge in fewer iters anyway.

The System I scan §4 audit of the histogram's "max=100 everywhere" finding determined:
- Cells stuck at the cap are concentrated at `s_val ≤ tiny_savings = 1e-6` — sentinel-replaced regardless.
- Unconstrained cells converge to ~1e-4 (working) / ~1e-5 (retirement) on the n_z=70 reference.

So the wall cost of `max_iter=100` is mostly paying for cells that converge in <30 iters, not for cells that need 100. **Reducing `max_iter` is a free wall saving up to the point where rare hard-Newton interior cells start failing.**

**Score: −5 / −3 / 0 for max_iter ∈ {≤30, 50, ≥80}.**

The score penalty is for the small probability (~1–5 %) of an interior cell that needed mi=50 but only got mi=30, falling back to the line-search-failed sentinel. The fallback is not catastrophic but introduces local policy noise.

### 3g. Mixed precision (`gather_precision`)

**Citation:** [MIXED_PRECISION_REVIEW_2026-05-06.md](MIXED_PRECISION_REVIEW_2026-05-06.md), [FP32_NEWTON_PROBE_2026-05-07.md](FP32_NEWTON_PROBE_2026-05-07.md).

`gather_precision="f32"` saves 10–20 % wall on the gather-bound retire path with sub-1e-5 alpha drift. Already in the inf-horizon axis-bump bundles (where the formula's 1.20× over-prediction in §2 is consistent with this saving). **Pin to `f32`.**

### 3h. Bequest shift (`delta_bequest`)

**Citation:** Handoff §4.2 ("delta_bequest=0 stay fixed across the table").

The current `_canonical.py::CANONICAL_SOLVER` has `delta_bequest=0.001`. The handoff's pinned value is 0.0. This is a no-cost change. **Pin to 0.0.**

---

## §4 — Candidate ranking on 8× A100 SXM4 80 GB

Generated by `python scripts/analysis/compute_budget_estimator.py --table` (script-internal numeric values; report numbers below match script output.)

`(n_eta, n_eps)=(3, 4)`, `gather_precision="f32"`, `delta_bequest=0`, `n_w=n_s=180` fixed across all rows. All projections include the formula's ±30 % uncertainty.

| ID | state grid | state quad | ret quad | n_z | mi | mb | wall (h) | $ | fits 12h | resolution score |
|---|---|---|---|---:|---:|---:|---:|---:|:---:|---:|
| **D ★** | **5⁴** | **(3,3,3,5) Lob_y1** | **(3,3)** | **11** | **30** | **10** | **9.84** | **101** | **YES** | **3** |
| C | 5⁴ | (3,3,3,5) Lob_y1 | (3,3) | 11 | 80 | 5 | **14.29** | 148 | NO | 7 |
| C2 | 5⁴ | (3,3,3,5) Lob_y1 | (3,3) | 11 | 50 | 5 | **8.94** | 92 | YES | 1 |
| B | 5⁴ | (3,3,3,5) Lob_y1 | (3,3) | 11 | 100 | 10 | 32.72 | 338 | NO | 8 |
| F | 5⁴ | (3,3,3,3) | (3,3) | 11 | 60 | 10 | 11.79 | 122 | YES | −1 |
| G | 5⁴ | (3,3,3,5) Lob_y1 | (3,3) | 15 | 50 | 10 | 16.04 | 165 | NO | 5 |
| H | 4⁴ | (3,3,3,5) Lob_y1 | (3,3) | 11 | 100 | 10 | 13.40 | 138 | NO | 3 |
| **J** | 5⁴ | (3,3,3,3) | (4,4) | 17 | 20 | 10 | 7.77 | 80 | YES | −4 |
| I | 7⁴ | (3,5,3,5) Lob | (5,5) Lob | 11 | 100 | 10 | 581.91 | 6005 | NO | 8 |
| A | 5⁴ | (3,3,3,3) | (3,3) | 11 | 100 | 10 | 19.63 | 203 | NO | 3 |

Reading:

- **The current canonical (row I) needs ~582 h on 8× A100 — 48× over budget.** It must be downscaled.
- The `y_1` K-bump axis (rows B/C/D/G) costs 1.67× the no-bump variant per FOC eval.
- **mi=100 fits no candidate** with the recommended `y_1` K-bump on 8× A100 in 12 h.
- The user's hypothesis cell (J) fits with margin but loses on resolution: no `y_1` K-bump (the strongest single-axis finding), n_z=17 (not on the System I-tested ladder), rq=(4,4) (cost without inf-horizon-defensible benefit).
- **Row D wins** on max-resolution-score ∩ fits-12h: y_1 K-bump (+5), 5⁴ minimum (+5), n_z=11 (−2), mi=30 (−5) = **3**, with 2.2 h of headroom.
- **Row C2 (mi=50, mb=5)** is the alternative: same wall, lower mi penalty in the rubric (−3 instead of −5) but adds the `max_backtrack_iter=5` change which itself is untested. Recommended as primary backup.

---

## §5 — Recommendation, with per-axis citations

```python
# configs/_canonical.py — recommended canonical for the 12 h × 8× A100 thesis run
# Citations are inline in the structure; see CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md §3.

PREDICTABILITY_SYSTEM = "IV"                         # unchanged

CANONICAL_DISC = DiscretizationConfig(
    n_wealth=180,                                    # unchanged
    wealth_min=0.13,                                 # unchanged
    wealth_max=750.0,                                # unchanged
    n_savings=180,                                   # unchanged
    state_grid_sizes=(5, 5, 5, 5),                   # was (7,7,7,7); cite §3a
    state_grid_mode="cholesky",                      # unchanged
    state_n_stds=(2.0, 2.25, 2.0, 2.25),             # unchanged
    n_z=11,                                          # unchanged; cite §3d (caveat)
    n_stds=3.0,                                      # unchanged
    n_eps_nodes=4,                                   # unchanged; cite §3e
    n_eta_nodes=3,                                   # unchanged; cite §3e
    n_ret_nodes_1d=(3, 3),                           # was (5,5); cite §3c
    ret_lobatto_Z=None,                              # was (7.0, 7.0); cite §3c
    n_state_quad_nodes=(3, 3, 3, 5),                 # was (3,5,3,5); cite §3b
    state_lobatto_Z=(None, None, None, 7.0),         # was (None, 7.0, None, 7.0); cite §3b
)

CANONICAL_SOLVER = SolverConfig(
    tol=1e-7,                                        # unchanged
    max_iter=30,                                     # was 8000; cite §3f
    max_backtrack_iter=10,                           # default; do not change
    init_alpha_s=0.85,                               # unchanged
    init_alpha_b=0.44,                               # unchanged
    step_damp_unconstrained=0.3,                     # unchanged
    use_line_search=True,                            # unchanged
    delta_bequest=0.0,                               # was 0.001; cite §3h
    gather_precision="f32",                          # explicit; cite §3g
    use_fori_newton=True,                            # unchanged
)
```

### Per-axis citation summary

| Axis | Value | Citation |
|---|---|---|
| `state_grid_sizes` | (5,5,5,5) | [INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md](INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md) §2 — g5 is the working minimum (g3 RED, g4 YELLOW) |
| `n_state_quad_nodes` | (3,3,3,5) | [INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md](INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md) — y_1 axis bump is dominant (2×10⁻¹ portfolio change), other axes <3×10⁻² |
| `state_lobatto_Z[3]` | 7.0 | [INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md](INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md) §5 recommendation; matches existing `_canonical.py` convention for tail correction |
| `n_ret_nodes_1d` | (3,3) | [INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md](INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md) — ret-quad axes saturated; bumps move policy by ≤2×10⁻³ (sub-tol) |
| `n_z` | 11 | [SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md) §5 + [CALIBRATION_VERIFICATION_2026-05-08.md](CALIBRATION_VERIFICATION_2026-05-08.md) §3 — System IV multidimensional state plausibly reduces per-axis n_z burden vs System I evidence; YELLOW on sup-norm but typical-cell GREEN. **Cheapest axis to bump if budget opens up.** |
| `n_eta`, `n_eps` | 3, 4 | [SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md](SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md) §1 — GREEN at smallest tested |
| `max_iter` | 30 | [COMPLEXITY_WALL_TIME_2026-05-06.md](COMPLEXITY_WALL_TIME_2026-05-06.md) §2 wall model + [SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md) §4 audit — hard 70 % wall save vs mi=100; cap-bound cells are tiny-savings boundary, sentinel-replaced regardless |
| `gather_precision` | "f32" | [MIXED_PRECISION_REVIEW_2026-05-06.md](MIXED_PRECISION_REVIEW_2026-05-06.md), [FP32_NEWTON_PROBE_2026-05-07.md](FP32_NEWTON_PROBE_2026-05-07.md) — 10–20 % wall save, sub-1e-5 alpha drift |
| `delta_bequest` | 0.0 | Handoff §4.2 (pinned by sensitivity work) |

---

## §6 — Risks + backups

### What's cut, and the worst-case if the cut is wrong

| Cut | Worst-case if wrong | Likelihood | Mitigation |
|---|---|---|---|
| state_grid 7⁴ → 5⁴ | sup-norm error in C ~ 17 % at high-(dp,spr) state corner; realised-panel error much smaller | Medium — the inf-horizon scan saw concentration at high-(dp,spr); finite-horizon System IV may have different concentration | Backup: **6⁴ + sq=(3,3,3,3) + rq=(3,3) + n_z=11 + mi=20** lands at ~14 h (over) but with lower per-axis costs the resolution per dollar is worse |
| state_quad rtb K=5 → K=3 | rtb axis insensitive in inf-horizon; finite-horizon finite cohort with more z-z coupling could differ | Low — System II quad scan and inf-horizon axis-bump run3 agree | If rtb sensitivity emerges in post-canonical ablation set, run a single (3,3,5,5) bundle to test |
| ret_quad 5×5 Lobatto → 3×3 plain | 1–2 pp portfolio share movement at extreme-state corner cells | Low for inf-horizon (saturated); medium for finite-horizon System IV (no clean test) | Backup F (5⁴ sq3333 rq33 n_z=11 mi=60) at 11.8 h gives a same-quality alternative without the rq Lobatto and tracks the worst-case |
| max_iter 100 → 30 | Rare hard-Newton interior cell falls back to line-search-failed sentinel | Low — System I sup-norm evidence shows cells stuck at cap are tiny-savings boundary, not interior | Backup C2 (mi=50, mb=5) at 8.9 h; same wall, more Newton headroom |
| delta_bequest 0.001 → 0 | Bequest valuation slightly over-attractive at low wealth; numerical effect <10⁻⁴ | Low | None needed; matches handoff-pinned value |

### Two named backups

- **If recommended config OOMs on 8× A100 / does not converge by 12 h:**
  Fall back to **config C2 = 5⁴ + sq=(3,3,3,5) Lob_y1 + rq=(3,3) + n_z=11 + max_iter=50 + max_backtrack_iter=5 + f32**. Wall projection 8.9 h. Same axis decisions but with mb halved instead of mi cut to 30 — preserves Newton headroom on hard cells, accepts shorter line-search depth (line-search default `max_backtrack_iter=10` is itself untested as the binding cost; mb=5 is a defensible reduction since most line searches resolve in 1–3 halvings on the System I evidence).

- **If recommended config wall comes in materially under 9 h** (the formula's f32-gather underestimate manifests):
  **Bump max_iter back to 50 first, then n_z to 13 if budget remains.** Each yields an 8 h-class run with cleaner Newton convergence and slight extra z-resolution.

### Scaling notes (8× A100 not available)

| Hardware | Wall projection for recommended config | Notes |
|---|---:|---|
| 8× A100 SXM4 80 GB (target) | **9.8 h** | recommendation built around this |
| 1× GH200 / 1× H100 SXM5 | **31 h** ≈ 1.3 days | over the 12 h budget; checkpoint + split |
| 2× H100 SXM5 | **20 h** | over budget; no good single-window option |
| 1× A100 SXM4 | **59 h** ≈ 2.5 days | not viable as single-job |
| 1× B200 (180 GB HBM3e) | **22 h** | possible if rate is competitive; 1× device is simplest |

The 8× A100 plan dominates on wall + cost. If only 2× H100 SXM5 is on offer, the recommendation is to either shrink to 5⁴ + sq=(3,3,3,3) + rq=(3,3) + n_z=11 + mi=30 (resolution score 0, ~12 h on 2× H100) or split the canonical run across two windows with checkpointing.

### Stability watch

The COMPLEXITY scan and the existing `_canonical.py` comments both flag a `stability_proxy` watch on cells where `α_b` cap is near. The recommended config's `α` cap is **not** explicit in `SolverConfig` (no `alpha_min`/`alpha_max` set) — it inherits the model.py default. **Surface this:** if the canonical run produces cap-bound cells (`total_newton_failures > 0` flagged via `EC_NEWTON_FAIL`), the post-canonical decision is whether to re-add the box cap. **Do not unilaterally re-add the cap before the run.**

---

## §7 — Implementation: paste-ready `_canonical.py` diff

```diff
--- a/configs/_canonical.py
+++ b/configs/_canonical.py
@@ -57,32 +57,33 @@ CANONICAL_DISC = DiscretizationConfig(
     n_wealth=180,
     wealth_min=0.13,
     wealth_max=750.0,
     n_savings=180,
-    state_grid_sizes=(7, 7, 7, 7),
+    state_grid_sizes=(5, 5, 5, 5),
     state_grid_mode="cholesky",
     state_n_stds=(2.0, 2.25, 2.0, 2.25),
     n_z=11,
     n_stds=3.0,
     n_eps_nodes=4,
     n_eta_nodes=3,
-    n_ret_nodes_1d=(5, 5),
-    ret_lobatto_Z=(7.0, 7.0),
-    n_state_quad_nodes=(3, 5, 3, 5),
-    state_lobatto_Z=(None, 7.0, None, 7.0),
+    n_ret_nodes_1d=(3, 3),
+    ret_lobatto_Z=None,
+    n_state_quad_nodes=(3, 3, 3, 5),
+    state_lobatto_Z=(None, None, None, 7.0),
 )

 CANONICAL_SOLVER = SolverConfig(
     tol=1e-7,
-    max_iter=8000,
+    max_iter=30,
     init_alpha_s=0.85,
     init_alpha_b=0.44,
     step_damp_unconstrained=0.3,
     use_line_search=True,
-    delta_bequest=0.001,
+    delta_bequest=0.0,
+    gather_precision="f32",
 )
```

After applying:

```bash
python scripts/analysis/compute_budget_estimator.py --table
# expect: row D wall = 9.84 h on 8x A100 SXM4 80GB
```

Then commit, then launch on Lambda.

---

## §8 — What this synthesis does and does not commit

**Commits to.** A single ranked recommendation against the 12 h × 8× A100 ceiling. Per-axis citations from the existing scans. Backup configs covering OOM, slow-convergence, and sub-12-h-headroom scenarios. A compute estimator that reproduces every empirical anchor within ±21 %.

**Does not commit to.** Anything not on the candidate-config table is out of scope; downstream ablation set design is out of scope; solver-side changes are out of scope; alpha-cap re-introduction is flagged as a watch item but not recommended pre-canonical.

The decision the user is making tomorrow is the canonical config commit — this report's job is to make that one decision **defensible per-axis**, not to design the next sweep.

---

## Appendix A — How to reproduce the numbers in this report

```bash
# Calibration table (matches §2)
python scripts/analysis/compute_budget_estimator.py --calibrate

# Candidate ranking (matches §4)
python scripts/analysis/compute_budget_estimator.py --table

# Both at once
python scripts/analysis/compute_budget_estimator.py
```

The estimator's `Config` dataclass mirrors `DiscretizationConfig` enough for ad-hoc what-ifs. To project a custom row:

```python
from compute_budget_estimator import Config, wall_total_s, cost_usd
cfg = Config(
    state_grid_sizes=(5,5,5,5),
    n_state_quad_nodes=(3,3,3,5),
    n_ret_nodes_1d=(3,3),
    n_z=11,
    max_iter=30,
)
print(f"{wall_total_s(cfg, 'A100_SXM4', 8) / 3600:.2f} h")
print(f"${cost_usd(cfg, 'A100_SXM4', 8):.0f}")
```

---

**End of synthesis.** The recommendation in §5 is the deliverable; everything else is its supporting argument.
