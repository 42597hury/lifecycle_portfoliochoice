# Handoff: Canonical Config Synthesis — Find the Best Resolution Within 12 h on 8× A100

**Branch:** `jax-rewrite`
**Mode:** **REPORT-PRODUCING with one optional script.** Read all the prior sweep / sensitivity work, fold it into one decision.
**Output:**
- `docs/scans/CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md` — the synthesis report. The user will use this to commit a canonical for the thesis run tomorrow.
- (Optional) `scripts/analysis/compute_budget_estimator.py` — a small parameterized estimator that converts `(config) → (wall on 8× A100 SXM4 80GB)` calibrated against the empirical anchors in §3. Producing this is in scope; not strictly required if the report's tables suffice.

**Time budget:** ~4–6 hours. Pure synthesis — no GPU runs, no new bundles.

**Commit obligation:** when the report is done, **you must `git add` + `git commit` + `git push`** to `origin/jax-rewrite`. If you cannot self-verify because of sandbox constraints, surface that in the commit body and push anyway — the user picks up from there. Do not leave the report uncommitted.

---

## 1. Purpose (read this twice)

The user is committing tomorrow to a **canonical config for the full thesis lifecycle run**. The constraint is hard:

> **It must fit in 12 hours of wall on 8× A100 SXM4 80 GB**, and it must be the highest-resolution config that does so.

You are not asked to run anything. You are asked to **synthesize every empirical and analytical finding produced over the past two days** into a single recommendation. Specifically:

1. Build a **calibrated wall-time formula** that converts a config tuple `(state_grid, n_z, n_state_quad, n_ret_quad, n_eta, n_eps, Newton max_iter, gather_precision)` into a wall projection on 8× A100. Anchor against the empirical timings in §3.
2. **Rank every config the user has considered** by `(economic resolution, wall, $cost)`. Use the sensitivity bundles (§2) to score "economic resolution" — i.e., how close each config is to the asymptote-truth implied by the data we already have.
3. Recommend **one canonical config** that maximizes resolution under the 12 h × 8× A100 constraint. Justify each axis of the config with a citation to a specific bundle / scan / handoff.

The user has spent days on the sensitivity work to enable exactly this decision. Do not duplicate that work; **use it**.

---

## 2. All available data sources

You have an unusual amount of converging evidence. Read what's relevant — don't re-do their analyses.

### 2.1 Saved bundles (with diagnostics + policies)

**Inf-horizon bundles** (1× A100/GH200, 2026-05-08, all at System IV / full state vector):

```
saved_runs/inf_horizon/
├── system_iv_inf_grid_g3_quad3334_ret44_calib1/   # 3⁴ state, (3,3,3,4) state quad, (4,4) ret quad
├── system_iv_inf_grid_g4_quad3334_ret44_calib1/   # 4⁴ state, same quad
├── system_iv_inf_grid_g5_quad3334_ret44_calib1/   # 5⁴ state, same quad — NEAR-CONVERGED (final stop 1.37e-5 vs tol 1e-5)
├── system_iv_inf_axisbump_run1_sq3333_rq33_calib1/   # 5⁴ state, (3,3,3,3) state, (3,3) ret — baseline
├── system_iv_inf_axisbump_run2_sq3335_rq33_calib1/   # +y_1 K-bump
├── system_iv_inf_axisbump_run3_sq5333_rq33_calib1/   # +dp K-bump
├── system_iv_inf_axisbump_run4_sq3533_rq33_calib1/   # +spr K-bump
├── system_iv_inf_axisbump_run5_sq3333_rq35_calib1/   # +xb ret K-bump
└── system_iv_inf_axisbump_run6_sq3333_rq53_calib1/   # +xr ret K-bump
```

**Finite-horizon ablation bundles** (2× H100, 2026-05-07):

```
saved_runs/ablations/
├── system_i_grid7_nz{10,15,30,70}_calib1/                # n_z sweep, System I (rtb-only)
├── system_i_grid7_nz30_eta{3eps4,4eps5,6eps6}_calib1/    # (n_eta, n_eps) sweep
└── system_ii_grid7x7_nz15_{sq3x3_rq3x3, sq4x4_rq3x3, sq3x3_rq4x4, sq3x5_rq3x3}_calib1/   # System II quad sweep
```

Each bundle has `policy_arrays.npz` (C/S/B), `metadata.json` (full config snapshot), `diagnostics.pkl` (convergence history, Newton/backtrack histograms, per-iter sup-norms, `total_newton_failures`, `stability_proxy`).

**⚠ Caveat:** the System II quad-sweep bundles were produced before commit 8bfaec9 — their `newton_iter_histogram` fields are unreliable (the per-cell `jnp.max` collapse bug). Policies are valid; treat the histograms as noise. See [docs/scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md](../scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md).

### 2.2 Sensitivity-analysis reports

Already-written reports — **read these before doing anything else**:

| Report | What it concluded |
|---|---|
| [docs/scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](../scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md) | n_z stops at minimum tested (≤10 adequate at System I) |
| [docs/scans/SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md](../scans/SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md) | (n_eta, n_eps)=(3,4) adequate at System I |
| [docs/scans/SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md](../scans/SYSTEM_II_QUAD_CONVERGENCE_2026-05-07.md) and [`..._2026-05-08.md`](../scans/SYSTEM_II_QUAD_CONVERGENCE_2026-05-08.md) | y_1 K-bump matters; dp/spr K-bumps don't (System II finite-horizon) |
| [docs/scans/INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md](../scans/INF_HORIZON_GRID_QUAD_CONVERGENCE_2026-05-08.md) | (Inf-horizon Sweep A analysis — read for state-grid verdicts) |
| [docs/scans/INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md](../scans/INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md) | (axis-bump analysis — y_1 K-bump validated at System IV inf-horizon) |
| [docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md](../scans/COMPLEXITY_WALL_TIME_2026-05-06.md) | Pre-existing wall-time estimator. Anchor: GH200 single-device 273 s/age at 5⁴ + reduced quad. **Reuse this; don't rewrite.** |
| [docs/scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md](../scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md) | Histogram-counting bug fix; explains why pre-fix bundles' histograms are unreliable |
| [docs/scans/CALIBRATION_VERIFICATION_2026-05-08.md](../scans/CALIBRATION_VERIFICATION_2026-05-08.md) | (Latest calibration check — read) |
| [docs/scans/MULTI_GPU_AUDIT_2026-05-07.md](../scans/MULTI_GPU_AUDIT_2026-05-07.md) | Multi-GPU sharding behavior; pmap chunking. Useful for translating single-A100 walls to 8×. |
| [docs/scans/MIXED_PRECISION_REVIEW_2026-05-06.md](../scans/MIXED_PRECISION_REVIEW_2026-05-06.md) and [docs/scans/FP32_NEWTON_PROBE_2026-05-07.md](../scans/FP32_NEWTON_PROBE_2026-05-07.md) | f32 gather: ~10–20% wall savings, sub-1e-5 alpha drift |
| [docs/scans/HLO_FUSION_AUDIT_2026-05-07.md](../scans/HLO_FUSION_AUDIT_2026-05-07.md) | Kernel fusion structure; informs the wall-time formula |

Plus structured metric JSONs at `docs/scans/system_i_*_metrics.json`, `system_ii_quad_*_metrics.json`, `inf_horizon_*_metrics.json` — direct input for the wall-time formula.

### 2.3 Sweep runner scripts (compute templates)

```
verify/benchmark_bundle.py                    # canonical single-bundle runner
verify/benchmark_inf_horizon.py               # single-cell inf-horizon
verify/benchmark_system_i_nz_sweep.py         # n_z sweep
verify/benchmark_system_i_eta_eps_sweep.py    # (n_eta, n_eps) sweep
verify/benchmark_system_ii_quad_sweep.py      # System II quad sweep
verify/inf_horizon_sweep_state_grid.py        # inf-horizon Sweep A (yesterday)
verify/inf_horizon_sweep_axis_bumps.py        # axis-bump sweep (yesterday)
verify/lambda_watchdog/watchdog.sh            # auto-terminate watchdog (cost-control reference)
```

These show how each cell's wall is structured (precompute, kernel call, save). Useful for parameterizing the estimator.

### 2.4 Analysis scripts (resolution-comparison templates)

```
scripts/analysis/system_i_nz_convergence.py       # n_z divergence + verdict
scripts/analysis/system_i_eta_eps_convergence.py  # eta-eps divergence + verdict
scripts/analysis/system_ii_quad_convergence.py    # quad-sweep divergence
scripts/analysis/inf_horizon_grid_quad_convergence.py
scripts/analysis/inf_horizon_resolution_investigation.py
scripts/analysis/state_grid_axis_sensitivity.py
scripts/analysis/quad_axis_sensitivity.py
```

### 2.5 Prior synthesis handoffs (know what's been planned)

```
docs/handoff/HANDOFF_COMPLEXITY_ANALYSIS.md
docs/handoff/HANDOFF_COMPLEXITY_WALL_TIME_ESTIMATOR.md       # template for the wall formula
docs/handoff/HANDOFF_INF_HORIZON_SENSITIVITY_PROGRAM.md      # designed yesterday's program
docs/handoff/HANDOFF_PERSISTENT_INCOME_DISCRETIZATION_TRADEOFF.md
docs/handoff/HANDOFF_NZ_ONE_FOR_INF_HORIZON.md
```

### 2.6 Canonical reference

[configs/_canonical.py](../../configs/_canonical.py) is the current canonical config (production target before this synthesis). Read the comments — they explain the "why" behind every existing knob.

```
state_grid_sizes=(7, 7, 7, 7),
state_n_stds=(2.0, 2.25, 2.0, 2.25),
n_z=11,
n_eps_nodes=4, n_eta_nodes=3,
n_ret_nodes_1d=(5, 5),  ret_lobatto_Z=(7.0, 7.0),
n_state_quad_nodes=(3, 5, 3, 5),  state_lobatto_Z=(None, 7.0, None, 7.0),
n_wealth=180, n_savings=180,
gather_precision="f32",
delta_bequest=0.0,
```

The synthesis question: **which axes can be cheaper than canonical without losing economic resolution, and which are at-or-below where we should be?**

---

## 3. Empirical anchors for the wall-time formula

You have direct timings to anchor the formula. Use these — don't infer from FLOPs alone.

| Anchor | Hardware | Wall | Source |
|---|---|---|---|
| 5⁴ retirement-only at reduced quad | GH200 single | **273 s/age** (warm JIT) | docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md |
| System II finite-horizon, 7×7 state, n_z=15, sq3x3, rq3x3, eta3eps4, 78 ages | 2× H100 SXM5 | **9.9 min** (full lifecycle) | yesterday's quad sweep |
| Inf-horizon g3 (3⁴, n_z=1, sq3334, rq44, tol=1e-5, 82 outer iters) | 1× A100 SXM4 | **16.3 min** | yesterday |
| Inf-horizon g4 (4⁴, otherwise as g3) | 1× A100 SXM4 | ~52 min (extrapolated; check bundle metadata) | yesterday |
| Inf-horizon g5 (5⁴, otherwise as g3) | 1× A100 SXM4 | ~127 min (extrapolated; not fully converged at iter 100) | yesterday |
| Inf-horizon axis-bump cell 1 (5⁴, sq3333, rq33, tol=1e-4, 62 outer iters) | 1× GH200 | **17.3 min** | yesterday |
| Inf-horizon axis-bump cells 2–6 (1.67× quad work each) | 1× GH200 | ~28 min each (extrapolated) | yesterday |

**Each bundle's `metadata.json` contains exact `wall_time_seconds`** — extract them rather than re-extrapolate.

**8× A100 SXM4 80 GB scaling.** Empirically (from MULTI_GPU_AUDIT and the canonical 5⁴ benchmark), inf-horizon scales ~near-linear in device count for cell-axis sharding when per-device cells ≥ 50. Your formula needs:

- Single-device → multi-device scaling factor (with per-device-cells dependent overhead)
- **Throughput ratios:** 1× A100 SXM4 ≈ 312 TFLOPS BF16; 1× H100 SXM5 ≈ 989; 1× GH200 ≈ 989 with 4 TB/s HBM3. So 1× H100 ≈ 3.2× 1× A100.
- **2× H100 vs 8× A100:** raw compute 8 × 312 = 2496 TFLOPS A100 vs 2 × 989 = 1978 TFLOPS H100 → 8× A100 has ~1.26× more aggregate compute. Memory: 8 × 80 = 640 GB vs 2 × 80 = 160 GB — 4× more.

This is **rough**. The user will accept ±30% accuracy on wall projections; aim for that.

---

## 4. Synthesis tasks

### 4.1 Build the wall-time formula

Parameterize:

```
wall_per_age(config, hw) = (
    cells_per_device(config, hw)
    × Newton_iters_avg(config)
    × cycles_per_FOC_eval(config, hw)
    × overhead_factor(config, hw)
)
```

Where:
- `cells_per_device` = `(n_z × N_state) / n_devices`
- `Newton_iters_avg` ≈ from the post-fix histograms (yesterday's bundles): p50 ≈ 1, mean ≈ 17 for the bumped-floor configs. Document where you got this from.
- `cycles_per_FOC_eval` ∝ `n_state_quad × n_ret_quad × (1 if retirement else n_eta × n_eps) × n_w`, scaled by hardware FLOPS
- `overhead_factor` = compile + scheduling. Calibrate from the difference between your formula's prediction and the empirical anchor.

For finite-horizon (lifecycle): multiply by 78 ages + the boundary/terminal kernels. For inf-horizon: multiply by `n_outer_iters` (~30–80 typical).

**Calibration target:** your formula must reproduce each anchor in §3 within ±30%. Run all anchors through the formula and report the calibration table.

### 4.2 Rank candidate configs

Build a table covering the user's likely candidates. Suggested sweep:

| state_grid | state_quad | ret_quad | n_z | Newton mi | wall on 8× A100 | resolution score | $cost |
|---|---|---|---|---|---|---|---|
| 5⁴ | (3,3,3,3) | (3,3) | 11 | 100 | ? | ? | ? |
| 5⁴ | (3,3,3,5) | (3,3) | 11 | 100 | ? | ? | ? |
| 5⁴ | (3,3,3,5)+Lob | (3,3) | 11 | 100 | ? | ? | ? |
| 5⁴ | (3,3,3,5) | (4,4) | 17 | 100 | ? | ? | ? |
| 6⁴ | (3,3,3,5) | (3,3) | 11 | 100 | ? | ? | ? |
| 7⁴ | (3,3,3,5)+Lob | (5,5)+Lob | 11 | 100 | ? | ? | ? |  ← canonical from `_canonical.py`
| (the user's hypothesis: 5⁴, (3,3,3,3), (4,4), n_z=17, mi=20, f32) | | | | | ? | ? | ? |

Add cells as relevance dictates. **n_eta=3, n_eps=4, gather=f32, delta_bequest=0** stay fixed across the table (already pinned by sensitivity work).

**"Resolution score":** compose from the sensitivity bundles. Suggested rubric:
- y_1 K-bump present → +X (axis-bump cell 2 vs cell 1: alpha_b max 8.82 vs 10.63 → 17% improvement)
- State grid ≥ 5⁴ → +Y (Sweep A: g4 vs g5 inferred convergence)
- n_z ≥ 11 → already saturated per System I sweep
- ret_quad bumps → 0 (axis-bump cells 5+6 showed no material change)
- Newton max_iter < 100 → −Z (fail-rate goes 8% → ~20%)

Document the rubric in the report; it becomes the methodology defense.

### 4.3 Pick the canonical

Apply the constraint: `wall ≤ 12 h on 8× A100 SXM4 80 GB`.

Among configs that fit:
- Highest resolution score wins
- Tie-break: prefer cheaper (lower wall, more budget remaining for retries / reruns)
- Tie-break 2: prefer configs that match `_canonical.py` more closely (institutional consistency)

Recommend ONE config. Format as a paste-ready `_canonical.py` diff or a `_replace(...)` snippet.

**Cite each axis** to a specific bundle / scan / handoff. Example:

> `n_state_quad_nodes=(3,3,3,5)` — y_1 K-bump only. **Citation:** axis-bump sweep run 2 (sq3335) reduced alpha_b max from 10.63 (run 1, baseline) to 8.82 — 17% reduction concentrated on the bond-tail policy. Other axis bumps (runs 3, 4, 5, 6) all changed alpha_b ≤ 0.6%. Bundle: `saved_runs/inf_horizon/system_iv_inf_axisbump_run2_sq3335_rq33_calib1/diagnostics.pkl`.

### 4.4 Surface risks + alternatives

The recommendation should include:
- **What you cut, and what's the worst-case if that cut is wrong.** E.g., "dropped Lobatto on state quad — saves ~10% wall — risk: bond-tail accuracy at γ=5 if the unbumped tail behavior matters more than axis-bump cell 2 implies."
- **Two backup configs:** "if the recommended config OOMs on 8× A100, fall back to X. If it doesn't converge in 12 h, fall back to Y."
- **Scaling notes:** if 8× A100 isn't available, what's the equivalent on 2× H100 / GH200?

---

## 5. The deliverable structure

Mandatory sections in the synthesis report:

1. **TL;DR** — one paragraph: recommended config + 12 h projection + which knob(s) the user should sanity-check before committing.
2. **§1 Methodology** — what data you used, what calibration anchors, what the resolution score weights are.
3. **§2 Wall-time formula** — the parameterized formula + the calibration table showing each anchor reproduced within ±30%.
4. **§3 Resolution rubric** — how each axis's "value" is scored. Cite bundles.
5. **§4 Candidate configs table** — the ranking table.
6. **§5 Recommendation** — the one canonical config, with per-axis citations.
7. **§6 Risks + backups** — what's cut, what's the worst case, fallback configs.
8. **§7 Implementation** — paste-ready `_canonical.py` diff.

---

## 6. Pause points

- **An anchor reproduces in your formula at >50% off.** The formula is wrong; pause and investigate before publishing the candidate-config table. Most likely culprits: wrong cells_per_device, missed `n_savings` factor, missed `n_eta × n_eps` factor on working ages.
- **The recommended config doesn't fit in 12 h** even at the cheapest realistic resolution. Surface as a finding; don't hand-tune to make it fit. Either the user accepts a longer wall or accepts lower resolution.
- **Stability proxy > 1 in any candidate's expected behavior.** Flag the bond-leverage interaction; the canonical may need an alpha box-cap (currently scratched). Don't unilaterally re-add the cap; surface it.

---

## 7. Out of scope

- **Running new bundles.** Pure synthesis. Use what's saved.
- **Solver-side changes.** This is a config recommendation, not a code change.
- **Downstream implications** (post-canonical ablation set design). One canonical at a time.
- **Hardware procurement.** Assume 8× A100 SXM4 80 GB available for 12 h; don't second-guess.
- **Re-running existing sensitivity reports.** Read them, cite them, build on them.

---

## 8. Why this matters

The user has spent two days running ~20 ablation cells across two systems and inf-horizon to enable exactly this decision. They committed an unbounded compute budget last night to surface the y_1 K-bump finding. They now need a single recommendation with **defensible per-axis citations** so the canonical can be committed in `_canonical.py` and the full thesis run launched without second-guessing.

Your synthesis is the deliverable that makes those two days pay off. **Do not duplicate the analyses.** **Do cite them everywhere.** **Do recommend ONE config.** **Do commit + push.**
