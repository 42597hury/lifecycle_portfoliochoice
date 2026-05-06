# Handoff: Euler-error diagnostic workflow for solver runs

## When to use this handoff

After any new policy bundle is solved (locally or on AWS), you need to grade it against the publication-grade Euler-error tests and compare it to prior bundles. This handoff describes the diagnostic battery, how to run it, and how to interpret and compare the results so the team can decide whether the bundle is publication-ready or what needs to change next.

This is the workflow used through the entire investigation that produced the current locked-in fixes (log1p grids, Path B, raised `wealth_min`, leverage cap). Apply the same battery to any new bundle.

---

## 1. Background — fixes already in the codebase

Before investigating any new bundle, know what's already been done:

- **log1p-stretched savings + wealth grids**: avoids wasted nodes near zero, denser at moderate wealth where curvature is real. Implemented in `precompute.py`.
- **Path B (bequest = 0 at bankruptcy)**: in FOC kernels (`solver.py`), EE diagnostic kernels (`scripts/diagnostics/_diag_euler_errors.py`), and the simulator (`simulation.py`). When `s · R_p ≤ 0`, the bequest term contributes zero and the alive branch uses `x_next = income_or_pension`. Eliminates the `(1e-10)^(-γ)` integrand explosion.
- **`wealth_min` raised** to a level above the EGM anchor segment, so `wealth_grid[0]` no longer interpolates across the degenerate anchor.
- **Numerical leverage cap** in the unconstrained Newton: `α_s, α_b ∈ [SolverConfig.alpha_min, SolverConfig.alpha_max]`. Numerical guardrail; configurable per run.

These are the *baseline* behavior of any new solve. If a new bundle's diagnostic doesn't reflect these (e.g., catastrophic invalidity from bequest pathology), suspect the bundle was produced from stale code.

---

## 2. The diagnostic battery

For every bundle, run **three** diagnostics in this order. They take minutes, not hours.

### Diagnostic A — Gridpoint EE under same quadrature

This is the cheap stress test. Probes every (age, iz, state-corner, wealth-index) tuple from a structured cube. Conservative because it includes extreme state corners that simulated agents may rarely visit.

```bash
python -m scripts.diagnostics._diag_gridpoint_ee \
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \
  <bundle-path> \
  --eval-mode same \
  --markdown-out diagnostics_reports/diagnostics_gridpoint_ee_<label>_same.md
```

Reports:
- Mean / P95 / max / min `log10|EE|` overall and by phase
- Worst grid points (age, iz, state, iw, x, c, α_s, α_b, EE)
- Validity counts

**Read for:**
- Mean: indicator of overall integrand consistency. Should be ≤ −5 to −6 on a healthy bundle.
- Max: dominated by extreme grid corners. Useful for spotting anchor pathologies.
- Validity: should be 100%. Anything less is a hard failure.
- Worst-points table: tells you the *pattern* of pathology — extreme leverage? low wealth? specific state corner?

#### Bimodal wealth decomposition (Lobatto bundles)

When `--eval-mode same` reports max log10|EE| > −3 on a Lobatto bundle but the mean is at solver tolerance, the failure population is bimodal. Distinguish the two modes with a dense low-wealth scan:

```bash
python -m scripts.diagnostics._diag_gridpoint_ee \
  <bundle> --model-bundle <bundle> --eval-mode same \
  --wealth-indices 0 1 2 3 4 5 10 20 50 100 149
```

Then read the failure share (cells with `log10|EE| > −4`) per wealth index. Two characteristic patterns:

- **Mode (1) — EGM anchor**: high failure share at iw ∈ {0, 1, 2} (often 20-40%) that drops sharply by the iw where wealth crosses the EGM constrained-region break. Diagnostic signature: failure share is a step function of x. Fix: raise `wealth_min` past the anchor break — to whatever x corresponds to the iw where the share first stabilises (typically x ≈ 0.10-0.20).

- **Mode (2) — leverage-corner kink**: ~2-3% failure share that *persists across all wealth levels* until x ~ 5-15, where it finally collapses to solver tolerance. Top cells consistently have `|α_b| ∈ [1, 3]` at extreme state corners. Mechanism: at the Lobatto ±Zσ tail node, `s · R_p ≈ 0` puts `x_next` near zero where the log1p wealth grid has its highest curvature; `C_{t+1}(x_next)` linear interp can't represent the bankruptcy kink even at gridpoint EE evaluation (the eval probes (z, state, x) at gridpoints but `x_next` is always off-grid). Fix is structural: smaller Z, smoother Path B clamp, or tighter `alpha_min/max` to keep the agent away from `s·R_p ≈ 0` at the tail node — note that *more* wealth nodes near zero usually doesn't help if log1p density is already saturated.

This decomposition matters because the two modes need different remediations. Reporting only "max log10|EE| at gridpoint" without separating them gives a single number that no single config change can move. Reporting per-iw failure share makes the structural fix legible.

For the v4_lobatto bundle (canonical Lobatto Z=7, `wealth_min=0.05`), the per-iw scan showed mode (1) at 41% (iw=0) → 22% (iw=1) → 5% (iw=2) and mode (2) at ~2.7% across iw=10..30. Raising `wealth_min` to 0.13 is predicted to kill mode (1); the mode (2) ~3% needs a separate Z or clamp intervention.

### Diagnostic B — Gridpoint EE under finer quadrature

Same probe set, **stricter grading rule**. This is the "honest" version of the same-Q test — it grades the policy against a quadrature rule strictly richer than the solver's own.

```bash
python -m scripts.diagnostics._diag_gridpoint_ee \
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \
  <bundle-path> \
  --eval-mode next_finer \
  --markdown-out diagnostics_reports/diagnostics_gridpoint_ee_<label>_nextfiner.md
```

The `next_finer` rule scales relative to the bundle's own quadrature: `(K_state, K_ret, n_eta, n_eps)` each get bumped (e.g., `(2,2,5) → (3,3,6)`, `(3,7,5) → (5,9,7)`, etc.). The eval rule is reported in the markdown header.

**Read for:**
- Mean / max gap vs same-Q: large gap means the policy's quadrature truncation is *active*. Small gap means the policy is honest.
- Invalidity rate: the *most* informative statistic from this diagnostic. > 1% means the policy has positions that produce ill-defined moments under richer sampling — typically extreme-leverage tails.

#### Hermite-Lobatto bundles: eval rule propagation (2026-05-04)

For bundles solved with the prescribed-tail (Hermite-Lobatto) rule on any return / state axis (`ret_lobatto_Z` or `state_lobatto_Z` set in metadata), the eval-rule construction now propagates that Lobatto config per-axis with a K-aware adjustment for the eval-mode K bump:

- `same`: identical to solver, including the same Z and K.
- `next_finer` / `double`: Lobatto axes keep their Z; K is clamped to the smallest valid Lobatto K (in `{3, 5, 7}`) at or above the bumped target. K=8/9/10 → 7 (the closed-form ceiling); K=4/6 → 5/7 respectively. Non-Lobatto axes follow the standard GH bump (`+2` for return, `+1` for state).
- If the solver Z falls outside every K's validity window (extremely rare), the axis falls back to pure Gauss-Hermite at the bumped K and a warning is printed once.

**Why this matters.** Pre-2026-05-04 the eval rule silently fell back to pure Gauss-Hermite even on Lobatto bundles, so the EE residual conflated genuine policy inaccuracy with the disagreement between Lobatto (with explicit ±Zσ tail nodes) and GH (without) about what's "optimal" at the boundary. Numbers reported by `_diag_euler_errors`, `_diag_gridpoint_ee`, `_diag_invalid_cells`, and `_diag_simpath_worst_cells` change for any Lobatto bundle from this date forward; bundles with no Lobatto config are unaffected (bit-identical to before).

**Opt-out.** Pass `--eval-disable-lobatto` to recover the pre-fix GH-only eval rule. The rule-mismatch term that produces is itself diagnostic: if `propagated max log10|EE|` is much worse than `GH-only max log10|EE|`, the policy is failing the FOC at the explicit tail node the solver claimed to have probed — that's policy inaccuracy, not eval-rule mismatch.

**Worked example (v4_lobatto, sim-path, 5000 households, eval cap 256/age, K=5 Lobatto Z=7 on axes 1,2):**

| run | mean log10\|EE\| | median | max |
|---|---:|---:|---:|
| pre-fix (GH eval, Lobatto solver) | −2.62 | −2.53 | −0.81 |
| `--eval-disable-lobatto` (GH-only) | −2.57 | −2.49 | −0.59 |
| default (Lobatto propagated) | −2.57 | −2.49 | **−0.08** |

The propagated-Lobatto run reveals the residual is concentrated at the ±7σ tail node — the policy's α_b at high-y_1 corners doesn't satisfy the FOC when the eval rule actually hits the same tail the solver was designed against. This is real policy inaccuracy at extreme cells, not a rule-mismatch artefact.

### Diagnostic C — Simulation-path EE under finer quadrature

**This is the test referees grade against.** Simulates a panel of agents through the policy and evaluates the FOC at the simulated states using `next_finer` quadrature.

```bash
python -m scripts.diagnostics._diag_euler_errors \
  <bundle-path> \
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \
  --eval-mode next_finer \
  --n-simulations 5000 \
  --eval-households-per-age 256 \
  --initial-x <reasonable-initial-wealth> \
  --initial-z stationary \
  --initial-state median \
  --partial-init-mode centered \
  --markdown-out diagnostics_reports/diagnostics_simpath_ee_<label>_nextfiner.md
```

**Initial conditions matter** — they determine which states the simulation visits:
- `--initial-x`: pick a value near typical SCF wealth at the youngest solved age. For age-65 starts, ~5–10. For age-50 starts, ~2–5. For full-lifecycle starts at age 22, use defaults.
- `--initial-z`: `stationary` covers the unconditional z distribution; `median` is a tight cross-section.
- `--initial-state`: `median` for centered tests; `stationary` to spread.
- `--partial-init-mode`: `centered` for partial bundles; `warm_start` to load from a full-lifecycle bundle for the initial cross-section.

Reports per-age and phase-aggregated statistics, plus pass/fail flags against the publication gates encoded in `scripts/diagnostics/_diag_euler_errors.py:1073–1078`.

### Publication gates (from `contextfiles/GRID_CONVERGENCE_CRITERIA.md`)

| phase | grade | mean log10\|EE\| | max log10\|EE\| |
|---|---|---:|---:|
| working | publication | < −4.0 | < −3.0 |
| working | welfare | < −5.0 | < −4.0 |
| retirement | publication | < −4.5 | < −3.0 |
| retirement | welfare | < −5.5 | < −4.0 |

---

## 3. The investigative process — read the diagnostics together

The three diagnostics are complementary. Don't read any one in isolation.

### Step 1 — Sanity check the bundle

Before running diagnostics, inspect the bundle's `metadata.json`:

```bash
python -c "
import json
m = json.load(open('<bundle-path>/metadata.json'))
ds = m['diagnostics_summary']
print('status:', ds['solve_status'])
print('ages:', ds['youngest_solved_age'], '-', ds['oldest_solved_age'])
print('newton_failures:', ds['total_newton_failures'])
print('mono_violations:', ds['total_mono_violations'])
print('worst_foc_resid:', ds['worst_foc_resid'])
print('disc_config:', ds['disc_config'])
"
```

**Bundle is unhealthy if** `total_newton_failures > 0`, `total_mono_violations > 0`, or `worst_foc_resid` is far from `tol`. Don't run diagnostics on an unhealthy bundle — fix the solver first.

### Step 2 — Read Diagnostic A (gridpoint same-Q)

This tells you whether the solver self-consistency is intact. Healthy bundle: mean < −5, validity 100%. **Anomalous patterns:**

- Max much worse than P95: a single pathological probe (often the iw=0 anchor or a state-corner outlier). Inspect the worst-points table.
- Validity < 100%: hard failure; bundle is broken.
- Phase imbalance (e.g., working ages much worse than retirement): could be income-quadrature truncation.

### Step 3 — Read Diagnostic B (gridpoint next_finer)

The **gap** between same-Q and next_finer is the headline finding here.

- Small gap (≤ 0.5 orders): policy is reasonably honest under refined Q.
- Large gap (1+ orders): policy is exploiting quadrature truncation. Refine quadrature.
- High invalidity (> 5%): policy has structural-tail pathology (extreme leverage). Tighten leverage cap or refine quadrature.

### Step 4 — Read Diagnostic C (sim-path next_finer)

This is the publication test. **Compare against the gates table above.**

- All four gates pass: bundle is publication-ready.
- Some pass, some fail: identify which phase and metric (mean vs max). Use the by-age table to spot which ages drive failure.
- All fail: large gap to gate; needs structural fix (quadrature, grid, or model).

The sim-path test is **less conservative** than the gridpoint test in some ways (it only probes states the simulation actually visits, weighted by equilibrium probability) but **more conservative** in others (it uses off-grid `x_next` values where interp truncation manifests). They're complementary; trust both.

### Step 5 — Identify which fix is needed

| symptom | likely cause | fix |
|---|---|---|
| Diagnostic A max binds at iw=0, x ≈ wealth_min | EGM anchor pathology | raise `wealth_min` |
| Diagnostic A binds at extreme state corners with normal α | grid coarseness or quadrature truncation in body | quadrature refinement |
| Diagnostic B large invalidity, worst probes have extreme α | leverage tail under coarse Q | tighten `alpha_min/max`; refine state quadrature |
| Diagnostic C max far from gate, sim-path traces leveraged tails | same as above, in-distribution | leverage cap + quadrature |
| Diagnostic C mean far from gate, but body of distribution OK | quadrature truncation in body | refine `n_state_quad_nodes`, `n_eta`, `n_eps` |
| Diagnostic A clean but C reports invalidity | sim-path reaches off-grid x_next where FOC-kernel lookup is biased | FOC-kernel-side wealth interp refinement |

---

## 4. Drill-down: investigating problematic states

When Diagnostic B (gridpoint next_finer) reports nonzero invalidity, or Diagnostic C (sim-path next_finer) reports a worst-tail max log10|EE| above the gate, the next question is **which states are causing it and why**. The headline numbers don't tell you whether the residual is concentrated at a few wide corners or spread across the lifecycle, whether it's a leverage-direction issue, or whether it's a quadrature-axis issue.

This section describes the workflow that turns "the bundle has a 14% relative error somewhere" into "the bundle has 167 cells at i₂=6 corners with α_b ≈ +2 because state[2] Lobatto Z=7 only covers 81% of the long-bond direction."

### Step 1 — Decide which set of cells to investigate

| diagnostic that flagged the issue | drill-down tool | what it produces |
|---|---|---|
| Diag B (gridpoint next_finer) reports invalidity | `_diag_invalid_cells.py` | Per-cell breakdown by age × iz × state-corner × iw, alpha distribution, cap-binding rate, Sharpe at worst corners, stationary-mass sanity check |
| Diag C (sim-path next_finer) reports max-tail problem | `_diag_simpath_worst_cells.py` | Top-N worst sim-path cells with full (age, iz, state-coords, wealth, alpha, savings, |EE|) tuples, plus pattern stats over the top-1% bad set |

Run the appropriate tool against the same bundle:

```bash
# For gridpoint-flagged invalidity
python -m scripts.diagnostics._diag_invalid_cells \
  --bundle <bundle-path> \
  --markdown-out diagnostics_reports/diagnostics_invalid_cells_<label>.md

# For sim-path-flagged tail problems
python -m scripts.diagnostics._diag_simpath_worst_cells
# (script is parameterised internally; edit `bundle_path` near the top)
```

Both tools support `--eval-disable-lobatto` to compare propagated-Lobatto vs GH-only eval (see §2 worked example).

### Step 2 — Read the worst-cell list as a population, not as point estimates

The drill-down tools print the top 15-20 worst cells. **Don't focus on the single worst cell.** Look at the top set as a sample and identify what they have in common:

- **State corner**: are the bad cells clustered at specific (i_0, i_1, i_2) values? Top buckets by count are reported. A typical pattern: "all 167 worst cells have i_2 ∈ {3, 4, 6}" — the issue is on state axis 2.
- **Alpha direction**: average and range of α_b. Two-sided pathology (mix of long and short) suggests a body-quadrature issue. One-sided pathology (consistently leveraged in the same direction) points to a tail-quadrature issue on a specific axis.
- **Age**: range and mean. Mid-late retirement (age >85) means the bequest motive interacts; full-window means the issue is more structural.
- **Wealth and savings ratio**: low wealth + high savings ratio (~85%) is a typical retired-cell signature. Concentration near `wealth_min` would point to anchor pathology.
- **Cap-binding / KKT-constrained**: cells with `is_constrained` are not real EE failures (KKT slack is honest). Filter them out and re-read the pattern.

### Step 3 — Cross-reference with quadrature-axis tools

Once you have the bad-cell signature (e.g., "i_2=6 corner, long bond, α_b ≈ +2"), use the per-axis quadrature tools to map symptom to cause:

| tool | what it tells you |
|---|---|
| `_diag_arbitrage_quadsweep.py` | Per-state min R_p at the per-state Merton alpha under the solver cloud and a finer eval cloud. "Worst min R_p (Merton)" should be comfortably negative on the solver cloud if Lobatto is doing its job. The sweep also tries dropping K or Z on each axis to see what changes. |
| `_diag_per_axis_tail.py` | For each state-quadrature axis individually, what fraction of currently-pathological cells get flipped from solver-safe to finer-bankrupt by adding a tail node at ±L on that axis. Use this to identify which axis is doing the work. |
| `_diag_tail_node_position.py` | Per-cell required tail-node position (in standardised σ units) for each cell, on the bond-residual axis. |
| `_diag_state_tail_node.py` | Same idea but for state-axis tail nodes — useful because state innovations have M-matrix amplification on bond returns. |
| `_diag_gridpoint_ee.py --eval-mode same --wealth-indices 0 1 2 3 4 5 10 20 50 100 149` | **Dense low-wealth gridpoint scan.** Decomposes a bad gridpoint-EE max into mode (1) EGM-anchor pathology (concentrated at iw ∈ {0,1,2}, fixable by raising `wealth_min`) vs mode (2) leverage-corner kink (persistent ~3% across iw, structural). See §2 *Bimodal wealth decomposition*. |

In our v4_lobatto investigation, the per-axis tool showed state[2] Lobatto Z=7 covers 95% of short-bond pathology but only 81% of long-bond pathology — a 14-percentage-point coverage gap that exactly mapped to the 167-cell long-bond cluster the worst-cells diagnostic flagged.

### Step 4 — Form a hypothesis about the mechanism

The bad-cell signature plus per-axis sensitivity should make the mechanism legible. Examples from past investigations:

| signature | mechanism | reference |
|---|---|---|
| Short bonds at low-spr corners (i_1=0), α_b ≈ −5, wide state support | Bond Sharpe extremely negative at low-spr → Merton wants huge short bond → solver cloud's K=5 GH bond axis doesn't see the bond-rally tail → discrete free lunch | v3, fixed by Lobatto Z=7 |
| Long bonds at high-y_1 corners (i_2=6), α_b ≈ +2, mid-late retirement | M[xb,2] = −8.72 makes high state z_2 → low μ_b → bonds still attractive at corner → but explicit ±7σ tail node creates near-discontinuity that Newton can't navigate cleanly → solver delivers warm-restart fallback policy with FOC residual ~0.5 | v4_lobatto, current |
| Cells at iw=0 with normal α | EGM anchor pathology (interpolation across the constrained-region anchor segment) | pre-`wealth_min` raise |
| Mid-distribution cells with mean log10|EE| > −3 | Body integration too coarse (n_eps, n_eta, or state-quadrature K) | recurring; usually n_eps in our case |
| Persistent ~3% gridpoint-EE failure across wealth, concentrated at \|α_b\| ∈ [1, 3] | Wealth-grid `C_{t+1}(x_next)` interp across the post-bankruptcy region: at the Lobatto ±Zσ tail node `s·R_p ≈ 0` → `x_next` lands on the steep low-x portion of the log1p wealth grid where the policy has high curvature; linear interp can't represent the kink even at gridpoint probes (the EE evaluator probes (z,state,x) at gridpoints but x_next is always off-grid) | new (2026-05-04) |

#### Why state-axis Lobatto matters: the two channels for v^s

A common mistake when reasoning about whether to keep state-axis Lobatto is to argue that state innovations only enter the integrand via `c_{t+1}(s_{t+1})` interpolation, and that state-grid clamping at `±state_n_stds` (≈ ±2.25σ canonically) makes any tail node beyond the grid carry the same `c_{t+1}` value as the grid edge — so Lobatto Z=7 on state is "dead weight." This is half right.

State innovations enter the integrand via **two** channels:

1. **Indirect (state-grid interp)**: `v^s` shifts `s_{t+1} = Φ_0 + Φ_11 · s_t + v^s`, used to interpolate `C_{t+1}(s_{t+1})`. For `v^s` outside the state grid the interpolation clamps to the grid edge. The state-grid-clamp argument applies **only here** — and it does kill the marginal value of state-Lobatto on this channel.

2. **Direct (M-coupling on R_p)**: `v^s` shifts the period-`t` return mean via `μ_r = const_r + A_r · s_t + M · v^s` (see [docs/DESIGN.md:424,434](../DESIGN.md)). With `M[xb, y_1] = −8.72` (bond duration), a `v^s_{y_1} = +Z` realisation shifts the bond return mean by ~`−8.72 · Z`. At Z=7 that's a 61-unit μ_b shift — a catastrophic bond crash that pure GH at the same K never approaches (GH K=5 max ≈ 2.86σ → 25-unit shift). **This is the channel that makes state-axis Lobatto bond-bankruptcy-relevant** and the reason state-axis tail nodes interact with leveraged α_b positions.

Channel 2 is **not** affected by the state grid because `M · v^s` is a smooth linear shift, not an interpolation. There is no "state grid edge" to clamp against. So when assessing whether to drop state Lobatto, the question is whether the M-coupled R_p tail at extreme `v^s` is doing work, not whether the c_{t+1} interp at extreme `s_{t+1}` is.

For the canonical calibration (`M[xb, y_1] = −8.72`, `M[xb, spr] = −8.51`), channel 2 is large enough that state-axis Lobatto on axes 1 and 2 is mechanism-essential for any bundle where the agent takes leveraged α_b positions. The right A/B test for state-axis Lobatto sufficiency at a given Z is "state Lobatto Z=Z₁ vs state Lobatto Z=Z₂" (does the value of Z matter?), not "state Lobatto vs state GH" (does the channel exist?) — the existence question is settled by the M loadings.

### Step 5 — Map mechanism to remediation

| mechanism | first-line fix | escalation path |
|---|---|---|
| Tail-quadrature gap on a specific axis | Add Lobatto Z on that axis with an empirically-chosen Z (cover ~95% of pathological cells per the tail-position diagnostic) | Larger Z if coverage incomplete; or Lobatto on a sister axis (state[1] is often the dominant bond-channel by per-σ effect, even when M[xb,2] looks more dominant by raw loading) |
| Newton fails at the explicit tail discontinuity | Smaller Z (move the tail node inward), smoother bankruptcy clamp (replace hard `if sR_p ≤ 0` with a soft penalty), or tighter alpha cap (limits leverage before reaching the discontinuity) | Larger code change: smooth-bankruptcy FOC kernel |
| Body integration too coarse | Bump K on the relevant axis (n_eps, n_eta, or state-quad K) | Verify polynomial exactness 2K-1 covers the integrand's dominant Taylor degree at production γ |
| Anchor pathology | Raise `wealth_min` past the EGM constrained-region anchor | Verify with `_diag_bundle_state_clipping` that no agent-years sit at the anchor in simulation |
| KKT-bound cells dominating reported max | Filter to `cell_set=unconstrained` (already in gate table); the EE there is real | Tighten alpha cap if cells are at the leverage cap; the KKT residual is honest |

### Step 6 — Make the recommendation actionable

Wrap up the drill-down with three things in the comparison report (§9):
1. **Bad-cell signature**: one-line summary like "167 cells at i_2=6 corners with α_b ≈ +2, mean wealth 7, age 71-98."
2. **Mechanism**: one-line attribution like "long-bond direction tail-quadrature gap on state axis 2."
3. **Recommended next test**: one specific config delta with predicted outcome, e.g., "Z=5 instead of Z=7 on state[2] — moves the discontinuity inward; expected to reduce max sim-EE from −0.08 to ~−1.5 by smoothing Newton convergence at the cost of slightly worse short-bond coverage."

If steps 4 and 5 don't converge to a single recommendation (e.g., two competing mechanisms could explain the pattern), say so and design the smallest possible A/B test to discriminate. The cheapest discriminating test is usually a one-config re-solve with a single knob changed.

### Worked example: v4_lobatto investigation

Full trace of how the workflow was applied:

1. **Diagnostic C (sim-path, default propagated-Lobatto eval)** flagged max log10|EE| = −0.08 (83% relative error at the worst cell).
2. **`_diag_simpath_worst_cells.py`** identified 167 cells with |EE| > 3% (1% of evaluated). Pattern: i_2 ∈ {3, 4, 6} (dominantly i_2=6), α_b mean +0.96 (long bonds), age mean 88.4, wealth mean 6.9, savings ratio 86%, zero cap-binding, zero KKT-constrained. Worst single cell: age 89, state (2,1,6), wealth 2.6, α=(−0.22, +2.12), |EE| = 14.4%.
3. **`_diag_per_axis_tail.py` cross-reference**: state[2] Lobatto Z=7 covers 95% of short-bond cells but only 81% of long-bond cells. The 167 long-bond cells map to the missing 14 percentage points of coverage.
4. **`_diag_arbitrage_quadsweep.py`** confirms the long-bond pathology is tail-localised, not a body-integration issue.
5. **Mechanism attribution**: at the high-y_1 corners, M[xb,2] × +2.25σ shifts bond mean down enough that bonds remain attractive; the agent goes long; the explicit ±7σ tail node makes R_bond × 0.64 (catastrophic for long bonds); FOC has a near-discontinuity there; Newton's warm-restart fallback delivers a non-converged α_b ≈ +2 with FOC residual ~0.5 (matches metadata `worst_foc_resid` = 0.672).
6. **Recommended next test**: smaller Z=5 on state[2] (or state[1] which has nearly identical M-loading). Predicted outcome: max sim-EE improves but short-bond coverage degrades. A/B test: re-solve with Z=5, run Diag C, compare max + invalidity.

This is the template; apply it whenever a bundle's headline numbers are above the gate.

---

## 5. Comparing across runs

When evaluating a sweep (e.g., the 10-config AWS sweep), build a comparison table.

### Comparison driver

Run all three diagnostics on every bundle, then aggregate. Quick driver pattern:

```python
# scripts/diagnostics/_compare_bundles.py (or inline in a notebook cell)
import json
from pathlib import Path
import re

BUNDLES = [
    "saved_runs/checkpoints/...config_1...",
    "saved_runs/checkpoints/...config_2...",
    # ... etc
]
MODEL_BUNDLE = "saved_runs/unconstrained_principal_grid5x5x5_nz9"

# Parse the markdown reports for each bundle.
def parse_report(md_path):
    text = Path(md_path).read_text()
    out = {}
    m = re.search(r"Mean `log10\|EE\|`:\s*`([-\d.]+)`", text)
    if m: out["mean"] = float(m.group(1))
    m = re.search(r"Max `log10\|EE\|`:\s*`([-\d.]+)`", text)
    if m: out["max"] = float(m.group(1))
    m = re.search(r"valid:\s*`(\d+)`", text)
    if m: out["valid"] = int(m.group(1))
    return out

# Or read by-age tables for sim-path bundles to extract phase-level gates.
```

For each bundle, produce a row with:
- Bundle label
- Solver health: `worst_foc_resid`, `total_newton_failures`
- **Diagnostic A** (gridpoint same-Q): mean, p95, max
- **Diagnostic B** (gridpoint next_finer): mean (valid only), max, **invalidity rate**
- **Diagnostic C** (sim-path next_finer): working mean, working max, retirement mean, retirement max
- **Gate pass/fail** flags (4 gates)

Sort by retirement-mean-next_finer (or working-mean if working is more binding for your study).

### What to compare

When isolating which change is doing the work, design A/B comparisons:

- Same config, with vs. without leverage cap → cap effect
- Same config, with `(2,2,5)` vs `(3,3,5)` state quadrature → state-quad effect
- Same config, with `n_eta=3` vs `n_eta=5` → income-quad effect
- Same config, 7×7×7 vs 9×9×9 grid → grid refinement effect

Cross-config, watch for **interaction effects**: does state-quad refinement help only when combined with the cap? Does income quad matter on its own?

### Reading sim-path EE history across the project

Past markdown reports under `diagnostics_reports/diagnostics_*.md` are part of the running history. New reports should follow the same naming convention `diagnostics_simpath_ee_<bundle_label>_<eval_mode>.md` and live in `diagnostics_reports/` so they're easy to grep and compare.

---

## 6. Common diagnostic pitfalls

1. **Same-Q EE is a self-grade.** A bundle that passes same-Q gates but fails next_finer is fooling itself, not honestly publication-ready. Always check next_finer.
2. **iw=0 anchor outliers** can dominate the headline max. Decompose by wealth bin to confirm whether the max is anchor pathology or genuine policy issue. The decomposition pattern: build a per-iw breakdown of mean/max/invalidity.
3. **State-corner artifacts.** Wide state support `(2.0, 2.25, 2.25)` makes the gridpoint sweep probe extreme economic corners that may have ~zero stationary probability. Don't read those probes as "the policy is bad" — they're stress-tests. Sim-path EE weighs by equilibrium probability and is the trustworthy headline.
4. **Partial bundles can't measure both phases.** A from-age-87 bundle gives no working-age data; a from-age-65 bundle gives 2 working ages (insufficient for working gate). Aim for from-age-22 or from-age-50 for honest publication-grade evaluation.
5. **Initial conditions matter for sim-path.** A `centered` initial cross-section is tighter than `stationary`. Different initial choices produce different EE distributions. Document the initial-condition choice in any reported numbers.
6. **`next_finer` invalidity is information, not noise.** Cases where `e_sum ≤ 0` reflect real positions in the policy that are infeasible under richer quadrature. Don't drop them silently — quantify the rate.
7. **Quadrature self-consistency illusion.** A bundle solved with `(2,2,5)` state quadrature evaluated under `(2,2,5)` looks great. Evaluated under `(3,3,6)` it can look terrible. The honest grade is the finer one. The gap is the policy's truncation bias.

---

## 7. Bundle naming convention

Use a self-describing path that lets you reconstruct the config without opening metadata:

```
saved_runs/checkpoints/<system>_<constraint>_<grid_mode>_grid<state_grid_sizes>_nz<n_z>_age<youngest>_<terminal>_kstate<...>_kret<...>_eta<n_eta>eps<n_eps>_cap<value>_<modifications>_v1
```

Example: `..._grid7x7x7_nz9_age50_99_kstate3x3x5_kret3x7x5_eta3eps3_cap5_log1p_pathB_wmin01_v1`

Diagnostic markdown files should mirror the bundle label so they're trivially associable: `diagnostics_simpath_ee_<bundle_label>_nextfiner.md`.

---

## 8. Quick-reference command sequence

For a new bundle at `<bundle>`:

```bash
# Sanity check
python -c "import json; m=json.load(open('<bundle>/metadata.json')); print(m['diagnostics_summary'])" | head -20

# Diagnostic A
python -m scripts.diagnostics._diag_gridpoint_ee \
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \
  <bundle> --eval-mode same \
  --markdown-out diagnostics_reports/diagnostics_gridpoint_<label>_same.md

# Diagnostic B
python -m scripts.diagnostics._diag_gridpoint_ee \
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \
  <bundle> --eval-mode next_finer \
  --markdown-out diagnostics_reports/diagnostics_gridpoint_<label>_nextfiner.md

# Diagnostic C
python -m scripts.diagnostics._diag_euler_errors \
  <bundle> \
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \
  --eval-mode next_finer \
  --n-simulations 5000 --eval-households-per-age 256 \
  --initial-x 5.0 --initial-z stationary --initial-state median \
  --partial-init-mode centered \
  --markdown-out diagnostics_reports/diagnostics_simpath_<label>_nextfiner.md
```

Then read the three markdown reports together and compare against gates.

---

## 9. What you should produce in a comparison report

When grading a sweep (or a single bundle), your final deliverable should be a markdown comparison table answering:

1. **Solver health:** does the bundle solve cleanly?
2. **Same-Q grade:** mean, p95, max, validity. Per phase if available.
3. **next_finer grade:** mean, max, **invalidity rate**.
4. **Sim-path grade:** working mean/max, retirement mean/max, **gate pass/fail**.
5. **Worst-point profile:** which (age, iz, state, iw) drives the max? What's the policy α at that point?
6. **Comparison to baseline / sibling configs:** which axis of variation matters?
7. **Recommendation:** does this bundle pass the publication test? If not, what's the dominant remaining issue and what's the recommended next fix?

Include the raw markdown reports as appendix or via links so anyone can re-derive the numbers.

---

That's the full workflow. The same battery applies to every solve, including the in-flight 10-config AWS sweep, future quadrature-refinement sweeps, and any model-level investigations.

---

## 10. How this compares to EGM codebases on GitHub

When defending diagnostic choices to a referee, it helps to know what the open-source EGM/lifecycle community does. Below is what the most widely cited GitHub repos actually compute. The short version: nobody has standardised on a single number, but **HARK's `make_euler_error_func` is the most explicit reference implementation**, and its formula is the same idea as ours up to sign.

### 10.1 HARK (`econ-ark/HARK`) — the closest reference implementation

File: `HARK/ConsumptionSaving/ConsIndShockModel.py`, method `IndShockConsumerType.make_euler_error_func(mMax=100, approx_inc_dstn=True)`.

**Formula** (verbatim):

```python
mNowMin = self.solution[0].mNrmMin + 10**(-15)   # avoid 0/0 at the constraint
mNowGrid = np.linspace(mNowMin, mMax, 1000)
cNowGrid = cFuncNow(mNowGrid)
# expected marginal value next period (scalar inversion of FOC):
ExvPnextGrid = self.DiscFac * self.Rfree[0] * self.LivPrb[0] * \
               self.PermGroFac[0]**(-self.CRRA) * \
               np.sum(PermShkVals_tiled**(-self.CRRA) * vPnextArray * ShkPrbs_tiled, axis=0)
cOptGrid = ExvPnextGrid**(-1.0/self.CRRA)
EulerErrorNrmGrid = (cNowGrid - cOptGrid) / cOptGrid
```

**Convention.** "Consumption error per dollar of consumption" — normalised by `cOpt` (the FOC-implied consumption), not by the policy. `EE > 0` means policy over-consumes vs FOC; `EE < 0` means under-consumes.

**Compared to ours** (`scripts/diagnostics/_diag_euler_errors.py`):
- We report `1 - c_implied / c_policy` = `−EE_HARK / (1 + EE_HARK)`. For small errors the magnitude is identical; the sign convention differs (HARK over-consumes positive, ours under-consumes positive). When citing AFV-RR (2006) or Den Haan (2010) thresholds, our `|EE|` is directly comparable to HARK's `|EE|`.
- HARK evaluates on a 1D grid in `m` only (single-asset model with idiosyncratic income shock). Our diagnostic is multi-D over `(t, z, s, x)` — Diagnostic A grid sweep — and along simulated paths — Diagnostic C.
- HARK lets you toggle the eval expectation between `IncShkDstn` (production quadrature) and "a very accurate discrete approximation". This is exactly our `--eval-mode same` vs `next_finer / double` toggle.
- HARK does not summarize: it stores `eulerErrorFunc` as an interpolated function. The user plots it. Our gates table is a HARK-style function summarized into mean / p99 / max.

**Constraint-region treatment in HARK.** The grid starts at `mNrmMin + 1e-15` (a defensive offset to avoid `0/0`) but does **not** explicitly filter the kink. In practice this means the lowest few `mNowGrid` points sit on the constrained branch; HARK's docstring describes the function as for "expository and benchmarking" use, leaving filtering to the reader. The unconstrained-cell-set we added to our gates table (Section 2 of `_diag_euler_errors.py`, `cell_set='unconstrained'`) is the explicit filter that AFV-RR and Den Haan apply implicitly.

Source: [`econ-ark/HARK` ConsIndShockModel.py](https://github.com/econ-ark/HARK/blob/master/HARK/ConsumptionSaving/ConsIndShockModel.py).

### 10.2 EconForge `dolo` / `dolark` — residuals at the steady state

Dolo defines an `arbitrage` function (Euler-equation residual) symbolically. Two diagnostic patterns:
- `model.residuals(s, x)` — evaluates the FOC residual at any (state, control) tuple. Used at the deterministic steady state as a sanity check.
- The solvers report convergence as `||arbitrage_residual||_∞` at the final iterate.

There's no GitHub-shipped equivalent of "log10|EE| along a simulated panel" in dolo. The dolo workflow leans on (a) steady-state residual, (b) iteration-level convergence norms, and (c) impulse-response comparisons against finite-difference references.

Source: [`EconForge/dolo.py`](https://github.com/EconForge/dolo.py), `model.residuals` and arbitrage block.

### 10.3 Iskhakov–Jørgensen–Rust–Schjerning DC-EGM (`fediskhakov/dcegm`)

Matlab implementation of the algorithm in QE 2017. The repo ships `solve_egm`, `solve_vfi`, and a stub `solve_euler` that errors out as `'Not implemented: homework'`. **Euler errors are not computed in the released code.** Validation in the paper itself relies on (a) cross-method agreement (EGM vs VFI vs Euler-equation iteration on the same model), and (b) max log10|EE| reported in the published tables, evaluated externally. The DC-EGM contribution is the upper-envelope construction at non-concave kinks; once that's correct, the EE collapses to a standard EGM diagnostic.

Source: [`fediskhakov/dcegm`](https://github.com/fediskhakov/dcegm).

### 10.4 Sequence-Space Jacobian (`shade-econ/sequence-jacobian`)

Auclert, Bardóczy, Rognlie, Straub (Econometrica 2021). The household block solves a Bellman with EGM but the **gating diagnostics in SSJ are different in kind**: the toolkit checks (a) policy-function convergence to fixed-point tolerance, (b) Jacobian accuracy via finite-difference vs analytical comparison, and (c) impulse-response Jacobian agreement against direct nonlinear-perfect-foresight simulation. The notebooks (`krusell_smith.ipynb`, `two_asset.ipynb`, `hank.ipynb`) do not report Euler errors per se — the IRF-vs-truth check serves the same purpose at the equilibrium-system level.

Source: [`shade-econ/sequence-jacobian`](https://github.com/shade-econ/sequence-jacobian).

### 10.5 QuantEcon EGM lectures

The IFP/EGM lectures (`ifp_egm.html`, `egm_policy_iter.html`, JAX variants) **do not compute Euler errors**. Validation is via:
- Comparison to closed-form cake-eating solutions where available.
- Cross-implementation agreement (NumPy vs JAX vs Numba).
- Stationary-distribution shape checks.

The QuantEcon-style `euler_diff(c, a, z)` function appears in time-iteration / Coleman operator material but as a *solver primitive*, not a diagnostic.

Source: [QuantEcon EGM lecture](https://python.quantecon.org/egm_policy_iter.html).

### 10.6 CGM (Cocco–Gomes–Maenhout 2005) replication (`econ-ark/REMARK`)

The REMARK replication of CGM uses HARK's portfolio solver and inherits HARK's Euler-error tooling (§10.1). The replication does not report a separate EE table; the validation criterion is reproducing the published lifecycle policy and equity-share figures.

Source: [`econ-ark/REMARK/CGMPortfolio`](https://github.com/econ-ark/REMARK).

### 10.7 What this means for our reported numbers

**The diagnostic battery in §2 is more aggressive than what most GitHub EGM repos ship.** Specifically:
- HARK reports a function, not a summary; we report mean / p99 / max gates.
- HARK evaluates on a 1D grid; we evaluate on a 4D structured cube (Diagnostic A) plus a stochastic simulation (Diagnostic C).
- HARK has one quadrature toggle (`approx_inc_dstn`); we have a full `same / next_finer / double` plus per-axis overrides.
- Nobody else publishes both gridpoint and simulation-path EE in the same workflow; we do, and the gap between them (Diagnostic A max far worse than Diagnostic C max) is itself information about which states matter for the headline.

**Where we should align with the literature**:
- The `unconstrained` cell-set in our gates table (filtering `savings/x < kink_tol`) matches the AFV-RR / Den Haan convention. Kept the `all` cell-set for back-compat.
- Sign convention: ours `1 − c_implied/c_policy` differs from HARK's `(c_policy − c_implied)/c_implied`. Both are equivalent on `|EE|`, which is what we and they report. No need to change.
- HARK's "very dense approximation" of the income shock distribution, used as the eval rule, is exactly our `next_finer` semantics (independent and finer than the production rule).

**Where we go beyond**:
- Multi-dimensional state EGM (3-D financial-state VAR + persistent income) — neither HARK nor any of the reference repos solves this; we built the diagnostic for it.
- Structured-probe Diagnostic A at extreme state corners — useful as a stress test even if those corners have ~zero stationary probability. Not standard but defensible because corner states are exactly where the trilinear interpolation truncation manifests.
- Decomposition by wealth bin / state corner / phase — the per-iw and per-state breakdowns we use to localize pathology aren't in any of the open-source toolkits.

### 10.8 Citations to defend the workflow in the paper

When justifying our gates and the unconstrained-cell convention, the load-bearing citations are:
- Aruoba, Fernández-Villaverde, Rubio-Ramírez (2006), *J. Econ. Dyn. Control* 30 — the unit-free-EE convention and the threshold table.
- Den Haan (2010), *J. Econ. Dyn. Control* 34 — what accuracy is achievable per method class, and the simulated-path gating practice.
- Carroll (2006), *Economics Letters* 91 — the EGM constraint-corner construction.
- Iskhakov, Jørgensen, Rust, Schjerning (2017), *Quantitative Economics* — the kink/upper-envelope convention at non-concave points (relevant if we ever extend to discrete-continuous choice).
- Judd (1998), Ch. 7 — quadrature polynomial-exactness thresholds (the basis for our `n_eps`/`n_eta` discussion in `GRID_CONVERGENCE_CRITERIA.md` §5.5).

Open-source code citations:
- HARK `make_euler_error_func`: closest implementation analog; cite for the formula and the same/finer-Q toggle.
- AFV-RR (2006) Table 4: cite for the headline format (mean and max log10|EE| by configuration).

