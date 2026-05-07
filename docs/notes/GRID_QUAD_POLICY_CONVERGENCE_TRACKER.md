# Grid/Quadrature Policy Convergence Tracker

Last updated: 2026-05-07

## Era

Active production state model (post-rtb-as-state migration, JAX branch):

- **State vector** (4-D): `(cy, spr, rtb, y_1)` — `rtb` is now a state innovation,
  not a return shock. `log_R_bill` is read from the next-period state vector.
- **Return block** (2-D): `(xr, xb)` — stock-excess and bond-excess log returns.
- **`state_grid_mode = "cholesky"`** — Cholesky-rotated state grid in
  standardised z-space (lower-triangular L), then transformed back to physical
  state coordinates.
- **`n_z = 11`** — income discretisation (Rouwenhorst).
- **`wealth_dynamics_spec = "ccv_log"`** — Campbell–Chacko–Viceira log-portfolio
  return; bankruptcy mathematically impossible (R_p = exp(r_p) > 0).
- **`use_fori_newton = True`** — Newton runs `lax.fori_loop` with mask;
  `max_iter` is literal wall cost per cell.

Anything in `docs/archive/` (e.g. `LOBATTO_CONFIG_TRACKER.md`) reflects the
pre-migration 3-D state era and does not transfer mechanically.

## In-flight migrations affecting future entries

- **CCV w8566 VAR re-estimation** (in flight, see
  `docs/CCV_IMPLEMENTATION_HANDOFF.md`): sample changes from 1963–2025 with
  `cy` to 1871–2011 (T=141) with `dp`. Re-estimation will change `Sigma_v`
  block-by-block. Bundles produced before this migration completes are
  optimal under a calibration the locked spec just declared wrong; they
  remain valid as system-validation artefacts but will be re-solved on the
  new VAR before any thesis figure is generated from them.
- **Cholesky/quadrature mathematical-correctness audit**
  (`docs/handoff/HANDOFF_CHOLESKY_QUADRATURE_AUDIT.md`): Phase A audit of
  the two-stage Cholesky decomposition + tensor-product GH discretisation.
  Findings will land in `docs/scans/CHOLESKY_QUAD_AUDIT_<DATE>.md` and may
  trigger a retroactive re-run of recent entries here.

---

## Purpose

Track the partial-solve convergence exercise for lifecycle policy functions
under the JAX branch. Vary one axis at a time:

- financial-state grid size (per-axis)
- return quadrature (`n_ret_nodes_1d`)
- state quadrature (`n_state_quad_nodes`)
- state-axis support widths (`state_n_stds`)
- Newton/quadrature stability levers (`max_iter`, `lobatto_Z`, …)

Goal: identify the smallest configuration that is "good enough" for the
portfolio-choice problem at publication accuracy on the headline EE
metrics, without burning compute on dimensions that have already saturated.

This tracker is the bookkeeping layer for:

- what has been run
- what is currently running
- which saved bundles are clean comparison candidates
- which tests we want to run on completed bundles
- the **iterative loop**: each bundle reveals something, we patch it, the
  next bundle confirms or refutes the patch, repeat until publication
  gates clear.

Companion: `docs/notes/GPU_TRIAL_FINDINGS.md` (hardware/wall/cost lessons
per run). Convergence and config-iteration findings live **here**;
hardware findings live there.

---

## Canonical Comparison Rules

To keep convergence comparisons interpretable:

- Hold `state_grid_mode="cholesky"` fixed.
- Hold `n_z=11`, `n_eta_nodes=4`, `n_eps_nodes=4`, `n_wealth=180`,
  `n_savings=180` fixed unless explicitly testing one of them.
- Hold `state_n_stds` fixed across a comparison family.
- Change one main axis at a time:
  - first return quadrature `n_ret_nodes_1d`
  - then state grid size `state_grid_sizes`
  - then state quadrature `n_state_quad_nodes`
  - support widths `state_n_stds` only if support clipping is the binding
    issue
- For cross-grid comparisons, do not compare native array entries directly.
  Compare policies on a common probe set.

---

## Current Sweep Strategy

The next bundle is sized as the calibration anchor for the run after it.
Each cycle is one full bundle solve plus a `max_iter` calibration pass on
the resulting Newton-iter histogram (canonical workflow:
`max_iter_next = max(20, ⌈1.5 × p99_newton⌉)`).

### Cycle 0 — 5⁴ retirement baseline ✅ DONE

`state_grid_sizes=(5,5,5,5)`, `n_state_quad_nodes=(2,3,2,3)`,
`n_ret_nodes_1d=(5,5)`, `state_n_stds=(2.0, 2.25, 2.0, 2.25)`,
`gather_precision="f32"`, `max_iter=100`, retirement-only (ages 67–99).
Hardware: 1× GH200. Wall: 273 s/age × 33 ages = 150 min.

### Cycle 1 — 6⁴ retirement on 8× GPU (PENDING LAUNCH)

Same per-cell config as Cycle 0 (reduced quad), at `state_grid_sizes=(6,6,6,6)`.
Runner: `verify/benchmark_bundle_6666.py`. Purpose:
- system-validate 8× pmap dispatch + cache + S3 + checkpointing in production
- establish 6⁴ wall baseline
- harvest first real Newton-iter histogram → calibrate `max_iter` for Cycle 2

### Cycle 2 — 6⁴ full-solve

Canonical full lifecycle on 6⁴. Calibrated `max_iter` from Cycle 1.
Use this bundle's grid-EE + sim-EE to decide whether 6⁴ is sufficient or
7⁴ is required for publication-grade gates.

### Cycle 3 — 7⁴ canonical full-solve (publication artefact)

Final publication bundle assuming Cycle 2 cleared the EE thresholds. If
Cycle 2 failed gates, an intermediate cycle revisits the binding axis
before going to 7⁴.

---

## Known Bundles

### Clean reference bundles

| Label | Bundle | Status | Ages solved | Key discretisation notes |
|---|---|---|---|---|
| 5⁴ retirement baseline | `system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_*` (S3 + laptop) | `complete` | `67-99` | `state_grid_sizes=(5,5,5,5)`, `n_state_quad_nodes=(2,3,2,3)`, `n_ret_nodes_1d=(5,5)`, `state_n_stds=(2.0,2.25,2.0,2.25)`, `gather_precision="f32"`, `max_iter=100`, Lobatto OFF, `delta_bequest=0.0` |

### Pending bundles

| Label | Planned bundle pattern | Status | Notes |
|---|---|---|---|
| 6⁴ retirement (calibration anchor) | `system_iv_full_var_unconstrained_cholesky_grid6x6x6x6_nz11_jax_benchmark` | `queued` | Runner: `verify/benchmark_bundle_6666.py`. Awaiting Lambda 8× capacity + decision on whether to wait for CCV-VAR migration first. |

### Bundles to treat with caution

| Bundle | Reason |
|---|---|
| Anything pre-CCV-VAR-migration once the migration lands | Solved against 1963–2025 cy-VAR; thesis figures must use post-migration 1871–2011 dp-VAR. Keep for system-validation reference, do not use for thesis numbers. |

---

## Patch Journal

The iterative loop. Each cycle: what we ran, what was off, what got
patched, what we expect from the next cycle.

### Cycle 0 — 5⁴ retirement baseline (2026-05-06, 1× GH200 on Lambda)

- **Config.** `state_grid_sizes=(5,5,5,5)`, `n_state_quad_nodes=(2,3,2,3)`,
  `n_ret_nodes_1d=(5,5)`, `state_n_stds=(2.0, 2.25, 2.0, 2.25)`,
  `wealth_dynamics_spec="ccv_log"`, `gather_precision="f32"`, `max_iter=100`,
  Lobatto OFF, `delta_bequest=0.0`. Retirement-only (ages 67–99, 33 ages).
- **Hardware.** Lambda `gpu_1x_gh200`, 97 GB HBM3 (the "480 GB" branding
  is unified Grace LPDDR5 + GPU HBM3; only the 97 GB HBM is addressable
  by the solver). Effective fp64 throughput ≈ 6.8 TFLOPS (70 % of GH200
  9.7 peak), per-age compute dominates over JIT compile cost.
- **What got tried before this config worked.**
  1. **9⁴ retirement-only with `n_state_quad=(3,4,3,4)` and `max_iter=400` →
     OOM.** XLA planned 96 GB, GH200 had 97 GB. Failed by allocator
     overhead.
  2. **7⁴ same per-cell config → worse OOM.** XLA's plan ballooned to
     1.06 TB — 11× worse than 9⁴ despite a smaller state grid. Same code,
     different shape → different XLA tile/fusion/remat decisions.
     Documented XLA non-monotonicity-in-input-dimensions.
  3. **5⁴ with reduced quad `(2,3,2,3)` and `max_iter=100` → fits in
     ~62 GB worst-case full-materialisation.** Compiled and ran cleanly.
- **Result.** Bundle `system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_*`
  saved locally + S3. `solve_status="complete"`, no Newton failures, no
  monotonicity violations. Per-age wall stabilised at 273 s/age flat
  (no acceleration after age 1 — JIT compile cost is small ≈ 5–15 s; the
  rest is real fp64 compute). Alphas economically coherent with backward
  warm-start producing textbook lifecycle pattern (α_s rises, α_b falls
  as we move backward in age).
- **Grid-EE on the bundle (39.6 M probes, eval_mode `same`):** median
  log10|EE| = −6.5, p95 = −3.5, p99 = −2.7, **max = −2.0** (worst abs
  EE = 9.65 × 10⁻³, ≈ 1 %). PASS at the worst-cell threshold but right
  at the PASS / CONCERNING boundary (main-branch convention:
  PASS<1e-2, FAIL>5e-2). Tighter than main-branch v3 on every directly
  comparable metric.
- **What was off (1) — bundle interrupted at age 94 (cost ≈ $0.85).**
  Run was launched as `ssh ubuntu@... 'bash scripts/gpu_run.sh' &` —
  background SSH. When the SSH session recycled, python received SIGHUP
  and was killed. No `nohup`, no tmux, no `disown`. Standard Linux
  gotcha. **Patched:** all subsequent runs go via
  `tmux new-session -d -s bench '...'`; `verify/benchmark_bundle.py`
  default `checkpoint_every_n_ages=5` so worst-case loss is bounded
  to 23 min ≈ $0.75 of GPU time. Documented in
  `docs/notes/GPU_TRIAL_FINDINGS.md` operating notes.
- **What was off (2) — solver/simulator CCV-vs-arithmetic mismatch
  (HIGH-severity bug).** [solver.py:657-679](../../lifecycle/solver.py#L657-L679)
  computed CCV log-portfolio return; [simulation.py:355](../../lifecycle/simulation.py#L355)
  was doing arithmetic combination of simple returns. **Solver-optimal
  policy under one return law was being simulated under a different
  return law.** Bias ≈ 50–150 bp per period, compounds over 33 retirement
  years. Bundle remains valid as a **CCV-optimal policy**, but
  path-based diagnostics (sim-EE, welfare numbers) computed before the
  fix were biased. **Patched:** `simulation.py:329-362` rewritten to
  CCV. Solver/simulator parity test (`verify/ccv_solver_sim_parity.py`)
  added and confirmed to 1e-12. **Implication for this bundle:** grid-EE
  numbers reported above are clean (FOC-grid based, not sim-path based).
  Sim-EE on this bundle has not been re-run since the simulator fix
  landed.
- **What was off (3) — no Newton iter histogram.** Bundle was solved
  before commit `051423a` exposed `newton_iter_histogram` and
  `backtrack_iter_histogram` in `diag`. So `max_iter=100` was chosen by
  sniff, not by p99 calibration. **Patched:** all bundles from Cycle 1
  onwards land histograms; cycle-N+1 `max_iter` calibrated as
  `max(20, ⌈1.5 × p99_newton⌉)`.
- **What was off (4) — Lobatto tails were OFF in this bundle.** Both
  `ret_lobatto_Z` and `state_lobatto_Z` are `None`; pure Gauss–Hermite
  throughout. For thesis-quality runs at small `delta_bequest`, Lobatto
  tails at ±3σ should be enabled to capture bankruptcy-boundary cells
  robustly. **Open** (deferred to a later cycle pending CCV-VAR migration).
- **Next cycle expectation.** First real p99 from Cycle 1 → first
  principled `max_iter` setting for Cycle 2. Multi-GPU pmap dispatch
  validated at production scale. 6⁴ wall baseline measured.

### Cycle 1 — 6⁴ retirement on 8× GPU (pending)

- *Awaiting launch.*

---

## Run Ledger

Chronological run-by-run notes. Each new bundle gets a date-stamped block
with: config tried, solve outcome, what was found, follow-up actions.
Start a fresh dated entry on every solve.

### 2026-05-06

- **First Lambda Labs GPU trial.** Spun up `gpu_1x_gh200`
  (Grace ARM + GH200 with 97 GB HBM3) in `us-east-3`, ~$1.49–$2.49/hr.
  Bootstrap clean: Lambda Stack 22.04 → Python 3.10.12 → JAX 0.6.2 with
  full aarch64 wheel resolution for `jax[cuda12]`. AWS credentials
  piped over SSH (`cat ~/.aws/credentials | ssh ...`) saves ~30 s vs
  `aws configure`.
- **Smoke (`verify/smoke.py` at the time, 6 ages, tiny config):** 30.2 s
  total, mostly JIT compile cold. `Cell-batching pattern: vmap-only
  (single-device)` printed correctly — first time the new dispatch
  path was exercised on real GPU silicon. No regressions.
- **Benchmark attempts (production sizing, `n_z=11`, `n_w=180`, `n_s=180`):**
  - **Attempt 1 — 9⁴ retirement-only with `(3,4,3,4)` quad, `max_iter=400`:**
    OOM in the terminal kernel. XLA planned 96 GB; GH200 has 97 GB. Failed
    by allocator overhead margin.
  - **Attempt 2 — 7⁴ same per-cell config:** OOM, XLA-plan 1.06 TB
    (16× the irreducible 67 GB working set). Documented XLA
    non-monotonicity in input dimensions — same code, different shape,
    different fusion/tile/remat decisions.
  - **Attempt 3 — 5⁴ + reduced quad `(2,3,2,3)` + `max_iter=100` +
    `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`:** compiled and ran cleanly
    at 273 s/age flat, GPU instrumentation steady at 92.9 / 97.9 GB HBM.
    First production bundle landed.
- **Bundle saved:**
  `system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_*`
  (laptop + S3). `solve_status="complete"`, ages 67–99, no Newton
  failures, no monotonicity violations. Backward warm-start producing
  textbook α_s rising / α_b falling pattern across ages.
- **Grid-EE on the bundle** (`verify/ee_residuals.py`, eval-mode `same`,
  39.6 M probes):
  - median log10|EE| = −6.5
  - p95 log10|EE| = −3.5
  - p99 log10|EE| = −2.7
  - **max log10|EE| = −2.0** (worst abs EE = 9.65 × 10⁻³, ≈ 1 %)
  - Verdict: bundle passes correctness gate; sits right at the
    PASS / CONCERNING boundary on worst-cell. Tighter than main-branch
    v3 on every directly comparable metric.
- **Run died at age 94 — SSH-attached python killed by SIGHUP.**
  Background SSH (`ssh ... 'bash scripts/gpu_run.sh' &`) recycled →
  python killed. No `nohup` / tmux / `disown`. Loss: 25 min ≈ $0.85.
  No partial bundle (the original benchmark script had
  `checkpoint_every_n_ages=None`). **Patched** for relaunch by switching
  to `tmux new-session -d -s bench '...'` and adding
  `checkpoint_every_n_ages=5` to `verify/benchmark_bundle.py` defaults.
- **Bug surfaced and fixed mid-run** (HIGH severity): solver / simulator
  portfolio-return mismatch.
  [solver.py:657-679](../../lifecycle/solver.py#L657-L679) was CCV-correct;
  [simulation.py:355](../../lifecycle/simulation.py#L355) was doing
  arithmetic combination of simple returns. Solver-optimal policy under
  one return law, simulated under a different return law. Bias ≈ 50–150
  bp per period; compounds over 33 retirement years. **Fixed:**
  `simulation.py:329-362` rewritten to CCV form. Parity verified to
  1e-12 with `verify/ccv_solver_sim_parity.py`. The bundle from this
  run remains valid as a CCV-optimal policy; sim-path EE on it has not
  yet been re-run post-fix.
- **Hardware/perf characterisation calibrated.** GH200 fp64 effective
  throughput ≈ 6.8 TFLOPS (70 % of 9.7 peak). Per-age wall is real
  compute (273 s); JIT compile cost ≈ 5–15 s — small relative.
  Implication: further wall reduction must come from reducing per-cell
  FLOPs (smaller `n_state_quad`, smaller `max_iter`), not JIT or
  fusion micro-optimisation. Single-cell `c_corners` at this quad
  ≈ 9 MB → 6875 cells × 9 MB ≈ 62 GB worst-case full materialisation,
  fits 97 GB with headroom; 7⁴ at full canonical quad would need
  ~950 GB and require manual cell-vmap chunking.
- **Saturation finding:** the `vmap-only (single-device)` 5.4× CPU
  speedup over pmap doesn't translate directly to GPU wall savings —
  per-age fp64 compute dominates over the dispatch overhead the
  vmap-only path eliminates. The fusion gain is real but masked at
  this scale.

---

## Planned Run Matrix

| Priority | Solve window | State grid | Return quadrature | State quadrature | `state_n_stds` | Purpose |
|---|---|---|---|---|---|---|
| Done | `67-99` | `(5,5,5,5)` | `(5,5)` | `(2,3,2,3)` | `(2.0, 2.25, 2.0, 2.25)` | Cycle 0 baseline (above) |
| 1 | `67-99` | `(6,6,6,6)` | `(5,5)` | `(2,3,2,3)` | `(2.0, 2.25, 2.0, 2.25)` | Cycle 1 calibration anchor + multi-GPU validation |
| 2 | full lifecycle | `(6,6,6,6)` | `(5,5)` | `(2,3,2,3)` | `(2.0, 2.25, 2.0, 2.25)` | Cycle 2: first full-solve on calibrated `max_iter` |
| 3 | full lifecycle | `(7,7,7,7)` | `(5,5)` | TBD | TBD | Cycle 3 publication artefact |

`gather_precision="f32"` and `cell_vmap_chunks=1` are held fixed across
the matrix; chunking-on-pmap is YELLOW-by-design and only matters at
≥11⁴ on multi-GPU.

---

## Tests To Run On Policy Bundles

Implemented tooling (post-migration):

- `verify/ee_residuals.py` — grid-EE diagnostic on a saved bundle.
- `verify/ee_simpath.py` — sim-path EE diagnostic (the headline thesis
  number once the simulator CCV fix is in production).
- `verify/ccv_solver_sim_parity.py` — solver-vs-simulator CCV-formula
  parity to 1e-12. Must pass after any change to either side.
- `verify/arbitrage.py` + `verify/invalid_cells.py` — discrete-arbitrage
  + invalid-cell preflight (precompute-level, no GPU).
- `lifecycle.diagnostics.diagnose_terminal_portfolio_states` — terminal-age
  FOC residual at the solved policy.

### 1. Bundle integrity

For every saved bundle:

- verify `metadata.json` exists
- verify `diagnostics.pkl` loads
- verify `policy_arrays.npz` loads
- verify `solve_status`, `is_partial`, and age coverage
- record whether the bundle is complete, checkpointed, or interrupted

### 2. Solver-health diagnostics

For every candidate comparison bundle, record:

- `total_newton_failures` (gate: ≤ 1% of cells)
- `total_mono_violations`
- `worst_foc_resid` (target ≤ 1e-7 at solved cells)
- `newton_iter_histogram` (`p50 / p95 / p99 / max`) — drives
  `max_iter` calibration for the next bundle
- `backtrack_iter_histogram` (`p50 / p95 / p99 / max`)
- `total_calls`

Bundles with solver pathologies should not be used as convergence
references.

### 3. Common-probe policy comparison

When comparing two bundles, evaluate policies on a common probe set
rather than raw array indices.

Suggested probe ages (post-migration):

- `66`, `67`, `75`, `85`, `95`, `99`
- and `22`, `35`, `50`, `65` once full-lifecycle solves are available

Suggested probe wealth points:

- low, middle, and high wealth regions
- include points near `wealth_min`
- include points where risky shares often move sharply

Suggested probe income states:

- several `z` values, including low / median / high

Suggested financial-state probes:

- median state
- several off-centre states
- if needed, a fixed list of economic states shared across bundles

### 4. Policy metrics to compare

For each pair of bundles, compare:

- consumption `C`
- stock share `S`
- bond share `B`
- implied bill share `1 - S - B`
- consumption ratio `C / W` where meaningful

Recommended summary statistics:

- max absolute difference
- median absolute difference
- 95th percentile absolute difference

### 5. Convergence decision rules

Working rule of thumb (carry over from pre-migration era — methodology
unchanged, support widths and dimensionality changed):

- If a coarser quadrature is materially close to a finer one on the
  common probe set, keep the coarser unless tail metrics diverge.
- Focus extra attention on the work-to-retire boundary (ages 65–67)
  once full-lifecycle solves are available — historically the hardest
  cells.
- For the multi-axis state quadrature, do not assume the four state
  axes need symmetric K. The dominant solver lever in pre-migration
  experience was the axis carrying the largest M-loading; under the
  4-D `(cy, spr, rtb, y_1)` ordering, `M[xb, y_1]` is the dominant
  entry and `y_1` is at the last (axis-3) Cholesky position.

---

## Open Questions

### Resolved

- *(empty — fresh tracker; resolved findings will land here as cycles
  complete and the patch journal accumulates)*

### Still open

- **`state_n_stds` choice for the 4-D era.** The 5⁴ baseline used the
  asymmetric `(2.0, 2.25, 2.0, 2.25)` (matching the old 3-D narrow
  support pattern). Whether the canonical setting `(2.93, 2.93, 2.93, 2.93)`
  is needed for publication-grade tail accuracy in 4-D is untested.
- **Per-axis `n_state_quad_nodes` sensitivity in 4-D.** All cycles so
  far use the symmetric `(2,3,2,3)`. Single-axis bumps would identify
  which of the four Cholesky-rotated axes is binding for the
  M-projection-onto-returns step.
- **Wealth-grid coverage at full-lifecycle horizon.** `wealth_max=750`
  is currently the production default. Whether simulated trajectories
  exceed it materially under the 4-D state needs measurement once a
  full-lifecycle bundle exists.
- **Retroactive impact of the CCV-VAR migration.** Once the new
  1871–2011 dp-VAR lands, every entry above will need re-solving on
  the new calibration. Patch journal entries should be re-examined for
  whether the original "what was off / patched" still applies.
- **Smolyak / sparse-grid alternatives:** investigation concluded as a
  dead end (handoff archived). Lobatto-prescribed-tails on selected
  axes remains a config option but is not currently exercised in any
  pending cycle.
