# Handoff: Inf-Horizon Discretization Sensitivity Program

**Branch:** `jax-rewrite`
**Effort:** ~3–5 h compute on 2× H100 SXM (~$3.29/h → ~$10–17). Plan-only handoff; implementation = a `verify/inf_horizon_sweep_*.py` runner per sweep + analysis scripts mirroring `scripts/analysis/system_ii_quad_convergence.py`.
**Output of this handoff:** dispatch instructions for three inf-horizon sweeps (state grid, state quadrature, return quadrature), each with a baseline, 3–5 cells, comparison strategy, pause points, and verdict criteria. **Plan only — do not implement.**

---

## TL;DR

Inf-horizon retirement steady state is structurally ~100× cheaper than finite-horizon at the same discretization (one 4096-cell kernel call per outer iter × ~25–30 iters; no 78 ages, no working-age FOC, no income-shock quadrature). That makes a systematic **3-axis sensitivity program** tractable in one work-session of compute.

The program runs in **3 sweeps × 4 cells** for a total of ~12 inf-horizon solves, plus a single shared "asymptote" reference at the canonical (or near-canonical) discretization. Each sweep isolates one dimension and answers a clean GREEN/YELLOW/RED question:

1. **State-grid density** — *Is `state_grid_sizes=(4,4,4,4)` enough, or do we need (5,5,5,5) / (6,6,6,6) / (7,7,7,7)?* → tells the user where to pin canonical for production.
2. **State-quadrature density** — *Is `n_state_quad_nodes=(3,3,3,3)` enough on the inert-z subspace, or does the y_1 K-bump pay off here too?* → confirms whether the System II quad-sweep finding generalizes to System IV.
3. **Return-quadrature density** — *Is `n_ret_nodes_1d=(3,3)` enough, or do we need (4,4) / (5,5) / Lobatto Z=7?* → tells the user whether the canonical ret-quad is over-specified for retirement-only policies.

These three questions, answered cleanly on inf-horizon, **transfer** to canonical lifecycle calibration: anything that doesn't matter at inf-horizon (where the bottleneck is the state Bellman, not income) certainly won't matter more once income shocks are layered in.

---

## Why inf-horizon is the right testbed

- **No mortality, no bequest, no pension, no working ages.** State is just (dp, spr, rtb, y_1) × wealth, and the Bellman operator is the JAX retirement kernel iterated to a fixed point. Income-shock quadrature `(n_eta, n_eps)` is irrelevant; `n_z=1` is exact under pension=0/psi=1 (per the 2026-05-07 N=1 guard). That **eliminates** two of the four canonical discretization knobs from the sensitivity analysis surface.
- **Cheap.** The 8⁴ benchmark in `verify/benchmark_inf_horizon.py` projects ~98 s/iter × ~25–30 iters ≈ 42–50 min on 2× H100. A 5⁴ cell at the same quad is ~625/4096 ≈ 0.15× → **~7 min**. A 7⁴ cell ≈ 2401/4096 ≈ 0.59× → **~28 min**. The full 12-cell program fits in ~3–4 h plus compile overhead.
- **Diagnostics already wired.** `lifecycle/inf_horizon_solver.py:_build_diagnostics` returns `converged`, `n_iter`, `final_stopping_supnorm`, per-iter sup-norm histories, post-fix Newton/backtrack histograms (post-8bfaec9), and the assumed-imminent `total_newton_failures` from `HANDOFF_NEWTON_FAILURE_COUNT_WIRING.md`. KeyboardInterrupt → partial save.
- **Asymptote reference is well-defined.** `configs/_canonical.py` pins the production-grade target — `state_grid_sizes=(7,7,7,7)`, `n_state_quad_nodes=(3,5,3,5)`, `state_lobatto_Z=(None,7.0,None,7.0)`, `n_ret_nodes_1d=(5,5)`, `ret_lobatto_Z=(7.0,7.0)`. Pin one inf-horizon run at this config and use it as the **shared anchor** for all three sweeps.

---

## Shared run convention

All cells in the program share the following knobs (deviations ⇒ retire to a different program). These mirror the System II quad-sweep convention plus the agreed inf-horizon settings.

```python
# Economics — System IV nominal VAR baseline (matches verify/benchmark_inf_horizon.py)
from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER

# Outer fixed-point loop
IH_TOL       = 1e-5
IH_MAX_ITER  = 100      # safety cap; convergence typically in 25-30 at canonical
IH_DAMPING   = 1.0      # full Bellman; lower only if a cell visibly diverges

# Discretization base (overridden per-sweep for the swept axis)
disc_base = CANONICAL_DISC._replace(
    wealth_min=0.05,
    n_wealth=180,
    n_savings=180,
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_stds=2.25,
    n_z=1,                         # inf-horizon guard; pension=0/psi=1
    n_eps_nodes=4, n_eta_nodes=3,  # irrelevant in inf-horizon, set to canonical
)

# Solver (canonical inner Newton; CCV log-wealth; f32 gather; chunks=1 for n_z=1)
solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
    max_iter=100, max_iter_unconstrained=100,
    delta_bequest=0.0,
    gather_precision="f32",
    cell_vmap_chunks=1,
)
```

**Bundle naming convention** (mirrors `system_iv_inf_horizon_grid8x8x8x8_nz1_y1lob_calib1`):

```
inf_horizon/system_iv_inf_<sweep>_<cell-tag>_calib1/
```

So e.g.:
- `inf_horizon/system_iv_inf_grid_g4_calib1/` — state-grid sweep, (4,4,4,4)
- `inf_horizon/system_iv_inf_sq_3333_calib1/` — state-quad sweep, (3,3,3,3)
- `inf_horizon/system_iv_inf_rq_4x4_calib1/` — ret-quad sweep, (4,4)
- `inf_horizon/system_iv_inf_anchor_canonical_calib1/` — shared canonical-asymptote anchor

Each cell uses `lifecycle/policy_io.save_policy_bundle` (already used by `verify/benchmark_inf_horizon.py`) so the npz / metadata.json / diagnostics.pkl layout matches the rest of the saved-runs ecosystem.

---

## Compute budget at a glance

Wall projections derived from the 8⁴ projection in `verify/benchmark_inf_horizon.py` (~98 s/iter × ~25–30 iters → ~45 min) scaled by cell-count and quad-node-count, plus a one-time JAX compile of ~60–90 s per distinct kernel shape:

| Axis | Cell | Cells × n_state_quad × n_ret_quad | Rel. cost vs 8⁴/(3,3,3,5)/(3,3) | Wall (per-iter) | Wall (full solve) |
|---|---|---|---|---|---|
| Anchor | 7⁴ + (3,5,3,5) + (5,5) | 2401 × 225 × 25 | 1.4× | ~140 s/iter | ~75 min |
| state-grid g3 | 3⁴ + (3,3,3,3) + (3,3) | 81 × 81 × 9 | 0.013× | ~3 s/iter | ~3 min |
| state-grid g4 | 4⁴ + (3,3,3,3) + (3,3) | 256 × 81 × 9 | 0.040× | ~6 s/iter | ~5 min |
| state-grid g5 | 5⁴ + (3,3,3,3) + (3,3) | 625 × 81 × 9 | 0.098× | ~12 s/iter | ~8 min |
| state-grid g6 | 6⁴ + (3,3,3,3) + (3,3) | 1296 × 81 × 9 | 0.20× | ~22 s/iter | ~15 min |
| state-quad 3333 | 5⁴ + (3,3,3,3) + (3,3) | 625 × 81 × 9 | 0.098× | ~12 s/iter | ~8 min (= g5) |
| state-quad 3535 | 5⁴ + (3,5,3,5) + (3,3) | 625 × 225 × 9 | 0.27× | ~30 s/iter | ~16 min |
| state-quad 4444 | 5⁴ + (4,4,4,4) + (3,3) | 625 × 256 × 9 | 0.31× | ~33 s/iter | ~17 min |
| state-quad 5555 | 5⁴ + (5,5,5,5) + (3,3) | 625 × 625 × 9 | 0.76× | ~80 s/iter | ~38 min |
| ret-quad 33 | 5⁴ + (3,3,3,3) + (3,3) | 625 × 81 × 9 | 0.098× | ~12 s/iter | ~8 min (= g5) |
| ret-quad 44 | 5⁴ + (3,3,3,3) + (4,4) | 625 × 81 × 16 | 0.17× | ~20 s/iter | ~12 min |
| ret-quad 55 | 5⁴ + (3,3,3,3) + (5,5) | 625 × 81 × 25 | 0.27× | ~30 s/iter | ~16 min |
| ret-quad 55-Lob7 | 5⁴ + (3,3,3,3) + (5,5)+Lob | 625 × 81 × 25 | 0.27× | ~30 s/iter | ~16 min |

**Notes on these projections:**
- Per-iter scaling is approximately linear in (N_state × n_state_quad × n_ret_quad) at the inner-Newton-bound regime (which is where the JAX retirement kernel sits). Small deviations from linearity come from the inner Newton's iteration count being state-dependent — bigger grids may push more cells into Newton-saturation, particularly along the y_1-tail (see `NEWTON_HISTOGRAM_AUDIT_2026-05-07.md` §"per-savings drill").
- Compile time is amortized **per distinct kernel shape**. The state-grid sweep recompiles 4× (one per cell-count). The state-quad and ret-quad sweeps share cell-count = 5⁴, so they recompile only on quad-shape changes — but a quad-shape change is enough to trigger a fresh compile. Budget ~60–90 s × 12 distinct shapes ≈ ~15 min compile overhead total.
- These are **rough** — first cell of any sweep is the load-bearing wall measurement. If the 5⁴/(3,3,3,3)/(3,3) baseline takes >25 min, halt and recheck.

**Total program estimate:** anchor (75) + state-grid (31) + state-quad (16+17+38; baseline shared with grid g5) + ret-quad (12+16+16; baseline shared with grid g5) + compile (~15) ≈ **~3.7 h compute ≈ ~$12 on 2× H100 SXM**. Comfortably inside the ~3–5 h budget.

---

## Recommended execution order

The dependencies are: every sweep needs a "safe-enough" baseline on the other two axes. The cleanest ordering is:

1. **Pre-flight: anchor run.** Solve the canonical-asymptote cell (7⁴/(3,5,3,5)+Lob7/(5,5)+Lob7) first as a quality gate. If this doesn't converge in 100 outer iters with `total_newton_failures==0`, the entire program is suspect — pause and diagnose the canonical config in inf-horizon mode before launching sweeps.
2. **Sweep A: state grid** (cells g3, g4, g5, g6, anchor-g7). Run cheapest first → most expensive. Once you see g4 vs g3 sup-norm < 1% and g5 vs g4 < 0.5%, you have early evidence of "stops at minimum tested" and can decide whether to skip g6.
3. **Sweep B: state quadrature** (cells 3333, 3535, 4444, 5555). Holds state-grid at g5 (the recommended "safe enough" — see §Sweep B baseline rationale below). Run order: 3333 first (cheapest; matches grid sweep g5 — **reuse that bundle**, don't re-solve). Then 3535 (the y_1 K-bump probe — load-bearing). Then 4444 (uniform refinement). Then 5555 only if 3535/4444 disagree with 3333 by more than the policy-noise floor.
4. **Sweep C: return quadrature** (cells 33, 44, 55, 55-Lob7). Holds state-grid at g5, state-quad at (3,3,3,3). Run order: 33 first (cheapest; reuse Sweep A's g5 bundle). Then 55-Lob7 (anchor-side bound). Then 44 and 55-no-Lob.

**Total wall (sequential):** ~3.7 h. If launched in parallel across the two H100s the per-iter is already cell-axis-sharded — no parallelism to gain at the program level. Run sequentially; one solve per Python process.

---

## Sweep A — State-grid density

**Question:** *How fast does the inf-horizon stationary policy converge as N_state grows along all four state axes uniformly?*

### Cells

| Tag | `state_grid_sizes` | N_state | What it tests | Wall |
|---|---|---|---|---|
| g3 | (3,3,3,3) | 81 | Lower bound — does anything sensible come out? | ~3 min |
| g4 | (4,4,4,4) | 256 | First serious cell — is this enough? | ~5 min |
| g5 | (5,5,5,5) | 625 | The "safe enough" cell that Sweeps B/C ride on | ~8 min |
| g6 | (6,6,6,6) | 1296 | Penultimate; flat curve here ⇒ done | ~15 min |
| g7 (anchor) | (7,7,7,7) + canonical quad | 2401 | Asymptote reference (canonical) | ~75 min |

All cells in this sweep hold `n_state_quad_nodes=(3,3,3,3)` and `n_ret_nodes_1d=(3,3)` — the cheapest defensible quad on both. This isolates state-grid effects only. The `g7` cell uses the **canonical quad** (3,5,3,5)+Lob/ (5,5)+Lob, not the cheap quad — it's not a fair g6 → g7 comparison; it's the asymptote reference for the cross-sweep verdict.

### Baseline rationale for the held-fixed quads

`(3,3,3,3)` state quad and `(3,3)` ret quad are the **smallest** defensible choices (< this and we lose polynomial exactness needed for the conditional VAR expectation). Holding both at the floor makes any state-grid divergence we see attributable to the grid, not to interactions with under-resolved quadrature. Sweeps B/C will tell us whether holding at the floor here was conservative enough; if Sweep B finds quad refinement matters at fixed g5, **revisit Sweep A** with the bumped quad to confirm the verdict isn't quad-confounded. (Pause point.)

### Cell config snippet

```python
disc_g4 = disc_base._replace(
    state_grid_sizes=(4, 4, 4, 4),
    n_state_quad_nodes=(3, 3, 3, 3),  # held at floor
    state_lobatto_Z=None,
    n_ret_nodes_1d=(3, 3),
    ret_lobatto_Z=None,
)
# g5: state_grid_sizes=(5,5,5,5); g6: (6,6,6,6); g3: (3,3,3,3)
```

### Comparison strategy

This is the **only** sweep where direct element-wise comparison fails — the policy arrays have different shapes across cells:
- g3: `(1, 81, 180)`
- g4: `(1, 256, 180)`
- g5: `(1, 625, 180)`
- g6: `(1, 1296, 180)`
- g7: `(1, 2401, 180)`

(Layout `(n_z, N_state, n_w)` with `N_state = prod(state_grid_sizes)`.)

**Strategy 1 (recommended): per-axis cross-section comparison at common state-grid coordinates.**

For each pair (g_k, g_k+1), identify the state-grid coordinates that **both grids share exactly**. Each axis is a Gauss-Hermite-like grid; `(3,3,3,3)` shares 1 point with `(5,5,5,5)` (the center, axis 0 = index 1 in the 3-grid = index 2 in the 5-grid), shares 0 axis points with `(4,4,4,4)`, and shares 1 point with `(7,7,7,7)`. So **odd-vs-odd grids share centers**; even-vs-odd share nothing. For odd-vs-odd, extract the (z, center_state, w) slice from each — both are `(1, 1, 180)` shaped. Compare element-wise on that slice.

**Strategy 2: interpolate finer onto coarser.** For each (g_k, g_ref) pair, interpolate the finer-grid policy onto the coarser-grid state coordinates via 4D linear interpolation in state-space (`scipy.interpolate.RegularGridInterpolator` over the per-axis state-grid coordinates exposed by `pc.state_bracket_grids`). Then element-wise compare. This is more flexible but introduces interpolation error; the spurious-divergence floor is roughly the local Lipschitz constant of the policy times the grid spacing.

**Strategy 3: simulation-based comparison.** Forward-simulate from each policy under a common shock path (e.g., `pc.state_bracket_grids` mid-state, ergodic distribution sampled), produce a 1D distribution of consumption / share / etc., compare distributions across cells via Wasserstein or sup-norm-on-quantiles. This integrates over the policy in a way that's robust to interpolation noise. **Recommended only if Strategies 1+2 give ambiguous answers** — costs ~1 extra hour of analysis time.

**Cross-axis per-axis profile:** for each cell, marginalize the policy along three of the four state axes (mean over those axes at each fixed value of the fourth) and plot the 1D profile vs. the kept axis. Four panels (one per state axis); one curve per cell. Visual overlap = converged. This handles mismatched grid resolutions naturally — the curves can be plotted on a common physical x-axis (rtb in std units, etc.).

### Pause points

- **g3 doesn't converge** (max_iter hit, `final_stopping_supnorm > 1e-5`, or NaN). Doesn't kill the program; it's the lower-bound stress test. Note in the verdict but proceed.
- **g7 (anchor) doesn't converge.** Stop the program. Either solver knobs (max_iter Newton, `step_damp_unconstrained`) need tuning at canonical scale, or the canonical config has a bug.
- **`total_newton_failures > 0` at any cell.** Surface the count + per-iter trajectory; this signals high-state-tail Newton saturation. With the failure-count wiring landing per `HANDOFF_NEWTON_FAILURE_COUNT_WIRING.md` and `cell_vmap_chunks=1`, this should be cheap to read.
- **Per-axis profiles diverge at one specific axis** (e.g., y_1 axis disagrees while dp/spr/rtb agree). That's actually a *finding*, not a problem — cross-link to Sweep B's K-bump result. Surface, don't editorialize.

### Verdict format

Three sub-verdicts, GREEN/YELLOW/RED:

1. **g4 vs g5 sup-norm divergence** (do they agree along shared centerline + per-axis profiles?) → if delta_C < 0.01, delta_S < 0.005, delta_B < 0.005 ⇒ GREEN: g4 converged.
2. **g5 vs g6 sup-norm** → if delta < same thresholds ⇒ GREEN: g5 converged.
3. **g6 vs g7 (canonical)** → ditto ⇒ GREEN: g6 = canonical.

The combined verdict tells the user **the smallest state-grid that's safe for inf-horizon**. The user expects "g5 enough" (mirroring the System I/II "stops early" pattern); if g4 is also enough, that's a stronger result.

---

## Sweep B — State-quadrature density

**Question:** *Does the K-bump on the y_1 axis (Smolyak-style asymmetric refinement) pay off in the inf-horizon retirement Bellman, the way it did at System II finite-horizon?*

### Cells

| Tag | `n_state_quad_nodes` | `state_lobatto_Z` | n_state_quad | What it tests | Wall |
|---|---|---|---|---|---|
| 3333 | (3,3,3,3) | None | 81 | Floor — same cell as Sweep A g5 (reuse) | (= g5: ~8 min) |
| 3535 | (3,5,3,5) | None | 225 | y_1 + spr K-bump (no Lobatto) — System II's load-bearing finding | ~16 min |
| 4444 | (4,4,4,4) | None | 256 | Uniform refinement — head-to-head with 3535 at near-equal node count | ~17 min |
| 5555 | (5,5,5,5) | None | 625 | Reference — top of the cheap-Lobatto-free family | ~38 min |

Optional 5th cell: `3535-Lob7` = (3,5,3,5) + state_lobatto_Z=(None,7.0,None,7.0) — the canonical-style state-quad with Lobatto tails. **Recommend skip** in the first pass: Lobatto's effect is a separate question, and the canonical anchor (Sweep A g7) already includes it. If Sweep B 3535-vs-3333 shows non-trivial divergence, run 3535-Lob7 as a follow-up to factor "K-bump effect" from "Lobatto-tail effect" — adds ~16 min.

### Baseline rationale

State grid: **g5 = (5,5,5,5)** (recommended). Reasoning:
- g4 might not be converged (Sweep A's job to confirm); riding state-quad analysis on a non-converged grid muddies the verdict.
- g6 is more expensive without obvious benefit if g5 is already converged.
- g5 also matches the natural midpoint between "cheap floor" and "canonical asymptote."

If Sweep A confirms g4 ⇒ converged (GREEN), revisit: re-running Sweep B at g4 saves ~50% wall and gives a tighter direct comparison. **Don't pre-commit to g5; let Sweep A's first 2 cells inform.**

Ret quad: **(3,3) = floor**. Same reasoning as Sweep A: minimize confound from the orthogonal axis.

### Cell config snippet

```python
disc_3535 = disc_base._replace(
    state_grid_sizes=(5, 5, 5, 5),
    n_state_quad_nodes=(3, 5, 3, 5),     # K-bump on spr (axis 1) and y_1 (axis 3)
    state_lobatto_Z=None,                # no Lobatto here — that's a separate cell
    n_ret_nodes_1d=(3, 3),
    ret_lobatto_Z=None,
)
# 4444: n_state_quad_nodes=(4,4,4,4); 5555: (5,5,5,5); 3333: (3,3,3,3) (= g5)
```

### Comparison strategy

All cells in this sweep have **identical policy-array shape** `(1, 625, 180)` (state grid is fixed at g5; only the quadrature varies → only `n_state_quad` changes; this affects compute, not output shape). **Direct element-wise sup-norm comparison; no interpolation.** Same simplicity as the System II quad sweep.

**Pairwise table** — pick `3333` as the cheapest baseline; the user's question is "is this enough?":

| Coarse | vs Reference | Effect isolated |
|---|---|---|
| 3333 | 3535 | y_1+spr K-bump value (does asymmetric refinement help?) |
| 3333 | 4444 | uniform refinement value |
| 3333 | 5555 | full asymptote-side refinement |
| 3535 | 4444 | **K-bump vs uniform** (the load-bearing economic comparison; same total ~225 vs 256 nodes) |
| 3535 | 5555 | does K-bump capture most of full refinement? |
| 4444 | 5555 | does uniform stop early? |

The key comparison is **3535 vs 4444 vs 5555** at near-equal node count (225 / 256 / 625):
- If 3535 ≈ 5555 < 4444 (in divergence): K-bump captures more of the asymptote per node spent → adopt (3,5,3,5) for canonical inf-horizon.
- If 4444 ≈ 5555 < 3535: uniform refinement is preferable; Smolyak rec doesn't apply at System IV inf-horizon.
- If 3535 ≈ 4444 ≈ 5555: all three are converged; pick cheapest = 3535 (225 < 256 < 625).

### Per-axis decomposition

For each pair, decompose divergence:
- **Per-state axis profile**: marginalize over (z, w); 625 numbers reshaped as 5×5×5×5; max-over-three-axes per axis. Identifies whether divergence is along the y_1 axis (which K-bump targets) vs spr axis (also K-bumped) vs dp/rtb axes (not bumped).
- **Per-wealth**: 180 numbers. Low-wealth cells often have stronger Newton saturation; expect divergence concentrated there.

### Pause points

- **3333 baseline is bizarrely far from 5555** (sup_C > 0.05, sup_S > 0.02): floor quad isn't capturing the conditional VAR expectation well at the y_1 tail. Consider re-running Sweep A at the bumped quad to refactor what we attributed to grid vs. quad.
- **3535 LARGER divergence than 4444 vs 5555** — Smolyak rec is anti-helpful at inf-horizon scale, contrary to System II finding. Surface as a finding; don't editorialize. Inflates the importance of running 3535-Lob7 as a follow-up.

### Verdict format

Three sub-verdicts:

1. **Floor adequacy**: GREEN if 3333 ≈ 5555 (≤ thresholds in §Sweep A); YELLOW if 3535 needed; RED if 5555 needed.
2. **K-bump value (load-bearing)**: GREEN (K-bump cheaper than uniform at equal accuracy); YELLOW (equivalent — pick by cost); RED (uniform better).
3. **Lobatto-tail effect** (only if 3535-Lob7 cell is run): GREEN if 3535-Lob7 ≈ 3535 (Lobatto unnecessary at inf-horizon); RED if 3535-Lob7 ≈ canonical anchor < 3535 (Lobatto materially tightens).

Combined recommendation: state-quad for inf-horizon canonical = (\<smallest "stops here"\>, with/without Lobatto).

---

## Sweep C — Return-quadrature density

**Question:** *Does the (xr, xb) return shock conditional expectation need 5×5 + Lobatto (canonical), or is 3×3 sufficient for retirement steady-state policies?*

### Cells

| Tag | `n_ret_nodes_1d` | `ret_lobatto_Z` | n_ret_quad | What it tests | Wall |
|---|---|---|---|---|---|
| rq33 | (3,3) | None | 9 | Floor — same cell as Sweep A g5 (reuse) | (= g5: ~8 min) |
| rq44 | (4,4) | None | 16 | First refinement | ~12 min |
| rq55 | (5,5) | None | 25 | Canonical density without Lobatto | ~16 min |
| rq55-Lob7 | (5,5) | (7.0, 7.0) | 25 | Canonical (matches `_canonical.py`) | ~16 min |

The Lobatto comparison (rq55 vs rq55-Lob7) tests whether the bond-tail discrete-free-lunch correction (per `docs/handoff/HANDOFF_VERIFY_RETURN_QUADRATURE.md` and the Lobatto handoffs) actually pays off in retirement, where there's no working-age bond-tail wealth-explosion incentive. **Plausibly Lobatto matters less here.** Worth confirming.

### Baseline rationale

State grid: **g5** (same as Sweep B). State quad: **(3,3,3,3)** (floor, same as Sweep A baseline). Same reasoning — minimize confounds from orthogonal axes.

### Cell config snippet

```python
disc_rq44 = disc_base._replace(
    state_grid_sizes=(5, 5, 5, 5),
    n_state_quad_nodes=(3, 3, 3, 3),
    state_lobatto_Z=None,
    n_ret_nodes_1d=(4, 4),
    ret_lobatto_Z=None,
)
# rq55: n_ret_nodes_1d=(5,5); rq55-Lob7: + ret_lobatto_Z=(7.0,7.0)
```

### Comparison strategy

All cells share shape `(1, 625, 180)`. **Direct element-wise sup-norm.** Same simplicity as Sweep B.

**Pairwise table** — `rq33` as cheapest baseline:

| Coarse | vs Reference | Effect isolated |
|---|---|---|
| rq33 | rq44 | first refinement |
| rq33 | rq55 | non-Lobatto asymptote |
| rq33 | rq55-Lob7 | full canonical asymptote |
| rq55 | rq55-Lob7 | **Lobatto-tail value** (does Z=7 prescribe-tail correction matter at inf-horizon?) |
| rq44 | rq55 | does (4,4) capture most of (5,5)? |

The key comparison is **rq55 vs rq55-Lob7** at equal node count: pure Lobatto-tail effect.
- If rq55 ≈ rq55-Lob7 (≤ thresholds): Lobatto doesn't help at inf-horizon retirement (no income shock interaction); save the cost.
- If rq55 < rq55-Lob7 in divergence vs canonical: Lobatto still load-bearing at inf-horizon — the bond-tail correction matters even without working-age dynamics.

### Per-axis decomposition

- **Per-state**: 625 numbers reshaped 5×5×5×5; max-over-three-axes per axis. Look for divergence concentrating at high-rtb cells (rtb is the inflation-surprise axis; bond tails are most sensitive there).
- **Per-wealth**: 180 numbers. High-wealth cells (above the SCF-median region) typically have larger leverage and so larger ret-quad sensitivity.

### Pause points

- **rq33 vs rq55 sup-norm > 0.02 in alpha_s**: floor ret-quad isn't capturing the stock-tail well at high-leverage cells. Cross-link to Sweep B; if state-quad refinement also moved alpha_s by similar amounts, the two effects may be aliased.
- **rq55 farther from canonical anchor (Sweep A g7) than rq55-Lob7 is**: Lobatto-tail correction is doing real work; canonical Z=7 setting is justified.
- **rq44 farther from rq55 than rq55-Lob7 is**: (4,4) is below floor adequacy; jump straight to (5,5) for canonical.

### Verdict format

Two sub-verdicts:

1. **Floor adequacy**: GREEN if rq33 ≈ rq55 (≤ thresholds); YELLOW if rq44 needed; RED if rq55 needed.
2. **Lobatto value at inf-horizon**: GREEN if rq55 ≈ rq55-Lob7 (Lobatto doesn't help here; save cost on inf-horizon-only ablations); YELLOW if Lobatto reduces divergence by ~50%; RED if Lobatto fully load-bearing.

Combined recommendation: ret-quad for inf-horizon canonical = (\<smallest "stops here"\>, with/without Lobatto).

---

## Diagnostic signals to log per cell

For each of the 12 cells, the bundle's `diagnostics.pkl` should be inspected for:

| Field | What to read | Healthy range |
|---|---|---|
| `converged` | True/False | True (else flag) |
| `n_iter` | outer iters used | 20–50 (typically) |
| `final_stopping_supnorm` | last (xi, share) sup-norm | < tol = 1e-5 |
| `policy_supnorm_history` | trajectory shape | monotone-ish decrease |
| `total_newton_failures` | per `HANDOFF_NEWTON_FAILURE_COUNT_WIRING.md` | 0 (else flag and read per-iter) |
| `newton_iter_histogram.p99` | per-savings Newton iter distribution | well below max_iter=100 |
| `newton_iter_histogram.max` | tail | flag if = max_iter (saturation) |
| `backtrack_iter_histogram.p99` | line-search aggressiveness | should be small (1–3 typical) |
| `max_z_slice_diff_*` | inf-horizon-specific (z-invariance) | ~0 (n_z=1 makes this trivial; sanity) |
| `max_xi_spread_across_w`, `max_share_spread_across_w` | wealth-homogeneity proxies | should be small at the fixed point |
| `stability_proxy` | β·E[exp((1-γ)r_p)] bound | < 1 (contraction; else policy unstable) |

Use these to triage suspicious cells before doing pairwise policy comparison. A cell with `total_newton_failures > 0` or `stability_proxy ≥ 1` should be flagged in the verdict, not buried in the pairwise table.

---

## Cross-sweep synthesis (optional §6)

Once all three sweeps are analyzed individually, produce a one-page synthesis:

- **Combined recommended canonical inf-horizon discretization** (state grid, state quad, ret quad).
- **Wall-time savings** vs running every inf-horizon ablation at canonical: the smallest "stops here" cell × the smallest quad × the smallest ret-quad gives the cost of a representative inf-horizon ablation. Compare to running each at canonical g7+(3,5,3,5)+Lob+(5,5)+Lob.
- **Cross-link to System I/II findings**: do the inf-horizon "stops early" patterns mirror what System I (n_z=10 adequate, (3,4) eta-eps adequate) and System II ((3,5) y_1 K-bump adequate) showed? If yes → strong evidence the discretization can be cheap **across the entire ablation set**, including finite-horizon. If no → the inf-horizon-only canonical may differ from finite-horizon canonical; document the mapping.
- **What this implies for the production lifecycle solve**: if inf-horizon needs g5, lifecycle likely needs ≥ g5 too (working ages add complexity, not subtract). If inf-horizon needs g4, that's a lower bound on lifecycle but lifecycle could still need more.

---

## Out of scope

- **Solver-side changes.** No edits to `lifecycle/solver.py`, `lifecycle/inf_horizon_solver.py`, or precompute. The `HANDOFF_NEWTON_FAILURE_COUNT_WIRING.md` change is assumed landed.
- **Re-running canonical lifecycle sweeps.** This program is inf-horizon-only. Cross-system comparisons are §6 synthesis; they're cross-link interpretation, not new solves.
- **Sim-based EE comparison.** Stay grid-based unless Strategies 1 + 2 in Sweep A give ambiguous answers.
- **Changes to the canonical config.** This program's verdict *informs* a future canonical update, but the update itself is a separate handoff.
- **Lobatto-tail Z-value sensitivity** (Z=4 vs Z=7 etc.). One-shot use Z=7 (canonical) or none. A Z-sensitivity sweep is a separate program.

---

## Implementation checklist (for the implementing agent)

For each sweep, the runner script mirrors `verify/benchmark_inf_horizon.py` almost exactly, parameterized by sweep cell:

- [ ] **`verify/inf_horizon_sweep_state_grid.py`** — loops over the 4 state-grid cells, calls `run_infinite_horizon_solver`, saves bundles via `save_policy_bundle`. Optional S3 upload (env-gated).
- [ ] **`verify/inf_horizon_sweep_state_quad.py`** — same, for 4 state-quad cells.
- [ ] **`verify/inf_horizon_sweep_ret_quad.py`** — same, for 4 ret-quad cells.
- [ ] **`verify/inf_horizon_anchor_canonical.py`** — single-cell runner for the canonical-asymptote anchor. (Or fold into one of the above as cell #5.)
- [ ] **`scripts/analysis/inf_horizon_state_grid_convergence.py`** — analysis script, mirrors `system_ii_quad_convergence.py` plus the cross-shape interpolation/cross-section helper.
- [ ] **`scripts/analysis/inf_horizon_state_quad_convergence.py`** — direct copy of `system_ii_quad_convergence.py` with relabeled bundle paths.
- [ ] **`scripts/analysis/inf_horizon_ret_quad_convergence.py`** — same template as state-quad.
- [ ] **`docs/scans/INF_HORIZON_STATE_GRID_2026-MM-DD.md`**, **`...STATE_QUAD...`**, **`...RET_QUAD...`** — three short reports with verdicts, plus a one-page synthesis.
- [ ] Pre-flight check: confirm `total_newton_failures` wiring landed (`grep "total_newton_failures = 0" lifecycle/`). If still hardcoded, pause and resolve `HANDOFF_NEWTON_FAILURE_COUNT_WIRING.md` first.

**Don't reinvent.** `verify/benchmark_inf_horizon.py` is the runner template; `scripts/analysis/system_ii_quad_convergence.py` is the analysis template (load_policy_bundle + same-shape pairwise + per-axis decomposition + per-cell line plots + verdict). The state-grid sweep needs an extra cross-shape comparison helper (Strategy 1 + Strategy 2 from Sweep A); Sweeps B and C use the same-shape template directly.

---

## Why this matters

Inf-horizon is the **calibration sandbox** for the whole solver. Every discretization knob has the same direction in inf-horizon as in lifecycle (state grid, state quad, ret quad all enter the retirement Bellman identically), but inf-horizon is ~100× cheaper per cell. **A clean inf-horizon sensitivity result transfers directly** to lifecycle: any axis that matters at inf-horizon also matters at lifecycle (because lifecycle's retirement step *is* the inf-horizon Bellman applied 32 times); any axis that doesn't matter at inf-horizon almost certainly doesn't matter at lifecycle either (income shocks add a separate quadrature, but the state-side sensitivity is preserved).

This program is the second leg of a three-leg calibration result:
- **System I × n_z**: stops at n_z=10 (already done).
- **System I × (n_eta, n_eps)**: stops at (3,4) (already done).
- **Inf-horizon × (state grid, state quad, ret quad)**: this program. Tells us where to pin canonical for the **state-side** dimensions.

Combined: every future ablation run uses minimal-resolution discretization with full defensibility from sensitivity data. The savings compound across Systems II, III, IV at every (n_z, eta-eps, state, ret) combination — typically 3–7× per axis, **15–50×** combined across the canonical ablation set. The thesis methodology section becomes "we picked these resolutions from data, not arbitrarily."
