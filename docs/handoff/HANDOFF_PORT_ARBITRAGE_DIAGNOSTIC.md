# Handoff: Port Arbitrage / Invalid-Cells Diagnostic from Numba `main`

**Branch:** `jax-rewrite`
**Status when this doc was written:** the JAX branch has **no spurious-arbitrage test and no invalid-cells sanity check.** The Numba `main` branch had `scripts/diagnostics/_diag_arbitrage_quadsweep.py` and `_diag_invalid_cells.py` — both deleted in JAX-rewrite handoff 1 along with the rest of `scripts/diagnostics/`. The `lifecycle/quadrature_with_tails.py` Lobatto infrastructure is the *fix* for spurious arbitrage at extreme returns; we currently have the fix's plumbing but no way to detect whether arbitrage exists in a given bundle.

**Why we need this:** Lobatto adds ±Zσ tail nodes specifically to prevent quadrature-induced spurious arbitrages (cells where the discrete integrator finds a "free lunch" portfolio that doesn't actually exist in continuous returns). Without a test, we don't know if a given bundle has arbitrage. With a test, we can decide whether Lobatto needs to be turned on for a given config — instead of always-on (wasteful) or always-off (dangerous).

**Companion to** the EE diagnostic (`verify/ee_residuals.py`) ported recently. Same structural template; same code-archaeology source on `main`.

**Effort:** ~1-1.5 days (port + adapt + integrate). Parallels the EE port closely.

---

## 1. Goal

Produce **two new artifacts** plus an integration touch-up:

1. **`verify/arbitrage.py`** — loads a saved policy bundle, evaluates per-cell arbitrage gap (the integrator's apparent expected excess return at any feasible portfolio that the model says shouldn't exist), and reports per-age statistics + a JSON summary.
2. **`verify/invalid_cells.py`** — loads a saved policy bundle and flags cells that are NaN, have extreme α (e.g., `|α_s| > 20` or `|α_b| > 50`), or have implausible `α_s + α_b + α_bill ≠ 1` violations beyond float tolerance.
3. **Integration into `docs/agents/PREFLIGHT_AGENT.md`** — wire these checks (plus the existing `verify/ee_residuals.py`) into the preflight protocol so any bundle gets characterised before it goes to downstream analysis.

**Naming convention:** the EE port chose `verify/ee_residuals.py`. Mirror that — `verify/arbitrage.py`, `verify/invalid_cells.py`. (Reasonable to combine into a single `verify_bundle_health.py` if the agent prefers, but keep them separable so a user can run just one if they want.)

---

## 2. Scope

### In scope

- Port the **structure** of `scripts/diagnostics/_diag_arbitrage_quadsweep.py` and `_diag_invalid_cells.py` from `main` (via `git show main:scripts/diagnostics/_diag_arbitrage_quadsweep.py`).
- Adapt for the JAX rewrite's economic features:
  - **4-D state** (was 3-D on main)
  - **CCV log returns** with continuous-rebalancing variance correction (was arithmetic on main, possibly)
  - **rtb-as-state**: rtb realisation comes from the next-period state vector at `rtb_index_in_state`, not from a return draw
  - **16 corners** (multilinear over 4 state axes), was 8 on main
- Reuse the JAX solver's existing CCV machinery: `_ccv_log_return_and_grad`, `retirement_foc_jac_ccv`, `working_foc_jac_ccv`, the precompute infrastructure. **Don't reimplement** the return formulas.
- Output: per-age JSON summary into the bundle directory, mirroring `ee_residuals.json` format.
- Update `docs/agents/PREFLIGHT_AGENT.md` to include arbitrage + invalid-cells checks alongside the existing flow. **This is a hard deliverable — the protocol is what makes the diagnostic actually get run before launches, instead of being optional.**

### Out of scope

- **Simulator-path-based variants** (`_diag_simpath_worst_cells.py` from `main`). Depends on simulator being CCV-correct, which it now is, but adds another diagnostic axis. Defer.
- **Visualisation / plotting.** JSON output only.
- **Auto-running from `verify/benchmark_bundle.py`.** Add later via a separate handoff that wires post-solve diagnostics together.
- **Modifying `lifecycle/quadrature_with_tails.py`** or any solver-side code. Pure read-only diagnostic.
- **The arbitrage *fix* (turning on Lobatto)** — this handoff produces the *test*. If the test reports arbitrage, that's the user's signal to rerun with Lobatto, separately tracked in `HANDOFF_EVAL_LOBATTO_PROPAGATION.md`.

### Hard constraints

- **Use existing CCV machinery.** Don't re-derive R_p formulas. Import and call solver kernels.
- **No solver-side code changes.** Pure diagnostic; read-only on the bundle.
- **Memory-bounded.** Use the chunk-outside-JIT pattern that the EE diagnostic already established (`_chunked_vmap_two` from `verify/ee_residuals.py`). Don't repeat the OOM lesson.
- **Adapted for 4-D state and CCV.** The `main` branch's diagnostic was 3-D + arithmetic returns; this port must use 4-D + CCV log returns end-to-end.

---

## 3. The arbitrage-gap math

### What "spurious arbitrage" means here

The solver's quadrature integrates over future returns to compute expected utility. A finite quadrature (Gauss-Hermite or otherwise) introduces some error. **If that error is asymmetric across the support — e.g., the upper tail is approximated more accurately than the lower tail — the discrete integrator can imply a "no-loss-possible" portfolio that doesn't exist in the underlying lognormal model.** This is "spurious arbitrage."

Concretely: at some cell `(z, state, w)`, evaluate `E[R_p(α)]` and `min_quad R_p(α)` across the quadrature nodes for some test α (e.g., the solved policy's α, or a sweep of αs). If `min_quad R_p > R_bill` for any α the model thinks is feasible, that's a spurious arbitrage — the discrete integrator says "you can guarantee a return above the bill rate" when the continuous model says no.

### The diagnostic computation

For each cell `(t, z_idx, state_idx, w_idx)` in the saved bundle:

1. Load the solved policy `(c, α_s, α_b)` for this cell. Compute `α_bill = 1 - α_s - α_b`. Skip cells where `c` is NaN or `α_s + α_b + α_bill` deviates from 1 by more than `1e-9`.
2. For each quadrature node `(state_quad_node, return_quad_node)`:
   - Read `rtb_{t+1}` from the bracketed next-period state vector at `rtb_index_in_state` (using `_build_step_log_returns` / `_build_step_state_brackets` from `solver.py`)
   - Compute `R_p` via CCV log-return formula (use `_ccv_log_return_and_grad` from solver, NOT arithmetic combination)
3. Aggregate per-cell:
   - `R_p_mean = Σ weight × R_p` (expected return at solved α)
   - `R_p_min = min_node R_p`
   - `R_p_max = max_node R_p`
   - `R_bill_mean = exp(rtb)` averaged over quad
4. **Arbitrage gap:** `gap = R_p_min - R_bill_mean`. If `gap > 0` for the solved α, that cell exhibits spurious arbitrage at the solved policy.

**Stronger test (from `_diag_arbitrage_quadsweep` on main):** sweep over α at each cell — not just the solved one. If `max_α (min_node R_p(α)) - R_bill_mean > tolerance`, the *quadrature* admits arbitrage even if the *solver* didn't choose to exploit it. The α-sweep version is what the Numba diagnostic does; it's the more meaningful test.

For initial port: compute the arbitrage gap **at the solved policy** for each cell. Per-age stats: max gap, p99 gap, fraction of cells with gap > 0. If time permits, add the α-sweep variant; if not, leave as a TODO in the script.

### Tolerance and threshold

- **Pass:** max arbitrage gap < `1e-6` (within float roundoff of zero; no cells exhibit spurious arbitrage)
- **Concerning:** max in `[1e-6, 1e-4]`, fraction-above-1e-6 > 1% — quadrature is borderline; consider Lobatto
- **Fail:** max > `1e-4` or fraction-above-1e-6 > 5% — clear quadrature failure; turn on Lobatto and re-solve before downstream analysis

### What invalid-cells checks

`verify/invalid_cells.py` is simpler. For each cell:

- `c` finite? `α_s, α_b` finite?
- `c > min_consumption`? If `c <= tiny_savings`, this is the documented tiny-savings fallback — flag separately.
- `|α_s|, |α_b|` within reasonable bounds (e.g., < 20 each)?
- `α_s + α_b + α_bill ≈ 1` (where `α_bill = 1 - α_s - α_b`, so this is automatically true *if* α_s and α_b are finite)?
- For retirement-only bundles: only check ages `>= youngest_age_to_solve`. Skip working-age cells which are NaN by design.

Per-age count of cells in each category. JSON output. Pass = zero NaN, zero extreme-α.

---

## 4. Implementation outline

### 4.1 `verify/arbitrage.py` — structural template

```python
"""verify/arbitrage.py — Spurious-arbitrage check for saved policy bundles.

Loads a bundle, evaluates per-cell arbitrage gap = min_quad R_p(α) - E[R_bill]
at the solved (α_s, α_b). Reports per-age stats and saves JSON summary alongside
the bundle.

Pass criterion: max gap < 1e-6 globally, no NaN, fraction-above-1e-6 < 1%.

Adapted for the JAX rewrite: 4-D state, CCV log returns, rtb-as-state.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import jax, jax.numpy as jnp
from jax import jit, vmap, lax

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.solver import (
    _ccv_log_return_and_grad,
    _build_step_log_returns,
    _build_step_state_brackets,
    _precompute_per_is_tensors,
    _pc_to_jnp,
    DELTA_BEQUEST,
)

# Use the same chunk-outside-JIT pattern as verify/ee_residuals.py:
# - `_chunked_arbitrage_runner` factored helper with for-loop in Python land
# - Each chunk is its own @jit'd computation
# - .block_until_ready() between chunks if needed
```

Reuse `_build_step_log_returns` and `_build_step_state_brackets` to avoid re-deriving the rtb-from-state-vector logic. The `verify/ee_residuals.py` agent already established the imports and pattern — copy the bundle-loading + precompute-rebuild scaffolding.

### 4.2 Memory pattern — copy from `verify/ee_residuals.py`

The EE diagnostic agent had the same memory issue tonight (chunks-inside-JIT bug → 68 GB allocation). After fix, it uses Python-level chunk loop with per-chunk `@jit`. **Mirror that exactly.** Don't repeat the lesson.

```python
# Pseudocode mirroring _chunked_vmap_two from verify/ee_residuals.py
@jit
def per_chunk_arbitrage(z_chunk, is_chunk, C_t, S_t, B_t, ...):
    def per_cell(z_idx, i_s):
        # Get policy at this cell, evaluate arbitrage gap at all quad nodes
        ...
        return cell_gap, cell_min_R_p, cell_R_bill_mean
    return vmap(per_cell)(z_chunk, is_chunk)

def per_age(...):
    # Python loop over chunks; each iteration is an independent JIT call
    chunk_results = []
    for i in range(n_chunks):
        chunk_results.append(per_chunk_arbitrage(...))
    return concatenate(chunk_results)[:n_cells]
```

### 4.3 `verify/invalid_cells.py` — much simpler

Doesn't need precompute or quadrature. Pure NumPy on the loaded `(C, S, B)` arrays:

```python
import numpy as np
def check(C, S, B, sc):
    # C, S, B shape (n_age, n_z, N_state, n_w)
    nan_mask = ~np.isfinite(C) | ~np.isfinite(S) | ~np.isfinite(B)
    extreme_alpha = (np.abs(S) > 20) | (np.abs(B) > 50)
    tiny_consumption = (C <= sc.min_consumption) & ~nan_mask  # excluded design fallback
    # per-age counts, return dict
```

Whole script ~80 lines. JSON output, pass = zero anomalies (excluding tiny-savings fallback by design).

### 4.4 Integration: update `docs/agents/PREFLIGHT_AGENT.md`

After implementation, update the preflight protocol to include:

```markdown
## Bundle health checks (run after every saved bundle, before downstream analysis)

Run all three diagnostics in sequence:

1. `python verify/invalid_cells.py <bundle-name>` — must pass (zero NaN, zero extreme-α)
2. `python verify/arbitrage.py <bundle-name>` — must pass (max gap < 1e-6)
3. `python verify/ee_residuals.py <bundle-name> --use-relative` — must pass (max < 1e-2)

If any fail: investigate before treating the bundle as valid. Do NOT skip; do NOT
proceed to plotting or simulation analysis with a non-passing bundle.

Combined wall: 5-15 min on laptop. Free.
```

The exact placement in the preflight doc is the agent's call — add it as its own section, or fold into an existing "post-solve checks" section if one exists. **The hard requirement is that the three diagnostics are documented as a unified pre-launch (or post-bundle) health check, not as one-off scripts.**

---

## 5. Verification gates

1. **`verify/invalid_cells.py`** runs cleanly on tonight's 5⁴ bundle. Reports zero anomalies (we already know `total_newton_failures=0` and policies looked sane).
2. **`verify/arbitrage.py`** runs cleanly on tonight's 5⁴ bundle. Reports per-age max gap. **Initial expectation is uncertain** — Lobatto is OFF in tonight's bundle, so spurious arbitrage might exist. The point is to *measure* it.
3. **Both scripts produce JSON output** in the bundle dir, mirror format of `ee_residuals.json`.
4. **`docs/agents/PREFLIGHT_AGENT.md`** updated with the three-step health-check section. Cross-references to all three scripts.
5. **Memory test** at production scale: each script processes the full 5⁴ bundle without OOM on the laptop (use the chunk-outside-JIT pattern).

---

## 6. Why this matters

Without this diagnostic:
- We don't know if tonight's 5⁴ bundle has spurious arbitrage anywhere
- The decision "should we turn Lobatto on for the next 7⁴ run" is a guess
- Any thesis result derived from policies with spurious arbitrage is silently biased
- We have no automated way to catch the issue before downstream analysis

With it:
- Every bundle gets a pass/fail signal on quadrature integrity
- Lobatto-on becomes data-driven (turn on if and only if gap > 1e-6)
- Preflight protocol catches issues before paid downstream work

---

## 7. Implementation checklist

- [ ] Read `git show main:scripts/diagnostics/_diag_arbitrage_quadsweep.py` for structure
- [ ] Read `git show main:scripts/diagnostics/_diag_invalid_cells.py` for structure
- [ ] Read `verify/ee_residuals.py` (current branch, post chunk-fix) for the chunk-outside-JIT pattern + precompute rebuild scaffolding
- [ ] Implement `verify/arbitrage.py` per §4.1 + §4.2
- [ ] Implement `verify/invalid_cells.py` per §4.3
- [ ] Test both on tonight's 5⁴ bundle (`saved_runs/system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark/`)
- [ ] Confirm both produce `<bundle>/arbitrage.json` and `<bundle>/invalid_cells.json` in the same format style as `ee_residuals.json`
- [ ] Update `docs/agents/PREFLIGHT_AGENT.md` with the unified health-check section per §4.4
- [ ] Single commit with all of the above:

```
diagnostics: port arbitrage + invalid-cells checks from main; preflight integration

Reintroduces the spurious-arbitrage and invalid-cells diagnostics that
existed on Numba main (scripts/diagnostics/_diag_arbitrage_quadsweep.py,
_diag_invalid_cells.py) and were deleted in JAX-rewrite handoff 1.

Adapted for the rtb-as-state migration: 4-D state grid, CCV log returns
with variance correction, rtb realisation read from next-period state
vector. Uses solver-internal CCV kernels (_ccv_log_return_and_grad,
_build_step_log_returns) so residuals are graded against the same math
the solver used.

Memory pattern: chunk-outside-JIT (mirrors verify/ee_residuals.py's
post-fix pattern). Per-chunk peak HBM bounded by user-chosen K.

Outputs:
- verify/arbitrage.py + <bundle>/arbitrage.json
- verify/invalid_cells.py + <bundle>/invalid_cells.json

Updated docs/agents/PREFLIGHT_AGENT.md to require all three bundle-
health checks (these two + verify/ee_residuals.py) before any downstream
analysis on a saved bundle. Preflight is the gate that makes the
diagnostics actually get run.

Verified on tonight's 5^4 retirement-only bundle. [Pass/concerning/fail
results depending on whether Lobatto-off introduces spurious arbitrage
at this config.]
```

---

## 8. Edge cases / gotchas

### 8.1 The α-sweep variant is more thorough but optional

The `main` branch's `_diag_arbitrage_quadsweep.py` swept α over a grid (e.g., `α_s ∈ {-2, -1, 0, 1, 2}` × `α_b` similarly) and asked "for ANY of these α, does the quadrature admit arbitrage?" That's a stronger test than just the solved α. **Initial port: solved α only. Add α-sweep as a follow-up if time permits or if first run shows clean.** Document the deferral in the commit message.

### 8.2 Bundle's `youngest_age_to_solve` skip

If the bundle is retirement-only (`youngest_age_to_solve=67`), only ages ≥ 67 have valid policies. The script must skip earlier ages (they're NaN). `verify/ee_residuals.py` already does this — copy the pattern.

### 8.3 No simulator dependence

This handoff explicitly does NOT depend on the simulator. Path-based diagnostics are deferred. The grid-based check is sufficient and avoids the simulator-CCV-fix dependency entirely.

### 8.4 Lobatto-off caveat for tonight's bundle

Tonight's bundle was solved with `ret_lobatto_Z = state_lobatto_Z = None`. **This is precisely the case where spurious arbitrage is most likely.** If `verify/arbitrage.py` reports gap > 0 for some cells, that's not a bug in the diagnostic — it's the bundle revealing that Lobatto should have been on. Document the result, don't try to "fix" it via the diagnostic.

### 8.5 Memory at 4-D + n_z=11 + n_state_quad=36

The arbitrage diagnostic's per-cell working set is similar to the EE diagnostic's (both evaluate FOC-style quad sums). Use the same chunk size logic — `_build_padded_cell_indices` from `verify/ee_residuals.py` with default `chunk_size=2048` (or whatever the EE agent landed on after their fix).

---

## 9. Ordering against the queue

When the agent picks this up, the EE port (`verify/ee_residuals.py`) and chunking fix should already be done — both provide the working pattern this port mirrors. Don't start before:

- ✅ EE agent's chunk-outside-JIT fix in `verify/ee_residuals.py` is committed
- ✅ Float agent's solver-side `_chunked_vmap_cells` fix is committed (so `lifecycle/solver.py` is in known-good state for the diagnostic to import from)

After both are in: this handoff can run in parallel with anything else (it doesn't touch `lifecycle/solver.py` or any in-flight code).

---

## 10. Why bundle this with preflight integration

A diagnostic that's not in the preflight protocol is a script someone might forget to run. **Bundling implementation + preflight integration into one commit means the next person who launches a bundle has the protocol nudge them toward the check.** No extra cognitive load on the user; the system enforces health-checking by being integrated.

`docs/agents/PREFLIGHT_AGENT.md` is the right home because preflight is consumed by automated agents (the one that fires off launches) and humans (the user about to commit money to a GPU run). Making it the canonical pre-launch checklist is the lever that keeps these diagnostics actually used.
