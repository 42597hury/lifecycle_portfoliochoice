# Handoff: Bug & Correctness Scan (Report-Only)

**Branch:** `jax-rewrite`
**Mode:** **REPORT ONLY.** Do not edit any code. Multiple other agents are editing the codebase simultaneously; introducing parallel edits will cause merge conflicts.

**Output target:** Create `docs/scans/BUG_SCAN_2026-05-06.md`. The user reviews and dispatches fixes as separate work.

**Scope:** Correctness, robustness, crash modes, silent-wrong-result risks. **Not** performance (separate handoff).

**Time budget:** 3-4 hours focused. The bug-scan is broader than the perf scan because the consequences of a correctness bug are higher than a 10% slowdown.

---

## 1. Goal

The lifecycle solver is about to run on GPU at scale, then will be extended (via the rtb-as-state agent) to a 4-D state. **Find anything that could:**

- Crash the solver mid-run on GPU (silent OOM, kernel error, infinite hang).
- Produce wrong policies without crashing (silent NaN propagation, wrong axis order, off-by-one).
- Leave the user with a corrupted bundle they don't notice.
- Break under the not-yet-implemented 4-D state extension because of a hardcoded 3-D assumption that the rtb-as-state handoff missed.

The work is constrained: a fixed checklist (§3) of specific suspect bug classes at specific locations. Each item has yes/no findings.

---

## 2. Scope

### In scope

- `lifecycle/solver.py` — primary focus, full file.
- `lifecycle/precompute.py` — full file (state grid + Cholesky + quadrature builders are bug-prone).
- `lifecycle/__init__.py` — cache + runtime check.
- `lifecycle/policy_io.py` — bundle save/load.
- `lifecycle/_compile_cache_sync.py` — S3 cache helpers.
- `run_lifecycle_solver` orchestrator and resume-from-checkpoint path.
- `scripts/gpu_run.sh`, `scripts/download_bundle.sh` — wrapper scripts (shell bug-scan).

### Out of scope

- `verify_*.py` runner scripts (configs, not the engine).
- The Newton math itself — assume the FOC formulas are correct (separate verification handoff exists).
- The VAR construction in `lifecycle/var.py` — being actively edited by the rtb-as-state agent. Don't touch.
- `lifecycle/diagnostics.py`, `simulation.py`, `inf_horizon_solver.py` — out of solver hot path; flag anything obvious but don't dive deep.

### Hard constraints

- **No code edits.** Findings go in the report only.
- **Do not run the solver.** Static read of code only.
- Do not run smoke or any tests.

---

## 3. Checklist (enumerated)

For each item: **YES** (bug or risk found), **NO** (checked, looks correct), **SKIPPED** (couldn't determine without running).

### Item 1 — `_newton_fori` infinite-iter mask correctness

**Where:** `lifecycle/solver.py`, `_newton_fori` function (around line 523).

**What to look for:**
- After the `lax.fori_loop` runs `max_iter` iterations, every cell either has `converged=True` or `ls_failed=True` (or hit max_iter without either). Verify the exit-code logic at the end correctly distinguishes:
  - **Converged**: `EC_INTERIOR`
  - **Hit max_iter without convergence**: should be `EC_NEWTON_FAIL`
  - **Line-search failed mid-iter**: should be `EC_NEWTON_FAIL`
- Particularly check that a cell which converged BUT had `ls_failed=True` set briefly during iteration before converging doesn't get marked as failure.
- Verify `n_iters_used` increments only when `is_active=True`. If it counts inactive iters, post-solve diagnostics report wrong Newton-iter counts.

**Severity criteria:**
- HIGH: exit-code mis-classifies converged cells as failed (false-negative on `n_newton_failures`)
- HIGH: `n_iters_used` includes masked-out iters (silently inflated diagnostic)
- MEDIUM: edge case where `ls_failed` and `converged` both set in the same body invocation

### Item 2 — `_backtracking_fori` first-success semantics

**Where:** `lifecycle/solver.py`, `_backtracking_fori` function (around line 465).

**What to look for:** The `improved_now = NOT found AND err_t < err_old` gate is the correctness-critical line. The `while_loop` version exits on first success and never evaluates more halvings. The fori version evaluates all halvings but masks the result.

Verify:
- After the first successful halving, `found=True` and the `improved_now` gate evaluates False on subsequent iterations regardless of `err_t`.
- The state held after `found=True` is the FIRST success, not a later (worse) halving.
- The `alpha` field is held at its first-success value, not subsequent halved values.

**Severity criteria:**
- HIGH: a later halving overwrites the first-success state (would change Newton trajectory vs while_loop, breaking equivalence)
- MEDIUM: alpha field updates after found=True but doesn't affect output (cosmetic mismatch)

### Item 3 — Backward-age warm-start under checkpoint resume

**Where:** `run_lifecycle_solver`, lines ~1903-1944 (resume) and ~1999-2052 (age loop with init arrays).

**What to look for:**
- When resuming from checkpoint, `S_list[t+1]`, `B_list[t+1]` are loaded from the bundle file.
- For `t = T-1` (first age below terminal), the previous-age policy IS the terminal solve's broadcast `S_list[-1]`. Verify the broadcast survives the resume path (the code uploads non-broadcast slabs).
- For ages between resumed and not-yet-solved, verify there's no off-by-one where the loop reads `S_list[t+1]` when `t+1` is itself unsolved.
- Verify `S_list[t+1]` cannot be a `None` (the `if solved_age_mask[t]: continue` guard skips solved ages, but the warm-start init reads `S_list[t+1]` BEFORE the solved-mask check).

**Severity criteria:**
- HIGH: a NaN/None propagates from an unsolved-but-attempted age into the next age's init
- HIGH: off-by-one where `S_list[t+1]` is read before populated
- MEDIUM: broadcast lost when round-tripping through the bundle

### Item 4 — `n_dev` dispatch correctness

**Where:** `_build_per_age_*_kernel` dispatcher functions (lines 1334, 1479, 1611).

**What to look for:**
- `if n_dev == 1: return _vmap_only(...)` else `_pmap(...)`. Verify:
  - The vmap-only path returns a `call(...)` with the SAME signature as the pmap path. Mismatched signatures cause `TypeError` at the call site.
  - The pmap path's `call(...)` doesn't try to do `pmap` reshaping when `n_dev == 1` somehow leaks through.
  - The terminal kernel's `call(_unused_age_idx=None)` signature is identical between pmap and vmap-only versions (it's called with no args).
- If `len(jax.devices()) == 0` (impossible but defensive check): does anything blow up?

**Severity criteria:**
- HIGH: signature mismatch between pmap and vmap-only paths
- MEDIUM: a path that misbehaves with edge values of `n_dev`

### Item 5 — Hardcoded 3-D state assumptions (forward-look for rtb-as-state)

**Where:** Anywhere in `lifecycle/solver.py` and `lifecycle/precompute.py`.

**What to look for:** The rtb-as-state agent will extend state from 3-D to 4-D. Anything that hardcodes `n_state == 3` or `8 corners` or `3-tuple` will break under their work. Specifically grep for:

- `bracket_state_3d_jax` — already known to be 3-D-only
- Literal `8` in shape contexts (the 8 trilinear corners)
- Literal `3` in tuple-length contexts (`state_grid_sizes`, `state_n_stds`, `n_state_quad_nodes`)
- Indexing `grids_0`, `grids_1`, `grids_2` as named scalars (rather than via `grids[i]`)
- `j_corners_i, w_corners_i` shape `(n_state_quad, 8)` — the `8` is `2**3`
- Comments referencing "trilinear" or "3D state"

For each finding, note location and whether the rtb-as-state handoff §3-§5 captures it. If a hardcode exists that the handoff DOESN'T explicitly call out, that's a future bug — flag it.

**Severity criteria:**
- HIGH: hardcoded 3-D assumption in code that the rtb handoff lists as "no change required"
- MEDIUM: hardcoded 3-D assumption in code listed for change but with subtle gotcha
- LOW: `8` appearing in a context unrelated to corners (just noise)

### Item 6 — NaN/Inf propagation paths

**Where:** Throughout `lifecycle/solver.py` hot path.

**What to look for:** Operations that can silently produce NaN/Inf and propagate without detection:

- `1.0 / x` where x can be zero or near-zero
- `jnp.sqrt(x)` where x can be slightly negative due to roundoff (e.g., `jnp.sqrt(fs**2 + fb**2 - 2*epsilon)`)
- `jnp.log(...)` of zero or negative
- `x ** (-1.0/gamma)` in `_egm_scan_cell` — what if `x` is zero?
- The `tiny_savings` fallback: does it correctly avoid the singularity?
- `bequest_mu_and_mup` clamps via `b_bar` but verify the chain doesn't NaN
- The interp denominators (`1.0 / (x1 - x0)` in `_interp_c_and_mpc_at_cell`) — what if `x0 == x1`?

**Severity criteria:**
- HIGH: a divide-by-zero path with no clamp upstream
- HIGH: `jnp.sqrt` of provably-can-be-negative quantity
- MEDIUM: a `min_consumption` floor missing somewhere it should be applied
- LOW: a NaN that's clearly clamped immediately downstream

### Item 7 — Integer overflow / dtype issues

**Where:** Index arithmetic anywhere.

**What to look for:**
- Any `np.arange(n_cells, dtype=np.int64)` — at canonical 4-D state `n_cells = 11 × 5832 = 64,152`, fits in int32 easily, but verify int64 is used everywhere a downstream gather might require it.
- `jnp.searchsorted(...)` returning int32 vs int64 — JAX defaults to int32; combined with `jnp.clip(... int(n_w - 2))` as a Python int, can silently cast.
- Memory size computations (`n_age * n_z * N_state * n_w * 8 bytes`) — at inflation 4-D this is `80 × 11 × 5832 × 180 × 8` ≈ 7.4e9. Fits in int64 but not int32 (~2.1e9 limit).

**Severity criteria:**
- MEDIUM: a multiplication that could overflow int32 silently
- LOW: defensive int64 casts that aren't strictly needed

### Item 8 — Resume-from-checkpoint shape/version mismatch

**Where:** `run_lifecycle_solver` lines ~1903-1944.

**What to look for:**
- The shape check: `if Cc.shape != shape: raise RuntimeError`. Verify `shape` is correctly assembled at runtime.
- What if the bundle was solved with `use_fori_newton=True` and resumed with `use_fori_newton=False` (or vice versa)? The policies are still valid; verify resume doesn't reject this.
- What if the bundle was solved with `use_backward_age_warm_start` set differently? Same question.
- What if the bundle was solved before the warm-start kill — does the new code error confusingly?

**Severity criteria:**
- HIGH: an invalid bundle gets accepted and produces wrong policies
- MEDIUM: a valid bundle gets rejected with a confusing error
- LOW: error messages that could be improved

### Item 9 — `lifecycle/__init__.py` import-time exception paths

**Where:** `lifecycle/__init__.py` `_configure_persistent_cache()` and `_check_runtime_platform()`.

**What to look for:**
- What if `_os.makedirs(cache_dir, exist_ok=True)` raises `PermissionError`? The current code catches `OSError` — does that include `PermissionError`? (Yes, but verify.)
- What if `jax.devices()` raises during `_check_runtime_platform()`? (Could happen if CUDA driver mismatch.) Currently no try/except.
- What if `LIFECYCLE_JAX_CACHE_MIN_COMPILE_SECS` is set to a negative number? The code falls back to 1.0 on ValueError but accepts a negative float.
- What if `LIFECYCLE_JAX_CACHE_DIR` contains shell metacharacters? `os.path.expanduser` is safe but worth noting.

**Severity criteria:**
- MEDIUM: an import-time crash that could kill the run before any solver work
- LOW: a silently wrong configuration setting

### Item 10 — `scripts/gpu_run.sh` shell bugs

**Where:** `scripts/gpu_run.sh`.

**What to look for:**
- `set -euo pipefail` is set — verify nothing later relies on a command's failure being recoverable
- The hardware-tier auto-detection regex: matches `"h100" && "sxm"` for SXM5, `"h100"` else for PCIe. What about H100 NVL? B200 SXM6 (mentioned in user's earlier exploration)? They'd fall through to `gpu-h100-pcie` or `gpu-other`. Acceptable, but flag.
- The `RC=${PIPESTATUS[0]}` capture: relies on bash. If the shebang says `#!/bin/bash` (it does), this is fine. If a shell environment lacks bashisms (POSIX sh), it breaks.
- `aws s3 sync` failures are all `|| echo "..."` — non-fatal as designed. But the script doesn't `exit 1` if the run script itself failed AND the cache push happens to also fail. Verify the final `exit "$RC"` is what's intended.

**Severity criteria:**
- MEDIUM: a shell pattern that breaks in a non-bash environment
- MEDIUM: AWS CLI not installed → script crashes with a non-obvious error
- LOW: edge-case GPU types not classified; correct fallback is "gpu-other"

### Item 11 — `scripts/download_bundle.sh` shell bugs

**Where:** `scripts/download_bundle.sh`.

**What to look for:**
- `BUNDLE_NAME="${1:?Usage: $0 <bundle-name>}"` — parameter expansion error if no arg, but the message is clear.
- The `LOCAL_DIR` and `BUNDLE_NAME` interpolation: any chance of path traversal if `BUNDLE_NAME` contains `..`? (Low risk; user-controlled string locally.) Note as defensive concern.
- What if the local dir already has a partial download? `aws s3 sync` is incremental and safe.

**Severity criteria:**
- LOW: defensive concerns only, this script is run on a trusted local laptop

### Item 12 — `_compile_cache_sync.py` failure modes

**Where:** `lifecycle/_compile_cache_sync.py`.

**What to look for:**
- `pull_from_s3`'s "non-empty cache → skip" check uses `os.scandir(cache_dir)` then `any(...)` — what if the cache dir contains only `.lock` files or other metadata? Could be technically non-empty without useful entries.
- `subprocess.run([...], check=False)` — non-fatal failure is correct, but verify the returncode is meaningfully reported and not lost.
- If `aws` CLI is missing → `subprocess.run` raises `FileNotFoundError` (Python 3.x). Currently no try/except for that case.

**Severity criteria:**
- MEDIUM: missing AWS CLI silently breaks cache reuse without a clear error
- LOW: `.lock`-only cache dirs falsely reported as "non-empty"

### Item 13 — Stale comments / documentation drift

**Where:** Anywhere in scope.

**What to look for:**
- Line ~1969-1971 in `solver.py`: comment says "pmap in_axes=None materialises the broadcast lazily" — but on `n_dev==1` the path doesn't go through pmap. Stale.
- Comments referencing "constrained Newton" / "leverage caps" / "simple_clamp" — those branches were removed in JAX rewrite handoff 2.
- Comments referencing handoff numbers ("handoff 1", "handoff 2") — those were transient labels.
- `tiny_savings: 1e-6` comment "below this, hold warm-start alphas" — but warm-start was removed; check if the comment is still accurate (it's now backward-age warm-start).

**Severity criteria:**
- LOW: documentation issues only. Group all stale-comment findings into one bullet for compactness.

### Item 14 — Numerical bound checks (interp / bracket)

**Where:** `interp_1d_lin_extrap` (line 236), `bracket_uniform` (line 259), `bracket_state_3d_jax` (line 267).

**What to look for:**
- `bracket_uniform` clips `iz` to `[0, n_z - 2]`. Verify there's no off-by-one (should be `n_z - 1`?). The grid has `n_z` points indexed 0..n_z-1; brackets need `lo` and `lo+1`, so `lo` ranges 0..n_z-2. Looks correct, verify.
- `interp_1d_lin_extrap` extrapolates linearly outside the grid bounds. At extreme wealth values this may produce nonsense (e.g., negative consumption). Currently clamped via `jnp.maximum(c, min_consumption)` downstream — verify chain.
- `bracket_state_3d_jax` clips fractions to `[0, 1]` — correct.

**Severity criteria:**
- HIGH: an off-by-one in bracket arithmetic (would produce wrong policies at boundary)
- MEDIUM: extrapolation that bypasses a needed clamp downstream

### Item 15 — Pmap path's padding unwind

**Where:** Each `*_pmap` builder's `collapse(a)` function (lines 1416, 1469, 1708).

**What to look for:**
- `pad_n = ceil(N_state / n_dev) * n_dev` then `flat[:n_cells]` reshape. Verify the `[:n_cells]` slice is what discards padding entries, not `[:N_state]` (terminal vs the others).
- For terminal: `arr[:N_state]` is correct (1D over states).
- For retirement/working: `flat[:n_cells]` then reshape to `(n_z, N_state, -1)` — verify the index ordering matches how cells were padded (z-first then state).

**Severity criteria:**
- HIGH: padding unwound incorrectly so policies appear at the wrong (z, state) index
- MEDIUM: padding survives into the final array

### Item 16 — Float64 vs Float32 silent casts

**Where:** Anywhere a `jnp.float32`, `jnp.int32`, or unannotated NumPy array enters the JAX device.

**What to look for:**
- `jnp.asarray(np.array_with_default_dtype)` — defaults to `np.float64` on most systems but can vary. Should be explicitly `dtype=jnp.float64`.
- Any `0.5` literal in a JAX expression where the rest is `float64` — JAX promotes correctly, but worth verifying.
- The `static` tuple's int values (`max_iter`, `max_backtrack_iter`) — these are Python ints, baked into the trace as constants. Fine.

**Severity criteria:**
- HIGH: a silent float32 cast in the FOC computation path (would degrade precision)
- LOW: dtype concerns in metadata

### Item 17 — Race conditions and concurrency

**Where:** Persistent cache directory, S3 sync helpers.

**What to look for:**
- If two solver processes share the same `LIFECYCLE_JAX_CACHE_DIR`, do they race on writes? JAX uses internal file locking — verify by checking the cache config.
- If `gpu_run.sh` is invoked twice on the same instance (overlapping), do the cache pull/push race? `aws s3 sync` is atomic per file but not per directory.
- The `verify_benchmark_bundle.py` writes a bundle to `./saved_runs/<name>/`. If two runs use the same name (e.g., re-running after a crash), does the second corrupt the first?

**Severity criteria:**
- MEDIUM: a foreseeable concurrent-runs scenario that produces a corrupted bundle
- LOW: theoretical race that requires manual interleaving

### Item 18 — Print-statement side effects in JIT context

**Where:** Any `print(...)` calls inside JIT'd functions or inside `vmap`/`pmap` bodies.

**What to look for:**
- A `print` inside a `@jit`'d function that wasn't traced-out. JAX would show an error at trace time; if it didn't crash already, none exist. Verify.
- `print` in the orchestrator (outside JIT context) is fine.

**Severity criteria:**
- HIGH: a print inside a JIT body (would have crashed at first run; if you find one, it means the path was never executed)
- LOW: cosmetic prints

---

## 4. Reporting format

Create `docs/scans/BUG_SCAN_2026-05-06.md` with this structure:

```markdown
# Bug & Correctness Scan — 2026-05-06

**Scanner:** [agent name]
**Branch HEAD:** <git rev-parse HEAD>
**Scope:** lifecycle/solver.py, precompute.py, __init__.py, policy_io.py, _compile_cache_sync.py, scripts/*.sh
**Mode:** static read; no execution; no code edits.

## Summary table

| # | Item | Verdict | Severity | One-line finding |
|---|---|---|---|---|
| 1 | _newton_fori mask | YES/NO/SKIPPED | HIGH/MED/LOW/— | ... |
| ... |

**Severity counts:** HIGH=N, MEDIUM=N, LOW=N

## High-severity findings (act before next run)

[Each high-severity finding gets its own subsection.]

### Item N — [Title]
**Location:** `lifecycle/solver.py:XXXX`
**Severity:** HIGH
**Description:** [what's wrong]
**Failure mode:** [how it manifests — crash / wrong result / silent corruption]
**Reproducibility:** [conditions under which it triggers]
**Recommended fix:** [one paragraph; do not implement]
**Confidence:** HIGH/MED/LOW

## Medium-severity findings

[Same format, briefer.]

## Low-severity findings (group as bullets)

- [Item N]: one-line finding, location, one-line fix.
- ...

## Items reported as NO

[Brief one-line per item.]

## Items reported as SKIPPED (need profiler / runtime check)

[List with reason for skipping.]
```

---

## 5. Don'ts

- **Don't fix anything.** Findings only.
- **Don't speculate beyond the checklist.** Off-checklist concerns go in a single "Other observations" bullet at the end.
- **Don't rerun the smoke** to verify findings. Static read only.
- **Don't propose redesigns.** Bug fixes only — and even those, you propose, the user dispatches.

---

## 6. Workflow

1. Pull the latest `jax-rewrite` (the GPU-prep commits should have just landed).
2. Read the in-scope files end-to-end before scanning. Especially `solver.py` from line 304 (Newton) through line 2080 (orchestrator).
3. Work through items 1-18 in order.
4. For each: YES/NO/SKIPPED with location + one-paragraph description.
5. Compile findings into `docs/scans/BUG_SCAN_2026-05-06.md`.
6. **Do not commit the report.** Leave it staged.
7. Report back with severity counts (HIGH/MED/LOW) and the path to the report. Stop.

Total expected report length: 1000-2500 words. Verbose is OK if findings warrant; don't pad.

---

## 7. What "good output" looks like

A HIGH finding example:

> ### Item 4 — Signature mismatch in pmap vs vmap-only kernel call
> **Location:** `lifecycle/solver.py:1462, 1598`
> **Severity:** HIGH
> **Description:** `_build_per_age_retirement_kernel_pmap.call()` accepts `(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr)` (5 positional args). `_build_per_age_retirement_kernel_vmap_only.call()` accepts `(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr)` — same 5 args, in the same order. Confirmed signatures match.
> **Verdict:** NO (false alarm; signatures do match). Promoting Item 4 to NO with this line as evidence.

A NO example:

> ### Item 18 — Prints inside JIT bodies
> **Verdict:** NO. Searched for `print(` inside any `@jit` / `@partial(jit, ...)` / `vmap`-decorated function body. Zero hits in scope. Clean.

Be specific. "Looks fine" is not an acceptable verdict. Cite a line number or a search command.
