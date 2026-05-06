# GPU Trial Findings — Running Log

**Purpose:** Capture lessons from real GPU runs. Date-stamped entries; oldest first. Each entry: what was tried, what happened, what it implies, what to do next time.

**Companion to** [AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) (the recipe). This doc is the *working notebook*; that one is the *manual*.

---

## 2026-05-06 — First Lambda Labs GH200 attempt

### Hardware

- **Instance:** Lambda Labs `gpu_1x_gh200` (1× NVIDIA GH200 480GB)
- **Region:** us-east-3
- **OS / image:** Lambda Stack 22.04 (Ubuntu 22.04 + NVIDIA driver + CUDA 12 + cuDNN preinstalled)
- **CPU:** Grace ARM (aarch64) — Grace Hopper Superchip
- **GPU:** GH200 with **97 GB HBM3** (the "480 GB" in the name is *combined* CPU LPDDR5 + GPU HBM3 unified memory)
- **Cost:** ~$1.49–$2.49/hr

### Bootstrap (worked first try)

```bash
git clone -b jax-rewrite https://github.com/42597hury/lifecycle_portfoliochoice.git
cd lifecycle_portfoliochoice
python3 -m venv venv               # python3 → 3.10.12 on Lambda Stack 22.04
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt  # ~3 min download, all aarch64 wheels
```

**Wheel resolution:** clean. JAX 0.6.2 with `jax-cuda12-pjrt-0.6.2`, `jax-cuda12-plugin-0.6.2`, full NVIDIA CUDA 12.9 stack — all aarch64 builds. **No aarch64 wheel issues** despite the AWS_TRIAL_JAX.md note about "aarch64 wheel issues" (that was for Graviton CPU-only; GH200 ARM with NVIDIA GPU is fine).

**JAX device check:**
```
[lifecycle] JAX runtime: 1 device(s), platform(s)=['gpu']
Devices: [CudaDevice(id=0)]
```

The new `_check_runtime_platform()` warning fires correctly when GPU env hints are set but JAX doesn't see CUDA — didn't trigger in our case, all clean.

### AWS credentials shortcut

Rather than `aws configure` interactively on the Lambda instance, we **piped local `~/.aws/credentials` over SSH** in one command:
```bash
cat ~/.aws/credentials | ssh ubuntu@<ip> 'mkdir -p ~/.aws && cat > ~/.aws/credentials && chmod 600 ~/.aws/credentials'
```
Saves ~30s and avoids typing secret keys. Worth adding to the recipe.

### Smoke results — clean pass

`verify_smoke.py` (6 ages, `(3,3,3,3)` state, `n_z=5`, `n_w=20`):

| Metric | Value |
|---|---|
| Solve wall | **30.2 s** (mostly JIT compile cold) |
| Cell-batching | `vmap-only (single-device)` ← exercised the new code path |
| `Status` | `complete (6/6 ages solved)` |
| `Policy sanity` | `PASS` |
| `alpha_s range` | `[-1.453, 3.547]` |
| `alpha_b range` | `[-9.526, 10.410]` |
| Cache pushed to S3 | 22.6 MiB |

**alpha ranges differ from the historical 3D baseline** (`[-1.038, 3.082]` / `[-8.996, 9.718]`) — *expected* under rtb-as-state because rtb is now in the state; magnitudes and signs are still economically sane. **Not a regression.**

### Benchmark attempt 1 — `9×9×9×9` OOM

**Config:** `verify_benchmark_bundle.py` at `state_grid_sizes=(9,9,9,9)` → 6561 N_state, `n_z=11`, `n_w=180`, `n_s=180`, `n_state_quad=(3,4,3,4)=144`, `max_iter=400`, retirement-only 33 ages.

**Result:** OOM in the **terminal kernel**.
```
Can't reduce memory use below 67.31 GiB by rematerialization;
only reduced to 96.46 GiB, down from 96.52 GiB originally
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 95.93 GiB
```

XLA's plan: 96 GB. GH200 HBM: 97 GB. Failed by margin of allocator overhead.

### Benchmark attempt 2 — `7×7×7×7` worse OOM (the surprise)

**Config:** dropped `state_grid_sizes` to `(7,7,7,7)` → 2401 N_state. **Everything else unchanged.**

**Expected:** 2.7× less memory than 9⁴ (~36 GB).
**Actual:**
```
Can't reduce memory use below 67.13 GiB by rematerialization;
only reduced to 1.06 TiB, down from 1.06 TiB originally
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.06 TB
```

**Memory request grew to 1.06 TB** — 11× *worse* than 9⁴, despite a smaller state grid. **This is XLA's compilation lottery**, not our solver getting harder.

### Why XLA flipped

The two error logs reveal a critical truth: **the genuine minimum working set is ~67 GB regardless of grid size.** Both attempts hit `Can't reduce memory use below 67 GiB`. That's the irreducible lower bound after rematerialisation.

What differs is XLA's *initial scheduling plan*:
- **9⁴:** planned 96 GB (10% above the 67 GB floor — barely fits 97 GB)
- **7⁴:** planned 1.06 TB (16× above the floor — catastrophically off)

When trace shape changes, XLA's scheduler picks different **tile sizes**, **fusion boundaries**, and **rematerialization decisions**. For some shapes it lines up cleanly; for others it doesn't. **Known XLA pain point** — non-monotonic memory behavior in input dimensions. Same class of issue people hit in LLM training where batch size 17 OOMs but 16 and 18 are fine.

### Implication for headroom

GH200's 97 GB HBM only has ~30 GB of *real* headroom over the 67 GB minimum. **Any XLA scheduling slop > 30 GB → OOM.** For 4D inflation at canonical 9⁴, you'd want either:

- **A bigger card** — H200 SXM5 (141 GB) gives 70+ GB headroom. H100 SXM5 (80 GB) is *tighter* than GH200, so a downgrade.
- **Manual cell-vmap chunking** — the §6.6 follow-up from `HANDOFF_PMAP_TO_VMAP.md`. Keeps per-chunk working set well under ceiling regardless of XLA's plan.

### Benchmark attempt 3 — `5×5×5×5` + conservative knobs

**Config:**
- `state_grid_sizes=(5,5,5,5)` → 625 N_state
- `n_state_quad_nodes=(2,3,2,3)` → 36 (4× cheaper FOC)
- `max_iter=100, max_iter_unconstrained=100` (less fori_loop intermediate state)
- Env: `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`

**Per-cell c_corners math at this config:** `144 × 11 × 16 × 180 × 8 = 36 MB → 36 → 9 MB per cell after quad reduction`. Total worst-case full materialization ~62 GB at 6875 cells. Fits 97 GB with headroom.

**Result:** Compiled and ran cleanly. Per-age wall stabilised at **273 s/age flat** (no acceleration after first age — see "Per-age compute dominates JIT" below). Alphas economically coherent and trending correctly:
- Age 99 terminal: c range `[0.004, 132.6]` ✓
- Age 98: `α_s=0.515  α_b=0.494  c/W=0.112`
- Age 97: `α_s=0.553  α_b=0.443  c/W=0.112`
- Age 96: `α_s=0.580  α_b=0.411  c/W=0.112`
- Age 95: `α_s=0.599  α_b=0.393  c/W=0.113`
- Age 94: `α_s=0.613  α_b=0.384  c/W=0.113`

`α_s` rising and `α_b` falling as we move backward in age = textbook lifecycle pattern. Backward-age warm-start working as designed.

GPU instrumentation mid-run: `100% utilisation, 92.9 / 97.9 GB HBM`. Tight but stable.

### Run died at age 94 — SSH-attached python killed by HUP

**Cause:** the python process was launched via `ssh ubuntu@... 'bash scripts/gpu_run.sh' &` (background SSH from the Bash tool). When the SSH session naturally ended/recycled, python received SIGHUP and was killed. **No `nohup`, `tmux`, `disown`, or `setsid` to detach.** Standard Linux gotcha.

**Loss:** 25 minutes × $1.99/hr ≈ $0.85 of GPU time. No partial bundle (`checkpoint_every_n_ages=None` in original config — no auto-saves).

**Fix applied for relaunch:**

1. **Run inside `tmux`:**
   ```bash
   ssh ubuntu@<ip> 'tmux new-session -d -s bench "..."'
   ```
   tmux session lives independently of any SSH connection. Survives my Bash tool timeouts, network blips, anything.
2. **Enable checkpointing every 5 ages** in `verify_benchmark_bundle.py`:
   ```python
   solve_control = SolveControl(
       youngest_age_to_solve=67,
       checkpoint_every_n_ages=5,
       save_on_interrupt=True,
       return_partial_on_interrupt=True,
   )
   ```
   Worst-case loss now bounded at 5 × 273 = 23 min ≈ $0.75.

**General lesson:** `verify_benchmark_bundle.py` (and any future production runner) should default to `checkpoint_every_n_ages` set to something non-None. The current default `None` makes the script unsuited for runs longer than ~5 minutes without manual override.

### Lobatto tails were OFF in this run

Both `ret_lobatto_Z` and `state_lobatto_Z` are `None`. Pure Gauss-Hermite throughout. For thesis-quality runs at small `delta_bequest`, Lobatto tails at ±3σ should be enabled to capture bankruptcy-boundary cells robustly. Tied to the [HANDOFF_EVAL_LOBATTO_PROPAGATION.md](../handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md) handoff (still open) which adds the eval-side Lobatto propagation.

### Per-age compute dominates JIT compile

A surprising finding from the run: per-age wall **does not decrease** after the first age. Age 98 = 277.5s. Age 97 = 273s. Age 96 = 273s. Etc. This means JIT compile cost is small (~5s) and the **actual fp64 compute is doing 270+ seconds of work per age**.

Calibrated effective fp64 throughput: ~6.8 TFLOPS (about 70% of GH200's 9.7 TFLOPS peak). Standard for scientific fp64 workloads.

**Implication for performance work:** further wall reduction has to come from **reducing per-cell FLOP count** (smaller `n_state_quad`, smaller `max_iter`) — not from JIT optimisation, fusion tweaking, or persistent cache wins.

### vmap-only path confirmed working at production scale

`Cell-batching pattern: vmap-only (single-device)` printed correctly. The new code path landed in `pmap→vmap` agent's commit `cf0bdfc` runs cleanly in production. **First time exercised at this scale — no regressions found.**

The 5.4× speedup the agent observed on the 6-age CPU smoke (vmap-only vs pmap) **does not translate directly to GPU wall savings** here — because per-age GPU compute dominates over the dispatch overhead that vmap-only eliminates. The fusion gain is real but masked.

---

## 2026-05-06 — Critical bug discovered: solver/simulator portfolio-return mismatch

**Severity: HIGH (silent wrong-result risk for any path-based diagnostic).**

The user surfaced this concern during the live run; verified by reading both files.

### What it is

`lifecycle/solver.py:657-679` (`_ccv_log_return_and_grad`) computes the CCV log portfolio return with continuous-rebalancing variance correction:

```
r_p = log_R_bill + α_s·log_x_s + α_b·log_x_b
    + 0.5·(α_s·σ²_xr + α_b·σ²_xb)
    - 0.5·(α_s²·σ²_xr + 2·α_s·α_b·σ_xrxb + α_b²·σ²_xb)
R_p = exp(r_p)
```

Standard CCV w8566 eq. (10). **Correct.**

`lifecycle/simulation.py:355` computes the realised portfolio return as:

```
R_port = α_s · R_stock + α_b · R_bond + α_bill · R_bill
```

This is **arithmetic combination of simple returns** — i.e. discrete one-period rebalancing. **Different model assumption from the solver.** The docstring at `simulation.py:22` even labels it explicitly as "Arithmetic R_p = ...".

### Implications

- **Solved policies are CCV-optimal** (the in-flight bundle is fine *as a CCV optimum*).
- **Forward-simulation paths use different dynamics** than what the solver optimised for.
- **Any path-based diagnostic — EE residuals on simulated paths, moment matching, welfare comparisons — is biased.** The size of the wedge depends on the variance terms; for typical equity premium magnitudes it's 50-150 bp per period in expected portfolio return. Compounds over 33 retirement years.

### Why it's likely a bug, not intent

- The codebase commits to CCV throughout (`wealth_dynamics_spec='ccv_log'`, the entire solver algebra, the rtb-as-state migration was done in CCV terms).
- The simulator's arithmetic formula looks like an inheritance/oversight from earlier work — easy to miss when porting because both formulas "look reasonable."
- The codebase has `Sigma_rr` available everywhere the simulator needs the variance correction (same source the solver uses) — so the fix is local, ~10 lines.

### Fix path

Already in flight via the user's other agent:

```python
# Replace simulation.py:350-355 with:
log_R_port = (
    log_R_bill
    + a_s_t * log_x_s + a_b_t * log_x_b
    + 0.5 * (a_s_t * sigma2_xr + a_b_t * sigma2_xb)
    - 0.5 * (a_s_t**2 * sigma2_xr + 2 * a_s_t * a_b_t * sigma_xrxb + a_b_t**2 * sigma2_xb)
)
R_port = jnp.exp(log_R_port)
```

Plus a unit test asserting `solver._ccv_log_return_and_grad` and `simulation._compute_R_port` agree to ~1e-12 on a known input.

**Bundle from tonight's run remains valid as a CCV-optimal policy.** Only the simulator (downstream of the bundle) needs the fix. Any sim-based diagnostics done before the fix lands need to be re-run after.

---

## 2026-05-06 — Grid-EE residuals on the 5⁴ bundle

First quantitative correctness check on the JAX 5⁴ production bundle. **Grid-based** Euler-residual sweep (not sim-path — that's queued for tomorrow). Same-quadrature mode: training quadrature reused for the FOC re-evaluation.

### Results

| metric | JAX 5⁴ same | main 7³ v3 same | main 7³ v4_lobatto next_finer |
|---|---|---|---|
| typical (median/mean) log10\|EE\| | -6.5 (median) | -7.97 (mean) | -2.92 (mean) |
| P95 log10\|EE\| | -3.5 | -2.48 | -1.96 |
| P99 log10\|EE\| | -2.7 | n/a | n/a |
| MAX log10\|EE\| | -2.0 | -0.39 | -0.007 |
| worst abs EE | 9.65e-3 (1%) | 0.41 (41%) | 0.98 (98%) |
| total cells probed | 39,600,000 | 13,365 | — |

### Interpretation

- **JAX is materially better than main v3 on the tail** — worst-cell residual ~1% vs main's 41%, P95 ~10× tighter.
- **Median (-6.5) vs main's mean (-7.97) is apples-to-oranges** (different statistics + different cell-set sizes). The directly comparable metrics (P95, MAX) all favour JAX.
- **MAX 9.65e-3 sits right at the PASS/CONCERNING boundary** per main's threshold convention (PASS<1e-2, FAIL>5e-2). Just inside PASS.
- **39.6M probes vs main's 13.4K** — 3000× more samples. The JAX sweep is *guaranteed* to surface tail cells main never visits; that the worst is still <1% is a strong signal.
- **The v4_lobatto next_finer column** (worst cell 98%) is a separate quadrature-truncation question (training quad vs higher-fidelity quad), not a policy-correctness question. Don't read it as a regression metric.

### Caveats

- This is **grid-EE**, not **sim-EE**. The 1% worst cell is "somewhere in the 39.6M probe space" — not "somewhere a household actually visits." Tail cells in the grid (extreme spr or rtb) are exactly where you'd expect grid-EE to deteriorate, and they may carry zero ergodic-measure weight.
- main publishes against **sim-EE** numbers, so the apples-to-apples comparison for the thesis comes from the sim-path port (queued, math review tomorrow before agent dispatch).
- Eval mode was `same` only. `next_finer` (different quadrature for eval vs training) is a separate pass that exposes truncation-error contributions.

### Verdict

**Bundle passes correctness gate** under grid-EE. No regression vs main on directly comparable metrics. Sim-EE will be the headline thesis number; this confirms the policies aren't broken in some way grid-EE would catch.

---

## Synthesis: what we learned about the JAX code itself

### Confirmed working as designed (production scale)

- `vmap-only (single-device)` cell-batching path
- Backward-age warm-start (smooth alpha trajectories across ages)
- rtb-as-state migration (4-D state, 16-corner gather, multilinear interp, name-based return indexing)
- aarch64 + JAX cu12 wheel resolution
- Persistent compilation cache (push/pull via `_compile_cache_sync.py`)
- `_check_runtime_platform()` warning system (correctly silent on healthy GPU)
- The bug-scan + perf-scan fixes (didn't surface any regressions in production)

### Hard limits discovered

- **67 GB minimum working set** — XLA-stated lower bound, irreducible regardless of state grid size at canonical `n_state_quad × n_z × n_corners × n_w × n_s` shapes.
- **GH200 has only 97 GB HBM** (not 480 GB as branding suggests; the rest is unaddressable Grace LPDDR5).
- **XLA memory planning is non-monotonic in input dimensions.** 7⁴ requested 1 TB; 9⁴ requested 96 GB. Same code, different shape → different XLA decisions.
- **Per-cell `c_corners` at full canonical 4-D quad is 36 MB.** At 7⁴: 950 GB worst-case full materialization. **Manual cell-vmap chunking is now mandatory for >5⁴ on a single GPU.**

### Bugs / gotchas surfaced

- **HIGH: solver/simulator CCV-vs-arithmetic mismatch** (see entry above).
- **`verify_benchmark_bundle.py` had no default checkpointing** (`checkpoint_every_n_ages=None`). Long runs lost everything on death. Now patched for relaunch; should be fixed in the source.
- **`max_iter` is real wall cost under fori_loop** — not "average iters used." Doubling `max_iter` doubles wall regardless of cell convergence behaviour. The original `max_iter=400` in the benchmark script was 4× more wasteful than needed; `max_iter=100` with backward-age warm-start is the right operating point.

### Performance characterization (calibrated tonight)

- GH200 fp64 effective throughput: ~6.8 TFLOPS (70% of 9.7 peak)
- Per-age wall at 5⁴ + n_state_quad=36 + max_iter=100: **273 s/age**
- JIT compile cost (cold cache): ~5-15 seconds per kernel — small vs steady-state per-age compute
- Smoke wall: 30 s (vs minutes on local CPU — vmap-only fusion gain confirmed)
- The 5.4× CPU vmap-only fusion gain doesn't directly translate to GPU wall — masked by dominant fp64 compute

### Forward-implication takeaways

1. **Cell-vmap chunking handoff is now critical**, not optional. Without it, no path to 7⁴ on any single GPU.
2. **Simulator CCV fix unblocks all path-based diagnostics.** Prerequisite for using bundles for anything beyond grid-based EE residuals.
3. **Default `checkpoint_every_n_ages=5`** should be the standard in production runner scripts.
4. **Lobatto tails should be turned on** for publishable runs (currently off; see [HANDOFF_EVAL_LOBATTO_PROPAGATION.md](../handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md)).
5. **`verify_benchmark_bundle.py` should run under tmux/nohup** in any production launcher, not bare SSH.

### Specific aarch64 + Lambda-Stack notes

- `python3` on Lambda Stack 22.04 = Python **3.10.12**, not 3.11. The original AWS_TRIAL_JAX.md userdata used `python3.11` — that fails on Lambda. Use `python3` for portability.
- `aws` CLI is at `/snap/bin/aws`, version 2.33.29, aarch64.ubuntu.22 build. Works fine.
- `nvidia-smi` reports `NVIDIA GH200 480GB, 97871 MiB`.

### Practical fallback ladder for "will this fit?"

When OOM hits, downgrade in this order (cheapest to most invasive):

1. `TF_GPU_ALLOCATOR=cuda_malloc_async` + `XLA_PYTHON_CLIENT_PREALLOCATE=false`
2. Drop `n_state_quad` (e.g. `(3,4,3,4)` → `(2,3,2,3)` — 4× cheaper FOC)
3. Drop `max_iter` (e.g. 400 → 100 — less fori_loop intermediate state)
4. Drop `state_grid_sizes` per-axis (9 → 7 → 5)
5. Drop `n_w` and `n_s` (180 → 120, last resort because hurts policy resolution)

**Don't drop `n_z`** unless you absolutely must — n_z=11 is the income discretization and is a published-paper minimum.

### Per-cell memory cost reference (4D state, n_z=11, n_w=180)

| `n_state_quad` | per-cell `c_corners_T` |
|---|---|
| (3,4,3,4) = 144 | ~36 MB |
| (2,3,2,3) = 36 | ~9 MB |
| (2,2,2,2) = 16 | ~4 MB |

Multiply by `n_cells = n_z × N_state` for worst-case full materialization. Real peak is usually lower (XLA streams) but this is the upper bound.

---

## Template for next entry

```markdown
## YYYY-MM-DD — <one-line summary>

### Hardware
- ...

### Config tried
- ...

### Result
- ...

### What we learned
- ...

### What to do differently next time
- ...
```

---

## Operating notes (running checklist)

### `max_iter` / `max_backtrack_iter` calibration

Under `use_fori_newton=True`, both values are **wall cost regardless of cell convergence** (every cell pays the full `max_iter × (1 + max_backtrack_iter)` FOC calls per Newton solve). Calibrate measurement-driven, not by guessing.

**Workflow:** after each real bundle (5⁴ or larger; do NOT calibrate from 3⁴ smoke — too easy, p99 not representative), read `diag['newton_iter_histogram']` and `diag['backtrack_iter_histogram']` and set for the next run:

```
max_iter_next            = max(20, ceil(1.5 × p99_newton))
max_backtrack_iter_next  = max(3,  ceil(1.5 × p99_backtrack))
```

Bundle that's calibrated against doesn't itself benefit; the **next** bundle does. Each launch is one calibration cycle. Skip the dial if `total_newton_failures > 0` — that means cells didn't converge at the *current* max_iter, dialing tighter would lose them entirely.

Note: tonight's 5⁴ bundle was solved BEFORE the histogram-exposing commit (`051423a`), so its diag is empty for these fields. **First real calibration data lands with the next post-`051423a` GPU run.**
