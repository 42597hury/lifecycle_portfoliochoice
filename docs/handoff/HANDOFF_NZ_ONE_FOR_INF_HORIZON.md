# Handoff: Enable `n_z=1` for inf-horizon (kill the wasted state)

**Branch:** `jax-rewrite`
**Effort:** ~3-4 hours. ~10-30 LOC + audit + validation.
**Output:**
- Modified `lifecycle/discretization.py` (N=1 guard).
- Modified `verify/benchmark_inf_horizon.py` (n_z=1).
- (If needed) minor downstream patches identified during the audit.
- `docs/scans/NZ_ONE_AUDIT_2026-05-07.md` — written audit confirming no other places break at n_z=1.

**Critical constraint:** **Do NOT modify the main lifecycle solver path.** Specifically:
- `lifecycle/solver.py` — read-only (no changes)
- `lifecycle/model.py` — read-only (no changes)
- `run_lifecycle_solver`, `_build_per_age_*_kernel*` — read-only
- `lifecycle/inf_horizon_solver.py` — minor changes ONLY if strictly necessary; prefer leaving it untouched

The fix lives in `lifecycle/discretization.py` and the inf-horizon launcher script. Lifecycle solver behavior at canonical n_z=11 must be **bit-identical** before/after this change.

---

## Why this matters

The infinite-horizon benchmark (`run_infinite_horizon_solver`) zeros out income (`pension_zero = jnp.zeros(pc.n_z)`) and mortality (`psi_one = jnp.ones(pc.n_z)`), and sets `b_bar=0` in its `ModelParams`. Under those conditions, **policies are mathematically identical across all z-states** — z is informationally inert.

Currently, `build_precompute(disc_config_with_n_z=1)` raises `IndexError: index 1 is out of bounds for axis 0 with size 1` at [`discretization.py:313`](../../lifecycle/discretization.py#L313):

```python
dz = z_grid[1] - z_grid[0]
```

`z_grid` has size `N=1`, so `z_grid[1]` is out of bounds. The whole Markov-chain construction (lines 316-332) is also degenerate for N=1.

**Workaround in flight tonight:** inf-horizon at 8⁴ × n_z=2 — paying 2× the compute for identical answers across the two z-states. Per-iter wall is ~195s; with n_z=1 it'd be ~98s. **For the planned ablation systems run (Systems I-IV inf-horizon), enabling n_z=1 saves ~half the compute time per run, $11+ per run, ~30+ min wall per run.**

---

## What to fix

### Step 1 — Guard `discretize_income_ar1_mixture` for N=1

In `lifecycle/discretization.py`, add a top-of-function early return:

```python
def discretize_income_ar1_mixture(rho, p, mu1, sigma1, mu2, sigma2, N, n_stds=3):
    """Discretize persistent income AR(1) with mixture-normal innovations."""
    if N == 1:
        # Degenerate single-state Markov chain — always stays in mean state.
        # Used by inf-horizon where pension=0 and psi=1 make z mathematically
        # inert (all z-slices produce identical policies). NOT a valid choice
        # for the lifecycle solver, where setting n_z=1 would silently zero
        # out persistent-income variation in the working-age FOC integrand.
        return np.array([0.0]), np.array([[1.0]])
    # ... existing code unchanged ...
```

This is a 5-line addition with a comment; no other change to the function.

### Step 2 — Audit downstream consumers of `n_z`, `z_grid`, `Pi_z`

The agent's audit must enumerate every place `n_z` is read or where shape `(n_z,)` / `(n_z, n_z)` is assumed, and confirm none break at n_z=1. Likely-affected files (search for `n_z`, `z_grid`, `Pi_z`, `pc.n_z`):

- `lifecycle/precompute.py` — likely uses `z_grid[1]`, `dz`, `Pi_z[i,j]` indexing for working-age income tensors. **Need to check** every site that touches z arrays for n_z=1 robustness.
- `lifecycle/solver.py` — kernel builders reshape policies as `(n_z, N_state, n_w)` and gather across z. With n_z=1, vmap'd code over z should still work (single-element batch axis is JAX-legal). **Likely fine**, but the audit should verify by spot-checking 2-3 kernel sites.
- `lifecycle/inf_horizon_solver.py` — uses `pc.n_z` directly for `pension_zero` and `psi_one`. Trivially scales (size-1 arrays). **Likely fine**.
- `lifecycle/simulation.py` — uses `Pi_z` for transition draws. With Pi_z=[[1.0]], all draws stay in state 0. Trivially fine.
- Other working-age income tensors (eta, eps): these are independent of n_z; the per-age `working_income` array has shape `(n_age, n_z, n_eps)` which is `(n_age, 1, n_eps)` at n_z=1 — should broadcast cleanly.

For each spot the audit identifies:
- If it works at n_z=1 already: note "OK"
- If it breaks: identify the exact failure mode and either patch (preferred) or document as a known limitation.

If a fix to anything beyond `discretization.py` is required, the agent must:
- Apply it minimally
- Confirm the change doesn't affect lifecycle behavior at n_z>=2 (bit-identity check)
- Document in the audit report

### Step 3 — Update `verify/benchmark_inf_horizon.py`

Change `n_z=2` → `n_z=1` in the `disc_config = CANONICAL_DISC._replace(...)` block. Update the docstring memory math accordingly:

- Per-cell c_corners: 135 × **1** × 16 × 180 × 8 = **3.1 MB** (vs 6.2 MB at n_z=2)
- 8⁴ × n_z=1: 4096 cells × 3.1 MB / 2 dev = ~6.3 GB per device working set (vs 25 GB at n_z=2)
- Per-iter wall: ~98s (vs 195s at n_z=2)
- 50 iters: ~82 min (vs ~162 min)

Update the `BUNDLE_NAME` to encode the change: `system_iv_inf_horizon_grid8x8x8x8_nz1_y1lob_calib1` (or similar — pick whatever's distinct from the in-flight n_z=2 bundle's name).

### Step 4 — Validation

**Run sequentially. Do not proceed if any fails.**

**Gate 1 — Lifecycle bit-identity at n_z>=2.**
Run `verify/smoke.py` (uses default n_z, currently 3 in the smoke). Capture alpha_s, alpha_b. They must be **bit-identical** to a pre-change capture. If they differ in any bit, the discretization.py change has unintended consequences for n_z>=2 — STOP and report.

**Gate 2 — Inf-horizon at n_z=1 produces a valid bundle.**
Run a tiny inf-horizon at n_z=1, state_grid=(2,2,2,2), n_w=10, max_iter=5 (smoke level). Should complete without errors and produce sane alpha ranges (no NaN, alphas in some reasonable bounded range).

**Gate 3 — Inf-horizon at n_z=1 vs n_z=2 produces (essentially) identical policies.**
Tightest gate. Run the same inf-horizon config twice — once at n_z=1, once at n_z=2 — same tol, same max_iter, same seeding, same everything else. Compare:
- Policy at z=0 (the only z-slice in the n_z=1 run) vs policy at z=0 (the first z-slice of the n_z=2 run).
- Should be **bit-identical** in principle (both compute the same expected utility under the same fixed-point map). In practice, expect **machine-precision drift** (~1e-13 to 1e-15) due to JAX scheduling differences across shape variations.
- Pass criterion: `np.max(np.abs(C_nz1 - C_nz2[0]))` < 1e-10. Same for S, B.

If Gate 3 deviates by more than machine precision: there's a real semantic difference that we need to understand. Pause and report.

---

## Out of scope

- **Modifying lifecycle solver paths.** Anything in `solver.py`, `run_lifecycle_solver`, kernel builders. Read-only.
- **Modifying SolverConfig fields.** No new flags.
- **Adding n_z=1 to the lifecycle solver as a model variant.** That would silently change the economics (no persistent income shocks). The fix is **inf-horizon-only** — flag this explicitly in the docstring if needed.
- **Performance optimization beyond the n_z reduction.** Just enable n_z=1; don't rewrite anything else.
- **Testing on real GPU.** CPU virtual-device validation is enough for correctness. The user runs the real-GPU launch separately.

---

## Pause points

The agent must STOP and ask the user when:
- **Gate 1 fails** (lifecycle behavior changes). Likely indicates a hidden dependency on the Markov-chain matrix that needs broader thinking.
- **Audit finds a downstream site that requires changes outside `discretization.py`.** Don't silently patch other files. Surface the issue.
- **Gate 3 deviation > 1e-10.** Indicates a model-semantic difference, not just numerical drift.

---

## Implementation checklist

- [ ] **Step 1**: add N=1 guard to `discretize_income_ar1_mixture` in `discretization.py`. ~5 LOC.
- [ ] **Step 2**: audit downstream consumers of n_z / z_grid / Pi_z. Document in `docs/scans/NZ_ONE_AUDIT_2026-05-07.md`. Patch any genuinely broken sites; surface ambiguous sites for user decision.
- [ ] **Step 3**: update `verify/benchmark_inf_horizon.py` to n_z=1, refresh docstring memory math, update BUNDLE_NAME.
- [ ] **Gate 1**: lifecycle smoke bit-identity check (must pass).
- [ ] **Gate 2**: tiny inf-horizon at n_z=1 runs without error.
- [ ] **Gate 3**: n_z=1 vs n_z=2 inf-horizon policies match within 1e-10.
- [ ] Commit:
  ```
  discretization+inf_horizon: enable n_z=1 for inf-horizon's degenerate income chain
  
  Inf-horizon zeros pension and sets psi=1, making z mathematically inert.
  Previously discretize_income_ar1_mixture failed at N=1 with IndexError on
  dz = z_grid[1] - z_grid[0]; the Markov-chain construction is also
  degenerate for a 1-state chain. Add a 5-line guard returning the trivial
  ([0.0], [[1.0]]) pair for N=1.
  
  Also update verify/benchmark_inf_horizon.py to n_z=1, halving per-iter
  wall (195s -> 98s) and per-device HBM (25 GB -> 6 GB) on the 8^4 grid.
  
  Lifecycle solver path untouched. Bit-identity verified at n_z>=2 vs
  pre-change. Inf-horizon at n_z=1 vs n_z=2 verified within 1e-10 across
  C, S, B at policy[0] z-slice.
  
  No math change to lifecycle. n_z=1 not a valid lifecycle config (would
  zero persistent-income variation); inf-horizon-only by design.
  ```
- [ ] Push.

---

## Why this is a separate handoff (not part of inf-horizon work)

- It's a **discretization-level fix**, not a kernel/orchestrator change. Cleaner to scope and audit independently.
- The current in-flight inf-horizon run (n_z=2) is **not affected** by this work — it'll complete on its own. This handoff prepares for FUTURE inf-horizon runs (ablation Systems I-IV, calibration cycle 2, etc.).
- The audit work is the bulk of the effort, not the 5-line patch — knowing that nothing else breaks at n_z=1 is the load-bearing deliverable.

---

## Why this matters

- **Halves wall time on every future inf-horizon run.** With Systems I-IV ablation planned (4 runs minimum), savings compound: 4 × ~80 min = 5+ hours saved, $40+ saved.
- **Halves memory pressure** on the 8⁴ inf-horizon path. Frees headroom for tighter quadrature or larger state grids in future runs.
- **Removes a confusing waste signal** in the run logs ("n_z=2 income states, but psi=1 and pension=0 make z inert" is a documented but ugly thing to explain).
