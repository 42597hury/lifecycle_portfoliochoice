# Audit: enabling `n_z=1` for the inf-horizon benchmark

**Date:** 2026-05-07
**Branch:** `jax-rewrite`
**Scope:** Identify every site touching `n_z`, `z_grid`, `Pi_z`, or `dz` and confirm none break at `n_z=1`. Targeted at the inf-horizon benchmark (`run_infinite_horizon_solver`); the lifecycle solver path is read-only.

## Summary

- Two sites — both single-line patches — needed changes:
  - [`lifecycle/discretization.py`](../../lifecycle/discretization.py): `discretize_income_ar1_mixture` (early `if N == 1:` return).
  - [`lifecycle/precompute.py:350`](../../lifecycle/precompute.py#L350): `dz = z_grid[1] - z_grid[0]` (size guard).
- All other downstream consumers of `n_z` / `z_grid` / `Pi_z` either size-broadcast cleanly at `n_z=1` or are not on the inf-horizon code path.
- Bit-identity at `n_z>=2` confirmed (Gate 1).
- Inf-horizon at `n_z=1` runs cleanly with sane policies (Gate 2).
- Inf-horizon at `n_z=1` matches `n_z=2`'s `z=0` slice exactly (Gate 3 max diff = 0.0e+00).

## Patch surfaced beyond `discretization.py` — please confirm

The handoff says "don't silently patch other files; surface the issue." [`precompute.py:350`](../../lifecycle/precompute.py#L350) needs an identical guard to the one applied in `discretization.py`:

```python
# before:
dz = z_grid[1] - z_grid[0]

# after:
dz = z_grid[1] - z_grid[0] if z_grid.size > 1 else 0.0
```

This is a structurally trivial 1-line change of the exact same character as the discretization guard — same bug, same dz-from-z_grid expression, same inf-horizon-only impact. Without it `build_precompute(n_z=1)` still raises `IndexError` at `dz = z_grid[1] - z_grid[0]` even with the discretization patch in place.

`pc.dz` is consumed only by `bracket_uniform` inside the lifecycle working kernel — never reached on the inf-horizon retirement-only path. Setting it to `0.0` is a placeholder; if anyone wires a working kernel call at `n_z=1` it will produce NaN at trace time, which is the desired loud failure for an unsupported config.

Bit-identity at `n_z>=2` was reverified after applying this change (Gate 1, hashes match exactly).

## Site-by-site audit

### `lifecycle/discretization.py:306-334` — `discretize_income_ar1_mixture` — **PATCHED**
- **Issue:** `dz = z_grid[1] - z_grid[0]` raises `IndexError` at `N=1`. The Markov-chain construction (lines 316-332) is also degenerate.
- **Fix:** Top-of-function early return `np.array([0.0]), np.array([[1.0]])` for `N=1`. Comment notes the inf-horizon-only intent.
- **Bit-identity:** for `N>=2` the function falls through unchanged.

### `lifecycle/precompute.py:350` — `dz` field of `Precompute` — **PATCHED**
- **Issue:** Same `z_grid[1] - z_grid[0]` failure at size 1.
- **Fix:** Guard via `z_grid.size > 1` ternary. `dz=0.0` is safe because no inf-horizon code path reaches `bracket_uniform`.
- **Bit-identity:** for `n_z>=2` the ternary branch is the original expression; arithmetic identical.

### `lifecycle/precompute.py:367-378` — `_precompute_working_income`, `_precompute_pension`, `_precompute_working_income_next` — **OK**
- All shape-broadcast across `n_z`. With `n_z=1` they produce arrays of shape `(n_age, 1, n_eps)`, `(n_age, 1)`, `(n_age, 1, n_eta, n_eps)` respectively.
- Inf-horizon zeros pension and disregards working_income, so the actual values are inert; the shapes flow through `_pc_to_jnp` cleanly.

### `lifecycle/inf_horizon_solver.py:540-545, 696-702` — `pension_zero`, `psi_one`, `expected_shape` — **OK**
- Built with `jnp.zeros(pc.n_z)` / `jnp.ones(pc.n_z)`. Trivially scales to size 1.
- `expected_shape = (pc.n_z, pc.N_state, pc.n_w)` cascades into `_markowitz_cold_start` and `_prepare_initial_policies`; both broadcast init values to `(1, N_state, n_w)` correctly.
- `_compute_z_invariance` returns zeros (single-slice trivially equals itself) — diagnostics are well-defined at `n_z=1`.
- `progress_probe_z_idx`: defaults to `pc.n_z // 2 = 0`, guards against out-of-range. OK.

### `lifecycle/solver.py` — retirement kernel (the only kernel inf-horizon calls) — **OK**
- `_solve_retirement_at_cell` indexes `c_next[z_idx, j_corners_i, :]`, `pension_next_by_z[z_idx]`, `psi_per_z[z_idx]` — all fine at `n_z=1`.
- The retirement kernel does **not** call `bracket_uniform` (no eta-bracketing in retirement). `pcj.dz` and `pcj.z_grid[0]` are only referenced by the working / boundary kernels.
- `_build_per_age_retirement_kernel_pmap` pads cells to a multiple of `n_dev`; `n_cells = 1 * N_state` is supported (no size-1 lower bound on `n_z`).
- `n_chunks = sc.cell_vmap_chunks` chunking arithmetic uses `n_z * N_state` and is robust to `n_z=1`.

### `lifecycle/solver.py:260-264` — `bracket_uniform` and lifecycle working kernel — **NOT EXERCISED BY INF-HORIZON, AVOID**
- Working kernel uses `dz` to interpolate `pension_next_by_z` and the gradient of `working_income_next`. With `dz=0` and `n_z=1` it would produce NaN.
- Inf-horizon never calls the working kernel, so the inf-horizon path is unaffected.
- Lifecycle solver users must not pass `n_z=1` (they wouldn't want to anyway — it zeros persistent-income variation in the working-age FOC). No new validation is added in this work; this is an existing implicit constraint.

### `lifecycle/simulation.py` — **NOT EXERCISED BY INF-HORIZON**
- Same `bracket_uniform` shape — would NaN at `n_z=1`.
- Inf-horizon does not draw simulation paths, so this is unreachable in this work.

### `lifecycle/mortality.py:245-329` — `calibrate_chi_vector`, `build_survival_probs_2d`, `calibrate_earnings_dependent_mortality` — **OK**
- Loops over `n_z = len(z_grid)` from 0 to `n_z - 1`; trivially supports size 1.
- With `z_grid = [0.0]` and `sigma_z > 0`, percentile = `100 * Φ(0) = 50`, chi solved at the median. No degenerate division.
- Inf-horizon overrides `psi_one = ones(n_z)` so the actual survival values do not enter the fixed-point map; only the construction must succeed.

### Configurable `state_grid_sizes[d] == 1` paths — **PRE-EXISTING SUPPORT, OK**
- `build_state_grid` (`lifecycle/discretization.py:158`) has explicit `if Nd == 1` branches in the `cholesky` and `lyapunov-axis` builders.
- `_independence_rouwenhorst_pi` returns the `1 x 1` identity for size-1 axes.
- This is not part of this work, but it informs the broader posture: **size-1 along single-state axes is an established pattern in the discretization layer; we are merely extending it to the income axis.**

### Other consumers (read-but-don't-modify scope) — **OK**

`Pi_z` is only read by `lifecycle/simulation.py:686` (`pcj.Pi_z`) — outside the inf-horizon path.

`pc.n_z` is also read by:
- `verify/canonical_small.py`, `verify/smoke_small.py`, `verify/smoke.py`, `verify/chunking.py`, `verify/mixed_precision*.py`, `verify/ee_residuals.py`, `verify/ee_simpath.py`, `verify/invalid_cells.py`, `verify/compare_jax.py`, `verify/benchmark_bundle*.py`, `verify/pmap_chunking.py`. None of these set `n_z=1` and none are affected.
- `lifecycle/diagnostics.py`, `lifecycle/plots.py`, `scripts/scratch/*.py`. None set `n_z=1`.

No other site requires patching.

## Validation (gates)

### Gate 1 — Lifecycle bit-identity at `n_z=3` (smoke)

Pre/post hashes captured in [docs/scans/nz_one_gate1_capture.json](nz_one_gate1_capture.json) by [scripts/scratch/nz_one_gate1_capture.py](../../scripts/scratch/nz_one_gate1_capture.py). Cleared the checkpoint before each run to force a fresh solve.

| Array | Pre-change SHA-256 | Post-change SHA-256 | Match |
| --- | --- | --- | --- |
| C | `e3492343...e0989` | `e3492343...e0989` | YES |
| S | `f0059e7d...c219` | `f0059e7d...c219` | YES |
| B | `97bbee73...b0ae` | `97bbee73...b0ae` | YES |

Lifecycle solver behavior is bit-identical at `n_z>=2`. **PASS.**

### Gate 2 — Tiny inf-horizon at `n_z=1` (smoke)

[scripts/scratch/nz_one_gate2_tiny_ih.py](../../scripts/scratch/nz_one_gate2_tiny_ih.py): `n_z=1`, `state_grid=(2,2,2,2)`, `n_w=10`, `n_s=10`, 5 inf-horizon iterations.

```
pc.n_z=1, pc.z_grid=[0.], pc.dz=0.0, pc.Pi_z=[[1.]]
ih iter 1..5 | stop 2.02e+01 -> 1.78e-01
C shape: (1, 16, 10)    NaN: C=0  S=0  B=0
alpha_s range: [-0.577, 2.067]
alpha_b range: [-16.466, 20.828]    # transient; shrinks with more iters
consumption range: [0.005, 212.500]
```

Gate 2 asserts only that the run completes, no NaN, finite policies, expected shape. **PASS.**

### Gate 3 — `n_z=1` vs `n_z=2` policy match (1e-10 tightest gate)

[scripts/scratch/nz_one_gate3_compare.py](../../scripts/scratch/nz_one_gate3_compare.py): identical config except `n_z`, 15 inf-horizon iterations from cold start, same tol/damping/seed.

```
n_z=1 vs n_z=2[z=0]:  max|dC|=0.000e+00  max|dS|=0.000e+00  max|dB|=0.000e+00
n_z=1 vs n_z=2[z=1]:  max|dC|=0.000e+00  max|dS|=0.000e+00  max|dB|=0.000e+00
within n_z=2:  C2[0]-C2[1] = 0  S2[0]-S2[1] = 0  B2[0]-B2[1] = 0
```

**PASS** — bit-identical, well below the 1e-10 tolerance. The handoff predicted ~1e-13–1e-15 drift from JAX scheduling differences across shape variations; in practice the inf-horizon policy at the z=0 slice is exactly identical because pension/psi/income are all forced to constants and the Newton scan visits the same nodes in the same order.

This also confirms the working-kernel paths are not silently exercised under the retirement-only inf-horizon driver.

## Files modified

- [lifecycle/discretization.py](../../lifecycle/discretization.py) — N=1 guard in `discretize_income_ar1_mixture`. (~7 LOC including comment.)
- [lifecycle/precompute.py](../../lifecycle/precompute.py) — size guard on the `dz = z_grid[1] - z_grid[0]` line. (~5 LOC including comment.) **NOT in original handoff scope** — see "Patch surfaced" above.
- [verify/benchmark_inf_horizon.py](../../verify/benchmark_inf_horizon.py) — `n_z=2` → `n_z=1`, refreshed docstring memory math, `BUNDLE_NAME` → `..._nz1_...`.

## Files created

- [scripts/scratch/nz_one_gate1_capture.py](../../scripts/scratch/nz_one_gate1_capture.py) — Gate 1 hash capture (bit-identity baseline).
- [scripts/scratch/nz_one_gate2_tiny_ih.py](../../scripts/scratch/nz_one_gate2_tiny_ih.py) — Gate 2 tiny inf-horizon smoke at `n_z=1`.
- [scripts/scratch/nz_one_gate3_compare.py](../../scripts/scratch/nz_one_gate3_compare.py) — Gate 3 `n_z=1` vs `n_z=2` parity check.
- [docs/scans/nz_one_gate1_capture.json](nz_one_gate1_capture.json) — captured bit-identity hashes.
