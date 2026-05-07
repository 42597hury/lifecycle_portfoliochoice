# Handoff: Analyze System I × n_z Sensitivity Sweep

**Branch:** `jax-rewrite`
**Effort:** ~half day. Pure analysis scripts + plots + a short written verdict. No solver-side changes.
**Output:**
- `scripts/analysis/system_i_nz_convergence.py` — main analysis script (load bundles + compare).
- `scripts/analysis/plot_nz_convergence.py` — visualization helpers (or a single combined script).
- `docs/scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md` — short report with convergence curves, sup-norm tables, and a verdict on whether n_z=10 is adequate.
- (Plots, embedded or in `docs/scans/figures/`.)

---

## Purpose

Quantify how much **labour-income state-grid resolution (`n_z`)** matters for consumption and portfolio policies in the simplest predictability system (System I, rtb-only state, iid returns). The economic claim under test:

> **At n_z=10, the policy is already converged.** Further refinement (15, 30, 70) produces only negligible changes.

If true, this lets the user defend `n_z=10` (or even smaller) for the broader ablation and lifecycle runs — saving ~7× compute over canonical `n_z=70` and ~3× over `n_z=30`. If false, the user needs `n_z>10` for publication-grade policies in System I.

The user expects YES (stopping at 10). The analysis must either confirm this or surface the cells where it doesn't hold.

---

## Inputs

Four bundles, all from the `system_i_grid7_nz<N>_calib1` sweep:

```
saved_runs/ablations/system_i_grid7_nz10_calib1/
saved_runs/ablations/system_i_grid7_nz15_calib1/
saved_runs/ablations/system_i_grid7_nz30_calib1/
saved_runs/ablations/system_i_grid7_nz70_calib1/
```

Each bundle contains:
- `policy_arrays.npz` — `C, S, B` arrays of shape `(78, n_z, 7, 180)`. Axes: (age, z, state, wealth).
- `metadata.json` — config snapshot, axis sizes
- `diagnostics.pkl` — including `newton_iter_histogram`, per-age FOC residuals, etc.

Also on S3 at `s3://hugo-thesis-runs/saved_runs/ablations/system_i_grid7_nz<N>_calib1/` if local copies are missing.

---

## Existing scripts to reuse (don't reinvent)

| Module | Function | Use |
|---|---|---|
| `lifecycle/policy_io.py` | `load_policy_bundle(bundle_dir)` | Returns `(C, S, B, diagnostics, run_config)`. Use this everywhere — don't manually `np.load` the npz. |
| `lifecycle/precompute.py` | `build_model`, `build_precompute` | If you need to rebuild the z-grid for any bundle (the actual z-grid coordinates aren't stored in the bundle, only n_z). |
| `lifecycle/discretization.py` | `discretize_income_ar1_mixture` | The function that produces z-grid coordinates from `(rho, mu_eta1, sigma_eta1, ..., n_z, n_stds)`. Useful for mapping between different n_z values. |
| `lifecycle/simulation.py` | `simulate_lifecycle` | Optional — if you want a sim-based comparison rather than grid-based. Forward simulation produces realized paths that integrate over the equilibrium distribution. |
| Bundle metadata | `disc_config.n_z`, `disc_config.n_stds` | Read these from each bundle's metadata to construct z-grids consistently. |

**Don't reinvent the wheel.** Specifically don't write a new bundle loader; `load_policy_bundle` exists for a reason.

---

## Analysis plan

### §1 — Load and align

For each of the 4 bundles:
1. Use `load_policy_bundle` to read `(C, S, B, diag, run_config)`.
2. Read `n_z` from `run_config["discretization_config"]["n_z"]`.
3. Reconstruct the z-grid via `discretize_income_ar1_mixture(...)` using parameters from `run_config["base_config"]` and `disc_config.n_stds`.

Now you have 4 tuples: `(z_grid_nz, C_nz, S_nz, B_nz)` with z_grid varying in length.

### §2 — Reference + interpolation framework

Pick **n_z=70 as the reference** (finest discretization). The hypothesis is "smaller grids equal-or-close to this."

For each coarser bundle (n_z ∈ {10, 15, 30}):
- Interpolate the coarser policy onto the n_z=70 z-grid via linear interpolation in z.
  - Specifically: at each fixed `(age, state, wealth)` triple, linearly interpolate `C[age, :, state, wealth]` from the coarser z-grid to z_grid_70.
- Now both bundles produce policies on the same shape `(78, 70, 7, 180)`.
- Compute `delta = | C_coarse_interp - C_70 |` element-wise.

Some thinking:
- The z-grid is constructed via `np.linspace(-n_stds*std_z, n_stds*std_z, N)`. With `n_stds=2.25` (per the sweep config), z values are evenly spaced over `[-2.25σ, +2.25σ]`. The endpoints match across bundles (modulo the `std_z` calculation, which doesn't depend on n_z). So interpolation between grids is well-defined.
- For consumption `C`, linear interp in z is reasonable. For portfolio shares `S, B`, you might want to compare in **dollar terms** (`α × W`) rather than share terms; some shares can flip sign between adjacent z-grid nodes in extreme regions. Use share interpolation for the headline result; flag dollar-term differences if they tell a different story.

### §3 — Convergence metrics

For each pair (n_z=N, n_z=70), compute:

1. **Sup-norm divergence**:
   - `sup_C(N) = max | C_interp(N) - C_70 |`
   - `sup_S(N) = max | S_interp(N) - S_70 |`
   - `sup_B(N) = max | B_interp(N) - B_70 |`

2. **L2 / RMS divergence** (weighted by some natural measure):
   - Either uniform across cells, or weighted by `wealth_grid` density (since extreme wealth cells may matter less).
   - L2 captures average-case error; sup captures worst-case.

3. **Per-axis max:**
   - `sup_C_per_age(N) = max over (z, state, wealth) of |C(N) - C_70|` (one number per age — shows where the divergence concentrates in time).
   - `sup_C_per_z(N)` — similar but max'd over (age, state, wealth) for each z. Reveals whether divergence is at extreme z values.
   - Same for S, B.

4. **Relative divergence** (normalize by mean policy at the same point):
   - `rel_C(N) = max | (C_N - C_70) / C_70 |` (where C_70 ≠ 0)
   - More economically interpretable than absolute differences.

### §4 — Convergence-curve visualization

Produce convergence-curve plots:
- X-axis: n_z (10, 15, 30, 70 — log scale or linear, your call).
- Y-axis: sup-norm divergence vs reference (n_z=70).
- One line per metric (C, S, B). Possibly separate panels.
- The "stops at 10" verdict is visible if the curve is FLAT below n_z=15 — i.e., n_z=10 already at the asymptote.

Also produce policy-vs-age line plots at one or two probe cells:
- z=0 (mean income), state=mid (state-grid center), wealth=SCF median.
- Plot C, S, B on three subpanels, one curve per n_z value (overlaid).
- If the four lines visually overlap → CONVERGED; if they fan out → NOT converged.

(The probe-index fix landed yesterday. Use the per-axis midpoint convention.)

### §5 — Distribution analysis (optional but useful)

Compare policy distributions across n_z:
- Histogram of `alpha_s` values across all `(age, z, state, wealth)` cells, one histogram per n_z.
- Histogram of `alpha_b` similarly.
- If distributions visually match → converged.
- If the n_z=10 distribution has a different shape than n_z=70 (e.g., missing tail mass) → that's the cells where coarser n_z fails.

### §6 — Summary verdict

Write a 1-2 page report:
- Overall verdict: GREEN (n_z=10 adequate) / YELLOW (n_z=10 close but n_z=15 better) / RED (n_z=10 visibly under-converged)
- Per-metric numbers: sup-norm and L2 divergences in a table
- Where the residual divergence lives (if any): age range, z range, wealth range
- Policy implication: "Use n_z=10 for X workflow; use n_z=Y for canonical publication."

---

## Implementation skeleton

```python
# scripts/analysis/system_i_nz_convergence.py
import numpy as np
from lifecycle.policy_io import load_policy_bundle
from lifecycle.discretization import discretize_income_ar1_mixture

NZ_VALUES = (10, 15, 30, 70)
BUNDLES = {nz: f"saved_runs/ablations/system_i_grid7_nz{nz}_calib1/" for nz in NZ_VALUES}

def load_one(nz):
    C, S, B, diag, run_config = load_policy_bundle(BUNDLES[nz])
    base = run_config["base_config"]
    n_stds = run_config["discretization_config"]["n_stds"]
    # The z-grid construction depends on (rho, mu_eta1, sigma_eta1, ..., nz, n_stds).
    # Pull params from base config; reconstruct via discretize_income_ar1_mixture.
    z_grid, _Pi_z = discretize_income_ar1_mixture(
        rho=base["rho"], p=base["pe"],
        mu1=base["mu_eps1"], sigma1=base["sigma_eps1"],
        mu2=base["mu_eps2"], sigma2=base["sigma_eps2"],
        N=nz, n_stds=n_stds,
    )
    return z_grid, C, S, B, diag

# Note: the z-grid construction uses eps params, not eta. Verify which matches
# the actual income process used in build_model — check precompute.py to be sure.

# Reference: nz=70
z70, C70, S70, B70, _ = load_one(70)

results = {}
for nz in (10, 15, 30):
    z_n, C_n, S_n, B_n, _ = load_one(nz)
    # Interpolate each cell's z-axis from coarse to fine
    # C_n shape: (78, nz, 7, 180); want to project to (78, 70, 7, 180)
    C_interp = np.zeros_like(C70)
    for age in range(C_n.shape[0]):
        for state in range(C_n.shape[2]):
            for w in range(C_n.shape[3]):
                C_interp[age, :, state, w] = np.interp(z70, z_n, C_n[age, :, state, w])
    # ... same for S, B (or use np.apply_along_axis / vectorized form for speed)

    delta_C = np.abs(C_interp - C70)
    delta_S = np.abs(S_interp - S70)  # similar
    delta_B = np.abs(B_interp - B70)
    results[nz] = {
        "sup_C": float(delta_C.max()),
        "sup_S": float(delta_S.max()),
        "sup_B": float(delta_B.max()),
        "p99_C": float(np.percentile(delta_C, 99)),
        # etc.
    }

# Print + save verdict
for nz, r in results.items():
    print(f"n_z={nz}: sup_C={r['sup_C']:.4f}, sup_S={r['sup_S']:.4f}, sup_B={r['sup_B']:.4f}")
```

The triple-nested loop is illustrative; vectorize it with `np.apply_along_axis(np.interp, axis=1, ...)` or similar for speed.

---

## Validation gates

- **Bundles all loaded successfully** — no NaN, no missing files.
- **Reconstructed z-grids are sensible** — symmetric around 0, monotone, expected ranges (e.g., `[-2.25σ, +2.25σ]`).
- **Reference bundle (n_z=70)** has sane policies — alpha ranges in a reasonable band, no NaN.
- **Sup-norm divergences for n_z=70 vs itself are exactly 0** — sanity check on the comparison code.

---

## Pause points

- **Bundle loading fails** for any of the 4: stop, report which one is broken (incomplete, corrupted, mismatched shape, etc.).
- **Reference (n_z=70) has NaN or extreme outliers**: the sweep itself may have failed silently. Surface this before doing analysis.
- **Sup-norm divergences are massive** (e.g., `sup_C > 1.0` in absolute terms): suggests the bundles aren't comparable (maybe different n_stds, different ages solved, different config). Verify config consistency before reporting numbers.
- **z-grid reconstruction doesn't match** what `build_precompute` would have produced: the analysis is built on the wrong reference. Pause and verify which discretizer to call (`discretize_income_ar1_mixture` vs an Eta-based discretizer if there are multiple in the codebase).

---

## Out of scope

- **Solver-side changes.** No code in `lifecycle/solver.py`, `lifecycle/precompute.py`, etc.
- **Re-running any of the 4 bundles.** Use only what's in `saved_runs/ablations/...`.
- **Sim-based EE comparison.** Different question (sim-EE assesses correctness; this analysis is convergence-with-resolution). Stay grid-based unless you find time and the conclusion is unclear.
- **Cross-system comparisons** (System I vs System II vs IV). Different deliverable; out of scope here.
- **Performance optimization** of the analysis code. It's a one-shot run on 4 bundles totaling <1 GB; even slow numpy is fine.

---

## Implementation checklist

- [ ] §1 — Load all 4 bundles; verify shapes + n_z; construct z-grids.
- [ ] §2 — Build interpolation framework; verify n_z=70 vs itself = 0 divergence.
- [ ] §3 — Compute sup-norm, L2, per-age, per-z, relative divergences for C/S/B.
- [ ] §4 — Convergence-curve plot + per-cell line plots.
- [ ] §5 (optional) — Distribution histograms.
- [ ] §6 — Write `docs/scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md` with verdict.
- [ ] Commit:
  ```
  docs+analysis: System I n_z sensitivity convergence study
  
  Loads the 4 ablation bundles (system_i_grid7_nz<N>_calib1) and
  computes policy convergence as n_z grows. Verdict: <GREEN: n_z=10
  adequate / YELLOW: n_z=15 recommended / RED: under-converged>.
  
  Sup-norm divergences at n_z={10,15,30} vs n_z=70 reference:
  C: ..., S: ..., B: ...
  
  Recommendation for downstream workflows: <one-line>.
  ```

---

## Why this matters

The user expects `n_z=10` to be adequate. If confirmed:
- All future ablation runs (Systems II, III, IV — all n_z values) can stop at n_z=10.
- Saves 3-7× compute on every future ablation run.
- Strengthens thesis methodology section (n_z choice is defensible from data, not arbitrary).

If NOT confirmed:
- The user needs to pick a higher n_z for canonical runs.
- The convergence curve tells them WHICH n_z value to pick.

Either way, the report is a durable methodology artifact for the thesis.
