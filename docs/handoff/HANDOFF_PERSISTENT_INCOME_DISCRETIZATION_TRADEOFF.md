# Handoff: Persistent-income discretization tradeoff

**Date:** 2026-05-07
**Branch:** `jax-rewrite`
**Recipient:** numerical-methods agent
**Effort:** 1-2 days. Pure investigation + comparative benchmark; mostly read-only of the production solver, plus a clean prototype of the alternative.
**Output:** a markdown report under `docs/scans/` with accuracy-per-compute curves and a recommendation.

---

## Question

In the current solver, persistent income `z` is represented on a uniform grid `z_grid` of size `n_z`. The continuation value `V(z, …)` is *stored* on that grid, and the conditional expectation `E[V(rho*z + eta, …) | z]` is computed by:

1. **Gauss-Hermite-Judd quadrature** on the continuous mixture-normal innovation `eta` (`n_eta_nodes` ≈ 3-4),
2. **Linear interpolation** of `V` between adjacent z-grid bins at each quadrature node `z_next = rho*z + eta_k`.

The candidate alternative is **Tauchen-style discretization**: precompute a transition matrix `Pi_z[i,j] = P(z_{t+1} ∈ bin_j | z_t = z_grid[i])` once at setup, then replace the inner expectation with a single dot product `V[next-state] @ Pi_z[i, :]`.

> **Find which method achieves the most accuracy per unit of compute, given the
> calibration we actually use** (high persistence ρ=0.991, two-component
> normal-mixture innovation that is highly skewed/leptokurtic, life-cycle
> backward-induction with Newton + EGM at every (z, state, savings) cell).
>
> The hope (from a recent convergence study; see `docs/scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md`)
> was that we could stop at n_z=10. We can't — n_z=10 produces 28 % relative
> consumption-policy error and 37 percentage-points stock-share error vs the
> n_z=70 reference. So we either solve at n_z ≥ 30 (current method, ~2× extra
> wall vs n_z=70 stops), or we change the discretization to extract more
> accuracy from a smaller n_z.

---

## Background — what the model does

### 1. Calibration (high persistence + skewed mixture innovation)

From `configs/_canonical.py::BASE_CONFIG`:

| Parameter | Value | Meaning |
|---|---|---|
| `rho` | 0.991 | AR(1) persistence on log persistent income z |
| `pz` | 0.176 | mass on mixture component 1 of the eta innovation |
| `mu_eta1` | -0.524 | mean of component 1 ("disaster" component) |
| `sigma_eta1` | 0.113 | std of component 1 |
| `mu_eta2` | (computed) ≈ +0.112 | mean of component 2; constrained so `E[eta]=0` |
| `sigma_eta2` | 0.046 | std of component 2 |

The mixture is **highly bimodal and skewed**: 17.6% mass at η ≈ -0.524 (a large negative jump), 82.4% mass at η ≈ +0.112 (a small positive drift). Standard deviation of the innovation is ≈ 0.25; stationary std of z is ≈ 1.87 (= 0.25 / √(1 − 0.991²)).

The transitory shock `eps` is a separate mixture, also analytically solvable with Judd (1998) Hermite-mixture quadrature; it does **not** persist and is not the subject of this handoff.

### 2. State / wealth / savings dimensions

Default lifecycle solve:

- 78 ages (22..99); 1 terminal age + 77 backward-induction ages.
- `n_z` = 10..70 (the variable under study).
- `N_state` = product of `state_grid_sizes`. System I has 1 axis (rtb only) with grid size 7; System IV has 4 axes (dp, spr, rtb, y_1) with sizes (5,5,5,5) for ~625 states.
- `n_w` = 180 wealth grid points; `n_s` = 180 savings grid points.
- Inner integrations: `n_eta` = 3-4 nodes, `n_eps` = 3-4 nodes, `n_state_quad` = 2-3 per axis (Cholesky tensor product), `n_ret_quad` = 3-5 per axis (tensor product over xr, xb).

Per-cell solve at one (z, state) pair: an EGM scan over 180 savings nodes, each invoking a 2-D Newton with line search. Inside the FOC kernel, an integrand is evaluated at every `(eta, eps, state-innov, return-shock)` quadrature node — i.e., the eta integration sits at the innermost level inside an already 4-D quadrature loop.

### 3. Solver architecture (where you'll be making changes)

The hot loop is:

```
backward induction over t in reversed(ages)
    vmap over (z_idx, i_state)
        EGM scan over savings-grid:
            Newton 2-D loop (alpha_s, alpha_b):
                FOC eval: integrate beta * E[u'(c') * R_p] - u'(c) over (eta, eps, v, ret) nodes
                          - reads V_next via linear-interp at z_next, multilinear at state_next
                          - bracket_uniform on z_next per eta-node
```

Relevant code:

- `lifecycle/solver.py:260-265` — `bracket_uniform(z, z_lo, dz, n_z)`: clips and computes `(iz_lo, frac_z)` for the linear interpolant on the uniform z-grid.
- `lifecycle/solver.py:2308-2311` — the eta-step inside the working-age kernel: `z_next = rho * z_now + eta_nodes` then `vmap(bracket_uniform)`.
- `lifecycle/solver.py:1304-1371` — `_solve_working_at_cell`: full per-cell driver. The FOC factory `working_foc_jac_ccv` consumes `eta_iz_lo, eta_frac_z, eta_weights` to weight the linearly-interpolated next-period continuation at each eta node.
- `lifecycle/solver.py:608-687` — `_newton_fori`: the `lax.fori_loop` Newton path. Currently `max_iter=100`, observed to be hit at the tiny-savings boundary on every (z, state) cell; this is a **separate** issue to track but it lives on the same hot path.
- `lifecycle/solver.py:1155-1186` — `_egm_scan_cell`: the `vmap(per_savings_point)(s_grid)` driver that produces `n_iters_egm` (max iters used) and `n_backtrack_egm`.

### 4. Current discretization machinery

- `lifecycle/discretization.py:306-342` — `discretize_income_ar1_mixture(rho, p, mu1, sigma1, mu2, sigma2, N, n_stds)`. Returns `(z_grid, Pi_z)`:
  - `z_grid = linspace(-n_stds * std_z, +n_stds * std_z, N)`. Currently `n_stds=2.25` for the System I sweep, giving `z_grid ∈ [-4.21, +4.21]` for the calibration's std_z ≈ 1.87.
  - `Pi_z[i, j]` is built via `mixture_cdf` on bin edges around `z_grid[j] - rho*z_grid[i]`. **This is exact for the mixture density** — there is no Tauchen "midpoint" approximation hidden in the bin probabilities.

- `lifecycle/precompute.py:336-345` — where `discretize_income_ar1_mixture` is called inside `build_precompute`. Both `z_grid` and `Pi_z` are stored on the precompute dataclass at `lifecycle/precompute.py:171-172`.

- `lifecycle/discretization.py:452-475` — `get_eta_quadrature_mixture(model, n_nodes)`: builds a Judd (1998) Gauss-mixture quadrature for the **continuous** eta density. Polynomial-exact up to order `2 * n_nodes - 1` against the mixture. This is what the solver currently uses; the bin-probability `Pi_z` is **not** consumed by the solver.

- `lifecycle/diagnostics.py:539` — explicit codebase comment: **`"NOTE: Pi_z is simulation-only — the solver uses eta quadrature."`** Pi_z is consumed only by `lifecycle/simulation.py:593` for drawing discrete z-paths during sim.

### 5. What "linear interpolation in z" really does

After `eta_k` is drawn from the Judd quadrature:

```
z_next_k       = rho * z_t + eta_k                   # continuous
iz_lo, frac    = bracket_uniform(z_next_k, z_lo, dz, n_z)
V_next_k       = (1 - frac) * V[iz_lo, ...] + frac * V[iz_lo+1, ...]
E[V_next | z_t] = sum_k eta_weights[k] * V_next_k
```

This is **piecewise-linear interpolation in z** combined with **exact-up-to-mixture-quadrature integration over eta**. Errors come from two sources:

1. **The continuous integral error from Judd quadrature is ≈ 0** at K=4 nodes for the mixture (15th-order polynomial exactness against the mixture density).
2. **The interpolation error of V along z is `O(dz²) * V''`** for smooth V; degrades to `O(dz)` near kinks (borrowing-constraint corner, tiny-savings boundary).

The empirically observed convergence rate of the *policy* is roughly **O(1/n_z)** — see §6 below — which is consistent with V having kinks at the constraint corners that linear interp can't exactly resolve.

---

## Findings from the n_z convergence study

Source: [docs/scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](../scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md)

The study solved System I (iid returns; N_state=7; rest of the calibration as canonical) at n_z ∈ {10, 15, 30, 70}, treated n_z=70 as reference, and computed both grid-vs-grid policy divergence and sim-path Euler residuals.

### Policy convergence (worst-cell vs reference, %-of-typical)

| n_z | sup ΔC (% of median C) | sup Δα_s (pp) | sup Δα_b (pp) | RMS ΔC (% of median) |
|---:|---:|---:|---:|---:|
| 10 | 78 % | 37 pp | 30 pp | 13 % |
| 15 | 39 % | 18 pp | 15 pp | 5.5 % |
| 30 | 14 % | 6 pp | 5 pp | 1.7 % |

Each ~doubling of n_z halves the divergence — clean **linear-in-1/n_z** convergence.

### Sim-path Euler residuals (mean log10|EE| on unconstrained cells; lower better)

| n_z | working | boundary | retirement |
|---:|---:|---:|---:|
| 10 | -2.13 | -2.75 | -4.54 |
| 15 | -2.49 | -2.97 | -4.82 |
| 30 | -2.92 | -3.32 | -5.14 |
| 70 | **-3.92** | **-4.24** | **-5.31** |

n_z=70 working-age residual (-3.92) sits just above the welfare gate (-4.0) and far from the publication gate (-5.0). The residual at n_z=70 is **not** dominated by z-resolution; it's dominated by the Newton iter cap (see §7).

### Where the divergence concentrates

- **Young working ages (22-25)** — z-distribution disperses fastest there.
- **Right tail of z (z ≈ +0.5σ to +1σ)** — moderately-high realised income with the deterministic age-earnings interaction creates fast-varying optimal-consumption surfaces.
- Two wealth modes: max wealth for C; lower-middle wealth band (~33% of grid range) for portfolio shares.

### Wall time per bundle (System I, full 78-age solve)

| n_z | wall |
|---:|---:|
| 10 | 33.5 s |
| 15 | 42.9 s |
| 30 | 72.4 s |
| 70 | 146.0 s |

Wall scales roughly linearly with `n_z` because the per-cell work doesn't change but the cell count does (`vmap` over `n_z * N_state`). For System IV (N_state=625) the n_z=70 wall would extrapolate to ~13 hours.

---

## What you need to investigate

### Hypothesis A — quadrature + linear-interp wins

The current method's per-cell cost on the eta axis:

- 1 `bracket_uniform` per eta node (≈ 2 multiplies + 2 adds + 1 cast + clip)
- 1 linear interpolant per eta node (1 mul + 1 add over the `(state, savings)` slice of V)
- Weighted sum over `n_eta_nodes` ≈ 3-4 nodes.

Total: ~`3 * (4 ops)` = O(15) FLOPs per cell on the eta axis (the rest of the FOC eval is the dominant cost).

Accuracy: O(dz²) for smooth V, O(dz) at kinks. Empirically O(1/n_z) for our calibration → kinks are dominant.

### Hypothesis B — Tauchen Pi_z dot product wins

Replace the eta-axis inner step with:

```
E[V_next | z_t = z_grid[i]] = sum_j Pi_z[i, j] * V[j, ...]
```

Pre-tabulated `Pi_z` is shape `(n_z, n_z)` — already built by `discretize_income_ar1_mixture`, exact for the mixture density via `mixture_cdf` on bin edges. **No quadrature is needed.**

Per-cell cost on the eta axis:
- 1 dot product of length n_z over the `(state, savings)` slice of V.
- O(n_z) FLOPs per cell.

For n_z=30, that's 30 FLOPs vs ~15 for quadrature. **Per-cell, Tauchen is roughly 2× more expensive** in FLOPs at our typical n_z. But it might enable smaller n_z by reducing the *interpolation* error component (Pi_z is "piecewise constant in V across bin j" rather than linear).

The honest comparison is **accuracy-per-compute on the SAME (n_z, calibration)**, plus seeing whether Tauchen at small n_z beats quadrature at larger n_z.

### Hypothesis C — neither wins outright; recommend a hybrid or a different discretization

Possibilities the agent should also consider:

- **Rouwenhorst** for high-persistence Gaussian AR(1). Standard recommendation when ρ > 0.95. **Does not natively handle mixture innovations** — would have to be applied to the matched-moments Gaussian approximation of the mixture, losing the skew/kurtosis. Probably not a fit, but worth one-paragraph dismissal in the report.
- **Cubic-spline interpolation in z** keeping the current quadrature path. Trades O(dz²) → O(dz⁴) accuracy on smooth V; doesn't help at kinks. Memory: 4 spline coefficients per (z, state, wealth) cell. Implementation: feasible inside JAX (`scipy.interpolate.CubicSpline` is not jit-friendly; need a hand-written Catmull-Rom or natural-spline tableau).
- **Adaptive z-grid** (denser nodes in [-1σ, +1σ] where stationary mass concentrates and policy curvature is highest). Easy: replace `linspace` with a custom non-uniform grid; `bracket_uniform` becomes `bracket_axis` (already exists in solver.py:268-274).
- **More Judd quadrature nodes** for the eta integration (lift `n_eta_nodes` from 3 to 5+) — orthogonal to the z discretization question. May or may not improve accuracy depending on whether the residual is quadrature-error-dominated or interp-error-dominated.

### Required deliverables

1. **A working prototype of method B** that swaps out the inner eta integration for `Pi_z @ V` on at least the working-age kernel. Must be JAX-jittable and produce policies in the same shape `(78, n_z, N_state, n_w)`. Suggested location: `lifecycle/solver_pi_z_variant.py` so the production solver stays untouched.
2. **A benchmark script** that:
   - Solves System I at e.g. {10, 15, 20, 30, 50} with **both** methods.
   - Reports wall time per bundle, sup-norm policy divergence vs the existing n_z=70 quadrature reference, RMS divergence, and (optionally, expensive) sim-path Euler residuals via `verify/ee_simpath.py`.
3. **A plot** of accuracy (sup-norm policy divergence) vs compute (wall) for both methods. The intersection — if any — tells you the regime where Pi_z wins.
4. **The recommendation** in `docs/scans/PERSISTENT_INCOME_DISCRETIZATION_2026-MM-DD.md`: which method, at which n_z, for which downstream use case. Specifically:
   - Is there a Pareto-dominant choice? (probably no)
   - For the canonical lifecycle solve (System I-IV, n_z producing publication-grade sim-EE), which is faster?
   - If we wanted to reduce n_z to 15-20 to save compute on the System II/III/IV ablations, does method B make that defensible?

### Constraints and gotchas

- **Stay JAX-jittable**. The hot loop is inside a chained `vmap` + `lax.fori_loop` + `jit`. `Pi_z` becomes a closure constant — fine, JAX traces it as a static array. But don't introduce host-side branching or Python loops over n_z inside the hot path.
- **`Pi_z` storage**: 70×70 = 4,900 floats = 39 KB. Trivial. But if you go higher than n_z=200 the matrix starts to compete with V for L1 cache; benchmark on actual hardware.
- **`Pi_z` per age vs once**: `Pi_z` does not depend on age — compute once at setup. The solver's existing `_pc_to_jnp` already moves it to device.
- **High-persistence row sums**: at ρ=0.991 with z_grid endpoints at ±2.25 σ_z = ±4.21, edge rows of `Pi_z` may have row sum < 1 because the conditional-eta CDF puts mass outside the grid. The current `discretize_income_ar1_mixture:338-340` renormalizes to row sum 1. **Verify this renormalization doesn't bias the conditional mean** — it should, slightly, by clamping outliers to the boundary. Compute the bias once: `sum_j Pi_z[i,j] * z_grid[j] - rho * z_grid[i]` per row i.
- **Mixture quadrature validity at n_eta < 4**: `get_eta_quadrature_mixture` uses Judd's Hermite-mixture construction; at n_eta=3 it's polynomial-exact up to order 5 against the mixture density. For our calibration's skew/kurtosis, that's marginally enough — most of the sim-EE error at n_z=70 likely comes from this, not from the z grid. Bump n_eta to 5 or 7 as part of the agent's investigation; the cost is linear in `n_eta_nodes` per FOC eval (cheap).
- **Newton iter cap is a separate issue**: `solver_config.max_iter=100` is currently hit at the tiny-savings boundary on every (z, state) cell. Don't try to fix it as part of this handoff; it's tracked separately. But your benchmarks should use the same `max_iter` for both methods so the comparison is apples-to-apples.

---

## Architecture-level guardrails

Things you should NOT change in scope of this work:

- The 4-D quadrature over `(eta, eps, v, ret)` — only the eta integration touches `n_z`. Leave the others untouched.
- The CCV log-portfolio return arithmetic — that lives at `_ccv_log_return_and_grad` in solver.py; never hot-path it.
- The state grid (Cholesky tensor product over the financial state vector). Different axis, different problem.
- The wealth grid construction (`np.expm1` on log-spaced indices, see `lifecycle/precompute.py:234-237`). It's well-tuned and orthogonal.
- The simulation pathway. `Pi_z` is already used in simulation; this work is about whether it should also be used in the *solver*. Don't co-modify `lifecycle/simulation.py`.

Things you SHOULD review but probably won't change:

- `BASE_CONFIG` calibration parameters. They are set by economic theory + data fitting; don't touch.
- `n_stds=2.25` for the System I sweep. Worth checking sensitivity (does going to 3.0 σ change the picture?), but don't conflate that with the method-comparison study.

---

## Reproducing the convergence-study findings

```sh
# Bundles on S3 at s3://hugo-thesis-runs/saved_runs/ablations/system_i_grid7_nz<N>_calib1/
#   for N in {10, 15, 30, 70}. Sync locally if absent:
aws s3 sync s3://hugo-thesis-runs/saved_runs/ablations/ saved_runs/ablations/

# Existing convergence study (you'll re-use its z-grid reconstruction logic):
python scripts/analysis/system_i_nz_convergence.py
python scripts/analysis/plot_nz_convergence.py

# Sim-path Euler residual diagnostic (the gold-standard correctness check).
# For System I bundles, must use the wrapper that picks the right VAR builder
# (verify/ee_simpath.py hardcodes System IV's builder):
for nz in 10 15 30 70; do
  python scripts/analysis/run_ee_simpath_system_i.py \
    saved_runs/ablations/system_i_grid7_nz${nz}_calib1 \
    --eval-mode same --n-simulations 500 \
    --eval-households-per-age 128 --seed 42 \
    --out-suffix _nz_convergence
done

# Re-running the sweep solver (only if you need to add new n_z values):
python verify/benchmark_system_i_nz_sweep.py
```

---

## Suggested implementation skeleton for method B

```python
# lifecycle/solver_pi_z_variant.py

from lifecycle.solver import _newton_fori, _build_step_log_returns, ...  # reuse
import jax.numpy as jnp
from jax import vmap, jit


def _build_per_age_working_kernel_pi_z(pcj, mp, sc, n_z, N_state, ...):
    """Drop-in alternative to _build_per_age_working_kernel_vmap_only that uses
    Pi_z @ V_next instead of bracket+linear-interp on eta-quadrature nodes.

    Pi_z is closed over from the precompute (pcj.Pi_z); becomes a static
    constant in the jit trace.
    """
    Pi_z_jnp = jnp.asarray(pcj.Pi_z)  # (n_z, n_z)

    def per_cell(z_idx, i_s, c_next, ...):
        # c_next has shape (n_z, N_state, n_w); for each (state, w) compute
        #   E_next[c_next | z_idx] = Pi_z[z_idx, :] @ c_next[:, state, w]
        # then bracket the state and return-shock dimensions as before.
        c_next_at_zexp = Pi_z_jnp[z_idx] @ c_next.reshape(n_z, -1)
        c_next_at_zexp = c_next_at_zexp.reshape(N_state, c_next.shape[-1])
        # ... rest of the FOC factory uses c_next_at_zexp, NOT eta-quadrature
        ...
```

The conceptual delta is small (~50 lines); the engineering delta is moderate (you have to re-thread the FOC kernel `working_foc_jac_ccv` to consume `E[c_next | z]` directly instead of bracketed eta-node values, which means a fresh derivative path for the Jacobian terms). **Validate against the existing kernel at n_z=70 first** — both methods should produce identical policies up to mixture-quadrature error if Pi_z is built from the same mixture density.

---

## What "good" looks like at the end of this handoff

A two-page report at `docs/scans/PERSISTENT_INCOME_DISCRETIZATION_2026-MM-DD.md` answering:

1. At the production calibration (ρ=0.991, two-component mixture innovation), does Tauchen-Pi_z beat quadrature+linear-interp on accuracy-per-compute?
2. If yes: at what n_z, for what use cases, by how much?
3. If no: where does Pi_z's accuracy advantage go, and why doesn't the FLOP cost trade work out?
4. Concrete recommendation: keep current method, switch wholesale, or hybrid (e.g., spline + quadrature)?
5. **One number:** what's the smallest n_z under method B at which we'd be comfortable defending policies as "publication-grade" (working-age sim-EE mean log10|EE| ≤ -4.0)? Compare to n_z=70 under the current method.

If the answer to (1) is "no" with high confidence, the report can be one page. If "maybe" — produce the plot showing accuracy-vs-compute Pareto frontier of both methods so we can decide later.

---

## Background reading

- Tauchen (1986), "Finite state Markov-chain approximations to univariate and vector autoregressions" — the canonical reference for the Pi_z construction.
- Kopecky & Suen (2010), "Finite state Markov-chain approximations to highly persistent processes" — argues Rouwenhorst > Tauchen for ρ > 0.95 in the Gaussian case.
- Judd (1998), "Numerical Methods in Economics" §6.5 — the Hermite-mixture quadrature construction we currently use.
- Internal: `docs/CCV_RETURN_IMPLEMENT.md` — the model spec; not directly relevant to z-discretization but you should skim §2 to understand how the FOC integrand depends on z'.
- Internal: `docs/scans/SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md` — the empirical baseline you'll be measuring against.
