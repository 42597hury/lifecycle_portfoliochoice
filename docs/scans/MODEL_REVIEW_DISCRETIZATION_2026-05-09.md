# Model review: discretization & approximation theory

**Date:** 2026-05-09
**Branch:** `jax-rewrite`
**Scope:** code-level correctness of grids, quadrature, and projection in
the post-pivot 3-axis (cape, spr, y_1) lifecycle model. Sensitivity findings
already covered in the existing scans are **not** revisited here; the focus
is on theory bugs, undocumented assumptions, and silent inconsistencies.

Files audited:
`lifecycle/precompute.py`, `lifecycle/discretization.py`,
`lifecycle/wealth_grid.py`, `lifecycle/model.py`,
`lifecycle/var.py` (partition only), `configs/_canonical.py`,
`lifecycle/solver.py` (lines 816–878, 1180–1300, 2721–2772 only).

---

## 1. State grid construction

**What I found.** `lifecycle/discretization.py:222–249`. In `cholesky` mode the
grid is `state_grid[i] = mu_s + L @ u`, with `L = chol(Σ_z)` and `Σ_z` solved
from the discrete Lyapunov equation (`stationary_covariance`,
`discretization.py:65–85`). `mu_s = (I − Φ)⁻¹ μ_intercept` is the unconditional
mean. Per-axis bounds `state_n_stds` operate in the orthonormal `u`-space, so
joint coverage = ∏ Φ-coverage of each axis.

State ordering is `(cape, spr, y_1)` (`lifecycle/var.py:441–445`), and
`L` is lower-triangular. Therefore `u_0` ↔ pure cape, `u_1` ↔ residual spr
after cape, `u_2` ↔ residual y_1 after both. The K-bump
`n_state_quad_nodes=(3,3,5)` refines `u_2`, which is the correct y_1-residual
direction once cape and spr have been orthogonalised away. **Order matters
and is consistent.**

`stationary_probs` in cholesky mode (`discretization.py:239–246`) computes
1D normal bin probabilities axis-by-axis on the standardised `u`-grid via
`_normal_bin_probs`, then takes the tensor product. This is correct **only
because** `u = L⁻¹ (z − μ_s) ~ N(0, I)` by construction, so the per-axis
marginals are independent standard normals. The non-orthogonal modes
(`naive`, `lyapunov-axis`) instead derive `stationary_probs` from the
Rouwenhorst Π_state's stationary distribution (`discretization.py:88–101`,
`280`) — that path is mathematically distinct, but is no longer the
production path (canonical sets `state_grid_mode="cholesky"`).

**Verdict.** CLEAR. Cholesky construction, ordering, and probability
weights are internally consistent.

---

## 2. State quadrature (innovations)

**What I found.** `lifecycle/discretization.py:690–751`. `get_state_quadrature`
draws tensor-product Gauss-Hermite nodes in standard-normal `z`-space,
maps to `v ~ N(0, Σ_ss)` via `v = z @ L^T` with `L = chol(Σ_ss)`. The
solver then forms `s_next = Φ_0 + Φ_11 s_t + v` (`solver.py:861`) — i.e.
quadrature is over the **conditional** one-step innovation, not the
unconditional Σ_z. That matches how Σ_z is used (state grid coverage) vs how
Σ_ss is used (transition density). **Correct separation.**

`_validate_state_quadrature` (`precompute.py:526–547`) verifies the
return-mean closure `Σ_k w_k μ_r_k = Φ_0_ret + Φ_21 s_i` to 1e-10. This
fires every build and would catch a Σ_ss mis-Cholesky or weight bug.

K-bump `(3,3,5)`: tensor-product, NOT Smolyak — total nodes = 45 per
source state. No vestigial Smolyak code lives in `lifecycle/`; the only
`smolyak_feasibility_jax.py` reference sits in `scripts/scratch/` and is
not imported anywhere on the live path. The closed-form Lobatto branch
(`quadrature_with_tails.gauss_hermite_prescribed_tails`) is reachable only
when `state_lobatto_Z` is not None; canonical sets it to None.

**One minor concern.** `get_state_quadrature` does NOT call
`_validate_state_quadrature` itself (the runtime closure check lives in
`build_precompute`); `v_weights.sum() == 1` is implicit by the GH 1D
weight normalisation (`_build_axis_grid`) but is never explicitly
asserted on the joint product. Robust enough in practice, but adding
`assert abs(v_weights.sum() - 1) < 1e-12` after the meshgrid would be
cheap insurance.

**Verdict.** CLEAR (with one suggested defensive assert).

---

## 3. Return quadrature (joint xr × xb)

**What I found.** `lifecycle/discretization.py:579–657`. Tensor-product
GH(4)×GH(4) standard-normal nodes, transformed by Cholesky of
`Σ_r_cond = Σ_rr − M Σ_sr` (the conditional residual return covariance).
**`r = z @ L^T` is the joint-distribution transform** — the cross-term
covariance enters via the off-diagonal `L[1,0]`, so the integration
respects the joint `(xr, xb)` distribution. The CCV scalar
`σ_xrxb` consumed by the FOC kernel (`precompute.py:340–342`) is sourced
from **Σ_rr** (unconditional), not Σ_r_cond, with an explicit comment
explaining why (CCV w8566 eq. 10 expectations are taken over the FULL
innovation, not the residual). **This is the right matrix and the right
source.**

The K=4 per axis on `n_ret=2` returns is consistent with the per-axis
sweep finding (refinement to (3,5)/(5,3) didn't move policy).
`n_ret_nodes_1d=(4, 4)` produces 16 joint nodes; under standard-normal
GH, exactness up to polynomial degree 7 in each marginal.

**Verdict.** CLEAR.

---

## 4. Income shock quadrature

**What I found.** `lifecycle/discretization.py:369–475`. `_judd_mixture_quadrature`
implements Judd (1998) Hankel-matrix construction: solves for the monic
orthogonal polynomial of degree `n` against the mixture density, computes
the polynomial roots (quadrature nodes), then solves the Vandermonde
weight system. Polynomial exactness is `2n-1` against the mixture.

**Zero-mean enforcement.** `get_eta_quadrature_mixture` and
`get_eps_quadrature_corrected` recompute `mu2_eff = -(p/(1-p))·mu1`
internally (`discretization.py:438–443`, `464–469`) — the model's
`mu_eta2` / `mu_eps2` fields are **silently ignored**.

For η this is fine: `_canonical.py:36` sets `mu_eta2 = -(pz/(1-pz))*mu_eta1`
explicitly, so model and effective values agree. **For ε this is a
silent inconsistency:** `_canonical.py:39` sets `mu_eps2 = 0.0`, but
the eps-quadrature internally uses `mu_eps2_eff = -(0.044/0.956)·0.134 ≈
-0.00617`. The two values differ; the model field is dead weight.

**Cross-coupling check.** `discretize_income_ar1_mixture` (the persistent
z-grid construction, `discretization.py:306–342`) **does** use
`model.mu_eta2` directly via the precompute call at `precompute.py:369`,
not the recomputed effective value. Because `_canonical.py:36` makes the
two equal, this happens to be safe for the canonical config — but a user
who manually overrides `mu_eta2` and forgets the constraint would put the
z-grid bin probabilities and the η Judd weights out of sync. **The
constraint should be enforced in one place** (either at config validation
time or by stripping `mu_eta2` / `mu_eps2` from the model entirely and
recomputing them from `(p, mu1)` everywhere).

**Working ages only.** `_precompute_working_income_next` (`precompute.py:585–634`)
populates the table only for `t+1 < retire_age_idx`; rows where
next-period is retirement get zeroed. The retirement-transition FOC uses
`pension_after_tax` with linear z-interpolation (per docstring) instead of
this table. **Confirmed correct.**

**Verdict.** UNDOCUMENTED. The dead `mu_eps2=0.0` field and the
"effective recomputation in two of three call-sites, raw model field in
the third" pattern are subtle landmines; suggest either deleting
`mu_eta2`/`mu_eps2` from `BASE_CONFIG` (and reconstructing internally) or
adding a `__post_init__` guard that raises when the constraint is
violated.

---

## 5. Wealth grid

**What I found.** `lifecycle/wealth_grid.py:16–28`. Default path is
`legacy_log1p_wealth_grid`: `expm1(linspace(log1p(min), log1p(max), n))`.
Confirmed live (`precompute.py:247–252`). The Bakhvalov / custom path
loads from `.npy`/`.npz` and runs `validate_wealth_grid`
(`wealth_grid.py:77–141`), which enforces strictly-increasing both in fp64
and after fp32 cast, plus an absolute and relative spacing floor — solid
defenses against a hand-edited grid breaking solver bracketing.

The solver bracketing path uses `searchsorted` (handled by
`jnp.interp` in `_lift_to_wealth_grid`, `solver.py:1249–1259`), so
non-uniform grids work; previously-confirmed as part of the Bakhvalov
handoff.

**`wealth_min=0.05` exposes the constraint.** With the legacy floor at
0.13, the smallest grid point sat above the EGM constraint kink for
median income; lowering to 0.05 makes the constrained branch the
solver's responsibility. The EGM scan (`solver.py:1211–1232`) handles
`s ≤ tiny_savings = 1e-6` via the cold-init fallback, which sits BELOW
`savings_min = 1e-8` — wait, this is backwards. Let me restate:
`tiny_savings = 1e-6 > savings_min = 1e-8`. So **the smallest 1–2
savings-grid points (s ≈ 1e-8 ... ~1e-7) fall inside the tiny branch**
and get sentinel-replaced with `(min_consumption, init_a_s, init_a_b)`.
Those rows then feed the EGM lift to wealth, where they show up as the
constrained-branch endogenous wealth at `c_min + s_tiny ≈ min_consumption`.
This is by design (per `verify/invalid_cells.py`), but the redundancy
between `savings_min` and `tiny_savings` is **undocumented** — the
`s_grid` could safely start at `tiny_savings` instead of going through
two sub-tiny points that immediately get overwritten.

**Verdict.** CLEAR on the grid construction itself; UNDOCUMENTED on the
`savings_min < tiny_savings` redundancy.

---

## 6. Savings grid

**What I found.** `lifecycle/precompute.py:268–272`. Same `expm1∘linspace∘log1p`
construction as the wealth grid, parameters
`(savings_min=1e-8, savings_max=wealth_max=750, n_savings=180)`. **No
custom-grid path** for savings — only the wealth grid supports a loaded
.npy.

`n_savings == n_wealth` is **not** required by the math. The EGM scan
solves at `n_savings + 1` endogenous points (anchor at s=0,
`solver.py:1236–1245`), and `_lift_to_wealth_grid` uses `jnp.interp` to
project onto the (possibly different-sized) wealth grid. Sharing
`n=180` is purely a thesis convention; could differ in principle.

`tiny_savings = 1e-6` (model.py:168) is the design fallback. The
savings-grid range `[1e-8, 750]` has the lowest 1–2 nodes at `s = 1e-8`,
`s ≈ 1.7e-8`, etc. — all below tiny_savings. As noted above, those get
sentinel-replaced. **Below tiny_savings, FOC is not solved; the policy is
fixed at `(c_min, init_a_s, init_a_b)`.** This is documented in
`solver.py:1227–1232` but the interaction with `savings_min` is not
called out in any one place.

**Verdict.** CLEAR (mathematically correct, design fallbacks are
deliberate). UNDOCUMENTED interaction between `savings_min` and
`tiny_savings`; consider promoting `savings_min := tiny_savings` as the
default to remove the redundant grid points.

---

## TL;DR

| Area | Verdict | Headline |
|---|---|---|
| State grid (cholesky) | CLEAR | Ordering & probabilities consistent |
| State quadrature | CLEAR | Conditional Σ_ss; closure check passes |
| Return quadrature | CLEAR | Joint, Cholesky-correct, Σ_rr in CCV |
| Income shock quadrature | UNDOCUMENTED | `mu_eps2 = 0.0` silently overridden; `mu_eta2` enforced manually |
| Wealth grid | CLEAR / UNDOCUMENTED | log1p default OK; `savings_min < tiny_savings` redundant |
| Savings grid | CLEAR / UNDOCUMENTED | `n_savings = n_wealth` is convention only |

## Single most important RED FLAG

**Income mixture parameter normalisation is enforced in three different
ways across the discretization stack** (`discretization.py:438–443`,
`464–469`, vs `precompute.py:369` calling `discretize_income_ar1_mixture`
with the **raw** `model.mu_eta2`). The η path is safe today only because
`_canonical.py:36` happens to set `mu_eta2 = -(pz/(1-pz))*mu_eta1`
manually. The ε path is already mismatched (`mu_eps2 = 0.0` in config vs
`-0.00617` effective in eps-quadrature) — the silently-ignored field
masks the inconsistency.

**Suggested fix.** Delete `mu_eta2`, `mu_eps2` from `BASE_CONFIG` and
`LifecyclePortfolioModel`; recompute the constrained value once inside
`build_model` from `(p, mu1)`. That removes any chance of a future user
overriding only `mu_eta1` and silently breaking the persistent-z bin
probabilities while the η-quadrature stays correct.
