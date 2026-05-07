# HANDOFF — Verify return-quadrature Cholesky implementation is correct

**Status:** open. Verification, not implementation. The code change landed
2026-04-30; the goal of this handoff is to confirm independently that the
Cholesky implementation produces a mathematically valid Gauss-Hermite
quadrature on `N(0, Σ_r_cond)` with honest per-axis interpretation.

**Scope:** verify (1) the math statement, (2) the code does what the math
says, (3) run the most informative numerical tests for accuracy. Do not
re-implement.

**Background context:** the previous implementation used eigendecomposition
of `Σ_r_cond`. The slot labels `(K_rtb, K_xr, K_xb)` in `n_ret_nodes_1d`
implied per-physical-asset axes, but eigendecomposition rotates by
ascending eigenvalue, so the labels did not match what `K[i]` actually
controlled. The 2026-04-30 switch to Cholesky aligns slot labels with
input return ordering. Both transforms are mathematically valid quadratures
on the same target distribution; the switch makes the labels honest and is
expected to be more accurate when per-axis `K` is asymmetric (because the
Cholesky axis ordering matches the input variable ordering).

---

## 1. Theoretical procedure

### 1.1 The integration target

The solver needs to integrate functions `f : R^3 -> R` against the
multivariate normal density

```
r ~ N(0, Σ_r_cond)
```

where `Σ_r_cond` is a `3×3` symmetric positive-definite matrix (the
residual return covariance after conditioning on the slow state). The
integral

```
I[f] := E[f(r)] = ∫ f(r) (2π)^(-3/2) |Σ|^(-1/2) exp(-½ r^T Σ^(-1) r) dr
```

is what `f = c_next^(-γ) · (R_s − R_bill)` and similar return-side FOC
integrands require.

### 1.2 Reparametrisation via a covariance square root

For any matrix `A` with `A A^T = Σ`, the linear map `r = A z` sends
`z ~ N(0, I_3)` to `r ~ N(0, Σ)`. The change-of-variable identity gives

```
E[f(r)] = E[f(A z)],   with  z ~ N(0, I_3)
```

This is exact regardless of which valid square root `A` is chosen. There
are infinitely many: the Cholesky factor (lower-triangular `L`),
`V √D` from the eigendecomposition `Σ = V D V^T`, or `A R` for any
orthogonal `R`.

### 1.3 Tensor-product Gauss-Hermite on the standard normal

Univariate Gauss-Hermite of order `K` provides nodes `(z_k, w_k)_{k=1..K}`
with `Σ_k w_k = 1` such that

```
∫ p(z) (2π)^(-1/2) exp(-z²/2) dz  =  Σ_k w_k p(z_k)
```

is **exact** for every polynomial `p` of degree `≤ 2K-1` (the standard
exactness theorem; see Stoer & Bulirsch §3.6, Judd §7.2).

Tensor-product the rule across the three independent standardised axes:

```
E[g(z)] ≈ Σ_{k_0, k_1, k_2}  w^(0)_{k_0} w^(1)_{k_1} w^(2)_{k_2}  g(z_{k_0}, z_{k_1}, z_{k_2})
```

with `K_d` nodes on axis `d` (per-axis order allowed). Polynomial exactness
is preserved separately along each axis: a polynomial of degree `≤ 2K_d − 1`
in `z_d` (with arbitrary lower-degree coupling to other axes) integrates
exactly.

### 1.4 The full quadrature on r

Substituting `r = A z`:

```
E[f(r)] = E[f(A z)] ≈ Σ_n w_n  f(A z_n)
```

where `n` indexes the joint nodes and `w_n = ∏_d w^(d)_{k_d}`. The
quadrature returns `(ret_nodes, ret_weights)` with `ret_nodes[n] = A z_n`
and `ret_weights[n] = w_n`.

### 1.5 Why Cholesky specifically

Both Cholesky and eigendecomposition deliver a valid square root, so
both yield convergent quadratures as `K -> ∞`. At finite `K` they place
nodes at different points in `r`-space and give different finite-sample
approximations. The choice matters for two reasons:

1. **Per-axis interpretation of K_d.** Refining `K_d` adds resolution
   along axis `d` in z-space, which maps to the d-th column of `A` in
   r-space. Under Cholesky's lower-triangular `L`, column `d` is
   non-zero only on rows `≥ d`. With input return ordering
   `(rtb, xr, xb)`:
   - `L[:, 0]` non-zero on row 0 only ⇒ `z_0` (and therefore `K_rtb`)
     contributes only to `r_rtb`. Refining `K_rtb` is a clean
     rtb-axis refinement.
   - `L[:, 1]` non-zero on rows 1, 2 ⇒ `z_1` (`K_xr`) refines the
     xr-residual after rtb has been orthogonalised away.
   - `L[:, 2]` non-zero on row 2 only ⇒ `z_2` (`K_xb`) is the pure
     xb-residual after rtb and xr have been orthogonalised away.

   So `(K_rtb, K_xr, K_xb)` slot names match what `K[i]` actually
   controls.

2. **Codebase consistency.** `build_state_grid` (state grid via `Σ_z`)
   and `get_state_quadrature` (state innovation via `Σ_ss`) already use
   Cholesky. Aligning `get_return_quadrature` makes the three
   square-root sites uniform.

### 1.6 Required preconditions

- `Σ_r_cond` must be **positive definite** (Cholesky requires PD;
  eigendecomposition tolerates PSD with zero eigenvalues). For our
  calibration this is always satisfied: residual returns are never
  collinear after the state partition.
- 1-D Gauss-Hermite weights must be non-negative (always true; Gauss-
  Hermite weights are positive for any K).
- Tensor product preserves non-negativity ⇒ joint weights `≥ 0`.
- Tensor product preserves the unit-mass property `Σ_k w_k = 1` per
  axis ⇒ joint weights sum to 1.

### 1.7 What the procedure delivers

For polynomial integrands the quadrature is **exact** along each axis up
to degree `2K_d − 1`. For general smooth integrands the error decays
super-algebraically with `K_d` (provided the integrand has bounded
moments under the Gaussian density). In particular:

- **First moment:** `E[r] = 0` exact at `K_d ≥ 1` per axis (zero is a
  polynomial of degree 0).
- **Second moment:** `E[r r^T] = Σ_r_cond` exact at `K_d ≥ 2` per axis
  (degree-2 polynomials in each axis pair).
- **Third central moment:** `E[(r_i − μ_i)(r_j − μ_j)(r_k − μ_k)] = 0`
  for the centred multivariate normal — exact whenever the polynomial
  degree per axis stays below `2K − 1` (so for `K=2`, exact on degree
  ≤ 3).
- **MGF identity:** `E[exp(a · r)] = exp(½ a^T Σ a)`. **Not** a
  polynomial; expect non-zero error that decays as `K -> ∞`. This is
  the most informative accuracy benchmark.

---

## 2. Implementation in the code

### 2.1 Function under test

`discretization.py::get_return_quadrature(model, n_nodes)` — returns
`(ret_nodes, ret_weights)` with shapes `(K_total, n_ret)` and
`(K_total,)`.

### 2.2 Step-by-step

1. **Normalise `n_nodes` to per-axis tuple.** `_normalize_ret_nodes`
   broadcasts a scalar to a length-`n_ret` tuple, validates length,
   and rejects `K_d < 1`.
2. **Per-axis 1-D Gauss-Hermite.** For each `K_d > 1`, call
   `scipy.special.roots_hermite(K_d)`. The function returns nodes and
   weights for the **physicist's** Hermite polynomial (weight
   `e^(-x^2)`), which the code rescales to the **probabilist's**
   convention (weight `(2π)^(-1/2) e^(-x^2/2)`):

   ```
   nodes_1d  = z * sqrt(2)      # convert from physicist to probabilist axis scaling
   weights_1d = w / sqrt(π)     # probabilist normalisation: ∫ N(0,1) = 1
   ```

   For `K_d = 1` the rule degenerates to a single zero node with
   weight 1 (handled as a special case to avoid `roots_hermite(1)`
   edge cases).
3. **Tensor product.** `np.meshgrid(..., indexing="ij")` produces the
   Cartesian product; `np.stack(...).ravel()` flattens to the
   `(K_total, n_ret)` shape and `np.prod(...)` gives weights.
4. **Cholesky transform.** Symmetrise `Σ_r_cond` (defensive against
   roundoff asymmetry) and compute `L = numpy.linalg.cholesky(Σ)`.
   `L` is lower-triangular with `L L^T = Σ_r_cond`.
5. **Map z to r.** `ret_nodes = z_nodes @ L^T`. This is the linear
   map of §1.2 with `A = L`. Each row of `ret_nodes` is one joint
   `r_n = L z_n`.
6. **Return** `(ret_nodes, ret_weights)`.

### 2.3 What downstream code consumes

The solver's FOC loops iterate

```python
for k_v in range(n_state_quad):
    for k_r in range(n_ret_quad):
        # use ret_nodes[k_r], ret_weights[k_r]
```

and treat both arrays as opaque. The shapes are `(K_total, 3)` and
`(K_total,)` respectively, with `K_total = prod(n_ret_nodes_1d)`. The
solver never inspects the transform underneath.

### 2.4 Saved-bundle compatibility

Saved bundles store `n_ret_nodes_1d` in metadata (e.g. `[3, 5, 3]`).
Reloading and rebuilding `Precompute` calls `get_return_quadrature` with
the saved value, so old bundles solved under eigendecomposition will
re-precompute under Cholesky on load. **The policy arrays in the saved
bundle were solved under eigendecomposition**; if you reload an old
bundle and use its `Precompute`, the policies don't match the new
quadrature. Best practice is to re-solve and save to a new path. The
codebase tags the new generation with `_v2` in `main.ipynb`.

---

## 3. Tests for accuracy

Run these in priority order. Tests 1–3 are the math integrity checks;
tests 4–6 quantify accuracy on representative integrands. Tests 7–8
guard the Cholesky-specific structure.

### Test 1 — Cholesky factor invariant

**What:** verify `L L^T == Σ_r_cond` to machine precision and that `L`
is lower-triangular.

```python
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model
import numpy as np

# Build model exactly as production does.
var_config, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
model = build_model(BASE_CFG, var_config, verbose=False)
Sigma = np.asarray(model.Sigma_r_cond)
Sigma_sym = 0.5 * (Sigma + Sigma.T)
L = np.linalg.cholesky(Sigma_sym)

assert np.allclose(L @ L.T, Sigma_sym, atol=1e-15, rtol=0)
assert np.allclose(np.triu(L, k=1), 0.0, atol=0.0)        # strictly upper-tri zero
assert np.all(np.diag(L) > 0)                              # positive diagonal
```

**Why informative:** if this fails, the entire quadrature is invalid
because the change-of-variable identity in §1.2 doesn't hold.

### Test 2 — Moment recovery (existing, mandatory)

**What:** verify the empirical moments of the quadrature match the
theoretical moments of `N(0, Σ_r_cond)`.

```python
from lifecycle.discretization import get_return_quadrature

for K_per_axis in [(2,2,2), (3,5,3), (2,2,5), (5,2,2), (3,3,3), (5,5,5)]:
    r, w = get_return_quadrature(model, n_nodes=K_per_axis)
    assert abs(w.sum() - 1.0) < 1e-15
    mean = w @ r
    cov  = (r.T * w) @ r
    assert np.max(np.abs(mean)) < 1e-15
    assert np.max(np.abs(cov - Sigma_sym)) < 1e-14
```

**Why informative:** the second moment is the most direct numerical
check that the quadrature is integrating `r r^T` correctly. Non-trivial
but Gauss-Hermite at K ≥ 2 nails it to machine precision; if a bug were
introduced (wrong factor, wrong indexing) this would catch it.

### Test 3 — Polynomial exactness per axis

**What:** Gauss-Hermite of order `K` integrates polynomials of degree
`≤ 2K-1` along its axis exactly. Verify:

```python
def true_E_zk(k):
    """E[z^k] for z ~ N(0,1)."""
    if k % 2 == 1:
        return 0.0
    # k even: (k-1)!!  (double factorial)
    from math import prod
    return prod(range(1, k, 2)) if k > 0 else 1.0

# Inline GH on a single axis at K nodes.
from scipy.special import roots_hermite
for K in (2, 3, 5):
    z, w = roots_hermite(K)
    z = z * np.sqrt(2.0); w = w / np.sqrt(np.pi)
    for d in range(2*K):  # exact through degree 2K-1
        emp = float(np.sum(w * z**d))
        truth = true_E_zk(d)
        assert abs(emp - truth) < 1e-13, (K, d, emp, truth)
```

**Why informative:** confirms the 1-D rule is correctly normalised
(probabilist's, not physicist's). A bug in the `* sqrt(2)` /
`/ sqrt(π)` rescaling would surface here as the moment recovery in
test 2 would still pass for low moments but fail higher up.

### Test 4 — MGF benchmark (the most informative accuracy test)

**What:** for a 3-vector `a`, the moment generating function gives a
**closed form** `E[exp(a · r)] = exp(½ a^T Σ a)`. The integrand is
non-polynomial, so quadrature error is non-zero and decays with K. Sweep:

```python
rng = np.random.default_rng(0)
def truth(a, S): return float(np.exp(0.5 * a @ S @ a))
def quad(a, K, model):
    r, w = get_return_quadrature(model, n_nodes=K)
    return float(np.sum(w * np.exp(r @ a)))

results = []
for trial in range(20):
    a = rng.standard_normal(3) * 5.0   # |a| ~ 5; representative of γ × |α|
    truth_val = truth(a, Sigma_sym)
    for K in [(2,2,2), (3,3,3), (5,5,5), (3,5,3), (2,2,5), (5,2,2)]:
        q = quad(a, K, model)
        rel_err = abs(q - truth_val) / abs(truth_val)
        results.append((trial, K, rel_err))
```

**Expected:** relative error decreases monotonically as `min(K_d)` rises.
At uniform `K=5`, errors should be `< 1e-6` for `|a| ≤ 5`. Asymmetric
`K=(3,5,3)` should give error in the regime between `K=3` uniform and
`K=5` uniform.

**Why informative:** this is the cleanest non-polynomial closed form
available for the multivariate Gaussian. If the relative error is
**larger** under Cholesky than under the previous eigendecomposition
implementation at the same total node count and uniform K, that's a
red flag. They should be the same to machine precision at uniform K (both
methods have identical accuracy on the MGF when nodes are placed
symmetrically across the variance ellipsoid).

### Test 5 — Asymmetric K accuracy ordering

**What:** at fixed total node count, asymmetric K should be more accurate
than uniform K when the integrand has more variation in one direction.

```python
# Pick a directional integrand: exp(λ · L[:, d] · z) for each d
# (this exercises only the d-th Cholesky axis).
for d in range(3):
    a_pure_d = L[:, d] * 5.0   # stress the d-th axis
    truth_val = truth(a_pure_d, Sigma_sym)
    
    K_uniform = (3, 3, 3)
    K_targeted = tuple(5 if i == d else 2 for i in range(3))
    # uniform: 27 nodes; targeted: 20 nodes (less compute) but more on axis d
    
    err_uniform = abs(quad(a_pure_d, K_uniform, model) - truth_val) / truth_val
    err_targeted = abs(quad(a_pure_d, K_targeted, model) - truth_val) / truth_val
    print(f"axis {d}: uniform K=3 err={err_uniform:.2e}  targeted K[{d}]=5 err={err_targeted:.2e}")
    # Expectation: err_targeted < err_uniform when the integrand is
    # axis-d-dominant.
```

**Why informative:** verifies the per-axis K knob does what it claims —
refining the right axis materially improves accuracy for axis-aligned
integrands. If `err_targeted >= err_uniform` at any d, the claim that
"K_d refines the d-th Cholesky axis" is wrong.

### Test 6 — Compare against eigendecomposition reference

**What:** independently re-implement the eigendecomposition transform,
compare the MGF errors at uniform K. They should be **identical** at
uniform K (both are valid square roots of the same Σ; tensor-product
Gauss-Hermite at uniform K covers the variance ellipsoid symmetrically
under either rotation).

```python
def quad_eigen(a, K, Sigma_sym):
    """Reference: eigendecomposition transform, uniform K."""
    z, w = roots_hermite(K)
    z = z * np.sqrt(2.0); w = w / np.sqrt(np.pi)
    # tensor product (3D)
    grid_z = np.meshgrid(z, z, z, indexing="ij")
    grid_w = np.meshgrid(w, w, w, indexing="ij")
    z_nodes = np.stack([g.ravel() for g in grid_z], axis=1)
    weights = np.prod(np.stack(grid_w, axis=0), axis=0).ravel()
    eigvals, eigvecs = np.linalg.eigh(Sigma_sym)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    r_nodes = z_nodes @ transform.T
    return float(np.sum(weights * np.exp(r_nodes @ a)))

# Compare at uniform K=2, 3, 5
for K in (2, 3, 5):
    err_chol  = abs(quad(a, (K,K,K), model) - truth(a, Sigma_sym)) / truth(a, Sigma_sym)
    err_eigen = abs(quad_eigen(a, K, Sigma_sym) - truth(a, Sigma_sym)) / truth(a, Sigma_sym)
    print(f"K={K}: Cholesky err = {err_chol:.2e},  Eigh err = {err_eigen:.2e}")
    # Expectation: both errors are within machine precision of each other
    # (different node placements but same coverage of the ellipsoid).
```

**Why informative:** independent reference implementation cross-check.
If Cholesky is significantly worse than eigendecomposition at uniform K
on the MGF, the implementation has a bug.

### Test 7 — Cholesky structural signature

**What:** under Cholesky, collapsing K on axis 0 to a single node should
zero out exactly the rtb component of `r` (lower-triangular structure).

```python
# K=(1, 1, 5): only z_2 active. Under Cholesky:
#   r[:, 0] = L[0, :] · z = L[0,0]·z_0 + 0·z_1 + 0·z_2.  z_0 = z_1 = 0 ⇒ r[:, 0] = 0.
#   r[:, 1] = L[1, :] · z = L[1,0]·z_0 + L[1,1]·z_1 + 0·z_2.  ⇒ r[:, 1] = 0.
#   r[:, 2] = L[2, :] · z = L[2,0]·z_0 + L[2,1]·z_1 + L[2,2]·z_2.  ⇒ r[:, 2] = L[2,2]·z_2.
r, _ = get_return_quadrature(model, n_nodes=(1, 1, 5))
assert np.allclose(r[:, 0], 0.0, atol=1e-15)
assert np.allclose(r[:, 1], 0.0, atol=1e-15)
assert np.allclose(r[:, 2], L[2, 2] * z_2_nodes, atol=1e-14)
```

**Why informative:** this signature is **unique to Cholesky** and would
fail under eigendecomposition (the 5 active nodes there would lie along
the largest eigenvector direction, which has non-zero projection on all
three return variables in our calibration). It's the regression-guard
that detects "did someone accidentally revert to eigendecomposition?"

### Test 8 — Axis labels match input ordering

**What:** confirm that `K_rtb` (slot 0) refines a direction that, when
isolated, only affects the rtb component of `r` (modulo Cholesky leakage
to xr and xb that's also from `z_0`, but the LABEL `K_rtb` is correct in
that slot 0 is the rtb-driving axis).

This is a doc-spec check more than a math check. Verify by inspection
that `model.ret_names[0] == 'rtb'` and that the first column of `L` has
non-zero only in row 0 ⇒ axis 0 of `z` controls only rtb.

```python
assert model.ret_names == ('rtb', 'xr', 'xb')
# Lower-triangular L: column 0 non-zero only in row 0
assert L[0, 0] != 0  # diagonal entry
assert L[1:, 0] is not None  # column may have entries below diagonal
# The KEY claim: row 0 of L has only L[0, 0] non-zero
assert np.allclose(L[0, 1:], 0.0)
```

If this assertion fails, the parameter naming `(K_rtb, K_xr, K_xb)` is
again misleading and the doc-spec is wrong.

---

## 4. Acceptance criteria

Pass all of:

1. Test 1 (Cholesky invariants) — strict equalities, no tolerance debate.
2. Test 2 (moment recovery) — `< 1e-14` on covariance.
3. Test 3 (polynomial exactness) — `< 1e-13` on `2K-1`-degree integrals.
4. Test 4 (MGF) — error decays with K; at uniform `K=5`, `< 1e-6` for
   moderate `|a|`.
5. Test 5 (asymmetric K helps) — targeted K beats uniform K of similar
   total-node count when the integrand is axis-aligned.
6. Test 6 (vs eigendecomposition) — Cholesky and eigh give the same
   MGF error at uniform K (within machine precision).
7. Test 7 (structural signature) — strict equalities, axes 0 and 1 are
   exactly zero when collapsed.
8. Test 8 (label match) — `ret_names[0] == 'rtb'` and first row of `L`
   has only `L[0,0]` non-zero.

Failures on tests 1, 2, 3, 7, 8 indicate a real bug. Failures on tests
4, 5, 6 in extreme regimes (very large `|a|`, very small `K`) are
expected and consistent with Gauss-Hermite limits.

---

## 5. Pointers

| What | Where |
|------|-------|
| Implementation under test | `discretization.py:get_return_quadrature` |
| Sister Cholesky on state innovation | `discretization.py:get_state_quadrature` |
| Existing moment-recovery test (scalar K) | `verify/discretization.ipynb` §A.4 |
| Existing per-axis tests | `tests/test_state_grid_modes.py::run_per_axis_n_ret_quad_checks` |
| Production caller | `precompute.py:Precompute.__init__` |
| Saved-bundle metadata convention | `policy_io.py:save_policy_bundle` |
| Math reference | Stoer & Bulirsch §3.6; Judd 1998 §7.2 |

---

## 6. Notes for the verifier

- Do not modify the implementation. The goal is independent confirmation.
- Tests 4 and 5 are quantitative; record the actual error numbers in the
  report so future maintainers can detect drift.
- If any of tests 1, 2, 3, 7, 8 fails, stop and flag — these are
  correctness assertions, not numerical-tuning observations.
- The previous eigendecomposition implementation is **also**
  mathematically correct; the Cholesky switch is a labelling-honesty
  upgrade, not a bug fix on the math. If the verifier finds equivalent
  accuracy under both transforms at uniform K, that is the expected and
  desired outcome.
