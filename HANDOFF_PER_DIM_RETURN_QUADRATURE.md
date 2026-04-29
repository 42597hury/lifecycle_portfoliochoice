# HANDOFF — Per-dimension return quadrature nodes

## Purpose

Allow `n_ret_nodes_1d` to be a 3-tuple `(K_rtb, K_xr, K_xb)` instead of a single
integer K, so the user can refine Gauss–Hermite resolution along the
stock-return axis (`xr`) without paying for K nodes on every other residual
dimension.

Backward compatibility is required: passing a plain `int` (current usage) must
continue to work and behave identically to the symmetric `(K, K, K)` tuple.

## Background — why this matters

At unconstrained CRRA (γ ≥ 1) some grid states have a discrete-quadrature
arbitrage: the convex hull of the joint excess-return cloud
`{(R_s^(n) − R_bill^(n), R_b^(n) − R_bill^(n))}_n` does not contain the origin,
so unconstrained Newton has no interior optimum and either fails (`EC_NEWTON_FAIL`
at max_iter) or "phantom-converges" at huge α. See
`contextfiles/RETURNS.md §6.12` for the diagnostic.

The arbitrage is killed cheapest by adding nodes on the **stock** residual axis,
not the bond axis, because (i) stock has the largest residual variance and
controls the principal eigenvector direction in `Sigma_r_cond`, and (ii) the
missing scenarios are joint stock+bond crashes, not marginal bond tail. Empirical
counts at `state_grid=5×5×5 principal/3.0σ`, `K_state=2`:

| `K_ret` (per dim) | joint return nodes | arbitrage states |
|---|---|---|
| (3, 3, 3) — current | 27 | 24 |
| (3, 3, 21) bond-only | 189 | 18 |
| (3, 9, 3) stock-only | 81 | 3 |
| (3, 15, 3) stock-only | 135 | 0 |
| (5, 5, 5) uniform | 125 | 11 |
| (9, 9, 9) uniform | 729 | 1 |

So per-dim K is roughly 5× cheaper than uniform K for the same arbitrage
elimination. Picking the actual K's for production is **not** part of this
task — runtime needs to be measured empirically and is the user's call. The
job here is just to make per-dim K **possible**.

## Scope

### In scope

- `DiscretizationConfig.n_ret_nodes_1d` accepts both `int` and `tuple/list of length n_ret`.
- `discretization.get_return_quadrature` accepts the new form and builds an
  asymmetric tensor product with the same Cholesky/eigendecomp transform.
- All log/print statements that currently format `K^n_ret` work with both forms.
- All save/load paths preserve the new form across JSON metadata.
- All existing tests and saved runs (which use `int`) continue to pass without
  modification — this is a strict backward-compat extension, not a refactor.
- A new validation test is added confirming weights/mean/covariance properties
  hold for an asymmetric configuration.

### Out of scope

- Implementing the pre-solve arbitrage diagnostic (separate task tracked in
  `contextfiles/RETURNS.md §6.12`).
- Picking the production value of `K_ret`. Leave the default at the current
  scalar (e.g. `n_ret_nodes_1d: int = 2`).
- Pruning grid states based on arbitrage. Separate task.
- State-quadrature dimension. `n_state_quad_nodes` stays a scalar; this handoff
  is **only** about return-residual quadrature.

## Implementation plan

### 1. Schema: `DiscretizationConfig` in [model.py:114](model.py#L114)

Change the type annotation but keep the default a scalar so existing call sites
remain valid:

```python
# Old:
n_ret_nodes_1d: int = 2

# New (typing-wise, accept Union[int, Tuple[int, ...]]):
n_ret_nodes_1d: Any = 2   # int OR (K_rtb, K_xr, K_xb); see normalize helper
```

Add a small helper next to the config (or in `discretization.py`) that
normalizes either form to a tuple of length `n_ret`:

```python
def _normalize_ret_nodes(value, n_ret: int) -> tuple:
    if isinstance(value, int):
        return (value,) * n_ret
    t = tuple(int(v) for v in value)
    if len(t) != n_ret:
        raise ValueError(
            f"n_ret_nodes_1d tuple length {len(t)} does not match n_ret={n_ret}")
    return t
```

This normalization is the single source of truth — every consumer calls it.
**Do not** change the field name. Renaming would break every saved run and
notebook.

### 2. `discretization.get_return_quadrature` at [discretization.py:447](discretization.py#L447)

Change signature and body. Keep the `n_nodes=1` "single zero residual" fast
path. Build an asymmetric Hermite product when any K_i > 1.

```python
def get_return_quadrature(model, n_nodes=1):
    """
    n_nodes : int OR sequence of ints of length model.n_ret
        Gauss-Hermite order per return dimension.
    """
    n_ret = int(model.n_ret)
    K_per_dim = _normalize_ret_nodes(n_nodes, n_ret)
    if any(k < 1 for k in K_per_dim):
        raise ValueError("All entries of n_ret_nodes_1d must be >= 1")

    # K=1 in every dim -> single zero node, weight 1 (preserved fast path)
    if all(k == 1 for k in K_per_dim):
        return np.zeros((1, n_ret), dtype=float), np.ones(1, dtype=float)

    # Build per-dim 1D GH nodes/weights
    grids_z, grids_w = [], []
    for K in K_per_dim:
        if K == 1:
            grids_z.append(np.zeros(1)); grids_w.append(np.ones(1))
        else:
            z, w = roots_hermite(K)
            grids_z.append(z * np.sqrt(2.0))
            grids_w.append(w / np.sqrt(np.pi))

    Z = np.meshgrid(*grids_z, indexing="ij")
    W = np.meshgrid(*grids_w, indexing="ij")
    z_nodes = np.stack([g.ravel() for g in Z], axis=1)
    ret_weights = np.prod(np.stack(W, axis=0), axis=0).ravel()

    Sigma = 0.5 * (np.asarray(model.Sigma_r_cond, dtype=float)
                   + np.asarray(model.Sigma_r_cond, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    if np.any(eigvals < -1e-12):
        raise ValueError("Sigma_r_cond must be positive semidefinite for return quadrature")
    eigvals = np.clip(eigvals, 0.0, None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    ret_nodes = z_nodes @ transform.T
    return ret_nodes, ret_weights
```

**Important** — the eigendecomp/transform step is unchanged. The asymmetric K
just changes how the standard-normal `z_nodes` lattice is built before being
mapped through `transform.T`. The contract (returned shapes
`(n_ret_quad, n_ret)` and `(n_ret_quad,)`) is preserved.

### 3. `Precompute.__init__` in [precompute.py](precompute.py)

Validation at [precompute.py:104](precompute.py#L104):

```python
# Old:
if disc_config.n_ret_nodes_1d < 1:
    raise ValueError("disc_config.n_ret_nodes_1d must be >= 1")

# New: delegate to the normalize helper which does its own validation
K_ret_per_dim = _normalize_ret_nodes(disc_config.n_ret_nodes_1d, model.n_ret)
if any(k < 1 for k in K_ret_per_dim):
    raise ValueError("All entries of disc_config.n_ret_nodes_1d must be >= 1")
```

Store the normalized tuple on the Precompute instance for downstream
consumers:

```python
self.n_ret_nodes_1d = K_ret_per_dim   # always a tuple of ints, length n_ret
```

The call to `get_return_quadrature` at [precompute.py:148](precompute.py#L148)
already passes through the raw config value — leave that line as-is, since
`get_return_quadrature` now normalizes internally.

Update the print at [precompute.py:380](precompute.py#L380) to handle both
forms gracefully:

```python
K_str = "x".join(str(k) for k in self.n_ret_nodes_1d)
print(f"Return quad  : ({K_str}) nodes/dim  ->  {self.n_ret_quad} joint nodes")
```

Update the comment at [precompute.py:153](precompute.py#L153) and
[precompute.py:249](precompute.py#L249) to reflect that the joint count is
`prod(n_ret_nodes_1d)` not `n_ret_nodes_1d ** n_ret`.

### 4. Diagnostics & simulation prints

Same treatment in:

- [diagnostics.py:708](diagnostics.py#L708) — replace `pc.disc_config.n_ret_nodes_1d` formatting with `"x".join(str(k) for k in pc.n_ret_nodes_1d)`.
- [diagnostics.py:828-829](diagnostics.py#L828-L829) — replace `f"= {pc.disc_config.n_ret_nodes_1d}^{model.n_ret}"` with `f"= prod({K_str}) = {pc.n_ret_quad}"`.
- [simulation.py:742](simulation.py#L742) — same treatment.

### 5. Tests and saved-run loaders

Saved runs at `saved_runs/*/metadata.json` store
`"n_ret_nodes_1d": <int>` (verified: see
[saved_runs/constrained_grid5x5x5_nz11/metadata.json:94](saved_runs/constrained_grid5x5x5_nz11/metadata.json#L94),
the `_diag_332_nz11`, `constrained_grid5x5x5_nz7`, `constrained_grid7x7x7_nz11`
runs all use scalar `int`). The four test files that reload from saved metadata
read this field directly into the `DiscretizationConfig` constructor:

- [tests/test_simulation.py:53](tests/test_simulation.py#L53)
- [tests/test_economics.py:51](tests/test_economics.py#L51)
- [tests/test_terminal_grad_hess.py:62](tests/test_terminal_grad_hess.py#L62)
- [tests/test_terminal_omega.py:83](tests/test_terminal_omega.py#L83)

These continue to work without modification because the new
`DiscretizationConfig` accepts `int`. Verify by running each test against
existing saved runs after the change. **Do not** rewrite saved-run metadata.

For new saves, JSON serialization is handled by `_to_jsonable` in
[policy_io.py:25](policy_io.py#L25) which already converts tuples to lists.
Loaders (e.g. saved-run reload paths) need to round-trip correctly:

- If `dc_raw["n_ret_nodes_1d"]` is `int` (legacy run) → pass through as int.
- If it's a `list` (new asymmetric run) → convert to `tuple(...)` before
  constructing `DiscretizationConfig`. This conversion belongs in the same
  test setup helpers above, in case anyone produces an asymmetric saved run
  later. A two-line `if isinstance(v, list): v = tuple(v)` shim is enough.

### 6. Notebooks

Three notebooks reference the field literally:

- [main.ipynb](main.ipynb)
- [main_part2.ipynb:98](main_part2.ipynb#L98)
- [verify_discretization.ipynb:109](verify_discretization.ipynb#L109) and [:495](verify_discretization.ipynb#L495)

All three currently pass an `int`. They keep working — do not modify them as
part of this PR. After the change, the user can edit them manually to switch
to `(3, 9, 3)` etc. when they want per-dim refinement.

`verify_discretization.ipynb:495` calls
`get_return_quadrature(MODEL, n_nodes=PROD_DISC.n_ret_nodes_1d)` which now
needs to work with the union type — it does, since `get_return_quadrature`
normalizes its argument. No edit required.

### 7. Other call sites

- [_run_sim_gap.py:25](_run_sim_gap.py#L25) — uses `int` literal. No change.
- Archived tests under `archive/test_state_quadrature*.py` — frozen, do not touch.

### 8. Documentation files (low priority, do at the end)

These hold reference text describing the field; update once the code is
working. None are imports/exec'd:

- [contextfiles/STATE_SPACE.md:258](contextfiles/STATE_SPACE.md#L258) and [:435](contextfiles/STATE_SPACE.md#L435) — change "K_r per dim" wording.
- [contextfiles/DESIGN.md:599](contextfiles/DESIGN.md#L599) — update example comment.
- [contextfiles/TODO.md:176](contextfiles/TODO.md#L176) — touch up example.
- [HANDOFF_COMPLEXITY_ANALYSIS.md:54](HANDOFF_COMPLEXITY_ANALYSIS.md#L54) — table row note.

`contextfiles/RETURNS.md §6.12` has an open task referencing this exact
implementation; mark as done after merge.

## Validation

Before declaring complete, all of the following must pass.

### A. Existing-behavior preservation (regression)

Run from repo root:

```bash
python -m pytest tests/test_terminal_omega.py tests/test_terminal_grad_hess.py
python tests/test_simulation.py
python tests/test_economics.py
python tests/test_state_grid_modes.py
python tests/test_terminal_failures.py
python tests/test_iter_diagnostics.py
python tests/test_unconstrained_stagnation.py
```

All must pass with **exact** same outputs as before the change when invoked
with the same `int` configuration. Any byte-level diff in
`saved_runs/*` reload paths is a regression — investigate.

### B. New asymmetric-form correctness

Add a test (e.g. extend `tests/test_judd_quadrature.py` or create
`tests/test_asymmetric_ret_quad.py`) that builds a Precompute with
`n_ret_nodes_1d=(3, 9, 3)` and verifies all of the existing return-quadrature
properties listed in `contextfiles/RETURNS.md §6.5`:

```python
# 1. weights sum to one
assert abs(ret_weights.sum() - 1.0) < 1e-12

# 2. all weights positive
assert (ret_weights > 0).all()

# 3. weighted mean is zero
wmean = (ret_weights[:, None] * ret_nodes).sum(axis=0)
assert np.max(np.abs(wmean)) < 1e-10

# 4. weighted covariance equals Sigma_r_cond exactly
cov = (ret_weights[:, None, None]
       * ret_nodes[:, :, None] * ret_nodes[:, None, :]).sum(axis=0)
assert np.max(np.abs(cov - model.Sigma_r_cond)) < 1e-12

# 5. node count is the product of the per-dim K's
assert ret_nodes.shape[0] == 3 * 9 * 3
assert ret_weights.shape == (3 * 9 * 3,)

# 6. (3,3,3) tuple matches scalar 3 exactly (bit-identical)
sym_n, sym_w = get_return_quadrature(model, n_nodes=3)
asym_n, asym_w = get_return_quadrature(model, n_nodes=(3, 3, 3))
assert np.allclose(sym_n, asym_n, atol=0)
assert np.allclose(sym_w, asym_w, atol=0)
```

Property (4) is the load-bearing one: the eigendecomp transform is supposed
to recover `Sigma_r_cond` *exactly* under any product GH rule with K_i ≥ 2,
because each 1D rule integrates polynomials up to degree `2*K_i − 1` exactly
and the second moments are degree 2.

### C. End-to-end smoke

Build a `Precompute` and run any short solve workflow already exercised by
the notebooks (the smallest-grid setup is fine — e.g. `state_grid_sizes=(3,3,3)`
with `n_ret_nodes_1d=(3, 5, 3)`). Confirm:

- Precompute constructs without error.
- The `Return quad : (3x5x3) nodes/dim → 45 joint nodes` print appears.
- Diagnostics report the new joint-node count consistently with `pc.n_ret_quad`.
- A short solver run (terminal age only is enough) returns the same shapes
  and finite values.

### D. Saved-run round-trip

Load `saved_runs/constrained_grid5x5x5_nz11` (which has `int` value), run
`tests/test_simulation.py` against it, confirm pass.

Then **separately** save a freshly-built Precompute that uses
`n_ret_nodes_1d=(3, 5, 3)`, reload its metadata, reconstruct the
DiscretizationConfig, and verify the round-trip produces an identical tuple.
Do not commit this saved run.

## Risk register

- **Numba/njit kernels**: solver code consumes `ret_nodes`, `ret_weights`,
  `exp_ret_bill/stock/bond` as flat arrays. None reference `n_ret_nodes_1d`
  directly. The asymmetric build preserves shapes and dtypes, so njit kernels
  should not need recompilation logic changes. Verify by running
  `tests/test_terminal_grad_hess.py` (which exercises njit FOC kernels) after
  the change.
- **JSON metadata round-trip**: tuples become lists in JSON. Any new code that
  reads the field must accept both `int` and `list`. The four test loaders
  listed above are the candidates. The fix is the two-line shim in §5.
- **Print-formatting strings**: there are exactly three locations that format
  `K^n_ret` (precompute, diagnostics ×2, simulation). Missing one will produce
  ugly output but won't break correctness. Grep for `n_ret_nodes_1d` in the
  source tree before declaring done — any remaining `^model.n_ret` or
  `K_r ** n_ret` formula needs updating.
- **Hidden assumptions about uniform K**: search the code for `** model.n_ret`,
  `** n_ret`, and `**3` near return-quad context. None should compute the joint
  count from `K^n_ret`; all should use `len(ret_weights)` or `pc.n_ret_quad`.
  If you find one, it's a latent bug — fix it.

## Non-goals (do not do)

- Do not reorder return columns. The convention `[rtb, xr, xb]` is hard-coded
  across the model, VAR partition, and identity verifications. Per-dim K just
  rebalances Hermite resolution; the column order stays.
- Do not change the eigendecomp transform. Switching to Cholesky (lower
  triangular) would change which physical residual the principal direction
  corresponds to, undermining the whole rationale ("K_xr controls the
  largest-variance principal direction").
- Do not auto-tune K_ret based on the arbitrage diagnostic. That is a separate
  task and must remain user-controlled here.
- Do not bump default values. Current default `n_ret_nodes_1d: int = 2` stays.

## Files touched (final checklist)

Code:

- [ ] `model.py` — type annotation
- [ ] `discretization.py` — `get_return_quadrature` body + new helper `_normalize_ret_nodes`
- [ ] `precompute.py` — validation, store normalized tuple, update print
- [ ] `diagnostics.py` — two prints
- [ ] `simulation.py` — one print

Tests:

- [ ] New test for asymmetric quadrature properties (location: caller's choice;
      `tests/test_judd_quadrature.py` is a reasonable home).
- [ ] Existing test loaders updated to accept `list` from JSON (4 files).

Docs (after code is working):

- [ ] `contextfiles/RETURNS.md §6.12` — mark task done
- [ ] `contextfiles/STATE_SPACE.md`, `contextfiles/DESIGN.md`,
      `contextfiles/TODO.md` — wording updates
- [ ] `HANDOFF_COMPLEXITY_ANALYSIS.md` — table note

Notebooks (intentionally untouched in this PR):

- `main.ipynb`, `main_part2.ipynb`, `verify_discretization.ipynb` — keep
  passing `int`, user will switch to tuple manually when adopting per-dim K's.

## Reference: what success looks like

After this PR, this code path must work:

```python
from model import DiscretizationConfig
from precompute import Precompute, build_model

# Legacy form — must continue to work unchanged
disc_old = DiscretizationConfig(n_ret_nodes_1d=3, ...)

# New form — refine stock axis only
disc_new = DiscretizationConfig(n_ret_nodes_1d=(3, 9, 3), ...)

pc_old = Precompute(model, disc_old)   # n_ret_quad = 27
pc_new = Precompute(model, disc_new)   # n_ret_quad = 81

# pc_old.n_ret_nodes_1d == (3, 3, 3)
# pc_new.n_ret_nodes_1d == (3, 9, 3)
# All other arrays (ret_nodes, ret_weights, exp_ret_*) keep their contracts.
```

And calling `Precompute(model, disc_new)` produces the print line
`Return quad  : (3x9x3) nodes/dim  ->  81 joint nodes`.
