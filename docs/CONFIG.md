# Production Configuration — Economics, Discretization, Numerical Solver

**Purpose.** Single source of truth for the run knobs we are using in production. Avoids drift across `configs/` files and `saved_runs/checkpoints/*/metadata.json`.

**Architecture.** [`configs/_canonical.py`](../configs/_canonical.py) holds every value. Each sweep cell in [`configs/sweep_main/`](../configs/sweep_main/) and the dev/smoke configs ([`configs/system_iv_5x5x5.py`](../configs/system_iv_5x5x5.py), [`configs/smoke_test.py`](../configs/smoke_test.py)) is a thin override of the canonical via `_replace(...)` on `CANONICAL_DISC` / `CANONICAL_SOLVER`. To dial a value across all cells, edit `_canonical.py` and re-run [`scripts/_gen_sweep_main.py`](../scripts/_gen_sweep_main.py).

**System.** The canonical production config uses `PREDICTABILITY_SYSTEM = "full"`:
the 3-axis real-yields VAR with state `(cape, spr, y_1)` and returns
`(xr, xb)`.

**Term-premium theta.** `TERM_PREMIUM_THETA = None` uses the active empirical
baseline at `data/var_dataset.csv`. Setting a numeric theta selects a generated
yield-stage term-premium-scaling dataset through
`resolve_var_csv_path(theta)`, for example
`data/term_premium_scaling/var_dataset_theta_0p00.csv`.

The theta datasets are created with:

```text
python data/build_var_dataset_term_premium_scale.py
```

Use the config helper when constructing a model:

```python
from configs._canonical import prepare_canonical_predictability_system

system = prepare_canonical_predictability_system(term_premium_theta=0.0)
var_config = system["var_config"]
disc_config_template = system["disc_config"]
```

---

## 1. Economics — `BASE_CONFIG` dict

Identical across **every** cell. Calibration sources are documented in [LABOUR.md](LABOUR.md), [RETURNS.md](RETURNS.md), and [DESIGN.md](DESIGN.md).

### 1.1 Preferences

| Key | Value | Meaning |
|---|---|---|
| `gamma` | `5.0` | CRRA risk aversion |
| `beta` | `0.96` | Annual discount factor |
| `b_bar` | `10` | Bequest horizon in years (Catherine 2025) |

### 1.2 Lifecycle

| Key | Value | Meaning |
|---|---|---|
| `start_age` | `22` | First decision age |
| `retire_age` | `67` | First retirement age (no labor income) |
| `terminal_age` | `99` | Last age (bequest only) |

→ 78 ages total (22..99 inclusive); 45 working ages (22..66) + 33 retirement ages (67..99).

### 1.3 Labor income — Catherine (2025) / Guvenen et al. (2022)

Age-earnings polynomial `b0 + b1·age + b2·(age/10)² + b3·(age/100)³`:

| Key | Value |
|---|---|
| `b0` | `-6.142` |
| `b1` | `0.3040` |
| `b2` | `-0.051` |
| `b3` | `0.002586` |

Persistent income process `z_{t+1} = ρ z_t + η`, `η` drawn from the two-component Gaussian mixture:

| Key | Value | Meaning |
|---|---|---|
| `rho` | `0.991` | AR(1) coefficient on persistent z |
| `pz` | `0.176` | Mixture weight on component 1 of η |
| `mu_eta1` | `-0.524` | Mean of η component 1 |
| `sigma_eta1` | `0.113` | Std of η component 1 |
| `mu_eta2` | `-(pz/(1-pz))·mu_eta1` ≈ `0.1118` | **Derived** (E[η]=0); recomputed in quadrature |
| `sigma_eta2` | `0.046` | Std of η component 2 |

Transitory shock `ε`, two-component Gaussian mixture:

| Key | Value | Meaning |
|---|---|---|
| `pe` | `0.044` | Mixture weight on component 1 of ε |
| `mu_eps1` | `0.134` | Mean of ε component 1 |
| `sigma_eps1` | `0.762` | Std of ε component 1 |
| `mu_eps2` | `0.0` | Mean of ε component 2 |
| `sigma_eps2` | `0.055` | Std of ε component 2 |

`mu_eps2` is also derived (E[ε]=0 ⇒ `mu_eps2 = -(pe/(1-pe))·mu_eps1`); the stored `0.0` is informational and the quadrature recomputes from `pe`, `mu_eps1`.

### 1.4 Constraint

| Key | Value | Meaning |
|---|---|---|
| `constrained` | `False` | Unconstrained portfolio choice (with numerical leverage cap, §2.6) |

---

## 2. Numerical solver — `CANONICAL_SOLVER`

> Tuned. Stays the same across every cell except `alpha_min/alpha_max` (§2.6 — the only swept solver field).

These are *not* the `SolverConfig` defaults from [model.py:126](../model.py#L126); they are the production values produced by tuning. Compare against `model.py` defaults so you can see what was tuned and why.

### 2.1 Newton iteration

| Key | Canonical | model.py default | Why |
|---|---|---|---|
| `tol` | `1e-7` | `1e-7` | FOC convergence tolerance |
| `max_iter` | `20` | `20` | Constrained Newton cap |
| `max_iter_unconstrained` | `8000` | `5000` | Raised; some leverage-cap-active points need many iterations |
| `edge_max_iter` | `8` | `8` | 1D edge Newton cap |

### 2.2 Initial guess

| Key | Canonical | model.py default | Why |
|---|---|---|---|
| `init_alpha_s` | `0.85` | `0.1` | Tuned to typical interior optimum (~0.85 stock, ~0.44 bond → implied −0.29 bill leverage); cuts iterations for cold starts at each new (i_s, z_i) chain and after warm-resets |
| `init_alpha_b` | `0.44` | `0.4` | Same |

### 2.3 Step control

| Key | Canonical | model.py default | Why |
|---|---|---|---|
| `step_damp_constrained` | `0.2` | `0.2` | Max Newton step length (constrained) |
| `step_damp_unconstrained` | `0.3` | `0.3` | Max Newton step length (unconstrained, line search off) |
| `grad_step_size` | `0.05` | `0.05` | Gradient-descent fallback step when Jacobian singular |

### 2.4 Backtracking line search (unconstrained branch)

| Key | Canonical | model.py default | Why |
|---|---|---|---|
| `use_line_search` | `True` | `True` | Backtracking line search with monotone decrease on ‖FOC‖. Safe with the stagnation-exit fix in `solver.py`: any failure to find a descent step exits cleanly with `EC_NEWTON_FAIL` instead of spinning |
| `max_backtrack_iter` | `10` | `10` | Max halvings: `α_min = 1/2¹⁰ ≈ 1e-3` |
| `line_search_max_step` | `2.0` | `2.0` | Raw step cap before backtracking |

### 2.5 Tolerances and safety clamps

All at defaults. Documented for reference; do not change without strong reason.

| Key | Value |
|---|---|
| `tiny_savings` | `1e-6` |
| `corner_tol` | `1e-8` |
| `edge_accept_factor` | `10.0` |
| `singular_det` | `1e-15` |
| `grad_denom_eps` | `1e-10` |
| `min_wealth_inv` | `1e-10` |
| `min_consumption` | `1e-10` |
| `min_return_power` | `1e-15` |
| `prob_skip_threshold` | `1e-12` |
| `euler_inv_floor` | `1e-20` |
| `egm_anchor` | `1e-10` |

### 2.6 Numerical leverage cap (unconstrained branch only) — **the only swept solver knob**

| Key | Canonical baseline | Why |
|---|---|---|
| `alpha_min` | `-6.0` | Box projection on (α_s, α_b) inside the unconstrained Newton. ±6 is a real cap (prior production hit max simulated \|α\| ≈ 9.25 at γ=5 on 7×7×7 wide-support); cap-bound cells surface as `EC_NEWTON_FAIL` in `diagnostics['total_newton_failures']` |
| `alpha_max` | `+6.0` | Same |

→ See §3.4 for the swept variants (±5, ±3). Note: `01_base`, `03_state33`, and `08_grid9_base` previously inherited ±10 (effectively no cap); they now inherit ±6 from canonical, so all 10 cells have a real cap.

---

## 3. Discretization — `CANONICAL_DISC`

The canonical block plus the per-cell overrides in §3.4.

### 3.1 Wealth and savings grids

| Key | Canonical | Notes |
|---|---|---|
| `n_wealth` | `150` | Wealth grid size |
| `wealth_min` | `0.05` | **Set explicitly in `_canonical.py`** (model.py default also `0.05`). Raised from `1e-4` on 2026-05-03 to skip the EGM constrained region — see [STATE_SPACE.md](STATE_SPACE.md) and [EE_DIAGNOSTIC_WORKFLOW.md](../EE_DIAGNOSTIC_WORKFLOW.md). `precompute.py:117` reads this directly; no override anywhere else. |
| `wealth_max` | `200.0` | Top of wealth grid (model.py default) |
| `n_savings` | `150` | EGM savings-grid size |
| `savings_min` | `1e-8` | Lower endpoint of EGM grid (model.py default) |
| `savings_max` | `None` | Falls back to `wealth_max` |

### 3.2 Income process discretization

| Key | Canonical | Notes |
|---|---|---|
| `n_z` | `9` | Persistent income grid (Rouwenhorst; must be odd) |
| `n_stds` | `3.0` | z-grid covers ±3 unconditional std devs |
| `n_eps_nodes` | `3` | Judd-mixture nodes for transitory ε (poly. exactness 2n-1 ⇒ 5). Matches mean/var/skew/kurt to machine precision |
| `n_eta_nodes` | `3` | Judd-mixture nodes for persistent η innovation. Same rationale |

### 3.3 Financial-state VAR discretization

| Key | Canonical baseline | Notes |
|---|---|---|
| `state_grid_mode` | `"cholesky"` | u-space half-width interpretation; see [STATE_SPACE.md](STATE_SPACE.md). Legacy alias `"principal"` is accepted for backward compat with old saved bundles. |
| `state_grid_sizes` | `(7, 7, 7)` | **Swept** (§3.4) |
| `state_n_stds` | `(2.0, 2.25, 2.25)` | u-space half-width per axis. Coverage = `2Φ(n_d)-1` per axis → `(95.5%, 97.6%, 97.6%)`, joint ≈ 91%. **Earlier value `(0.6, 1.75, 2.0)` gave joint ≈ 40% and produced unusable simulator moments — do not regress.** Production-grade 99% per-axis would need `(~2.93, ~2.93, ~2.93)`. |
| `n_state_quad_nodes` | `(2, 2, 5)` | **Swept** (§3.4); axis-3 always 5 (volatility axis needs refinement) |
| `n_ret_nodes_1d` | `(3, 7, 5)` | Held fixed in current sweep; arbitrage-free minimum is `(3, 5, 3)` |

### 3.4 Sweep matrix (current — [`scripts/_gen_sweep_main.py`](../scripts/_gen_sweep_main.py))

10 production cells. All inherit `BASE_CONFIG`, `CANONICAL_DISC`, `CANONICAL_SOLVER` and apply only the overrides shown.

| # | Label | Disc overrides | Solver overrides | Est. hr |
|---|---|---|---|---|
|  1 | `base`              | (none) | (none) |  2 |
|  2 | `cap_only`          | (none) | α-cap ±5 |  2 |
|  3 | `state33`           | quad=(3,3,5) | (none) |  5 |
|  4 | `state33_cap`       | quad=(3,3,5) | α-cap ±5 |  5 |
|  5 | `state44_cap`       | quad=(4,4,5) | α-cap ±5 |  8 |
|  6 | `inc55_cap`         | n_eps=5, n_eta=5 | α-cap ±5 |  7 |
|  7 | `mid_rich_cap`      | quad=(3,3,5), n_eps=5, n_eta=5 | α-cap ±5 | 12 |
|  8 | `grid9_base`        | grid=(9,9,9) | (none) |  5 |
|  9 | `grid9_state33_cap` | grid=(9,9,9), quad=(3,3,5) | α-cap ±5 | 12 |
| 10 | `tight_cap`         | quad=(3,3,5) | α-cap ±3 |  5 |

Override grammar is open: any field of `DiscretizationConfig` or `SolverConfig` can be put in a cell's override dict. To sweep `state_n_stds`, add `{"state_n_stds": (2.5, 2.5, 2.5)}` to a cell's `disc_overrides`.

### 3.5 Production target

When the sweep concludes, the **production reference** is the cell that satisfies the [GRID_CONVERGENCE_CRITERIA.md](GRID_CONVERGENCE_CRITERIA.md) thresholds at the lowest compute cost. Until that selection is locked, treat **`01_base`** (7×7×7, no cap) as the working baseline for diagnostics and PR-time smoke runs.

---

## 4. Smoke / development configs

Both inherit `BASE_CONFIG` + `CANONICAL_SOLVER` and apply only the discretization overrides shown.

### 4.1 [`configs/system_iv_5x5x5.py`](../configs/system_iv_5x5x5.py) — full-lifecycle dev solve

Disc overrides relative to canonical: `state_grid_sizes=(5, 5, 5)`. Bundle suffix `_v2`. Everything else (including the tuned solver block and the wider `state_n_stds`) inherits.

### 4.2 [`configs/smoke_test.py`](../configs/smoke_test.py) — minimum-viable CI smoke (~2–5 min)

Disc overrides relative to canonical:

| Key | Smoke | Canonical |
|---|---|---|
| `n_wealth` | `40` | `150` |
| `n_savings` | `40` | `150` |
| `state_grid_sizes` | `(3, 3, 3)` | `(7, 7, 7)` |
| `n_z` | `5` | `9` |
| `n_state_quad_nodes` | `(2, 2, 2)` | `(2, 2, 5)` |
| `n_ret_nodes_1d` | `(3, 5, 3)` | `(3, 7, 5)` |

`wealth_min`, `state_n_stds`, `n_eps_nodes`, `n_eta_nodes`, and the entire solver block inherit canonical. **Do not shrink** `n_eps_nodes`, `n_eta_nodes`, or the third return-quad axis below smoke values — moment match degrades.

---

## 5. How to use this document

- **Dialing a value across all cells** (e.g. widen `state_n_stds`, raise `wealth_min`, tune the solver): edit [`configs/_canonical.py`](../configs/_canonical.py), then re-run `python scripts/_gen_sweep_main.py`. Every cell that doesn't explicitly override that field inherits the new value.
- **Adding a new sweep cell:** edit the `SWEEP` matrix in [`scripts/_gen_sweep_main.py`](../scripts/_gen_sweep_main.py) and re-run it. Hand-edits to `configs/sweep_main/*.py` are overwritten.
- **One-off experimental run:** copy [`configs/system_iv_5x5x5.py`](../configs/system_iv_5x5x5.py) → new file, change the override dict, give it a unique `BUNDLE_SUFFIX`. Do not commit one-offs into `configs/sweep_main/`.
- **Bumping the canonical solver tuning:** update §2 here and `configs/_canonical.py` in the same PR. Re-run the generator. No per-cell edits needed.
- **Bumping the economics block:** update §1 here and `configs/_canonical.py`. Also update `_reference_base_config()` in `tests/test_partial_solve.py` (still has its own copy — flagged drift).

---

## 6. Source-of-truth pointers

| Block | Authoritative location |
|---|---|
| All three blocks (canonical) | [`configs/_canonical.py`](../configs/_canonical.py) |
| `SolverConfig` semantics | [`model.py:126`](../model.py#L126) |
| `DiscretizationConfig` semantics | [`model.py:95`](../model.py#L95) |
| Sweep matrix | [`scripts/_gen_sweep_main.py`](../scripts/_gen_sweep_main.py) (`SWEEP`) |
| Run driver | [`scripts/run_solve.py`](../scripts/run_solve.py) |
| Last full-lifecycle production checkpoint | `saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid9x9x9_nz9_grid9_base/metadata.json` (2026-05-03 — but with `state_n_stds=(0.6, 1.75, 2.0)`, the pre-fix value; rerun under new canonical before relying on it) |
