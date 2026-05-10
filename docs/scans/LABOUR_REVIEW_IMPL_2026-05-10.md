# Labour Income Subsystem — Implementation/Pipeline Review (2026-05-10)

Reviewer angle: implementation/pipeline correctness (the math/spec angle is
covered by the parallel reviewer in `LABOUR_REVIEW_MATH_2026-05-10.md`).

Branch: `jax-rewrite`. Repo: `c:\Users\carlh\Projekt\thesisscripts_JAX\`.
This is a read-only pass. No source files were modified. One small Python
probe was used to confirm shapes/dtypes/zero-mean — output is reproduced
inline; nothing was written to disk other than this scan file.

---

## §1 Scope and angle

The audit covers the *pipeline* path of the labour-income subsystem from
config dicts through the precompute factory to the JIT-compiled solver
kernels:

1. Income-table construction in `lifecycle/precompute.py`.
2. The status of `Pi_z` after the 2026-05-09 drop from `Precompute`.
3. Wiring of `working_income_next`, `pension_after_tax`, and the
   bracketed-z bridge through `_build_per_age_working_kernel` /
   `working_foc_jac_ccv`.
4. Single-source-of-truth audit on the `mu_eta2` / `mu_eps2` derivation
   (Fix A).
5. End-to-end shape/dtype invariant audit and `gather_precision`
   interaction.
6. Pension wiring across retire / boundary / working ages, including the
   `t+1` indexing convention.
7. `DiscretizationConfig` knob propagation (`n_z`, `n_eta_nodes`,
   `n_eps_nodes`).
8. Bundle/diagnostics serialization — what is recoverable from
   `metadata.json` after a solve.

What is explicitly out of scope: the economic correctness of the
mixture parameters (Catherine 2025 vs GKOS), the calibration accuracy
of `b0..b3`, the choice of payroll/income-tax brackets, and the
mortality-calibration spec. Those are the math reviewer's territory.

---

## §2 Income-table construction in precompute

All construction lives in
[lifecycle/precompute.py:230-554](../../lifecycle/precompute.py#L230)
inside `build_precompute(model, disc_config)` and three helpers.

### 2.1 Deterministic age component

[precompute.py:438-441](../../lifecycle/precompute.py#L438):

```python
log_det_profile = (model.b0
                   + model.b1 * ages
                   + model.b2 * ages**2 / 10.0
                   + model.b3 * ages**3 / 100.0)
```

Shape `(n_age,)`, dtype float64. `ages = np.arange(start_age, terminal_age + 1)`
([precompute.py:289](../../lifecycle/precompute.py#L289)) — int64. The
`avg_det` field at [precompute.py:443-448](../../lifecycle/precompute.py#L443)
is a *separate* recomputation of `exp(f)` over the working sub-range only
(`np.arange(start_age, retire_age)`); it's used by
`compute_pension_after_tax` ([precompute.py:452](../../lifecycle/precompute.py#L452)).
Probe: at canonical 22..99 with `b0=-6.142, b1=0.3040, b2=-0.051, b3=0.002586`,
`avg_det = 0.50686` — matches Catherine's documented ~0.50.

### 2.2 `working_income` table

Built by `_precompute_working_income`
([precompute.py:603-620](../../lifecycle/precompute.py#L603)):

```python
z   = z_grid[None, :, None]            # (1, n_z, 1)
eps = eps_nodes[None, None, :]         # (1, 1, n_eps)
det = log_det_profile[:, None, None]   # (n_age, 1, 1)
y_gross = np.exp(det + z + eps)        # (n_age, n_z, n_eps)
return disposable_income_working(y_gross)
```

Output shape: `(n_age, n_z, n_eps)`, dtype float64. Probe (canonical
ages 22..99, n_z=5, n_eps=3): shape `(78, 5, 3)`, dtype `float64`. The
tax application is **after** the broadcast — `disposable_income_working`
([model.py:421-444](../../lifecycle/model.py#L421)) consumes the full
3-D array element-wise and returns the same shape, so the table is
*after-tax* throughout. There is **no eta channel here** because the
working-age FOC consumes income at *current* (z, eps) but at *next-period*
(z_next, eps_next); the orthodox lookup table for FOC consumption is
`working_income_next`, see §2.3.

`working_income` itself is consumed by:
- `_build_discrete_income_next_table` in the variant solver
  ([solver_pi_z_variant.py:64-80](../../lifecycle/solver_pi_z_variant.py#L64))
  — it shifts `pc.working_income[1:]` to align with `t+1` semantics for
  the `(t, iz_current, iz_next, eps)` Pi_z-based table.
- The simulator does **not** read `working_income`; instead it
  reconstructs gross income live with `mu_eta2 / mu_eps2` mixture draws
  ([simulation.py:379-394](../../lifecycle/simulation.py#L379)) and
  passes through `jax_disposable_income`
  ([simulation.py:188-216](../../lifecycle/simulation.py#L188)).

### 2.3 `working_income_next` table — FOC-consumed lookup

Built by `_precompute_working_income_next`
([precompute.py:638-687](../../lifecycle/precompute.py#L638)):

```python
log_det_next = log_det_profile[1:][:, None, None, None]   # (n_age-1, 1, 1, 1)
z_term       = (rho * z_grid)[None, :, None, None]         # (1, n_z, 1, 1)
eta_term     = eta_nodes[None, None, :, None]              # (1, 1, n_eta, 1)
eps_term     = eps_nodes[None, None, None, :]              # (1, 1, 1, n_eps)
y_gross      = np.exp(log_det_next + z_term + eta_term + eps_term)
table[:-1]   = disposable_income_working(y_gross)
```

Output shape: `(n_age, n_z, n_eta, n_eps)`, dtype float64. This is the
table indexed as `pc.working_income_next[t]` to give next-period income
when transitioning *out of* row t.

**Mathematically** the gross-income expression is

```
log y_{t+1} = f(age_{t+1}) + (rho * z_t + eta_{t+1}) + eps_{t+1}
```

i.e. z_next is an *exact* mixture quadrature node `rho * z_grid[iz] +
eta_nodes[k_eta]`. There is no policy-side bracket at construction;
the bracket happens later in the FOC kernel
([solver.py:2486](../../lifecycle/solver.py#L2486),
[solver.py:2601](../../lifecycle/solver.py#L2601)) when interpolating
the next-age value function at z_next.

**Boundary handling (probe-confirmed):** rows where `t + 1 >=
retire_age_idx` are filled with zeros
([precompute.py:684-687](../../lifecycle/precompute.py#L684)). At
canonical retire_age=67, start_age=22 → `retire_age_idx = 45`. Probe
output (n_age=78):

```
retire_age_idx = 45
next_is_working[ridx-2..ridx+1]:  [(43, True), (44, False), (45, False), (46, False)]
table[ridx-2] all-zero?: False
table[ridx-1] all-zero?: True
table[ridx]   all-zero?: True
```

This matches the orchestrator's branch logic at
[solver.py:3075-3093](../../lifecycle/solver.py#L3075):
- Working ages (`age < retire_age - 1`): orchestrator slices
  `working_income_next_jnp[t + 1]` and dispatches to `working_kernel`.
- Boundary age (`age == retire_age - 1`, i.e. last working year):
  orchestrator passes a `jnp.zeros((n_z, n_eta, n_eps))` placeholder
  for `income_table` *and* the next pension slice; dispatches to
  `boundary_kernel`. The placeholder is *ignored* inside the kernel
  because `use_pension_next=True` selects the pension branch.

### 2.4 `pension_after_tax` table

Built by `_precompute_pension`
([precompute.py:623-635](../../lifecycle/precompute.py#L623)):

```python
base_pension = compute_pension_after_tax(z_grid, avg_det)   # (n_z,)
return np.broadcast_to(base_pension, (n_age, n_z)).copy()
```

Shape `(n_age, n_z)`, dtype float64. The pension is **constant across
ages** (Catherine eq. 19/20), so the broadcast over the age axis is
purely a layout convenience for the orchestrator to slice
`pension_table_jnp[t + 1, :]` uniformly. This is intentional;
`compute_pension_after_tax` takes `z_grid` *raw* and applies AIME and
PIA formulas plus the same 7-bracket tax schedule used for working
income. So pension is **after-tax** at construction time, identical to
`working_income`.

### 2.5 Tax application sites

Tax is applied at exactly one place per table: inside
`disposable_income_working`
([model.py:421-444](../../lifecycle/model.py#L421)) for both
`working_income` and `working_income_next` (and the simulator's
initial-period fallback at
[simulation.py:634](../../lifecycle/simulation.py#L634)), and inside
`compute_pension_after_tax`
([model.py:447-501](../../lifecycle/model.py#L447)) for the pension
table. The simulator's per-period working-age branch uses the JAX port
`jax_disposable_income`
([simulation.py:188-216](../../lifecycle/simulation.py#L188)) which is
documented as bit-identical to the host helper. The host and JAX
versions both implement the same hard-coded TCJA brackets — there is
**no shared lookup table**, so any future bracket change must be made
in two places. Not a current bug; a future-discipline footgun (LOW
severity).

### 2.6 Layout consistency check

| Table                | Shape                       | Indexed in solver as              | Source                                    |
|----------------------|-----------------------------|-----------------------------------|-------------------------------------------|
| `working_income`     | `(n_age, n_z, n_eps)`       | NOT consumed by production solver | docstring [precompute.py:115](../../lifecycle/precompute.py#L115) |
| `working_income_next`| `(n_age, n_z, n_eta, n_eps)`| `working_income_next_jnp[t + 1]`  | [solver.py:3087](../../lifecycle/solver.py#L3087) |
| `pension_after_tax`  | `(n_age, n_z)`              | `pension_table_jnp[t + 1, :]`     | [solver.py:3067, 3077](../../lifecycle/solver.py#L3067) |

Note that the production solver **does not consume `working_income`**.
It is constructed but only the variant solver and host-side diagnostics
read it. This is documented at the docstring but worth flagging — it's
a candidate for removal once the variant is retired or refactored.

---

## §3 Pi_z status

`Pi_z` was dropped from `Precompute` in the post-2026-05-09 refactor.
The current `Precompute` NamedTuple
([precompute.py:65-216](../../lifecycle/precompute.py#L65)) contains:

```
z_grid       (n_z,)
init_z_probs (n_z,)        # stationary Gaussian approx
eps_nodes    (n_eps,)
eps_weights  (n_eps,)
eta_nodes    (n_eta,)
eta_weights  (n_eta,)
dz           float
```

**No `Pi_z` field.** The construction site at
[precompute.py:389-398](../../lifecycle/precompute.py#L389) calls
`discretize_income_ar1_mixture` but unpacks only `z_grid` and discards
the returned matrix:

```python
z_grid, _ = discretize_income_ar1_mixture(...)
```

### Grep audit — who consumes Pi_z?

```
$ grep -rn "Pi_z" lifecycle/ verify/ scripts/ configs/ tests/ 2>/dev/null
```

Production source files referencing `Pi_z`:

- `lifecycle/precompute.py:380-398` — comments + drop site only.
- `lifecycle/simulation.py:609-613` — comment in the "array of indices"
  fallback branch noting Pi_z is gone.
- `lifecycle/solver_pi_z_variant.py` — the *variant* solver
  ([solver_pi_z_variant.py:8-17](../../lifecycle/solver_pi_z_variant.py#L8))
  acknowledges this and reconstructs `Pi_z` locally via
  `_build_pi_z_local`
  ([solver_pi_z_variant.py:37-55](../../lifecycle/solver_pi_z_variant.py#L37)).
  That local Pi_z is then attached to the per-call PCJaxPi tuple via
  `_pc_to_jnp_with_pi_z`
  ([solver_pi_z_variant.py:58-61](../../lifecycle/solver_pi_z_variant.py#L58))
  and read inside `per_cell` at
  [solver_pi_z_variant.py:450, 603](../../lifecycle/solver_pi_z_variant.py#L450).

So the variant is the *only* consumer and it self-builds Pi_z. Any
reducibility caveat (canonical (rho=0.991, n_z=11, n_stds=3.0) — see
INCOME_PIPELINE_REVIEW_2026-05-09.md MED-1) applies *only* to the
variant solver, not to production. The variant docstring already calls
this out and recommends `n_stds=2.25` for variant runs.

**Conclusion:** Pi_z is correctly contained. Production solver and
simulator are clean. The drop is a clear example of correctly removing
a fragile non-load-bearing array.

---

## §4 Working/boundary kernel wiring

### 4.1 Builder dispatch

[solver.py:2845-2846](../../lifecycle/solver.py#L2845):

```python
working_kernel  = _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next=False, ...)
boundary_kernel = _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next=True,  ...)
```

Same builder, two compilations, gated on the `use_pension_next` Python
bool which closes over the JIT trace. Bit-identical math at every
non-boundary cell because the bool is a Python-level switch, not a
runtime branch.

### 4.2 Per-cell kernel body — `working_kernel` path

[solver.py:2597-2631](../../lifecycle/solver.py#L2597) (vmap-only path):

```python
z_now = pcj.z_grid[z_idx]
z_next = mp.rho * z_now + pcj.eta_nodes                            # (n_eta,)
iz_lo, frac_z = vmap(bracket_uniform, in_axes=(0, None, None, None))(
    z_next, pcj.z_grid[0], pcj.dz, pcj.z_grid.shape[0]
)                                                                  # both (n_eta,)
income_table = income_next_table_z[z_idx]                          # (n_eta, n_eps)
```

So inside the kernel, `income_table` is exactly the precomputed table
sliced at the current z_idx — this is the (k_eta, i_e) value-function
inputs *at the next age*. The shape contract:

| Variable     | Expected shape          | Confirmed                        |
|--------------|-------------------------|----------------------------------|
| `z_next`     | `(n_eta,)`              | yes, exact value at quadrature node |
| `iz_lo`      | `(n_eta,)`, int32       | clip to [0, n_z - 2]             |
| `frac_z`     | `(n_eta,)`, float64     | clip to [0, 1]                   |
| `income_table` | `(n_eta, n_eps)`     | gathered from precomputed table  |

### 4.3 Per-cell kernel body — `boundary_kernel` path

Same builder but `use_pension_next=True`. Lines
[solver.py:2605-2610](../../lifecycle/solver.py#L2605):

```python
pension_at_eta = (
    (1.0 - frac_z) * pension_next_by_z[iz_lo]
    + frac_z * pension_next_by_z[iz_lo + 1]
)
income_table = pension_at_eta[:, None] * jnp.ones_like(pcj.eps_weights)[None, :]
```

So in the boundary case, the income at next age is the **pension**
linearly interpolated in z at each eta-induced z_next, broadcast over
the eps axis. There is no eps draw (pension is z-only) — the broadcast
is the canonical "tile across eps" device that lets the FOC summation
`sum(eta_w * eps_w * ...)` reduce to `sum(eta_w * pension_at_eta)` at
eps_w-row exposed.

This is correct: at age = retire_age - 1 (the last working age), next
age is age = retire_age, the first retired age, so income = pension. The
eps shock is not active in retirement.

### 4.4 FOC integration structure

[solver.py:1188-1230](../../lifecycle/solver.py#L1188):

```python
x_next = sR_p[:, :, None, None] + income_next_table[None, None, :, :]
# (n_state_quad, n_ret_quad, n_eta, n_eps)
...
weight_full = (
    weight_kv_kr[:, :, None, None]
    * eta_weights[None, None, :, None]
    * eps_weights[None, None, None, :]
)
alive_factor = weight_full * psi_z
```

So the working FOC integrates over four discrete dimensions:

```
sum_{k_v, k_r} sum_{k_eta, i_e} weight_kv_kr * eta_w * eps_w * psi(z) * [...]
```

`weight_kv_kr` itself is built once at
[solver.py:1767](../../lifecycle/solver.py#L1767):

```python
weight_kv_kr = jnp.asarray(pc.v_weights)[:, None] * jnp.asarray(pc.ret_weights)[None, :]
```

shape `(n_state_quad, n_ret_quad)`. So the full integration weight is
a Kronecker product of four 1-D quadrature rules:

```
weight_kv_kr[k_v, k_r] * eta_weights[k_eta] * eps_weights[i_e]
```

Each summed over its own axis, all mass = 1. **All four sum to 1
independently** — checked at canonical via probe:

```
E[eta] from quadrature: 4.163e-17
E[eps] from quadrature: 3.469e-18
```

(machine-precision zero — both Judd quadratures correctly normalize the
mixture mean to zero given the derived `mu_eta2`/`mu_eps2`).

### 4.5 JIT trace stability

Each kernel is built once per solve via `_build_per_age_working_kernel`
([solver.py:2423](../../lifecycle/solver.py#L2423)). The closure
captures `pcj`, `mp`, `sc`, `n_z`, `N_state`, `use_pension_next`, and
`per_is_tensors`. None of these vary across the per-age loop, so the
JIT cache hits the same key every age. The orchestrator passes
`(c_next_jnp, income_table, pension_next, psi_t, init_a_s_arr,
init_a_b_arr)` — all of which have *fixed shapes* across ages:

| Argument                    | Fixed shape                          |
|-----------------------------|--------------------------------------|
| `c_next_jnp`                | `(n_z, N_state, n_w)`                |
| `income_table`              | `(n_z, n_eta, n_eps)` (or zero placeholder of same shape on boundary) |
| `pension_next`              | `(n_z,)`                             |
| `psi_t`                     | `(n_z,)`                             |
| `init_a_s_arr/init_a_b_arr` | `(n_z, N_state, n_savings)`          |

Confirmed by reading the orchestrator branch at
[solver.py:3066-3093](../../lifecycle/solver.py#L3066). Shape stability
across all working / boundary ages → one JIT trace per kernel.

### 4.6 The boundary-kernel placeholder caveat

[solver.py:3078](../../lifecycle/solver.py#L3078):

```python
income_table = jnp.zeros((n_z, pc.n_eta, pc.n_eps))   # ignored on this branch
```

This zero array is allocated *every* boundary call and dispatched into
the boundary kernel where the kernel internally selects the pension
branch via `use_pension_next=True` (Python bool, baked into trace).
The placeholder *is* literally ignored — there's no `where` mux that
could cause numerical contamination. The cost is an n_z × n_eta × n_eps
device allocation per orchestrator iteration, which is one out of the
n_age iterations (so 1 allocation in a full life-cycle solve). Trivial.

A cleaner pattern would be `pension_only_kernel(c_next, pension_next,
psi_t, ...)` with no income_table argument at all, but that would
require the boundary kernel to be a *separate* JIT trace, doubling
compile cost. The current design accepts the cosmetic ugliness in
exchange for one fewer JIT cache slot. Not worth changing.

---

## §5 Fix A single-source-of-truth audit

The `mu_eta2` / `mu_eps2` derivation is the canonical "Fix A" change
called out in
[INCOME_PIPELINE_REVIEW_2026-05-09.md](INCOME_PIPELINE_REVIEW_2026-05-09.md).

### 5.1 Single derivation site

[precompute.py:867-893](../../lifecycle/precompute.py#L867):

```python
pz_v = float(base_config["pz"])
mu_eta1_v = float(base_config["mu_eta1"])
pe_v = float(base_config["pe"])
mu_eps1_v = float(base_config["mu_eps1"])
mu_eta2_derived = -(pz_v / (1.0 - pz_v)) * mu_eta1_v
mu_eps2_derived = -(pe_v / (1.0 - pe_v)) * mu_eps1_v

# Self-check
_e_eta = pz_v * mu_eta1_v + (1.0 - pz_v) * mu_eta2_derived
_e_eps = pe_v * mu_eps1_v + (1.0 - pe_v) * mu_eps2_derived
assert np.isclose(_e_eta, 0.0, atol=1e-12), ...
assert np.isclose(_e_eps, 0.0, atol=1e-12), ...
```

The derived values are passed into `LifecyclePortfolioModel(...)` at
[precompute.py:910, 915](../../lifecycle/precompute.py#L910). This is
the **only** place where the constraint is materialised. The doc
strings at [model.py:49, 54](../../lifecycle/model.py#L49) document
the field as DERIVED and authoritative.

### 5.2 Consumers (audit)

I grep'd `mu_eta2|mu_eps2|mu_eta2_eff|mu_eps2_eff` across the codebase
and triaged each non-comment hit. Source files only:

| File:line                                                               | Form                                | Compliant? |
|-------------------------------------------------------------------------|-------------------------------------|------------|
| [discretization.py:441](../../lifecycle/discretization.py#L441)         | `model.mu_eps2`                     | yes        |
| [discretization.py:467](../../lifecycle/discretization.py#L467)         | `model.mu_eta2`                     | yes        |
| [precompute.py:394](../../lifecycle/precompute.py#L394)                 | `model.mu_eta2` (Pi_z site, dropped result) | yes |
| [precompute.py:483](../../lifecycle/precompute.py#L483)                 | `model.mu_eta2` (mortality calibration) | yes    |
| [precompute.py:910, 915](../../lifecycle/precompute.py#L910)            | derivation site                     | yes        |
| [solver_pi_z_variant.py:50](../../lifecycle/solver_pi_z_variant.py#L50) | `model.mu_eta2` (variant Pi_z)      | yes        |
| [simulation.py:251-252, 382, 392, 631, 742, 747](../../lifecycle/simulation.py#L251) | `model.mu_eta2` / `model.mu_eps2` | yes |
| [diagnostics.py:120, 122, 128, 131, 147, 155](../../lifecycle/diagnostics.py#L120) | `model.mu_eta2` / `model.mu_eps2` | yes |
| [mortality.py:284, 290, 292, 338, 348, 361](../../lifecycle/mortality.py#L284) | function-arg `mu_eta2` (caller passes `model.mu_eta2`) | yes |

Every production-source consumer reads `model.mu_eta2` / `model.mu_eps2`
directly. **No source file re-derives the constraint.** Probe confirms
the simulator-side draws use the model's stored value and produce
machine-precision-zero E[eta], E[eps] under the canonical:

```
mu_eta1, mu_eta2: -0.524 0.11192233009708737
mu_eps1, mu_eps2: 0.134 -0.006167364016736402
E[eta] from quadrature: 4.163e-17
E[eps] from quadrature: 3.469e-18
```

### 5.3 Non-source orphan re-derivations (out of production hot path)

Hits that *do* re-derive the formula manually:

| File:line                                                                                | Status                            |
|------------------------------------------------------------------------------------------|-----------------------------------|
| [verify/ee_simpath_inf_horizon.py:221](../../verify/ee_simpath_inf_horizon.py#L221)      | Diagnostic only; reads model.pz/mu_eta1, computes locally |
| [scripts/analysis/system_i_nz_convergence.py:88-99](../../scripts/analysis/system_i_nz_convergence.py#L88) | Convergence sweep; reads BASE_CONFIG (which no longer has mu_eta2) and rederives — correct workaround given the canonical now omits the key |
| [scripts/analysis/verify_income_process_sim.py:135, 196](../../scripts/analysis/verify_income_process_sim.py#L135) | Standalone analysis script |
| [docs/archive/old_tests/test_discretization.py:219](../../docs/archive/old_tests/test_discretization.py#L219) | Archived; not run |
| [docs/archive/QUADRATURE_REFACTOR_HANDOFF.md, GH_MIXTURE_HANDOFF.md, HANDOFF_UNCONSTRAINED_LEVERAGE.md] | Documentation; archived |
| [inf_horizon_benchmark.ipynb] (top-level)                                                | Notebook only; uses literal numerics |
| [tests/test_income_normalization.py:142, 184, 203-206](../../tests/test_income_normalization.py#L142) | The test suite that locks in Fix A — *intentionally* re-derives to verify the model agrees |

The only actively-run re-derivation in the analysis pipeline is in
`scripts/analysis/system_i_nz_convergence.py`. Now that BASE_CONFIG no
longer carries `mu_eta2`, that script is forced to re-derive from
(pz, mu_eta1) — which is correct. Same for the diagnostic script under
`verify/` and the standalone analysis at
`scripts/analysis/verify_income_process_sim.py`. None of these feed
into the production solver.

**Verdict: Fix A is in place and clean.** Single source of truth lives
at [precompute.py:879-880](../../lifecycle/precompute.py#L879).

### 5.4 Legacy bundle consideration

Older saved bundles under `saved_runs/` (e.g.
`saved_runs/ablations/system_ii_grid5x5_*/metadata.json`) contain
`mu_eta2` and `mu_eps2` keys in `base_config` because they were saved
*before* the BASE_CONFIG dict was trimmed. The deserialization path
through `build_model` would *silently ignore* those keys — see the
docstring at [precompute.py:867-873](../../lifecycle/precompute.py#L867).
Any rehydrated model from an old bundle gets the freshly-derived value,
not the saved literal. This is the documented and correct policy.

---

## §6 Shape/dtype invariant audit

Probe under canonical-tier config (n_age=78, n_z=5, n_eta=3, n_eps=3,
N_state=27, n_w=20, n_s=20, JAX_PLATFORMS=cpu):

| Array                              | Declared at                 | Actual shape              | Dtype     | Consumed at (key sites)             |
|------------------------------------|-----------------------------|---------------------------|-----------|-------------------------------------|
| `pc.working_income`                | precompute.py:115           | `(78, 5, 3)`              | float64   | variant only                        |
| `pc.working_income_next`           | precompute.py:118           | `(78, 5, 3, 3)`           | float64   | solver.py:3027, 3087                |
| `pc.pension_after_tax`             | precompute.py:123           | `(78, 5)`                 | float64   | solver.py:3024, 3067, 3077          |
| `pc.z_grid`                        | precompute.py:98            | `(5,)`                    | float64   | solver.py:2484, 2599 etc.           |
| `pc.init_z_probs`                  | precompute.py:99            | `(5,)`                    | float64   | simulation.py only                  |
| `pc.eps_nodes` / `eps_weights`     | precompute.py:105           | `(3,) / (3,)`             | float64   | solver.py FOC sums                  |
| `pc.eta_nodes` / `eta_weights`     | precompute.py:192-193       | `(3,) / (3,)`             | float64   | solver.py FOC sums                  |
| `pc.dz`                            | precompute.py:194           | scalar                    | float64   | bracket_uniform                     |
| `pc.log_det_profile`               | precompute.py:195           | `(78,)`                   | float64   | simulation, _precompute_pension     |
| `pc.avg_det`                       | precompute.py:196           | scalar                    | float64   | _precompute_pension only            |
| `pc.survival_probs_2d`             | precompute.py:204           | `(78, 5)`                 | float64   | solver.py:3026, 3050                |
| `pc.chi_vec`                       | precompute.py:205           | `(5,)`                    | float64   | mortality diagnostics               |
| `pc.ages`                          | precompute.py:144           | `(78,)`                   | int64     | orchestrator banner only            |
| `pc.s_grid`                        | precompute.py:143           | `(20,)`                   | float64   | EGM scan                            |
| `pc.wealth_grid`                   | precompute.py:138           | `(20,)`                   | float64   | interp                              |
| `weight_kv_kr` (constructed once)  | solver.py:1767              | `(n_state_quad, n_ret_quad)` | float64 | solver.py:1188-1230                 |

**Inside the JIT kernel** (working FOC):

| Variable                      | Expected shape                       | Source                                   |
|-------------------------------|--------------------------------------|------------------------------------------|
| `z_next`                      | `(n_eta,)`                           | solver.py:2485                           |
| `iz_lo`, `frac_z`             | `(n_eta,) int32`, `(n_eta,) float64` | solver.py:2486 (vmap of bracket_uniform) |
| `income_next_at_z`            | `(n_eta, n_eps)`                     | solver.py:2500 (gather at z_idx)         |
| `c_corners` (pre-gather)      | `(n_z, n_state_quad, n_corners, n_w)`| solver.py:1580                           |
| `c_corners_T` (transpose)     | `(n_state_quad, n_z, n_corners, n_w)`| solver.py:1581                           |
| `x_next` (working)            | `(n_state_quad, n_ret_quad, n_eta, n_eps)` | solver.py:1190                     |
| `weight_full`                 | `(n_state_quad, n_ret_quad, n_eta, n_eps)` | solver.py:1221                     |
| `c_at_xn`, `mpc_at_xn`        | `(n_state_quad, n_ret_quad, n_eta, n_eps)` | solver.py:1215                     |
| `mu_alive`, `mup_alive`       | `(n_state_quad, n_ret_quad, n_eta, n_eps)` | solver.py:1218-1219                |

All explicit; no silent broadcast hazards. The boundary kernel's
`pension_at_eta[:, None] * jnp.ones_like(eps_weights)[None, :]` at
[solver.py:2497, 2610](../../lifecycle/solver.py#L2497) is a clean
broadcast — `pension_at_eta` is `(n_eta,)` and the resulting
`income_table` has shape `(n_eta, n_eps)`, exactly matching the working
case's contract.

### 6.1 `gather_precision` interaction

The `gather_precision="f32"` toggle (canonical default per
[_canonical.py:150](../../configs/_canonical.py#L150)) routes through
`_resolve_gather_dtype`
([solver.py:386-401](../../lifecycle/solver.py#L386)) and casts inside
`_interp_c_and_mpc_at_cell`
([solver.py:974-1019](../../lifecycle/solver.py#L974)) and the inline
`per_kv_kr` in `retirement_foc_jac_ccv`
([solver.py:1064-1090](../../lifecycle/solver.py#L1064)).

The cast scope is **closed**: c_corners, w_corners, wealth_grid,
frac_z, x_next_scalar are cast f64→f32 on entry; `c_g` and `mpc_g` are
cast back f32→f64 *before* the `min_consumption` floor, the `[0,1]`
MPC clip, and any FOC arithmetic.

**Income tables are unaffected by gather_precision.** They live at f64
throughout. Confirmed by tracing `income_next_at_z` from the
orchestrator (f64) through `working_kernel.call()` into
`_solve_working_at_cell` ([solver.py:1591](../../lifecycle/solver.py#L1591))
and into the FOC. The income channel never touches the gather cast.
The `x_next = sR_p[:, :, None, None] + income_next_table[None, None, :, :]`
combine at [solver.py:1190](../../lifecycle/solver.py#L1190) produces an
f64 `x_next`; that f64 scalar is then cast to gather_dtype *inside*
`_interp_c_and_mpc_at_cell` per its own internal protocol.

So under `f32`, the precision boundary is exactly:
`{c_corners, w_corners, wealth_grid, frac_z, x_next_scalar} → f32`
inside the multilinear interp; everything else stays f64. The income
construction, eta-bracketed z-fraction, FOC sums, and Newton step all
remain f64.

The `_validate_state_quadrature` self-check at
[precompute.py:579-600](../../lifecycle/precompute.py#L579) verifies
the state quadrature returns reproduce conditional moments at <1e-10
error — runs at every solve, gates on `verbose`. Income quadrature has
no analogous self-check beyond the inline mean-zero assertion in
`build_model` ([precompute.py:888-893](../../lifecycle/precompute.py#L888)).

---

## §7 Pension wiring across retire / boundary / working

### 7.1 Orchestrator dispatch logic

[solver.py:3066-3094](../../lifecycle/solver.py#L3066):

```python
if age >= retire_age:
    pension_next = pension_table_jnp[t + 1, :]           # (n_z,)
    ... -> retirement_kernel(c_next, pension_next, psi_t, ...)
else:
    use_pen = (age == retire_age - 1)
    if use_pen:
        pension_next = pension_table_jnp[t + 1, :]       # (n_z,)
        income_table = jnp.zeros((n_z, pc.n_eta, pc.n_eps))   # ignored
        ... -> boundary_kernel(c_next, income_table, pension_next, psi_t, ...)
    else:
        pension_next = pension_dummy_z_jnp               # (n_z,) zeros
        income_table = working_income_next_jnp[t + 1]    # (n_z, n_eta, n_eps)
        ... -> working_kernel(c_next, income_table, pension_next, psi_t, ...)
```

### 7.2 The `t + 1` indexing convention

The convention is: `t` is the *current* age index (0..n_age-1). The
table at index `t + 1` corresponds to the *next* age. In all three
branches the orchestrator slices at `t + 1`. Importantly,
`c_next_jnp = C_list[t + 1]` ([solver.py:3052](../../lifecycle/solver.py#L3052))
is the *next age's converged consumption policy*, so all three
slicings are consistent: the kernel sees pension/income at the next
age.

### 7.3 Boundary case correctness

At `age == retire_age - 1` (the last working year, t = retire_age -
start_age - 1 = 44 at canonical), the agent is alive and working at age
66; next period (t+1 = 45 = retire_age_idx) the agent is retired. The
income at t+1 is therefore pension(z_{t+1}). z_{t+1} is the bracketed
quadrature point `rho * z_t + eta`, so the pension lookup is

```
pension_at_eta[k_eta] = (1 - frac_z[k_eta]) * pension(iz_lo[k_eta])
                     +      frac_z[k_eta]  * pension(iz_lo[k_eta] + 1)
```

i.e. linear-in-z interpolation of the pension table at each of the
n_eta z_next quadrature evaluation points. Confirmed at
[solver.py:2493-2496, 2606-2609](../../lifecycle/solver.py#L2493).

### 7.4 Retirement-age slicing

In the retirement case [solver.py:3067](../../lifecycle/solver.py#L3067),
`pension_next = pension_table_jnp[t + 1, :]` is a `(n_z,)` slice. The
retirement kernel then reads `pension_next_z = pension_next_by_z[z_idx]`
([solver.py:1495](../../lifecycle/solver.py#L1495)) — *exact* lookup,
no interpolation. This matches the design: retired agents have z frozen
(no eta innovation), so z_{t+1} = z_t and pension(z_{t+1}) = pension(z_t).
The bracketed-z interpolation only applies during the work→retirement
boundary because z_{t+1} is still being shocked by eta in that one age.

This is correct.

### 7.5 The `pension_dummy_z_jnp` zeros placeholder

[solver.py:3025](../../lifecycle/solver.py#L3025), used by the working
kernel at non-boundary ages. It is a `(n_z,)` jnp.zeros of float64, and
inside the working kernel the `pension_next_by_z` argument is **never
read** when `use_pension_next=False` is baked into the trace. So the
zeros literally do nothing. Same trick as the income_table placeholder
in the boundary case.

The only visible cost is one extra dispatch input per age (a `(n_z,)`
zeros device array, ~88 bytes at canonical n_z=11). Acceptable.

### 7.6 Pension table is broadcast across age — semantically constant

The table at `pension_table_jnp[t + 1, :]` is **identical for every
retired t** because `_precompute_pension` does
`np.broadcast_to(base_pension, (n_age, n_z)).copy()`
([precompute.py:635](../../lifecycle/precompute.py#L635)). Every
retirement-age slice fetches the same `(n_z,)` vector. The host-side
copy ensures the table is C-contiguous and not a broadcast view — safe
to ship to JAX device.

---

## §8 Config-knob propagation

### 8.1 `DiscretizationConfig` fields involved

[model.py:88-122](../../lifecycle/model.py#L88):

```python
n_z: int = 7                     # persistent income grid points
n_stds: float = 3.0              # z-grid covers ±n_stds * sigma_z
n_eps_nodes: int = 3             # Judd-mixture nodes for eps
n_eta_nodes: int = 3             # Judd-mixture nodes for eta
```

Canonical override at
[_canonical.py:108-124](../../configs/_canonical.py#L108):

```python
n_z=11
n_stds=3.0
n_eps_nodes=4
n_eta_nodes=3
```

### 8.2 Propagation to Precompute

| DiscretizationConfig field | Read at                    | Used to size       |
|----------------------------|----------------------------|--------------------|
| `n_z`                      | precompute.py:396          | `z_grid` (and downstream pc.n_z) |
| `n_stds`                   | precompute.py:397          | z_grid spread      |
| `n_eps_nodes`              | precompute.py:428          | `eps_nodes`, `eps_weights` |
| `n_eta_nodes`              | precompute.py:429          | `eta_nodes`, `eta_weights` |

The Judd-mixture quadrature constructors
([discretization.py:426-475](../../lifecycle/discretization.py#L426))
receive the count and pass it through to `_judd_mixture_quadrature`
which builds an n-point rule with polynomial exactness `2n - 1`. The
default `n_eps_nodes=3, n_eta_nodes=3` give degree-5 exactness against
the mixture density.

### 8.3 Propagation to kernels

In the JAX kernel the actual sizes appear as *trace-time constants*:
the kernel closes over `pcj.eta_nodes.shape[0]`, `pcj.eps_weights.shape[0]`,
`pcj.z_grid.shape[0]` etc. via the closures around `per_cell` /
`per_kv` etc. Changing `n_z` or `n_eta_nodes` triggers a fresh JIT
trace because all the `(n_eta,)`, `(n_eps,)`, `(n_z,)` shapes are part
of the kernel's input signature.

### 8.4 The n_stds caveat

`disc_config.n_stds` controls *only* the z-grid spread (and Pi_z
construction inside the variant). It does **not** affect:

- The eta or eps Judd quadrature (those are built from the *mixture*
  parameters, not from a tail bound).
- The state-grid spread (that uses `state_n_stds`, distinct).

The variant solver caveat at
[solver_pi_z_variant.py:13-16](../../lifecycle/solver_pi_z_variant.py#L13)
recommends `n_stds=2.25` for variant runs at canonical rho=0.991. **This
is variant-only; production is unaffected** because production reads
`pcj.eta_nodes` (Judd quadrature, not Pi_z).

### 8.5 Production canonical (after-pivot) sanity

At canonical (n_z=11, n_eta=3, n_eps=4) the integration order is
`n_state_quad * n_ret_quad * n_eta * n_eps = 75 * 16 * 3 * 4 = 14,400`
quadrature points per (k_v, k_r, k_eta, i_e) summation, per cell, per
Newton iter. This is fully baked into the trace.

---

## §9 Bundle serialization gaps

### 9.1 Canonical save path

`save_policy_bundle`
([policy_io.py:76-229](../../lifecycle/policy_io.py#L76)) writes:

- `policy_arrays.npz` — C, S, B
- `wealth_grid.npy`
- `diagnostics.pkl`
- `metadata.json` — includes `run_config` if provided

The `run_config_snapshot` dict in `verify/test_baseline.py:151-163`
contains `base_config`, `discretization_config`, `solver_config`,
`predictability_ablation`, `bundle_name`, `wall_time_seconds`.

**`var_config` is NOT in `run_config_snapshot`.** Confirmed by reading
all `verify/*` and `configs/*` save sites — the standard pattern
omits `var_config` from the snapshot. Instead, the load path at
`verify/_diag_helpers.py:build_bundle_var_config`
([_diag_helpers.py:57-112](../../verify/_diag_helpers.py#L57))
**rebuilds** the var_config from
`metadata.run_config.predictability_ablation.system_code` by
dispatching to `build_real_full_var_config_hardcoded` /
`build_real_system1_var_config` / `build_real_system2_var_config`.

This is sound: the VAR is hardcoded (no state-dependent calibration),
and the system_code suffices to identify which builder to call. But
it does mean the bundle's metadata.json **does not record** the actual
VAR coefficients used — it records only the discriminator. If someone
edits `lifecycle/var.py` between save and load, the saved bundle will
silently re-load with whatever the *new* hardcoded values are. Not a
silent corruption hazard *today*, but worth flagging for future
calibration changes.

### 9.2 Income config recoverability

Confirmed inspection of `metadata.json` of an actual saved bundle at
`saved_runs/ablations/system_ii_grid5x5_*/metadata.json`:

```
run_config keys: ['base_config', 'bundle_name', 'discretization_config',
                  'predictability_ablation', 'solver_config', 'solver_kind',
                  'sweep', 'wall_time_seconds']

base_config keys: ['b0', 'b1', 'b2', 'b3', 'b_bar', 'beta', 'gamma',
                   'mu_eps1', 'mu_eps2', 'mu_eta1', 'mu_eta2', 'pe', 'pz',
                   'retire_age', 'rho', 'sigma_eps1', 'sigma_eps2',
                   'sigma_eta1', 'sigma_eta2', 'start_age', 'terminal_age']
```

This is a **legacy** bundle (saved before Fix A). It carries
`mu_eta2` and `mu_eps2` keys in `base_config`. The current
`build_model` at
[precompute.py:867-873](../../lifecycle/precompute.py#L867) will
*silently ignore* those keys and re-derive — so loading this bundle
into the current code works correctly.

For a **post-Fix-A** bundle (e.g. one produced by `verify/test_baseline.py`
today), the `base_config` dict would *not* contain `mu_eta2` /
`mu_eps2` keys (they were removed from `BASE_CONFIG` in
`configs/_canonical.py`). On rehydration, `build_model` would re-derive
them from `(pz, mu_eta1)` and `(pe, mu_eps1)`. So:

- The discretization config (`n_z`, `n_eta_nodes`, `n_eps_nodes`,
  `n_stds`) is fully recoverable from `metadata.run_config.discretization_config`.
- The five free income parameters (`pz, mu_eta1, sigma_eta1, sigma_eta2,
  pe, mu_eps1, sigma_eps1, sigma_eps2`) are fully recoverable from
  `metadata.run_config.base_config`.
- The two derived means (`mu_eta2, mu_eps2`) are recovered by
  re-derivation in `build_model`.

So the income config is **fully recoverable** from a bundle. No gap.

### 9.3 Diagnostics pkl includes Precompute?

[solver.py:_build_diagnostics] (not shown above) embeds disc_config in
the diagnostics dict (via `_dictify_namedtuples`) but **does not**
embed the materialized Precompute itself (no `working_income_next` or
`pension_after_tax` arrays in the bundle). For consumers like
`verify/ee_residuals.py` which need pc, the path is to rebuild
`Precompute` from `(model, disc_config)` via `build_precompute`. This
is legitimate: the precompute is fully deterministic given (model,
disc_config), so saving it would be redundant.

### 9.4 What is *not* recoverable

- The actual JIT trace cache key (depends on JAX version and code
  state).
- The exact fp32 vs fp64 round-off pattern under `gather_precision="f32"`
  is not bit-reproducible across JAX versions.
- Variant-specific PCJaxPi is not saved, but the variant solver is
  out-of-tree of the canonical path so this is fine.

---

## §10 Findings table

Severity scale: CRITICAL = currently produces wrong results;
HIGH = structural fragility / silent footgun; MEDIUM = robustness
issue; LOW = cosmetic / documentation.

| #  | Sev    | Location                                   | What's wrong / risky                                                                                                       | Fix sketch |
|----|--------|--------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|------------|
| 1  | LOW    | precompute.py:603-620 + variant only       | `pc.working_income` is built but **not consumed** by the production solver. Consumed only by the variant's `_build_discrete_income_next_table` helper and by host-side diagnostics. Wasted memory ~ n_age*n_z*n_eps*8 bytes (~9 KB at canonical) and a bit of construction time. | Either drop from production Precompute (keep variant-local construction), or document the asymmetry inline at [precompute.py:115](../../lifecycle/precompute.py#L115). |
| 2  | LOW    | model.py:421 vs simulation.py:188          | `disposable_income_working` (host) and `jax_disposable_income` (JAX) hard-code the 7 TCJA bracket boundaries and rates **independently**. Future bracket edits must be made in two places. | Centralize as a Python tuple of `(threshold, base, rate)` and have both functions consume it. |
| 3  | LOW    | solver.py:3078                             | `income_table = jnp.zeros((n_z, pc.n_eta, pc.n_eps))` allocated per boundary call but ignored inside the kernel. Cosmetic, ~negligible cost (1 alloc per solve). | Acceptable; don't change unless adding a separate `pension_only_kernel` trace is justified. |
| 4  | LOW    | solver.py:3025 + solver.py:3086            | `pension_dummy_z_jnp = jnp.zeros(n_z)` placeholder for working_kernel calls (when `use_pension_next=False` it's not read inside the trace). Same as #3. | Same comment. |
| 5  | LOW    | precompute.py:438-441 vs precompute.py:443-447 | `log_det_profile` and `avg_det` are built from the same polynomial twice. The second time over a sub-range only. Minor duplication. | Compute once, slice the working sub-range to take its mean. |
| 6  | LOW    | precompute.py:482-486                      | `mortality.calibrate_earnings_dependent_mortality` is called with `model.mu_eta2` raw. Now correct (post Fix A) because `model.mu_eta2` is authoritative. Was a fragile site pre-Fix-A; the API still *takes* `mu_eta2` as an argument rather than computing internally. | If desired: change `compute_sigma_z` and the calibration entry-point to take only `(rho, pz, mu_eta1, sigma_eta1, sigma_eta2)` and re-derive `mu_eta2` inside (Fix E from INCOME_PIPELINE_REVIEW). Defense-in-depth. |
| 7  | LOW    | policy_io.py: `var_config` not saved       | The bundle's `metadata.json` does not record the actual VAR coefficient matrices. Reload depends on `lifecycle/var.py` not being edited between save and load. The discriminator `predictability_ablation.system_code` is recorded and dispatches to a hardcoded builder. | Optionally serialize the actual `Phi`, `Omega`, `z_bar` arrays into the bundle metadata for reproducibility under future calibration edits. |
| 8  | LOW    | solver_pi_z_variant.py:13-16               | Variant inherits the canonical `disc_config.n_stds=3.0` from the production config, but at canonical rho=0.991 this produces a reducible Pi_z. Caveat documented in the docstring; not enforced programmatically. | Add a runtime warning when the variant detects `n_stds >= 3.0` AND `rho >= 0.99`. |
| 9  | LOW    | precompute.py:435 (`dz` placeholder)       | At `n_z=1`, `dz=0.0` is a placeholder. The lifecycle working kernel's `bracket_uniform` would divide by zero if reached. n_z=1 is "inf-horizon-only" by convention (z is inert). | Already documented; an explicit `assert n_z >= 2 or inf_horizon_only` in build_precompute would make the contract enforceable. |
| 10 | NONE   | discretization.py:441, 467 + 67 others     | Single source of truth for `mu_eta2`, `mu_eps2` is `lifecycle/precompute.py:build_model`. All production consumers read `model.mu_eta2` / `model.mu_eps2` directly. | No change needed. |
| 11 | NONE   | solver.py:3027 + 3087                      | `working_income_next_jnp[t + 1]` slicing is consistent with the `(t, iz, k_eta, i_e)` semantics — table[t] is "next-period income when transitioning out of t", and the orchestrator gathers t+1 to be "next-period income when transitioning out of (t+1)" = used in solving age t's FOC. Indexing convention correct. | No change needed. |
| 12 | NONE   | precompute.py:684-687                      | Boundary zero-fill of `working_income_next` rows where `t+1 >= retire_age_idx` matches the orchestrator's branching. Probe-confirmed at canonical retire_age_idx=45. | No change needed. |

**Independent observation** (not on the math reviewer's likely list):
the `working_income` table being unused by the production solver is a
genuine pipeline-level oddity. The variant solver's
`_build_discrete_income_next_table` re-shifts it
([solver_pi_z_variant.py:64-80](../../lifecycle/solver_pi_z_variant.py#L64))
to align with `t+1` semantics. If the variant is ever retired,
`working_income` becomes pure dead code in the production Precompute —
worth a cleanup pass at that point.

**Independent observation 2:** the orchestrator's
`pension_next = pension_table_jnp[t + 1, :]` slicing during retirement
loops over every retired age but always returns the same `(n_z,)`
vector (because the table is broadcast-constant across ages). The
device-side dispatch could in principle hoist this slice out of the
loop. As written, JAX's compilation cache does keep the slice op
cached, so the cost is one shader-cache hit per age, ~negligible. Worth
flagging only because it indicates an opportunity for a tiny
hoisting micro-opt.

---

## §11 Verdict

**PASS-WITH-CAVEATS.**

The labour-income subsystem's *implementation pipeline* is correct and
clean post-2026-05-09 Fix A:

- Single source of truth for `mu_eta2` / `mu_eps2` is in place at
  `lifecycle/precompute.py:879-880` with an internal mean-zero
  assertion. Every production consumer reads `model.mu_eta2` /
  `model.mu_eps2` directly.

- `working_income_next` and `pension_after_tax` shapes/dtypes are
  consistent end-to-end. The probe confirms the boundary zero-fill at
  `t + 1 >= retire_age_idx` matches the orchestrator's branch logic.
  No silent broadcast hazards.

- The `t + 1` indexing convention is consistent across all three
  branches (working, boundary, retirement). The bracketed-z pension
  interpolation in the boundary kernel is correctly limited to that
  one age.

- `Pi_z` is correctly off the production hot path. The variant solver
  reconstructs it locally with a documented caveat about `n_stds`
  reduction at canonical rho=0.991.

- `gather_precision` interaction is bounded: income tables stay f64
  throughout; the f32 cast is restricted to the multilinear-state
  interp closure and casts back to f64 before any FOC arithmetic.

- Bundle reload recovers the income config in full: the five free
  parameters are stored in `base_config`, the two derived means are
  re-derived on load.

The "WITH-CAVEATS" qualifier reflects the LOW-severity items
in §10 — primarily that `working_income` is unused by the production
solver and is candidate dead code if the variant is ever retired; the
host-vs-JAX dual implementation of `disposable_income_working`; the
absence of the actual VAR coefficients in saved bundle metadata; and a
defensive-coding opportunity to harden `mortality.compute_sigma_z`
against future drift in `mu_eta2`. None of these affect the canonical
solve. None require immediate action.

If a future change re-adds `mu_eta2` / `mu_eps2` to BASE_CONFIG (e.g.
to support a sweep over off-zero-mean alternatives), the
`build_model` assertion at
[precompute.py:888-893](../../lifecycle/precompute.py#L888) would *not*
catch it because the assertion checks the *derived* values, not the
config. A defensive guard (Fix B from the prior INCOME_PIPELINE
review) that compares any user-provided `mu_eta2` to the derived value
and raises on mismatch would close that one residual fragility surface.
Recommend adding it the next time the income surface is touched.
