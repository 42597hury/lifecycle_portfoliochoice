# HANDOFF: Labour Income Discretization Validation

**To:** next labour-income validation agent
**From:** validation session (2026-04-16)
**Status:** Theory/feature side closed out. Discretization side untouched.

---

## Critical user constraint — READ THIS FIRST

**Do NOT write out validation checklists or TODO items without explicit user
approval.** Past attempts turned into "spray and pray" — too many items
dumped at once with no closure.

Work **incrementally**: propose one check at a time, wait for the user to
direct the next step. This is logged as a feedback memory at
`C:\Users\carlh\.claude\projects\c--Users-carlh-Projekt-thesisscripts\memory\feedback_labour_validation.md`.

The user does not want big bulk lists. Talk through one thing, verify, check
it off, then ask what's next.

---

## What has been validated already (don't redo)

All theory / feature-side items are closed. See
[LABOUR.md Section 5](LABOUR.md#L336) for the full checklist. Highlights:

- Income process matches Guvenen et al. (2021) / Catherine (2025) spec
- Tax brackets (2019 TCJA, 7 brackets) correct to within 1.3%
- Payroll tax rate 10.6% justified (OASI only, no DI — Catherine adjustment)
- Pension exempt from payroll tax (income tax only) — correct
- PIA bend points (0.21, 1.25) and rates (90/32/15) correct
- AIME approximation documented as our deviation from Catherine (rho ≈ 1
  justifies dropping her cumulative-earnings state variable)
- Model units = SSA AWI ≈ $54,100 anchor established
- L̄ = 1 stationary assumption noted
- z initial condition handled correctly (solver agnostic, sim has 3 options)
- Pension formula numerically verified at all 11 z-grid points

## One open item (deferred, not for this handoff)

**Labour income ↔ return covariance** — deferred to after returns and state
variables have been validated. See [LBAR_HANDOFF.md](LBAR_HANDOFF.md) and
[INCOME_RETURN_COV_HANDOFF.md](INCOME_RETURN_COV_HANDOFF.md). Do not touch.

---

## Your scope: discretization of labour income

Section 4 of [LABOUR.md](LABOUR.md#L232) documents the current
discretization. It has not been validated. Potential checks (NOT a TODO
list — just a map of the territory, propose one at a time):

**z-grid**
- Is `linspace(-3*std_z, +3*std_z, n_z)` appropriate? (vs Rouwenhorst/Tauchen)
- Is `var_eta` computed correctly from the mixture?
- Does n_z = 11 (or whatever default) give enough resolution?
- Uniform spacing vs alternatives

**Gauss-Hermite quadrature for eta (persistent innovation)**
- Two sets of nodes, one per mixture component, concatenated
- Weights: `GH_weights * pz` and `GH_weights * (1-pz)` — do they sum to 1?
- Zero-mean enforced by construction — verify
- Variance reproduced — verify
- n_eta_nodes = 5 per component (10 total) — is that enough?

**Gauss-Hermite quadrature for eps (transitory shock)**
- Same structure, n_eps_nodes = 3 per component (6 total)
- `mu_eps2_eff` zero-mean override — is it applied everywhere?

**Precomputed income table** ([precompute.py](precompute.py))
- `_precompute_working_income()` at precompute.py:366
- Shape (n_age, n_z, n_eps), evaluated at grid points only
- Spot-check values against manual computation

**Precomputed pension table**
- `_precompute_pension()` at precompute.py:387
- Age-invariant (tiled) — is that right?
- Already numerically verified at all 11 z-grid points ✓

**Solver interpolation**
- Working age: linear interpolation in z between bracketing grid points
- `z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]` → floor, fraction, blend
- Retirement: direct index lookup, no interpolation
- Is linear interp accurate enough at the top/bottom tails?

**Solver vs simulation gap**
- Simulation computes income directly from continuous z (no table, no interp)
- Solver uses table + linear interp
- Known gap flagged in `issues.md` (~17%?) — needs investigation

---

## Code files to read

Core:
- [model.py:268-348](model.py#L268-L348) — `disposable_income_working` and
  `compute_pension_after_tax`. Already validated, just reference.
- [discretization.py](discretization.py) — `discretize_income_ar1_mixture`,
  `get_eta_quadrature_mixture`, `get_eps_quadrature_corrected`. **Main focus
  for this handoff.**
- [precompute.py](precompute.py) — `_precompute_working_income` (line 366),
  `_precompute_pension` (line 387), `log_det_profile` (line 197),
  `avg_det` (line 211).
- [solver.py](solver.py) — `compute_foc_jac_working` (how income enters the
  FOC with GH integration over eta and linear interp in z) and
  `compute_foc_jac_retirement`.
- [simulation.py](simulation.py) — `_scalar_disposable_income`,
  `_scalar_pension_after_tax`, and the main simulation loop. Direct continuous
  computation — contrasts with solver.

Config:
- [model.py:91-116](model.py#L91-L116) — `DiscretizationConfig`. Defaults:
  `n_z=7`, `n_stds=3.0`, `n_eps_nodes=3`, `n_eta_nodes=3`. Note defaults in
  config file differ from what's documented in LABOUR.md (says n_z=11).
  **Worth double-checking which value is actually used in current runs.**

## Context documents to read

**Essential:**
- [LABOUR.md](LABOUR.md) — full validation doc. Sections 0-3 for context,
  Section 4 for discretization specifics, Section 5 for checklist.
- [LBAR_HANDOFF.md](LBAR_HANDOFF.md) — L̄ context (short).

**Useful background:**
- `PENSION_FIX_HANDOFF.md` — earlier pension work. Contains Catherine eq.
  citations.
- `DESIGN.md` — overall architecture.
- `issues.md` — if it exists, check for the solver/simulation income gap.

## User preferences recap

- Terse responses, no preambles, no trailing summaries
- Incremental validation: one check, get approval, next check
- When the user pushes back on reasoning, they're usually right — think again
  rather than defending
- Use markdown link syntax `[file.py:42](file.py#L42)` for code references
- Today's date: 2026-04-16. Thesis deadline: 2026-05-18.

## First action when resumed

Do **not** dive in. Tell the user you've read the handoff, summarize what's
left to validate at a high level, and **ask which piece to start with**.
Let them direct the order.
