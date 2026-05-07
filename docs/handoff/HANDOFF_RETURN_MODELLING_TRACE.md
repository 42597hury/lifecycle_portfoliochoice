# Handoff: Return-Modelling Pipeline Trace (Theory Review Prep)

**Branch:** `jax-rewrite`
**Effort:** 2-3 days. Read-only audit + numerical reproduction + documentation. No code changes.
**Output:**
- `docs/scans/RETURN_MODELLING_TRACE_2026-05-07.md` — main pipeline trace document.
- `docs/scans/RETURN_MODELLING_PARAMS_2026-05-07.md` — numerical appendix: side-by-side VAR parameter reproduction (estimated vs hardcoded), residual covariance Σ_rr, state-innovation Σ_state, **cross-block Σ_(state,return)**, drift means.

**Purpose:** the user + a theory agent will use these documents to vet that the codebase's return-modelling implementation matches the CCV paper's mathematical specification end-to-end. This handoff produces the *evidence base* for that review. The agent here does NOT make correctness judgments — that's the user's and theory agent's job.

---

## What "return modelling" means in this codebase

Five layers, top to bottom. The trace must walk all five:

1. **Raw data sources.** Where do nominal/real return series, yield series, and predictor series originate? (Probably CSVs of historical series — Treasury yields, equity returns, inflation, dividend-price, etc.)
2. **Variable construction.** How are raw series transformed into the model's state/return variables: `log_R_bill`, `log_x_s` (excess log return on stocks), `log_x_b` (excess log return on bonds), `cy`, `spr`, `rtb`, `y_1`. Exact algebraic definitions of each.
3. **VAR estimation.** The reduced-form VAR system. Specification (state-space form, restrictions, sample period), estimation method (OLS / MLE / GLS), residual structure. Coefficient values: `Phi_0_state`, `Phi_11`, `A_r`, `const_r`, `Sigma_rr`, plus drift means.
4. **CCV continuous-rebalancing approximation.** The variance-corrected log portfolio return formula:
   ```
   r_p = log_R_bill + α_s·log_x_s + α_b·log_x_b
       + 0.5·(α_s·σ²_xr + α_b·σ²_xb)
       - 0.5·(α_s²·σ²_xr + 2·α_s·α_b·σ_xrxb + α_b²·σ²_xb)
   ```
   Where the formula appears in code, what each term represents, what assumption it implements (continuous rebalancing under log-normal returns).
5. **Discretization + solver consumption.** Gauss-Hermite + Lobatto-tail quadrature for return shocks. State grid via Cholesky factorization of the joint state covariance. How `_build_step_log_returns` and `_build_step_state_brackets` construct per-cell `(log_R_bill, log_x_s, log_x_b)` tensors. Critically: the **rtb-as-state migration** means `log_R_bill` is now read from `s_next[k_v, rtb_idx]` (i.e., from the next-period state vector at a quadrature node), not drawn from a return shock. This needs to be made explicit.

---

## Trace document structure (what the agent must produce)

The output must be **self-contained** — readable by a theory reviewer who doesn't see the code. Every claim about "what the code computes" needs a `file:line` citation. Every formula must be written out with explicit symbols (LaTeX or clear Unicode). Variable-name mapping (paper symbol ↔ code symbol) is mandatory.

Structure (these become the section headings):

### §1. Data sources and preprocessing
- What CSVs / series feed the pipeline? Search `scripts/`, `data/`, `configs/`, top-level `.py`, anywhere a `pd.read_csv` or similar lives.
- Frequency (monthly? quarterly? annual?). Sample period.
- Any preprocessing: log differences, demeaning, deflating, gross-vs-net return convention.
- For each series: source citation if available in code comments (FRED ticker, CRSP file name, etc.).

### §2. Variable construction
- Exact algebraic definition of each model variable in terms of raw series. Examples to chase down precisely:
  - `log_R_bill = ?` (log gross nominal bill return? log gross real?)
  - `log_x_s = log(R_s) - log(R_bill)` — confirm sign convention and gross/net
  - `log_x_b = log(R_b) - log(R_bill)` — same
  - `cy = ?` (consumption-wealth ratio? Lettau-Ludvigson cay residual? log dividend-price?)
  - `spr = ?` (term spread = long yield − short yield? specific maturities?)
  - `rtb = ?` (real T-bill rate? inflation-surprise residual? confirm definition by reading construction code, not by guessing from name)
  - `y_1 = ?` (1-year nominal yield? 1-year real yield?)
- Map each model variable to its code home (likely `lifecycle/var.py` or a data-loading script).
- Flag any *non-obvious* transformation (e.g. centering, scaling) that affects interpretation.

### §3. VAR specification
- Write the system equations in math:
  ```
  s_{t+1} = Phi_0_state + Phi_11 · s_t + ε_state,t+1
  r_{t+1} = const_r + A_r · s_t + ε_r,t+1
  Cov(ε) = Σ_joint   (block-structured: state, return, and cross-block)
  ```
  (or whatever the actual specification is — confirm from code).
- Dimensions: state vector size, return block size, coefficient matrix shapes.
- Innovation distribution (Gaussian? heavy-tail?).
- **Cross-block covariance Σ_(state,return)** — this is a **deliberate departure from textbook CCV** in this codebase. Document explicitly:
  - Whether the joint covariance has a non-zero off-diagonal block linking state innovations and return innovations.
  - Where in the code this cross-block term is constructed and consumed.
  - In CCV (textbook): state and return innovations are typically jointly normal with a non-zero cross-correlation — the same draw of macro shocks moves both. Confirm or contradict from the code.
  - If the code zeroes the cross-block: that's a deliberate simplification to flag.
  - If the code retains it: document how the simulator and solver consume it (state-conditional return distribution? Cholesky on the full joint Σ vs block-by-block?).
- Restrictions imposed (zeros in coefficient matrices, block-diagonal Σ, eigenvalue constraints, etc.).
- Where the VAR is constructed: probably `build_nominal_system1_var_config_hardcoded()` in `lifecycle/var.py`. Read it carefully.

### §4. VAR estimation procedure — REPRODUCE FROM RAW DATA

This section requires **end-to-end numerical reproduction**, not just documentation. The user needs to verify that the hardcoded values in `build_nominal_system1_var_config_hardcoded()` actually come from the documented data + estimation procedure.

Steps:
1. **Locate the raw-data files** (CSVs, source-cited series). Document their content and provenance.
2. **Locate the estimation script(s)** — could be in `scripts/var/`, a notebook (`.ipynb`), or a top-level python file. Search exhaustively if not obvious. Look for any code that produces VAR coefficients.
3. **Run the estimation** end-to-end. If the estimation script exists, execute it and capture its output. If it doesn't run as-is (because, e.g., the JAX-rewrite branch hasn't kept it current), DO NOT silently skip — report the breakage and what minimal fix would be needed to run it. **Don't fix anything; just describe.**
4. **Compare estimated values to the hardcoded values** in `build_nominal_system1_var_config_hardcoded()`. Print element-wise comparison: estimated vs hardcoded, absolute diff, relative diff.
5. **Document estimation method** (OLS by equation? MLE? Stambaugh / SUR? Bayesian shrinkage?).
6. **Document sample period** (start date, end date, frequency).
7. **Document restrictions** imposed during estimation (zeros in coefficient matrices, non-explosive eigenvalue constraint, positive-definiteness on Σ, etc.).

If estimated values DO NOT match the hardcoded values: do not editorialize; just report the discrepancy and let the theory reviewer judge.

If the estimation script doesn't exist at all (i.e., the hardcoded numbers have no traceable provenance in this branch): flag explicitly. That's a finding in itself.

### §5. VAR parameter values (numerical)
This goes into `RETURN_MODELLING_PARAMS_2026-05-07.md` as a separate appendix file. **Print TWO sets of numbers side-by-side**: the values reproduced from the estimation pipeline (§4 step 4) and the hardcoded values from `build_nominal_system1_var_config_hardcoded()`. Show element-wise diff.

Required tables:
- `Phi_0_state` — drift constants (vector, one per state element)
- `Phi_11` — state transition matrix (one row per state element)
- `A_r` — return loading matrix (one row per return element, columns = state elements)
- `const_r` — return intercepts (vector, one per return element)
- `Sigma_rr` — return-residual covariance (symmetric, full)
- `Sigma_state` — state-innovation covariance
- **`Sigma_state_return` — cross-block covariance** (state ↔ return innovations). Print regardless of whether it's zero — its value is a documented design choice.
- Joint covariance `Σ_joint` (the full block-structured matrix combining the three blocks above) — print eigenvalues to confirm positive-semidefinite.
- `mu_r` (return drift means, if separately stored)
- Eigenvalues of `Phi_11` (verify all are < 1 in modulus → stationary)

Format: 6 sig figs minimum, presented as labeled tables. Don't paraphrase — copy the numbers directly from the source. For the side-by-side reproduction:

| Element | Estimated (from raw data) | Hardcoded (in var.py) | Abs diff | Rel diff |
|---|---|---|---|---|
| Phi_11[0,0] | 0.991234 | 0.991234 | 0.0 | 0.0 |
| ... | ... | ... | ... | ... |

If the estimation pipeline can't be run (per §4 step 3), print only the hardcoded column and document the gap.

### §6. Continuous rebalancing — the CCV log-return formula
- Quote the exact code at `lifecycle/solver.py:694-710` (`_ccv_log_return_and_grad`).
- Write the formula in math notation.
- Identify each term:
  - `log_R_bill` — riskless rate (from state via rtb-as-state, see §9)
  - `α_s · log_x_s + α_b · log_x_b` — the linear excess return
  - `+ 0.5·(α_s·σ²_xr + α_b·σ²_xb)` — Jensen-correction additive term
  - `- 0.5·(α_s²·σ²_xr + 2·α_s·α_b·σ_xrxb + α_b²·σ²_xb)` — variance penalty (concavity of log of weighted gross return)
- Source: what equation in CCV (Campbell-Chacko-Viceira) does this match? If a citation comment exists, quote it. If not, flag the gap.
- The companion in the simulator: `lifecycle/simulation.py` (search for the same formula). Confirm parity (the recent CCV-fix commit verified 1e-12 parity to the solver — note the commit hash in your trace).
- Where do `σ²_xr`, `σ²_xb`, `σ_xrxb` come from? They're the (xr, xb) sub-block of `Sigma_rr`. Trace the path: VAR estimation → Sigma_rr matrix → which indices correspond to (xr, xb)?

### §7. Discretization of the return distribution
- Gauss-Hermite quadrature on `(xr, xb)` residuals. Where built: `lifecycle/quadrature_with_tails.py` or `lifecycle/precompute.py`. Number of nodes per axis (`n_ret_nodes_1d`).
- Lobatto tails: when active (controlled by `ret_lobatto_Z`), how it modifies the standard Gauss-Hermite rule. Why it's there (bond-tail bankruptcy correction).
- Cholesky factorization to make the bivariate quadrature account for `Sigma_rr` correlation: confirm exactly how `M_v_nodes` and `ret_nodes` are built.
- For state-block discretization: `state_lobatto_Z`, `state_grid_sizes`, `state_grid_mode='cholesky'`. The state grid is constructed in `lifecycle/discretization.py` — trace it.

### §8. Solver-side construction of per-cell return tensors
- Walk through `_build_step_log_returns` in `lifecycle/solver.py`. What it returns: `(log_R_bill, log_x_s, log_x_b)` of shape `(n_state_quad, n_ret_quad)` per cell.
- Walk through `_build_step_state_brackets`. What it returns: 16-corner indices for multilinear interpolation onto the policy grid at the next-period state.
- The arithmetic that combines them:
  - `log_x_s = const_r[xr_idx] + A_r[xr_idx] @ s_t + xr_residual_at_(k_v, k_r)`
  - `log_x_b = const_r[xb_idx] + A_r[xb_idx] @ s_t + xb_residual_at_(k_v, k_r)`
  - `s_next[k_v] = Phi_0_state + Phi_11 @ s_t + state_residual_at_k_v` (then bracket onto policy grid)
- **Confirm how state and return residuals are sampled jointly given the cross-block covariance.** Specifically:
  - Is `state_residual` sampled from `Sigma_state` and `(xr, xb)_residual` sampled from `Sigma_rr` independently (cross-block ignored)?
  - Or is the joint draw made from the full block-structured `Σ_joint`, with return residuals conditional on the state residual?
  - If conditional: trace the conditioning math. The conditional return distribution given a state shock is Gaussian with mean `Σ_(r,s) · Σ_state^(-1) · ε_state` and covariance `Σ_rr - Σ_(r,s) · Σ_state^(-1) · Σ_(s,r)`. Confirm whether this conditional structure is implemented or skipped.
  - This is a key consequence of the §3 cross-block design choice — it determines whether return draws "see" the state shock or are independent.
- Confirm the Cholesky factorization used: full block-structured Σ, or block-diagonal (state-only and return-only)?

### §9. The rtb-as-state semantics (CRITICAL)
This is recent and changes the return-modelling semantics. Make it visible:
- `log_R_bill` used to be sampled per return-quadrature node from a return-block component. After the rtb-as-state migration, `log_R_bill = s_next[k_v, rtb_idx]` — i.e., the riskless rate next period is **deterministic given the next-period state**, not stochastic given a return draw.
- Trace the migration commits: search `git log --oneline | grep -i "rtb\|rtb-as-state"`. Quote the commit message that documents the rationale.
- Implications:
  - `Sigma_rr` is now 2×2 (just `xr`, `xb`), not 3×3 (no longer includes a `rtb` component).
  - The return distribution is now conditional on `(s_t, s_{t+1})`, not just `s_t`.
  - `log_R_bill` enters the FOC via `s_next[rtb_idx]` rather than a quadrature draw.
- This is non-standard vs textbook CCV. The theory reviewer needs this called out explicitly. Don't editorialize; just describe what the code does and how it differs from §3's vanilla VAR specification.

### §10. Variable-name dictionary
Final table mapping every paper-side symbol to its code-side name and source location. Example rows:
| Paper / theory symbol | Code identifier | File:line | Definition |
|---|---|---|---|
| s_t (state vector) | `state_grid` row, also `s_t` in solver | precompute.py:NNN | (cy, spr, rtb, y_1) |
| Φ_0 (state drift) | `Phi_0_state` | var.py:NNN | additive constant in state transition |
| Φ_11 (state transition) | `Phi_11` | var.py:NNN | multiplicative transition matrix |
| A_r (return loading) | `A_r` | var.py:NNN | return predictability loading on state |
| Σ_r (return innovation cov) | `Sigma_rr` | var.py:NNN | reduced to 2×2 post rtb-as-state |
| ε_t^r (return innovation) | `ret_nodes` × `Cholesky(Sigma_rr)` | precompute.py:NNN | discretized via Gauss-Hermite |
| log_R_bill | `log_R_bill = s_next[k_v, rtb_idx]` | solver.py:NNN | rtb-as-state |
| log_x_s, log_x_b | `log_x_s`, `log_x_b` | solver.py:NNN | excess log returns over riskless |
| α_s, α_b | `alpha_s`, `alpha_b` | solver.py:NNN | portfolio shares |

Build the full table during the trace — every symbol that appears in §3-§9 should have a row.

---

## How to write each section

For every code reference:
- Quote a few key lines (5-15 max per quote — enough to convey what the code does).
- Annotate inline what each line accomplishes.
- Tie back to the math.

For every parameter value:
- Print the actual value to 6 sig figs.
- Note the units (annual vs monthly, log vs gross, demeaned vs raw).

For every mathematical claim:
- Write the full equation.
- State assumptions (Gaussian innovation, log-normal returns, continuous rebalancing).
- Cite source if known (CCV paper, equation N).

**Do NOT skip steps because they "look obvious."** The point is to make every step visible to a reviewer who hasn't read the code.

**Do NOT make value judgments.** Avoid "this is correct" or "this matches CCV." Just describe what the code computes; the user/theory agent will judge correctness.

**DO flag ambiguity.** If a code comment cites a paper equation that you can't verify against the actual code, write a "FLAG" note that the reviewer should resolve. If a variable definition seems inconsistent across files, flag it.

---

## Files to read (starting points)

The agent should load these as the primary reading set, then follow citations outward:

- `lifecycle/var.py` — VAR specification + parameter values (`build_nominal_system1_var_config_hardcoded`)
- `lifecycle/precompute.py` — quadrature build, return-block construction, state-grid construction
- `lifecycle/solver.py:694-710` — `_ccv_log_return_and_grad` (the CCV variance-corrected formula)
- `lifecycle/solver.py` — `_build_step_log_returns`, `_build_step_state_brackets` (search by name)
- `lifecycle/quadrature_with_tails.py` — Gauss-Hermite + Lobatto construction
- `lifecycle/discretization.py` — state grid (Cholesky mode)
- `lifecycle/simulation.py` — simulator-side CCV implementation (parity check vs solver)
- `configs/_canonical.py` — canonical state grid + quadrature parameters
- `lifecycle/model.py` — `BASE_CONFIG` economic parameters; `DiscretizationConfig`, `SolverConfig` field definitions
- `lifecycle/predictability_ablation.py` — for context on which subset of state we're using (System I vs IV)

Plus any companion data-ingestion or VAR-estimation script you find under `scripts/` or `data/`.

---

## Pause points

This handoff is **read-end-to-end without pausing.** Produce the full trace document in one delivery. Reasons:
- The theory reviewer wants the whole pipeline at once, not in chunks.
- Pausing mid-trace risks the agent forgetting context across rounds.
- No code changes means no risk of cascading bugs from intermediate decisions.

**Exception:** if you find an outright error in the code (math that doesn't compile, a comment contradicting the math, a variable used inconsistently), pause and flag immediately. Do NOT fix it. Do NOT continue past the broken section without the user's input.

---

## Implementation checklist

- [ ] §1 — data sources and preprocessing identified, with file paths to raw CSVs / source series.
- [ ] §2 — every model variable has an algebraic definition tied to raw series.
- [ ] §3 — VAR system equations, dimensions, restrictions written out, **including cross-block covariance** Σ_(state,return).
- [ ] §4 — estimation pipeline located, **executed end-to-end**, output captured. If it can't run, document the breakage (don't fix).
- [ ] §5 — numerical parameter appendix produced as separate file, with **side-by-side reproduction-vs-hardcoded comparison** (or hardcoded-only if §4 step 3 failed). Includes Σ_(state,return) cross-block.
- [ ] §6 — CCV log-return formula traced with full math + code citation + paper-equation reference.
- [ ] §7 — Gauss-Hermite + Lobatto + Cholesky discretization documented.
- [ ] §8 — solver-side per-cell tensor construction walked through with arithmetic.
- [ ] §9 — rtb-as-state semantics explicitly called out vs textbook CCV.
- [ ] §10 — full variable-name dictionary populated.
- [ ] Both files committed in a single commit:
  ```
  docs: return-modelling pipeline trace (theory review prep)

  Walks the data → variable construction → VAR estimation → CCV
  continuous-rebalancing approximation → discretization → solver
  consumption pipeline. Self-contained for theory reviewer; every
  claim about code behaviour cited to file:line. Numerical appendix
  prints VAR parameter estimates, residual covariance, drift means.

  No code changes. No correctness judgments — those are for the
  user + theory agent.
  ```

---

## Why a separate handoff (not part of any other)

- Distinct from arbitrage / EE / Newton-diag work: those are *runtime correctness* checks (does the solver produce the right numbers given the model). This is *model specification* check (does the model match the paper).
- Distinct from inf-horizon repair / multi-GPU audit: those are about code structure / dispatch. This is about math.
- Distinct from any code-improvement handoff: this is read-only documentation.

The output is a permanent artifact (lives in `docs/scans/`) that future theory revisions can re-use.

---

## Out of scope

- **Don't propose fixes** even if you spot a bug. Flag it; let the user/theory agent decide.
- **Don't refactor** for clarity or consistency.
- **Don't re-estimate the VAR** from scratch even if the data + scripts exist. Just document what the codebase does.
- **Don't compare to other papers** beyond CCV. Stick to what's claimed in code comments and the canonical CCV reference.
- **Don't write a tutorial.** This is a precise technical trace, not a pedagogy doc. Assume the reader knows continuous-time finance and VAR estimation.
