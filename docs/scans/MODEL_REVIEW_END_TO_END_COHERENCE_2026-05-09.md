# Model Review — End-to-End Coherence (2026-05-09)

Scope: cross-component coherence at the seams of the post-pivot lifecycle
model. Solver / simulator / precompute / VAR / canonical / docs / tests /
diagnostics. Read-only. Excludes Bellman/FOC, solver internals, and
discretization theory (other agents).

Branch: `jax-rewrite`. All file references absolute under
`c:\Users\carlh\Projekt\thesisscripts_JAX\`.

---

## 1. Solver ⟷ Simulator parity

**What I found.**
- `lifecycle/solver.py:747` `_ccv_log_return_and_grad` and
  `lifecycle/simulation.py:356` (`log_R_port = log_R_bill + …`) implement
  the *same* CCV w8566 eq. 10 closed form, term by term. The dedicated
  parity test `verify/ccv_solver_sim_parity.py:48` already pins this to
  1e-12 over 1000 random realisations.
- Bill rate: solver reads `state_grid_i[y_1_idx]` at `solver.py:840`;
  simulator reads `s_t[y_1_idx]` at `simulation.py:334`. Same convention.
- Volatility scalars `sigma2_xr/sigma2_xb/sigma_xrxb` flow from
  `pc.Sigma_rr` (precompute.py:340) into both solver `pcj` and the
  simulator (`simulation.py:730`). Single source.
- Wealth update: solver implicit, simulator `estate = savings * R_port`
  with `R_port = exp(log_R_port)`; consistent because solver's FOC is
  derived from the same CCV r_p.

**Verdict.** GREEN. The seam is well-tested.

---

## 2. Precompute ⟷ Solver

**What I found.**
- `state_grid` shape is uniformly `(N_state, n_state)` across precompute
  (`precompute.py:299`), `_pc_to_jnp` (`solver.py:1594`), inf-horizon
  (`inf_horizon_solver.py:154,344`), simulator (`simulation.py:666`),
  and diagnostics (`diagnostics.py:692`). No transpose drift.
- `pcj.y_1_idx`, `pcj.xr_pos`, `pcj.xb_pos` are derived in `_pc_to_jnp`
  by name lookup against `model.ret_names` (`solver.py:1586-1587`) —
  same pattern as `precompute.py:326-327`, `simulation.py:690-691`,
  and `inf_horizon_solver.py:150-151`. Robust to any future
  re-permutation.
- `M_v_nodes = v_nodes @ model.M.T` set in precompute (line 321);
  consumed unchanged in solver/diagnostics. `A_r = Phi_21 - M @ Phi_11`
  defined in precompute and propagated as-is.
- `_pc_to_jnp` enforces `n_state in {1,2,3}` and refuses
  `y_1_index_in_state is None` (lines 1569-1577). Canonical's 3-state
  setup falls in range. Hard-stops on misuse rather than silent breakage.

**Verdict.** GREEN.

---

## 3. Precompute ⟷ Simulator

**What I found.**
- `simulation._build_return_factor` Choleskies `model.Sigma_r_cond`
  (`simulation.py:104`) — the *conditional* covariance. Used only for
  the Monte-Carlo residual draw `ret_resid = ret_factor @ z`
  (line 345). State innovation Cholesky is built fresh from
  `model.Sigma_ss` (`simulation.py:667`) — matches the source
  precompute uses for state quadrature (`discretization.py:746-747`).
- Income shock simulation uses
  `(pz, mu_eta1, sigma_eta1, sigma_eta2, mu_eta2_eff, pe, mu_eps1,
  sigma_eps1, sigma_eps2, mu_eps2_eff)` directly off `model`
  (`simulation.py:582-583`, kernel signature 251-252) with
  `mu_eta2_eff = -(pz/(1-pz))*mu_eta1` recomputed — exactly matches
  the model docstring at `model.py:49-54`. Same mixture the precompute
  Judd quadrature consumes.
- `_validate_state_quadrature` (`precompute.py:526`) verifies the
  state quadrature reproduces conditional return moments before the
  `pc` is handed off, so the simulator and solver inherit a
  cross-checked precompute.

**Verdict.** GREEN.

---

## 4. VAR ⟷ Precompute ⟷ Solver

**What I found.**
- New VAR builders order: `[cape, spr, y_1, xr, xb]` columns,
  `state_indices=(0,1,2)`, `return_indices=(3,4)` (`var.py:441-446`).
  `y_1_index_in_state=2`, `spr_index_in_state=1`, `rtb_index_in_state=None`.
- In the partition (`partition_var`, `var.py:64-77`),
  `state_idx`/`ret_idx` are used with `np.ix_`, so `state_names` and
  `ret_names` are returned in the *order of `state_indices`/`return_indices`*.
  For the Full system this gives `state_names=("cape","spr","y_1")`,
  `ret_names=("xr","xb")`. Confirmed against `predictability_ablation.py:58`
  which asserts the spec.
- All consumers (`solver.py:1586`, `precompute.py:326`,
  `simulation.py:690`, `inf_horizon_solver.py:150`, `arbitrage.py:296`)
  do `ret_names.index("xr")` / `index("xb")` — order-agnostic.
- Restricted vs unrestricted: canonical builders default to
  `estimation="restricted"` (`var.py:413,455,484`), and the hardcoded
  fallback's Phi has `xr/xb` columns explicitly zeroed (`var.py:541-545`).
  Canonical run path uses the restricted estimator; this matches the
  pivot doc.
- `build_real_full_var_config_hardcoded()`'s frozen z_bar and Phi appear
  consistent with the new state/return ordering — no orphaned `dp`/`rtb`
  rows.

**Verdict.** GREEN.

---

## 5. Canonical ⟷ Implementation

**What I found.**
- `CANONICAL_DISC` fields (n_wealth=180, wealth_min=0.05, wealth_max=750,
  state_grid_sizes=(5,5,5), state_grid_mode="cholesky",
  state_n_stds=(2.0,2.25,2.25), n_z=11, n_eps_nodes=4, n_eta_nodes=3,
  n_ret_nodes_1d=(4,4), n_state_quad_nodes=(3,3,5)) are all wired
  through `precompute.build_precompute` and `discretization.*`.
  `state_lobatto_Z=None` / `ret_lobatto_Z=None` reach the quadrature
  builders (`precompute.py:307,314`).
- `CANONICAL_SOLVER` fields all present in `SolverConfig`. Two are
  **dead code** (orphans):
  - `step_damp_unconstrained` defined in `model.py:159` and set on the
    canonical (line 129) but never read anywhere in `lifecycle/`.
    Doc-comment at `model.py:159` already calls it "legacy; line search
    supersedes this".
  - `max_iter_unconstrained` defined `model.py:143`, kept "for
    backwards compatibility"; not consumed in `lifecycle/` either
    (`max_iter` is the live field).
- `delta_bequest=0.0` in canonical (`_canonical.py:131`) → solver
  resolves `delta = sc.delta_bequest if sc.delta_bequest >= 0.0 else
  DELTA_BEQUEST` at `solver.py:2518`, so `0.0` means "use 0.0" (NOT
  the sentinel `DELTA_BEQUEST=0.005`). This is correct; comment at
  `_canonical.py:118-120` ("drop the luxury-bequest shifter") is true.
- `wealth_min=0.05` comment claims "the EGM constrained branch is
  solved, NOT skipped" — solver / EGM does run on the lowest grid
  point; the wealth grid simply doesn't include `0.0`. Code path exists.
- "see docs/STATE_SPACE.md §wealth_min" (`_canonical.py:88`) — the
  reference target exists at `STATE_SPACE.md:218-220` and matches the
  0.05 value. The *rest* of STATE_SPACE.md is stale (next section).

**Verdict.** AMBER. Two orphan SolverConfig fields, doc references
otherwise OK.

---

## 6. Tests ⟷ Implementation

**What I found.**
- `tests/test_real_yields_pivot_smoke.py` — full end-to-end smoke for
  System 1 / 2 / Full on the new 3-axis model, includes
  `simulate_lifecycle` execution. Skips gracefully if CSV missing.
- `tests/test_solver_real_yields_foc.py`,
  `tests/test_simulator_real_yields_returns.py`,
  `tests/test_precompute_real_pivot.py`,
  `tests/test_sigma_rr_sourcing.py`,
  `tests/test_var_estimator_equivalence.py` — all post-pivot.
- `verify/ccv_solver_sim_parity.py` is a parity check between the seam.
- `_RETIRED_SYSTEM_CODES` in `predictability_ablation.py:94-104` raises
  on every legacy Roman-numeral / `dp`/`rtb`/etc. system code; mirrored
  by `_diag_helpers._LEGACY_SYSTEM_CODES` (`_diag_helpers.py:30-36`).
  Loud-fail, not silent-fall-through.
- No 4-axis tests remained; old tests appear to have been pruned.

**Verdict.** GREEN.

---

## 7. Diagnostics ⟷ Implementation

**What I found.**
- `verify/_diag_helpers.build_bundle_var_config` rejects legacy bundles
  with a clear pivot-explanation message (lines 39-55) and dispatches
  System 1/2/Full to the new builders.
- `verify/ee_residuals.py`, `verify/ee_simpath.py`, `verify/arbitrage.py`,
  `verify/invalid_cells.py` all consume `pcj.y_1_idx`,
  `pcj.xr_pos`, `pcj.xb_pos`, `pcj.sigma2_xr`, `pcj.sigma2_xb`,
  `pcj.sigma_xrxb` from the same `_pc_to_jnp` packing the solver uses.
  Same return formula (`ee_simpath.py:249-272`, `arbitrage.py:312-322`).
- `arbitrage.py:288-294` raises if `model.y_1_index_in_state is None`,
  matching the solver's hard-stop convention.
- `invalid_cells.py` is bundle-shape-agnostic (just scans NaNs/extreme
  alphas); needs no axis-count assumption.

**Verdict.** GREEN.

---

## RED FLAG — single most important

**Stale documentation: `docs/STATE_SPACE.md` and `docs/CONFIG.md` still
describe the pre-pivot 4-axis nominal model.**

`docs/STATE_SPACE.md`:
- §3.1 (line 112-117): `s_t = (cy, spr, y_1)` with state_names
  `('cy', 'spr', 'y_1')` — pre-pivot nominal. The new model's state
  is `(cape, spr, y_1)`.
- §3.6 (line 211): `mu_r : (N_state, N_state, 3)` — three returns.
  The post-pivot `n_ret = 2` (only `xr, xb`; the bill is real-risk-free
  off `state[y_1_idx]`).
- §6 (line 422-423): `state_names`/`ret_names` listed as
  `('y_1','spr','cy')`, `('rtb','xr','xb')` — both wrong.

`docs/CONFIG.md`:
- Line 7: `PREDICTABILITY_SYSTEM = "IV"` — the value is now
  `"full"` (`_canonical.py:25`); systems "I/II/III/IV" are explicitly
  retired (`predictability_ablation.py:94-104`).
- Line 71: claims a `constrained` BASE_CONFIG key — not present in
  current `BASE_CONFIG` (`_canonical.py:42-52`).

`docs/UTILITY.md` line 26: "γ = 3" — canonical is γ = 5.0
(`_canonical.py:31`).

These docs are referenced by `_canonical.py:20` ("See docs/CONFIG.md
for the rationale behind every value"), and by code comments that
still point readers to STATE_SPACE.md. Anyone onboarding will read the
wrong model.

**Severity.** Not a runtime bug. Code itself is internally coherent.
But documentation of record contradicts the implementation, which is
exactly the "two sources of truth" red flag pattern the brief asked
to surface. Fix is a doc-only sweep; no code changes required.

---

## TL;DR

| Area | Verdict |
|---|---|
| 1. Solver ⟷ Simulator parity | GREEN — pinned by `verify/ccv_solver_sim_parity.py` |
| 2. Precompute ⟷ Solver | GREEN |
| 3. Precompute ⟷ Simulator | GREEN |
| 4. VAR ⟷ Precompute ⟷ Solver | GREEN — restricted estimator, by-name index lookup |
| 5. Canonical ⟷ Implementation | AMBER — `step_damp_unconstrained` and `max_iter_unconstrained` are orphan SolverConfig fields |
| 6. Tests ⟷ Implementation | GREEN — full smoke for Systems 1/2/Full; `_RETIRED_SYSTEM_CODES` loud-fail |
| 7. Diagnostics ⟷ Implementation | GREEN |

**Single most important RED FLAG:** `docs/STATE_SPACE.md`,
`docs/CONFIG.md`, `docs/UTILITY.md` still describe the pre-pivot 4-axis
nominal model (state `(cy,spr,y_1)`, ret `(rtb,xr,xb)`,
`PREDICTABILITY_SYSTEM="IV"`, γ=3). Code is coherent; docs are not.
Doc-only fix.
