"""Canonical production config — single source of truth.

Every other config in `configs/` (the sweep cells in `sweep_main/` and the
dev/smoke configs) imports the three exports below and overrides only the
fields it intentionally varies. Dialing the canonical here propagates to
every cell on the next regen / next solve.

Exports:
  - PREDICTABILITY_SYSTEM
  - BASE_CONFIG          (dict — economics)
  - CANONICAL_DISC       (DiscretizationConfig — discretization)
  - CANONICAL_SOLVER     (SolverConfig — numerical solver)

See `docs/CONFIG.md` for the rationale behind every value.
"""

from lifecycle.model import DiscretizationConfig, SolverConfig, SolveControl

PREDICTABILITY_SYSTEM = "IV"


# ── Economics ───────────────────────────────────────────────────────────────
# NOTE: bequest spec is the shifted (luxury) form b̄·(W/A + δ)^{1-γ}/(1-γ);
# the shift parameter DELTA_BEQUEST is defined in lifecycle/model.py.
gamma, beta, b_bar = 5.0, 0.96, 10
start_age, retire_age, terminal_age = 22, 67, 99
b0, b1, b2, b3 = -6.142, 0.3040, -0.051, 0.002586
rho, pz = 0.991, 0.176
mu_eta1, sigma_eta1 = -0.524, 0.113
mu_eta2 = -(pz / (1.0 - pz)) * mu_eta1
sigma_eta2 = 0.046
mu_eps1, sigma_eps1 = 0.134, 0.762
mu_eps2, sigma_eps2 = 0.0, 0.055
pe = 0.044

BASE_CONFIG = {
    "beta": beta, "gamma": gamma, "b_bar": b_bar,
    "start_age": start_age, "retire_age": retire_age, "terminal_age": terminal_age,
    "b0": b0, "b1": b1, "b2": b2, "b3": b3,
    "rho": rho, "pz": pz,
    "mu_eta1": mu_eta1, "sigma_eta1": sigma_eta1,
    "mu_eta2": mu_eta2, "sigma_eta2": sigma_eta2,
    "pe": pe,
    "mu_eps1": mu_eps1, "sigma_eps1": sigma_eps1,
    "mu_eps2": mu_eps2, "sigma_eps2": sigma_eps2,
}


# ── Discretization ──────────────────────────────────────────────────────────
# state_n_stds: u-space half-width per axis in cholesky mode.
#   Per-axis coverage = 2*Phi(n_d) - 1; joint = product.
#   Current value (2.0, 2.25, 2.25) -> per-axis (95.5%, 97.6%, 97.6%),
#   joint ~91%. Earlier value (0.6, 1.75, 2.0) gave joint ~40% and
#   produced unusable simulator moments — do not regress to it.
#   Production-grade 99% per-axis would need (~2.93, ~2.93, ~2.93).
# wealth_min: explicit so it can never silently inherit a stale model.py
#   default. Raised to 0.05 on 2026-05-03 to skip the EGM constrained
#   region (see docs/STATE_SPACE.md §wealth_min and
#   docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md). precompute.py reads this directly.
CANONICAL_DISC = DiscretizationConfig(
    n_wealth=180,
    wealth_min=0.13,
    wealth_max=750.0,
    n_savings=180,
    state_grid_sizes=(7, 7, 7),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=11,
    n_stds=3.0,
    n_eps_nodes=4,
    n_eta_nodes=4,
    # Lobatto on stock/bond return-residual axes (ret[1], ret[2]) and on the
    # spr/y_1 state-innovation axes (state[1], state[2]). These are the four
    # axes where the empirical bond-tail discrete-free-lunch lives. See
    # diagnostics_reports/diagnostics_quadrature_arbitrage_nz11_v3.md and
    # scripts/diagnostics/_diag_per_axis_tail.py for the per-axis evidence.
    # Z=7 (sigma units of the standardised orthogonalised axis) puts the
    # explicit tail node at +/-7 sigma => +/-17% R_bond surprise on ret[2].
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=(None, 7.0, 7.0),
    n_state_quad_nodes=(3, 5, 5),
    state_lobatto_Z=(None, 7.0, 7.0),
)


# ── Numerical solver ────────────────────────────────────────────────────────
# Tuned. Do not change knobs other than alpha_min/alpha_max without rerunning
# the small-bundle smoke. See docs/CONFIG.md §2 for the "why" on each
# override of the model.py defaults.
# alpha_min/alpha_max: numerical leverage cap (unconstrained branch only).
#   Box projection (alpha_s, alpha_b) in [alpha_min, alpha_max]^2 inside the
#   unconstrained Newton. Canonical ±6 is a real cap (prior production hit
#   max simulated |alpha| ~9.25 at gamma=5, 7x7x7 wide-support). Cap-bound
#   cells surface as EC_NEWTON_FAIL in diagnostics['total_newton_failures'].
CANONICAL_SOLVER = SolverConfig(
    tol=1e-7,
    max_iter=8000,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    step_damp_unconstrained=0.3,
    use_line_search=True,
    delta_bequest=0.001,
)


# ── Solve control ───────────────────────────────────────────────────────────
# Canonical defaults for partial solves and crash recovery. Configs that need
# to set youngest_age_to_solve should _replace() this rather than instantiate
# SolveControl directly, so checkpointing stays on by default.
CANONICAL_SOLVE_CONTROL = SolveControl(
    checkpoint_every_n_ages=1,
    save_on_interrupt=True,
    return_partial_on_interrupt=True,
)
