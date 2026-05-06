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
    "constrained": False,
}


# ── Discretization ──────────────────────────────────────────────────────────
# state_n_stds: u-space half-width per axis in cholesky mode.
#   Per-axis coverage = 2*Phi(n_d) - 1; joint = product.
#   Canonical (2.93, 2.93, 2.93) -> per-axis 99.66%, joint ~99% coverage of
#   the stationary state distribution. Adopted 2026-05-05 with the move to
#   ccv_log wealth dynamics (which removes the bankruptcy-boundary kink that
#   made narrow grids necessary under simple_clamp).
#   Earlier (0.6, 1.75, 2.0) gave joint ~40% and produced unusable simulator
#   moments — do not regress to it.
# state_grid_sizes: 9x9x9. With n_stds=2.93 this gives u-space cell spacing
#   2*2.93/(9-1) = 0.733 sigma, matching the body resolution of earlier 7x7x7
#   narrow-envelope configs (which had spacing 0.75 sigma). 9x9x9 at 99%
#   coverage strictly Pareto-improves on 7x7x7 at 91% coverage under CCV.
#   See ccv_wide9_gh_k4 outcome in LOBATTO_CONFIG_TRACKER.md for the pivot.
# Lobatto OFF on all axes: under wealth_dynamics_spec="ccv_log" there is no
#   bankruptcy boundary, so the integrand is smooth jointly in (state, return)
#   and pure Gauss-Hermite is optimal (polynomial-exactness 2K-1 vs Lobatto's
#   2K-3). See LOBATTO_CONFIG_TRACKER.md §3.7 / §11 for the empirical case.
# n_state_quad_nodes=(3, 4, 4): K=4 GH on the bond-loaded axes 1 and 2
#   (polynomial-exactness 7) was empirically nearly identical to K=5 (degree
#   9) under CCV, at 36% lower compute cost. K=3 on axis 0 (cy) is sufficient
#   because M[xb,0]=-0.005 is negligible.
# wealth_min: 0.05. precompute.py reads this directly. Below min(pension)~0.0015
#   for very-low-z agents (~0.5% of stationary mass) — those agents land in
#   wealth-grid extrapolation territory; same constraint under simple_clamp
#   and ccv_log (see CCV_RETURNS.md).
CANONICAL_DISC = DiscretizationConfig(
    n_wealth=180,
    wealth_min=0.05,
    wealth_max=750.0,
    n_savings=180,
    state_grid_sizes=(9, 9, 9),
    state_grid_mode="cholesky",
    state_n_stds=(2.93, 2.93, 2.93),
    n_z=11,
    n_stds=3.0,
    n_eps_nodes=4,
    n_eta_nodes=4,
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 4, 4),
    state_lobatto_Z=None,
)


# ── Numerical solver ────────────────────────────────────────────────────────
# Tuned. Do not change knobs other than alpha_min/alpha_max without rerunning
# the small-bundle smoke. See docs/CONFIG.md §2 for the "why" on each
# override of the model.py defaults.
# alpha_min/alpha_max: numerical leverage cap (unconstrained branch only).
#   Box projection (alpha_s, alpha_b) in [alpha_min, alpha_max]^2 inside the
#   unconstrained Newton. Canonical ±6 is a real cap (prior production hit
#   max simulated |alpha| ~9.25 at gamma=5, 7x7x7 wide-support). Cap-bound
#   cells surface as EC_NEWTON_FAIL in diagnostics['total_newton_failures'];
#   under CCV most "Newton failures" are these cap-bound cells where the
#   warm-restart fallback finds the correct KKT-bound policy.
# wealth_dynamics_spec="ccv_log": use Campbell-Viceira (NBER w8566 eq.10)
#   continuous-rebalancing log-portfolio approximation with Jensen + Itō
#   corrections. R_p = exp(r_p^CCV) > 0 strictly, so no bankruptcy boundary
#   and no integrand discontinuity. Solver and simulator both use this same
#   spec internally — the docstring at lifecycle/model.py:188-194 enforces
#   the constraint (otherwise sR_p disagrees and EE diagnostics break).
#   Adopted 2026-05-05 as canonical wealth dynamics; see
#   LOBATTO_CONFIG_TRACKER.md §11 for the empirical case (worst_foc_resid
#   30x better than simple_clamp; max log10|EE| 6x better at v4_lobatto-
#   equivalent disc settings).
CANONICAL_SOLVER = SolverConfig(
    tol=1e-7,
    max_iter=20,
    max_iter_unconstrained=8000,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    step_damp_constrained=0.2,
    step_damp_unconstrained=0.3,
    use_line_search=True,
    alpha_min=-6.0,
    alpha_max=6.0,
    delta_bequest=0.001,
    wealth_dynamics_spec="ccv_log",
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
