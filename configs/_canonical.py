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

Real-yields pivot (2026-05-08):
  The 4-axis nominal model (System I/II/III/IV on rtb-as-state) has been
  retired. The new canonical is the 3-axis real-yields Full System with
  state vector (cape, spr, y_1). System 1 / System 2 ablations select a
  subset of axes via configs.predictability_ablation.

See `docs/CONFIG.md` for the rationale behind every value.
"""

from lifecycle.model import DiscretizationConfig, SolverConfig, SolveControl

PREDICTABILITY_SYSTEM = "full"


# ── Economics ───────────────────────────────────────────────────────────────
# NOTE: bequest spec is the shifted (luxury) form b̄·(W/A + δ)^{1-γ}/(1-γ);
# the shift parameter δ now lives entirely on CANONICAL_SOLVER.delta_bequest.
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
# State vector is (cape, spr, y_1) for the Full System; System 1 / System 2
# slice this template via project_predictability_disc_config (cape and/or
# spr dropped, the same ordering preserved).
#
#   Axis 0 = cape  (slow equity-yield predictor)
#   Axis 1 = spr   (term spread)
#   Axis 2 = y_1   (real bill yield; bond-return refinement target, K-bump)
#
# state_grid_sizes / state_n_stds: PLACEHOLDERS per the real-yields pivot
#   handoff. Validate before production by re-running the System I / Full
#   sensitivity sweeps on the new VAR (analog of the old grid+nstd sweeps,
#   but on (cape, spr, y_1) instead of (dp, spr, rtb, y_1)). The current
#   defaults follow the handoff suggestion of (5,5,5) and (2.0, 2.25, 2.25).
#
# n_state_quad_nodes=(3, 3, 5): K-bump preserved on the y_1 axis (last),
#   the long-standing bond-return refinement convention (M[xb, y_1] is the
#   dominant entry of M = Sigma_rs Sigma_ss^-1).
#
# state_lobatto_Z / ret_lobatto_Z = None: prescribed-tails Lobatto removed
#   on every axis; standard Gauss-Hermite throughout.
#
# n_ret_nodes_1d=(4, 4): bumped from (3, 3) per the per-axis quadrature
#   sensitivity finding (verify/inf_horizon_sweep_axis_bumps).
#
# n_z=11, n_eps_nodes=4, n_eta_nodes=3: validated under the System I
#   nz / eta-eps sensitivity sweeps (carry over from the legacy canonical).
#
# wealth_min: 0.01 AWI (~$540 2019). Lower bound of the wealth grid;
#   below the smallest real interior W_implied, the constrained-corner
#   clamp in _lift_to_wealth_grid (Path B, commit e6b5448) sets c=W,
#   alpha_s=alpha_b=0 — the standard borrowing-constrained corner per
#   Carroll (2006) / Druedahl & Jorgensen (2017). Previous values
#   (0.13, 0.05) were a legacy artefact: the constrained branch was
#   never actually implemented, jnp.interp blended the artificial anchor
#   with the smallest interior solution, so wealth_min had to stay above
#   the kink to avoid the meaningless blend region. With the clamp now
#   in place, wealth_min can drop below typical first-real-W_implied,
#   exposing the corner to downstream consumers (simulator, EE residuals,
#   arbitrage diagnostics). f32 spacing safety: 180 log1p-spaced points
#   0.01..750 give min_rel_diff32 = 3.77e-2 (~40,000x above the
#   8*eps_f32 ~= 9.5e-7 floor); enforced at runtime by
#   validate_wealth_grid() in precompute.py.
#   Initial-wealth defaults across notebooks/tests/scripts are 0.1..10.0
#   AWI — all >=10x above the floor.
CANONICAL_DISC = DiscretizationConfig(
    n_wealth=180,
    wealth_min=0.01,
    wealth_max=750.0,
    n_savings=180,
    state_grid_sizes=(5, 5, 5),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=11,
    n_stds=3.0,
    n_eps_nodes=4,
    n_eta_nodes=3,
    n_ret_nodes_1d=(4, 4),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 3, 5),
    state_lobatto_Z=None,
)


# ── Numerical solver ────────────────────────────────────────────────────────
# tol=1e-6: per docs/scans/NEWTON_FAILURE_STRUCTURE_2026-05-08.md, tol=1e-7
#   was unreachable at high-savings cells where FOC residual scale falls
#   below fp64 precision relative to tol*scale. Loosening to 1e-6 declares
#   those structurally-doomed cells as converged without changing actual
#   policy precision.
# init_alpha_s=0.85, init_alpha_b=0.44: warm starts at the long-run mean
#   portfolio under Full-System carryover; revisit after the new VAR's
#   stationary alphas are characterized.
# delta_bequest=0.0: drop the luxury-bequest shifter. The pivot baseline
#   is the un-shifted bequest; the shifter can be dialled back per-run if
#   the cliff at zero wealth is binding in production.
# gather_precision="f32": fp32 c_corners gather + interpolation; cast back
#   to fp64 BEFORE FOC / Newton arithmetic. Captures memory-bandwidth
#   savings without touching policy precision (verified to ~1e-4 relative
#   agreement vs f64 on prior canonicals).
CANONICAL_SOLVER = SolverConfig(
    tol=1e-6,
    max_iter=8000,
    init_alpha_s=0.85,
    init_alpha_b=0.44,
    step_damp_unconstrained=0.3,
    use_line_search=True,
    delta_bequest=0.0,
    gather_precision="f32",
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
