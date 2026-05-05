"""Canonical economics + discretization + CCV log-wealth dynamics.

Solves retirement years only (age 67 .. 99), unconstrained.

Inherits everything from `configs/_canonical.py`, with three deliberate
overrides:

  1. solver_config.wealth_dynamics_spec = "ccv_log"
       Switches the solver/simulator to the Campbell-Viceira (CCV) log
       portfolio return: R_p = exp(r_p^CVC) instead of simple+clamp.
       See HANDOFF/IMPLEMENTATION_HANDOFF_CVC_RETURNS.md.

  2. disc_config_template.wealth_min = 0.01 (was 0.13 in canonical)
       Extends the wealth-grid lower edge to handle tail-shock interpolation
       in the alive branch of the FOC kernel. Per the §6.3 check; see
       lower-edge analysis on the canonical+ccv investigation.
       wealth_max stays at 750 (verified equal under both specs at
       constrained alpha; unconstrained alpha can produce sR_p up to ~5x
       the grid top at +7sigma but the upper-edge linear extrapolation is
       homothetic-stable for CRRA at high wealth — see PR2 review note).

  3. SOLVE_CONTROL.youngest_age_to_solve = 67
       Stops backward induction at the retirement boundary: solves ages
       67..99 inclusive. Working ages remain NaN in the bundle (use the
       solved-age mask in diagnostics to filter).

Bundle name template (build_bundle_name, scripts/run_solve.py):
    system_iv_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_ccv_retire"

# Unconstrained is already the canonical default (BASE_CONFIG["constrained"] = False),
# but pin it explicitly here so the choice is documented in the config.
base_config = dict(BASE_CONFIG)
base_config["constrained"] = False

# Wealth grid: extend lower tail to 0.01 per the user's directive on the
# wealth-grid edge-check investigation. wealth_max stays at canonical 750
# (the upper tail is bit-identical between simple+clamp and CCV at the
# corner alphas and within 0.02% at interior — verified empirically).
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.01,
)

# Switch the wealth dynamics from simple+clamp to CCV log returns. Every
# other knob stays at canonical: tol, alpha caps, line-search, delta_bequest.
solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
)

# Solve retirement years only. youngest_age_to_solve=67 stops the backward
# sweep at the retirement boundary, leaving working-age policies as NaN.
# Checkpointing stays on (canonical default) so an EC2 interruption resumes
# from the last completed age rather than restarting from terminal.
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(
    youngest_age_to_solve=67,
)
