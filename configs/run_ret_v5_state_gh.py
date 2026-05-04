"""Retirement-only A/B test: STATE GAUSS-HERMITE branch (matched cloud cost).

Sister of `run_ret_v5_state_lobatto.py`. Tests whether explicit Lobatto tail
nodes on state-quadrature axes are doing real bankruptcy-detection work, or
whether plain Gauss-Hermite at the same K does the same job.

The argument for state-GH (this arm): the bankruptcy clamp is a kink in the
return integrand, not the state-innovation integrand. State innovations enter
only via `c_{t+1}(s_{t+1})` interpolation, which is smooth. The state grid
covers ±2.25 sigma; GH K=5 already maxes at ±2.86 sigma (already past the
grid edge, so interp is clamped anyway). Putting a Lobatto node at +-7 sigma
buys nothing on the state axes -- the interpolated `c_{t+1}` value is the
same as at the grid corner -- while paying a polynomial-exactness cost on
the bulk integrand (Lobatto K=5 is 7th-degree exact on N(0,1); GH K=5 is
9th-degree exact). See HANDOFF_EVAL_LOBATTO_PROPAGATION.md follow-up
discussion (2026-05-04) for the analytical argument.

This file replaces the deprecated `run_ret_v5_state_gh_k7.py` which used
K=7 GH and so confounded "no state Lobatto" with "more state nodes".
Matched K (K=5 on the historically-Lobatto axes) keeps the comparison clean:
both arms have 3*5*5 = 75 state nodes per cell; only Z differs.

All other knobs are identical to `run_ret_v5_state_lobatto.py`.
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_ret_v5_state_gh"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=(None, 5.0, 5.0),
    n_state_quad_nodes=(3, 5, 5),
    state_lobatto_Z=None,
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)
