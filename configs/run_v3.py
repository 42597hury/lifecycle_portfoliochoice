"""Production run v3.

Grid:        7×7×7 cholesky, state_n_stds=(2.0, 2.25, 2.25)
Quadrature:  state_quad=(3,3,5), n_eps=5, n_eta=5
Wealth/sav:  180×180, wealth_min=0.05, wealth_max=750
Solver:      canonical (tol=1e-7, alpha ±6)
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_v3"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC
solver_config = CANONICAL_SOLVER
