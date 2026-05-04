"""Production run v4 — canonical with Lobatto on tail axes.

Inherits canonical: 7x7x7 principal, n_z=11, n_eps=4, n_eta=4,
ret_quad=(3,5,5) + ret_lobatto_Z=(None,7,7), state_quad=(3,5,5) +
state_lobatto_Z=(None,7,7).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_v4_lobatto"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC
solver_config = CANONICAL_SOLVER
