"""Dev config: full lifecycle on a 5x5x5x5 state grid.

Mirrors `configs/sweep_main/01_base.py` but with a smaller state grid for
faster local iteration. Inherits canonical economics + solver from
`configs/_canonical.py`; overrides only `state_grid_sizes`. State vector is
4D post rtb-as-state ((cy, spr, rtb, y_1)).

Run with:
    python scripts/run_solve.py configs/system_iv_5x5x5.py --no-upload
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_v2"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(state_grid_sizes=(5, 5, 5, 5))
solver_config = CANONICAL_SOLVER
