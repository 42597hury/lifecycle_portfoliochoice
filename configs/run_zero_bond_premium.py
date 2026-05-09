"""Experiment config: full system with zero simple bond excess return.

The VAR keeps the same state vector, bond-return state dependence, residual
volatility, and covariance as the active AR(1)-matched 10-year baseline, but
the unconditional mean of ``xb`` is shifted so ``E[exp(xb)] - 1 = 0``.
"""

from configs._canonical import (
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)
from lifecycle.var import build_real_full_var_config_zero_bond_simple_excess_mean

PREDICTABILITY_SYSTEM = "full"
BUNDLE_SUFFIX = "_zero_bond_premium"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL

var_config = build_real_full_var_config_zero_bond_simple_excess_mean(
    target_simple_mean=0.0
)
