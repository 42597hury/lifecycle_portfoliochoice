"""Experiment config: CCV-style no long-yield residual (theta = 0).

Build the CCV nominal-yield theta datasets first:

    python data/build_var_dataset_ccv_nominal_yield_theta.py

This config selects
data/ccv_nominal_yield_scaling/var_dataset_theta_0p00.csv for the financial
VAR.  The real short rate, real 10-year yield, spread, and bond return are
constructed from the same CCV-style nominal-yield decomposition.
"""

from configs._canonical import (
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
    PREDICTABILITY_SYSTEM,
    prepare_canonical_predictability_system,
    resolve_var_csv_path,
)

TERM_PREMIUM_DATASET = "ccv_nominal"
TERM_PREMIUM_THETA = 0.0
VAR_CSV_PATH = resolve_var_csv_path(
    TERM_PREMIUM_THETA,
    term_premium_dataset=TERM_PREMIUM_DATASET,
)
BUNDLE_SUFFIX = "_ccv_nominal_theta0"

base_config = BASE_CONFIG
system_setup = prepare_canonical_predictability_system(
    term_premium_theta=TERM_PREMIUM_THETA,
    term_premium_dataset=TERM_PREMIUM_DATASET,
    system=PREDICTABILITY_SYSTEM,
    disc_config_template=CANONICAL_DISC,
)
var_config = system_setup["var_config"]
disc_config_template = system_setup["disc_config"]
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL
