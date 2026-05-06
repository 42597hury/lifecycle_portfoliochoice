"""Retirement-only run: CCV + 9x9x9 state grid + 99% coverage + pure GH.

Tests the trilinear-interpolation-spacing hypothesis from
LOBATTO_CONFIG_TRACKER.md (Q1 of the ccv_retire vs ccv_wide_gh comparison).

The hypothesis: ccv_wide_gh's degraded body EE (mean -1.95 vs ccv_retire's
-2.22) is driven by wider trilinear-cell spacing in physical state space,
not by the Lobatto-vs-GH choice. Going to a 9x9x9 grid at the same wide
envelope keeps coverage at 99% but tightens cell spacing back to roughly
ccv_retire's level.

Diff from ccv_wide_gh:
  - state_grid_sizes: (7, 7, 7) -> (9, 9, 9)              [+~2.1x cells]

State-grid u-space spacing comparison (n_stds=2.93):
  - 7^3 grid: 2*2.93/(7-1) = 0.978sigma per cell
  - 9^3 grid: 2*2.93/(9-1) = 0.733sigma per cell  [matches ccv_retire's 0.750]

Joint coverage: 99% (per-axis 99.66%, joint = 0.9966^3).

Cloud cost: 75 ret * 75 state = 5,625 nodes/cell (unchanged from ccv_wide_gh).
Total compute: ~2.1x ccv_wide_gh (729 vs 343 state cells; cloud per cell same).

Predicted outcome (per LOBATTO_CONFIG_TRACKER §11 update):
  - mean log10|EE|: -2.20 to -2.35  (body penalty 0.96x ccv_retire)
  - max log10|EE|:  -0.5 to -0.7    (similar to ccv_wide_gh tail)
  - worst_foc_resid: ~0.02          (CCV stays clean)
  - Newton failures: ~3.5M          (same scale as ccv_wide_gh)

If actual mean lands near -2.4 or better, the trilinear-spacing hypothesis
is confirmed and this becomes the new CCV canonical. If mean stays near
ccv_wide_gh's -1.95, something other than spacing is driving the body
degradation at wide envelopes.

Pre-flight arbitrage check pending — predicted clean (interpolates between
ccv_wide_gh's empirical pass and ccv_retire's empirical pass).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_ccv_wide9_gh"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(9, 9, 9),
    state_n_stds=(2.93, 2.93, 2.93),
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 5, 5),
    state_lobatto_Z=None,
)
solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
)
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)
