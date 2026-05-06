"""Retirement-only run: CCV + 9x9x9 state grid + state quad K=(3,4,4) GH.

Cost-constrained version of the trilinear-spacing hypothesis test.

Hypothesis (from LOBATTO_CONFIG_TRACKER §11 update): ccv_wide_gh's body-EE
degradation vs ccv_retire is driven by trilinear-cell width in physical
state space, not by the Lobatto-vs-GH choice. Going to 9x9x9 at the same
99% envelope tightens cell spacing back toward ccv_retire's level.

This config conflates two changes vs ccv_wide_gh:
  - state_grid_sizes: (7,7,7) -> (9,9,9)              [tighter trilinear cells]
  - n_state_quad_nodes: (3,5,5) -> (3,4,4)            [coarser body integration]

The conflate is deliberate: pure 9^3 + K=5 state was 2.1x ccv_wide_gh's
compute, too expensive. Cutting state quad K from 5 to 4 brings cost down
to 1.36x ccv_wide_gh while keeping the dominant change (the grid bump).

Predicted: the trilinear-cell-width effect dominates the quad-node-reduction
cost, so net result is body EE improvement vs ccv_wide_gh.

If results IMPROVE over ccv_wide_gh: hypothesis confirmed (grid spacing was
the binding body penalty). Adopt this config or 9^3+K=5 if compute permits.

If results MATCH ccv_wide_gh: the two effects cancelled; need follow-up to
disentangle (e.g., 7^3+K=4 to isolate quad-K effect alone).

If results REGRESS: quad-K effect dominated, K=4 is too coarse on state
axes 1, 2. Try 9^3 + K=5 (the original expensive proposal).

State-grid u-space spacing comparison:
  - 7^3 at n_stds=2.93: 2*2.93/(7-1) = 0.978 sigma per cell
  - 9^3 at n_stds=2.93: 2*2.93/(9-1) = 0.733 sigma per cell  [matches ccv_retire's 0.750]

State-quad polynomial exactness:
  - K=5 GH: degree 9 (ccv_wide_gh)
  - K=4 GH: degree 7 (this config)
  - K=3 GH: degree 5

For CCV-smooth integrand (no boundary kinks), degree 7 should be plenty.

Cloud cost: 75 ret * 48 state = 3,600 nodes/cell. Total state cells = 729.
Roughly 1.36x ccv_wide_gh compute (~2.5-3 hours retirement-only).

Pre-flight arbitrage prediction: clean (factorized moment recovery per §3.7;
GH K=4 still spans both signs of state innovations).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_ccv_wide9_gh_k4"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(9, 9, 9),
    state_n_stds=(2.93, 2.93, 2.93),
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=None,
    n_state_quad_nodes=(3, 4, 4),
    state_lobatto_Z=None,
)
solver_config = CANONICAL_SOLVER._replace(
    wealth_dynamics_spec="ccv_log",
)
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)
