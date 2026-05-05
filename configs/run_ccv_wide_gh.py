"""Retirement-only run: CCV + wider state grid + pure GH (no Lobatto).

Tests the §3.7 / §3.8 / §11 hypothesis from LOBATTO_CONFIG_TRACKER.md:
under CCV (no bankruptcy boundary), Lobatto is unnecessary and pure GH is
optimal. Combined with 99% joint state coverage (n_stds=2.93) which §3.4
says only mattered under simple_clamp.

Diff from `ccv_retire`:
  - state_n_stds:    (2.0, 2.25, 2.25)  -> (2.93, 2.93, 2.93)   [99% joint]
  - ret_lobatto_Z:   (None, 7.0, 7.0)   -> None                 [drop Lobatto]
  - state_lobatto_Z: (None, 7.0, 7.0)   -> None                 [drop Lobatto]

Wealth dynamics: ccv_log (matches ccv_retire). Retirement only (ages 67-99).

Cloud cost: 75 ret * 75 state = 5,625 nodes/cell — identical to ccv_retire
(K=5 Lobatto and K=5 GH have the same node count; the redistribution moves
nodes from explicit Lobatto tails to extra GH body nodes).

Polynomial-exactness gain: GH K=5 has 2K-1=9 vs Lobatto K=5's 2K-3=7.
Strict body-integration upgrade under CCV (no boundary to detect).

Pre-flight arbitrage check: T-Q1 max gap = 0 across all 343 state corners
on both log-excess and arithmetic-excess clouds. Confirmed before proposing
this config.

See `docs/notes/LOBATTO_CONFIG_TRACKER.md` §11 for the CCV regime context
and the prediction table.
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_ccv_wide_gh"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.05,
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
