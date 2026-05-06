"""configs/_canonical_jax.py — JAX-pure production baseline (iterated).

Single source of truth for sweep cells running against the JAX backend.
Tracks the current best-known JAX-feasible config as we iterate. Each ratified
version bumps __JAX_CANONICAL_VERSION__ and is documented in the changelog
below; sweep cells override only what they intentionally vary, exactly the way
they already do against ``configs/_canonical.py``.

Why a separate file (and not editing _canonical.py):
  - ``_canonical.py``'s ``CANONICAL_SOLVER.max_iter=8000`` was tuned for Numba's
    ``while_loop`` with cheap data-dependent early termination.
  - Under JAX's ``lax.fori_loop`` (the production path; SolverConfig default
    ``use_fori_newton=True``), max_iter is unconditional wall cost — every cell
    runs all max_iter iterations regardless of convergence (mask-based exit).
    See ``HANDOFF_NEWTON_FORI_LOOP_MASK.md`` and §3 of
    ``HANDOFF_COMPLEXITY_WALL_TIME_ESTIMATOR.md``.
  - ``_canonical.py`` remains the source of truth for the Numba reference path
    (``verify_benchmark_bundle.py``) and existing sweep cells. New JAX sweep
    cells import from this file instead.

Exports (mirrors ``_canonical.py`` so sweep cells only swap the import path):
  PREDICTABILITY_SYSTEM, BASE_CONFIG,
  CANONICAL_DISC, CANONICAL_SOLVER, CANONICAL_SOLVE_CONTROL

Changelog (bump __JAX_CANONICAL_VERSION__ on each ratified config):
  v0.1  2026-05-06  UNRATIFIED — initial baseline targeting H200 full-lifecycle.
                    Knob choices come from §7 of COMPLEXITY_WALL_TIME_2026-05-06.
                    Estimated wall: ~16 h on H200, ~$54 at $3.29/h.
                    Aggressive: max_iter=20, tol=1e-7, n_eta=n_eps=3, n_savings=120.
                    Validate by smoke at 5⁴ before committing to full lifecycle.
  v0.2  2026-05-06  UNRATIFIED — bump max_iter 20→30 and loosen tol 1e-7→1e-6.
                    Rationale: max_iter=20 was below the cold-init Newton iter
                    count for terminal age 99 and the work→retire boundary at
                    age 66; 30 gives a safer 50% margin without breaking the
                    budget. Loosening tol to 1e-6 reduces the count of cells
                    that fori_loop marks "not converged" (cells run all
                    max_iter iters either way; tol only sets the mask), which
                    makes the failure-share metric meaningful at the lower
                    max_iter cap. Estimated wall: ~24.5 h on H200, ~$80 at
                    $3.29/h (1.5× v0.1 wall from max_iter scaling alone).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC as _NUMBA_CANONICAL_DISC,
    CANONICAL_SOLVER as _NUMBA_CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

__JAX_CANONICAL_VERSION__ = "v0.2"


# ── Discretization ──────────────────────────────────────────────────────────
# Diff from _canonical.CANONICAL_DISC:
#   n_z              : 11        → 7         (productivity grid)
#   n_state_quad_nodes: (3,5,3,5) → (3,3,3,3) (state quadrature: K_v 225 → 81)
#   n_ret_nodes_1d   : (5, 5)    → (4, 4)    (return quadrature: K_r 25 → 16)
#   n_eta_nodes      : 4         → 3         (income shock; 2-component mixture, K=3 is min defensible)
#   n_eps_nodes      : 4         → 3         (income shock; same reasoning)
#   n_savings        : 180       → 120       (savings grid; smooth in s, accuracy ~unchanged)
# Unchanged:
#   state_grid_sizes = (7, 7, 7, 7)            (canonical 4-D state grid; thesis target)
#   n_wealth         = 180,  wealth_min = 0.13 (validated EGM region)
#   state_n_stds, state_grid_mode, ret_lobatto_Z, state_lobatto_Z (canonical settings)
#
# CAVEAT: state_lobatto_Z=(None, 7.0, None, 7.0) is preserved from canonical.
# At K=3 nodes/axis the Lobatto rule is valid (K>=3 is the minimum, see
# lifecycle/discretization.py:531) but tail nodes at ±7σ carry near-zero
# Gaussian weight, so the rule is roughly equivalent to Hermite-3 + endpoint
# anchors. Acceptable for now; revisit if the run fails sensitivity checks.
# If you want to drop Lobatto at K=3, set state_lobatto_Z=None.
CANONICAL_DISC = _NUMBA_CANONICAL_DISC._replace(
    n_z=7,
    n_state_quad_nodes=(3, 3, 3, 3),
    n_ret_nodes_1d=(4, 4),
    n_eta_nodes=3,
    n_eps_nodes=3,
    n_savings=120,
)


# ── Solver ──────────────────────────────────────────────────────────────────
# Diff from _canonical.CANONICAL_SOLVER:
#   max_iter              : 8000 → 30    (fori_loop is wall cost; v0.2 bump
#                                         from 20, see changelog)
#   max_iter_unconstrained: 5000 → 30    (legacy alias; pin both for safety)
#   tol                   : 1e-7 → 1e-6  (v0.2 loosen; tol controls the
#                                         convergence mask, not wall — under
#                                         fori_loop cells always run all
#                                         max_iter iters; looser tol just
#                                         marks more cells "converged" so the
#                                         failure-share metric stays
#                                         meaningful at the lower iter cap)
#   use_fori_newton       : True (already SolverConfig default; pinned explicit
#                                so a future default flip can't silently change
#                                canonical wall semantics)
#   wealth_dynamics_spec  : "ccv_log" (also default; pinned explicit)
#
# RISK: max_iter=30 still gives only a 50% margin over the cold-init Newton
# iter count for cells without backward-age warm-start (terminal age 99 cold
# init from (init_alpha_s, init_alpha_b) typically wants 30-50 iters;
# work→retire boundary at age 66 similarly stresses warm-start quality).
# Watch diagnostics['total_newton_failures'] and per-age age_newton_fail
# counts on the first run. Failures clustered at age 99 + boundary age 66 are
# expected at this cap; failures spread across ages mean bump max_iter to
# 50. Wall cost scales as (1 + max_iter*11), so 30→50 is ~1.67× slower.
CANONICAL_SOLVER = _NUMBA_CANONICAL_SOLVER._replace(
    max_iter=30,
    max_iter_unconstrained=30,
    tol=1e-6,
    use_fori_newton=True,
    wealth_dynamics_spec="ccv_log",
)


# ── Solve control ───────────────────────────────────────────────────────────
# Re-exported unchanged from _canonical. Default = full lifecycle (no
# youngest_age_to_solve) with per-age checkpointing. Sweep cells override with
# ``CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)`` for
# retirement-only runs.
# (Re-export already happens at the top via the import statement; this comment
# block is the documentation pin.)
