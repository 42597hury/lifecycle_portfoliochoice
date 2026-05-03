"""Verify n_iter diagnostic plumbing through the JIT period dispatch.

Three checks, focused on the changes introduced for #2 (per-call iteration diagnostics):

  1. The conditional-unpack pattern compiles in Numba.
     (constrained branch returns 5-tuple, unconstrained returns 6-tuple, dispatch
      site has different unpacking in each branch.)
  2. _reduce_diag handles the new (N_state, 14) / (N_state, 10) shapes.
  3. End-to-end iter accounting is correct: synthetic diag arrays roll up to the
     expected per-age aggregates (DI_SUM_ITER summed; DF_MAX_NEWTON_ITER maxed).

We deliberately do NOT call the full _solve_retirement_step_quad_jit or
_solve_working_age_step_quad_jit functions — the user has flagged full solver
runs as impractical, and the Numba JIT compile alone takes 30-60s. Instead we
confirm:
  - module imports cleanly (catches syntax errors)
  - the existing test_unconstrained_stagnation.py test still passes (Newton
    returning 6-tuple is structurally sound)
  - a minimal Numba conditional-unpack mirror of the pattern compiles and runs
"""

import os
import sys
import numpy as np
from numba import njit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importing solver triggers Python parse of every JIT function (but not codegen).
# A syntax error in the conditional-unpack would surface here.
import lifecycle.solver  # noqa: F401  -- triggers parse of every JIT function
from lifecycle.solver import (
    DI_SUM_ITER,
    DI_WARM_RESET,
    DI_TOTAL_CALLS,
    DI_NEWTON_FAIL,
    DF_MAX_NEWTON_ITER,
    N_DIAG_INT,
    N_DIAG_FLOAT,
    _reduce_diag,
)


# ----------------------------------------------------------------------
# Test A — module imports and constants are wired
# ----------------------------------------------------------------------
def test_constants_and_import():
    assert N_DIAG_INT == 15, f"N_DIAG_INT was {N_DIAG_INT}, expected 15"
    assert N_DIAG_FLOAT == 10, f"N_DIAG_FLOAT was {N_DIAG_FLOAT}, expected 10"
    assert DI_SUM_ITER == 13
    assert DI_WARM_RESET == 14
    assert DF_MAX_NEWTON_ITER == 9
    print(f"  [OK] solver module imports; "
          f"N_DIAG_INT={N_DIAG_INT}, N_DIAG_FLOAT={N_DIAG_FLOAT}, "
          f"DI_SUM_ITER={DI_SUM_ITER}, DI_WARM_RESET={DI_WARM_RESET}, "
          f"DF_MAX_NEWTON_ITER={DF_MAX_NEWTON_ITER}")


# ----------------------------------------------------------------------
# Test B — _reduce_diag with new shapes
# ----------------------------------------------------------------------
def test_reduce_diag_new_shape():
    """Synthesise per-i_s arrays of the new shape and verify aggregation."""
    N_state = 5

    diag_int = np.zeros((N_state, N_DIAG_INT), dtype=np.int64)
    diag_float = np.zeros((N_state, N_DIAG_FLOAT))

    # Plant test values: DI_SUM_ITER per i_s, DF_MAX_NEWTON_ITER per i_s
    iter_per_is = np.array([3, 7, 0, 12, 2], dtype=np.int64)
    max_per_is = np.array([2.0, 5.0, 0.0, 10.0, 1.0])

    diag_int[:, DI_SUM_ITER] = iter_per_is
    diag_float[:, DF_MAX_NEWTON_ITER] = max_per_is

    # Also set total_calls so the average is computable
    diag_int[:, DI_TOTAL_CALLS] = np.array([10, 10, 5, 10, 5], dtype=np.int64)

    ti, tf_sum, tf_max, tf_min = _reduce_diag(diag_int, diag_float)

    assert ti.shape == (N_DIAG_INT,), f"ti.shape={ti.shape}"
    assert tf_max.shape == (N_DIAG_FLOAT,), f"tf_max.shape={tf_max.shape}"

    # Sum of iter counts across i_s = expected aggregate
    assert ti[DI_SUM_ITER] == int(iter_per_is.sum()), (
        f"sum mismatch: {ti[DI_SUM_ITER]} vs {iter_per_is.sum()}")
    # Max of per-i_s maxes = global max for this age
    assert tf_max[DF_MAX_NEWTON_ITER] == max_per_is.max(), (
        f"max mismatch: {tf_max[DF_MAX_NEWTON_ITER]} vs {max_per_is.max()}")

    avg_per_call = ti[DI_SUM_ITER] / max(int(ti[DI_TOTAL_CALLS]), 1)
    print(f"  [OK] _reduce_diag (N_state={N_state}): "
          f"sum_iter={ti[DI_SUM_ITER]}, max_iter={tf_max[DF_MAX_NEWTON_ITER]:.0f}, "
          f"avg_iter/call={avg_per_call:.2f}")


# ----------------------------------------------------------------------
# Test C — Numba conditional-unpack pattern compiles
# ----------------------------------------------------------------------
@njit
def _five_tuple(x):
    return 1.0, 2.0, 3.0, 4, 5.0  # mimics constrained Newton: 5 values


@njit
def _six_tuple(x):
    return 10.0, 20.0, 30.0, 40, 50.0, 7  # mimics unconstrained: 6 values, last int


@njit
def _conditional_unpack_mirror(constrained, x, n_calls):
    """Mirror the conditional-unpack pattern used in the JIT period solvers.

    Numba must:
      - accept different-arity unpacking in each branch
      - assign n_iter consistently as int in both branches
      - allow downstream arithmetic on n_iter regardless of branch
    """
    total_iter = 0
    max_iter_seen = 0
    for _ in range(n_calls):
        if constrained:
            a, b, c, d, e = _five_tuple(x)
            n_iter = 0
        else:
            a, b, c, d, e, n_iter = _six_tuple(x)
        total_iter += n_iter
        if n_iter > max_iter_seen:
            max_iter_seen = n_iter
    return total_iter, max_iter_seen


def test_conditional_unpack_compiles():
    # Compile both branches
    t_unc, m_unc = _conditional_unpack_mirror(False, 1.0, 3)
    t_con, m_con = _conditional_unpack_mirror(True, 1.0, 3)

    # Unconstrained: 3 calls each returning n_iter=7 → total=21, max=7
    assert t_unc == 21, f"unconstrained total {t_unc} != 21"
    assert m_unc == 7, f"unconstrained max {m_unc} != 7"
    # Constrained: 3 calls each contributing 0 → total=0, max=0
    assert t_con == 0, f"constrained total {t_con} != 0"
    assert m_con == 0, f"constrained max {m_con} != 0"
    print(f"  [OK] Numba accepts conditional 5/6-tuple unpack with n_iter; "
          f"unc=({t_unc},{m_unc}), con=({t_con},{m_con})")


# ----------------------------------------------------------------------
# Test D — warm-start reset on EC_NEWTON_FAIL
# ----------------------------------------------------------------------
# Mirror the JIT period-solver pattern: a chain of solves with warm-starts.
# We feed in a synthetic exit_code sequence and verify:
#   - on FAIL, last_a_* resets to init
#   - on success, last_a_* carries the returned (opt_s, opt_b)
#   - constrained branch never resets (mirrors the (not constrained) guard)
EC_NEWTON_FAIL_LITERAL = 8  # matches solver.py EC_NEWTON_FAIL


@njit
def _warm_start_chain(constrained, init_s, init_b, exit_codes, opt_s_returned, opt_b_returned):
    """Simulate the warm-start chain in the JIT period dispatch.

    For each step i in exit_codes:
      - solver "returns" (opt_s_returned[i], opt_b_returned[i], exit_codes[i])
      - we apply the patched warm-start update logic
    Returns the final (last_a_s, last_a_b) plus the reset count.
    """
    last_a_s = init_s
    last_a_b = init_b
    n_resets = 0
    n = exit_codes.shape[0]
    for i in range(n):
        opt_s = opt_s_returned[i]
        opt_b = opt_b_returned[i]
        ec = exit_codes[i]
        if (not constrained) and ec == EC_NEWTON_FAIL_LITERAL:
            last_a_s = init_s
            last_a_b = init_b
            n_resets += 1
        else:
            last_a_s = opt_s
            last_a_b = opt_b
    return last_a_s, last_a_b, n_resets


def test_warm_start_reset():
    init_s, init_b = 0.40, 0.33

    # Scenario 1: unconstrained, alternating success/fail/success/fail
    #   exit_codes:        [INTERIOR, FAIL,    INTERIOR, FAIL]
    #   solver returns:    (0.5,0.6) (1.5,-2.0) (0.7,0.8) (-3,4)
    # After step 0: warm = (0.5, 0.6)        (success)
    # After step 1: warm = (init_s, init_b)  (FAIL → reset)
    # After step 2: warm = (0.7, 0.8)        (success)
    # After step 3: warm = (init_s, init_b)  (FAIL → reset)
    ecs = np.array([7, 8, 7, 8], dtype=np.int64)
    opt_s_seq = np.array([0.5, 1.5, 0.7, -3.0])
    opt_b_seq = np.array([0.6, -2.0, 0.8, 4.0])

    last_s, last_b, n_resets = _warm_start_chain(
        False, init_s, init_b, ecs, opt_s_seq, opt_b_seq)
    assert abs(last_s - init_s) < 1e-12, f"final last_s {last_s} != init_s {init_s}"
    assert abs(last_b - init_b) < 1e-12, f"final last_b {last_b} != init_b {init_b}"
    assert n_resets == 2, f"expected 2 resets, got {n_resets}"
    print(f"  [OK] unconstrained chain: 4 steps with 2 fails -> n_resets=2, "
          f"final warm=({last_s:.2f}, {last_b:.2f})=init")

    # Scenario 2: unconstrained, all success — no resets
    ecs2 = np.array([7, 7, 7], dtype=np.int64)
    opt_s2 = np.array([0.5, 0.6, 0.7])
    opt_b2 = np.array([0.4, 0.5, 0.6])
    ls2, lb2, n2 = _warm_start_chain(False, init_s, init_b, ecs2, opt_s2, opt_b2)
    assert n2 == 0
    assert abs(ls2 - 0.7) < 1e-12 and abs(lb2 - 0.6) < 1e-12
    print(f"  [OK] unconstrained chain, all success: n_resets=0, warm=(0.70,0.60)")

    # Scenario 3: constrained — even on FAIL, no reset (guard kicks out)
    ls3, lb3, n3 = _warm_start_chain(True, init_s, init_b, ecs, opt_s_seq, opt_b_seq)
    assert n3 == 0, f"constrained should NEVER reset; got n_resets={n3}"
    # final warm = (-3.0, 4.0) — last (opt_s, opt_b) carried forward
    assert abs(ls3 - (-3.0)) < 1e-12 and abs(lb3 - 4.0) < 1e-12
    print(f"  [OK] constrained chain: same fail pattern, n_resets=0 "
          f"(guard prevents reset), final warm=({ls3:.2f}, {lb3:.2f})")


if __name__ == "__main__":
    print("Test A: constants and module import")
    test_constants_and_import()
    print("\nTest B: _reduce_diag with new diag shapes")
    test_reduce_diag_new_shape()
    print("\nTest C: Numba conditional-unpack compiles")
    test_conditional_unpack_compiles()
    print("\nTest D: warm-start reset on EC_NEWTON_FAIL")
    test_warm_start_reset()
    print("\nAll tests passed.")
