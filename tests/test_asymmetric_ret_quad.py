"""Per-dimension return-quadrature tests.

Verifies that `discretization.get_return_quadrature` accepts a per-dimension
tuple `(K_1, ..., K_{n_ret})` and that all properties from
`docs/RETURNS.md §6.5` continue to hold:

  1. weights sum to one
  2. all weights positive
  3. weighted mean is zero
  4. weighted covariance equals model.Sigma_r_cond exactly
  5. node count equals prod(K_i)
  6. uniform tuple `(K, K, ..., K)` matches scalar `K` bit-identically

Also verifies the legacy scalar form remains valid and that Precompute
threads the asymmetric tuple through to its print/diagnostic surfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.discretization import _normalize_ret_nodes, get_return_quadrature
from lifecycle.model import DiscretizationConfig
from lifecycle.precompute import Precompute, build_model
from lifecycle.var import build_nominal_system1_var_config


N_PASS = 0
N_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global N_PASS, N_FAIL
    if condition:
        N_PASS += 1
        print(f"  PASS  {name}")
    else:
        N_FAIL += 1
        print(f"  FAIL  {name}  -- {detail}")


def reference_base_config() -> dict:
    return {
        "beta": 0.96,
        "gamma": 3.0,
        "b_bar": 10,
        "start_age": 22,
        "retire_age": 67,
        "terminal_age": 99,
        "b0": -6.142,
        "b1": 0.3040,
        "b2": -0.051,
        "b3": 0.002586,
        "rho": 0.991,
        "pz": 0.176,
        "mu_eta1": -0.524,
        "sigma_eta1": 0.113,
        "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524),
        "sigma_eta2": 0.046,
        "pe": 0.044,
        "mu_eps1": 0.134,
        "sigma_eps1": 0.762,
        "mu_eps2": 0.0,
        "sigma_eps2": 0.055,
        "constrained": True,
    }


def build_reference_model():
    var_config, _, _data = build_nominal_system1_var_config(
        csv_path=str(ROOT / "data" / "var_dataset.csv")
    )
    return build_model(reference_base_config(), var_config, verbose=False)


def run_normalize_helper_checks() -> None:
    print("\n" + "=" * 70)
    print("NORMALIZE HELPER")
    print("=" * 70)

    check("scalar int -> uniform tuple", _normalize_ret_nodes(3, 3) == (3, 3, 3))
    check("list -> tuple", _normalize_ret_nodes([3, 9, 3], 3) == (3, 9, 3))
    check("tuple passes through", _normalize_ret_nodes((3, 9, 3), 3) == (3, 9, 3))
    check("np.int64 scalar accepted", _normalize_ret_nodes(np.int64(2), 3) == (2, 2, 2))

    try:
        _normalize_ret_nodes((3, 3), 3)
        check("wrong-length tuple raises", False, "no error")
    except ValueError:
        check("wrong-length tuple raises", True)

    try:
        _normalize_ret_nodes("bad", 3)
        check("non-numeric raises", False, "no error")
    except (TypeError, ValueError):
        check("non-numeric raises", True)


def run_quadrature_property_checks(model) -> None:
    print("\n" + "=" * 70)
    print("ASYMMETRIC QUADRATURE PROPERTIES (3, 9, 3)")
    print("=" * 70)

    K = (3, 9, 3)
    ret_nodes, ret_weights = get_return_quadrature(model, n_nodes=K)

    expected_count = int(np.prod(K))
    check(
        "node count == prod(K)",
        ret_nodes.shape == (expected_count, model.n_ret),
        f"shape = {ret_nodes.shape}",
    )
    check(
        "weight count == prod(K)",
        ret_weights.shape == (expected_count,),
        f"shape = {ret_weights.shape}",
    )

    w_sum = float(ret_weights.sum())
    check("weights sum to 1", abs(w_sum - 1.0) < 1e-12, f"sum = {w_sum:.16f}")

    check("weights positive", bool((ret_weights > 0).all()))

    wmean = (ret_weights[:, None] * ret_nodes).sum(axis=0)
    err_mean = float(np.max(np.abs(wmean)))
    check("weighted mean is zero", err_mean < 1e-10, f"max|mean| = {err_mean:.3e}")

    cov = (
        ret_weights[:, None, None] * ret_nodes[:, :, None] * ret_nodes[:, None, :]
    ).sum(axis=0)
    err_cov = float(np.max(np.abs(cov - model.Sigma_r_cond)))
    check(
        "weighted covariance == Sigma_r_cond",
        err_cov < 1e-12,
        f"max|cov - Sigma| = {err_cov:.3e}",
    )


def run_symmetric_equivalence_checks(model) -> None:
    print("\n" + "=" * 70)
    print("SYMMETRIC TUPLE == SCALAR (bit-identical)")
    print("=" * 70)

    for K_scalar in (1, 2, 3, 5):
        sym_n, sym_w = get_return_quadrature(model, n_nodes=K_scalar)
        K_tuple = (K_scalar,) * model.n_ret
        asym_n, asym_w = get_return_quadrature(model, n_nodes=K_tuple)
        check(
            f"K={K_scalar}: nodes match scalar",
            np.array_equal(sym_n, asym_n),
            f"max|diff| = {float(np.max(np.abs(sym_n - asym_n))):.3e}",
        )
        check(
            f"K={K_scalar}: weights match scalar",
            np.array_equal(sym_w, asym_w),
            f"max|diff| = {float(np.max(np.abs(sym_w - asym_w))):.3e}",
        )


def run_precompute_integration_checks(model) -> None:
    print("\n" + "=" * 70)
    print("PRECOMPUTE INTEGRATION (asymmetric)")
    print("=" * 70)

    disc = DiscretizationConfig(
        n_wealth=20,
        n_savings=20,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=3.0,
        n_z=5,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=(3, 5, 3),
        n_state_quad_nodes=2,
    )
    pc = Precompute(model, disc, verbose=False)

    check("pc.n_ret_nodes_1d is tuple", isinstance(pc.n_ret_nodes_1d, tuple))
    check("pc.n_ret_nodes_1d preserves order", pc.n_ret_nodes_1d == (3, 5, 3))
    check("pc.n_ret_quad == prod(K)", pc.n_ret_quad == 3 * 5 * 3)
    check(
        "ret_nodes shape matches prod(K)",
        pc.ret_nodes.shape == (3 * 5 * 3, model.n_ret),
    )
    check(
        "exp_ret_* lengths match n_ret_quad",
        pc.exp_ret_bill.shape == (pc.n_ret_quad,)
        and pc.exp_ret_stock.shape == (pc.n_ret_quad,)
        and pc.exp_ret_bond.shape == (pc.n_ret_quad,),
    )

    # Legacy scalar still works.
    disc_scalar = disc._replace(n_ret_nodes_1d=3)
    pc_scalar = Precompute(model, disc_scalar, verbose=False)
    check(
        "scalar disc -> uniform tuple in pc",
        pc_scalar.n_ret_nodes_1d == (3, 3, 3),
    )
    check(
        "scalar disc -> n_ret_quad == K^n_ret",
        pc_scalar.n_ret_quad == 3 ** model.n_ret,
    )


def main() -> int:
    model = build_reference_model()
    run_normalize_helper_checks()
    run_quadrature_property_checks(model)
    run_symmetric_equivalence_checks(model)
    run_precompute_integration_checks(model)

    print("\n" + "=" * 70)
    print(f"RESULT: {N_PASS} passed, {N_FAIL} failed")
    print("=" * 70)
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
