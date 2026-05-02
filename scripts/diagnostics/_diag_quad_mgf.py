"""
_diag_quad_mgf.py -- MGF exactness check for state and return quadrature axes.

Tests whether a candidate K integrates exp(p*u) accurately enough against the
standard normal density, for the range of effective exponents p that the
lifecycle integrand actually sees on each axis.

Effective exponent per axis:
    p_eff(axis, ret) = gamma * |portfolio_loading| * |per-axis log-return shift|

For the K_state axes the per-axis shift is the row of M @ L_state.
For the K_ret axes it is the row of L_ret = chol(Sigma_r_cond).

Reads the production VAR from a saved bundle metadata.json so the test runs
without re-estimating anything.

Usage
-----
python -m scripts.diagnostics._diag_quad_mgf \\
    --bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 \\
    --gamma 10 --alpha-max 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import roots_hermite


def _arr(d):
    return np.asarray(d["values"], dtype=float)


def gh_nodes(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Gauss-Hermite: returns (u_k, w_k) with u_k in N(0,1) units."""
    if K == 1:
        return np.zeros(1), np.ones(1)
    z, w = roots_hermite(K)
    return z * math.sqrt(2.0), w / math.sqrt(math.pi)


def mgf_error(K: int, p: float) -> float:
    """Relative error of K-node Hermite integrating exp(p*u) vs analytic exp(p^2/2)."""
    u, w = gh_nodes(K)
    approx = float(np.sum(w * np.exp(p * u)))
    truth = math.exp(p * p / 2.0)
    return (approx - truth) / truth


def smallest_K_for_tol(p: float, tol: float, K_max: int = 12) -> int:
    """Return smallest K with |relative MGF error| <= tol; K_max if none."""
    for K in range(1, K_max + 1):
        if abs(mgf_error(K, p)) <= tol:
            return K
    return K_max + 1


def load_var(bundle: Path) -> dict:
    meta = json.loads((bundle / "metadata.json").read_text())
    v = meta["run_config"]["var_config"]
    Phi = _arr(v["Phi"])
    Omega = _arr(v["Omega"])
    state_idx = np.asarray(v["state_indices"])
    ret_idx = np.asarray(v["return_indices"])
    state_names = [v["variable_names"][i] for i in state_idx]
    ret_names = [v["variable_names"][i] for i in ret_idx]

    Sigma_ss = Omega[np.ix_(state_idx, state_idx)]
    Sigma_rs = Omega[np.ix_(ret_idx, state_idx)]
    Sigma_rr = Omega[np.ix_(ret_idx, ret_idx)]
    M = Sigma_rs @ np.linalg.inv(Sigma_ss)
    Sigma_r_cond = Sigma_rr - M @ Omega[np.ix_(state_idx, ret_idx)]
    L_state = np.linalg.cholesky(0.5 * (Sigma_ss + Sigma_ss.T))
    L_ret = np.linalg.cholesky(0.5 * (Sigma_r_cond + Sigma_r_cond.T))

    return dict(
        state_names=state_names, ret_names=ret_names,
        Sigma_ss=Sigma_ss, Sigma_r_cond=Sigma_r_cond,
        M=M, L_state=L_state, L_ret=L_ret, ML=M @ L_state,
    )


def axis_effective_p(loadings: np.ndarray, gamma: float, alpha_max: float) -> float:
    """Effective MGF exponent on an axis = gamma * alpha_max * max|loading|.

    Loadings: per-asset log-return shift per +1 std-normal innovation on this axis.
    """
    return gamma * alpha_max * float(np.max(np.abs(loadings)))


def report(bundle: Path, gamma: float, alpha_max: float, tol: float):
    data = load_var(bundle)
    ML = data["ML"]              # (n_ret, n_state) - K_state axis loadings on each return
    L_ret = data["L_ret"]        # (n_ret, n_ret)   - K_ret axis loadings on each return
    state_names = data["state_names"]
    ret_names = data["ret_names"]

    cholesky_state_labels = ["u_0 (cy pure)", "u_1 (spr-dom)", "u_2 (y_1 resid)"]
    cholesky_ret_labels = ["e_0 (rtb pure)", "e_1 (xr-resid orth)", "e_2 (xb-resid orth)"]

    print(f"Bundle: {bundle.name}")
    print(f"State ordering: {state_names}")
    print(f"Return ordering: {ret_names}")
    print(f"gamma = {gamma}, alpha_max = {alpha_max}, tolerance = {tol:g}")
    print()
    print("=" * 76)
    print("STATE-INNOVATION AXES  (M @ L  shows per-axis log-return shift per +1 std-N)")
    print("=" * 76)
    print(f"{'axis':30s} {'shift on rtb':>12s} {'shift on xr':>12s} {'shift on xb':>12s}")
    for i, lbl in enumerate(cholesky_state_labels[:ML.shape[1]]):
        print(f"{lbl:30s} {ML[0, i]:>12.4f} {ML[1, i]:>12.4f} {ML[2, i]:>12.4f}")
    print()

    print(f"{'axis':30s} {'p_eff':>8s} {'min K':>7s}   MGF rel-err at K = 2,3,4,5,6,7")
    for i, lbl in enumerate(cholesky_state_labels[:ML.shape[1]]):
        p_eff = axis_effective_p(ML[:, i], gamma, alpha_max)
        K_need = smallest_K_for_tol(p_eff, tol)
        errs = [mgf_error(K, p_eff) for K in range(2, 8)]
        err_str = " ".join(f"{e:+.1e}" for e in errs)
        flag = "  <-- POLICY K=2/5" if i in (0, 1) else ""
        print(f"{lbl:30s} {p_eff:>8.3f} {K_need:>7d}   {err_str}{flag}")
    print()

    print("=" * 76)
    print("RETURN-RESIDUAL AXES  (L_ret rows = per-axis residual log-return shift)")
    print("=" * 76)
    print(f"{'axis':30s} {'shift on rtb':>12s} {'shift on xr':>12s} {'shift on xb':>12s}")
    for i, lbl in enumerate(cholesky_ret_labels[:L_ret.shape[1]]):
        print(f"{lbl:30s} {L_ret[0, i]:>12.4f} {L_ret[1, i]:>12.4f} {L_ret[2, i]:>12.4f}")
    print()

    print(f"{'axis':30s} {'p_eff':>8s} {'min K':>7s}   MGF rel-err at K = 2,3,4,5,6,7")
    for i, lbl in enumerate(cholesky_ret_labels[:L_ret.shape[1]]):
        p_eff = axis_effective_p(L_ret[:, i], gamma, alpha_max)
        K_need = smallest_K_for_tol(p_eff, tol)
        errs = [mgf_error(K, p_eff) for K in range(2, 8)]
        err_str = " ".join(f"{e:+.1e}" for e in errs)
        print(f"{lbl:30s} {p_eff:>8.3f} {K_need:>7d}   {err_str}")
    print()

    print("=" * 76)
    print("SENSITIVITY: minimum K vs alpha_max (gamma fixed, all axes shown)")
    print("=" * 76)
    alpha_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    print(f"{'axis':30s}  " + " ".join(f"a={a:>3.1f}" for a in alpha_grid))
    for i, lbl in enumerate(cholesky_state_labels[:ML.shape[1]]):
        Ks = [smallest_K_for_tol(axis_effective_p(ML[:, i], gamma, a), tol) for a in alpha_grid]
        print(f"{lbl:30s}  " + " ".join(f"{K:>5d}" for K in Ks))
    for i, lbl in enumerate(cholesky_ret_labels[:L_ret.shape[1]]):
        Ks = [smallest_K_for_tol(axis_effective_p(L_ret[:, i], gamma, a), tol) for a in alpha_grid]
        print(f"{lbl:30s}  " + " ".join(f"{K:>5d}" for K in Ks))
    print()

    print("Interpretation:")
    print("  p_eff = gamma * alpha_max * max|axis loading on any log-return|.")
    print("  'min K' = smallest Hermite order with |E_K[exp(p*u)] - exp(p^2/2)| <= tol.")
    print("  This is a NECESSARY (not sufficient) condition: if K is below 'min K' the")
    print("  rule cannot integrate even the bare lognormal MGF at the integrand-relevant")
    print("  exponent, so it certainly cannot integrate mu(c_next)*R_p.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path,
                   default=Path("saved_runs/unconstrained_principal_grid5x5x5_nz9"))
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--alpha-max", type=float, default=3.0)
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()
    report(args.bundle, args.gamma, args.alpha_max, args.tol)
