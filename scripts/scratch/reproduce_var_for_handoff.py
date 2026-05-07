"""Reproduce the System I VAR estimation end-to-end and compare to hardcoded.

Writes a side-by-side report of estimated vs hardcoded coefficients,
intercepts, residual covariance, and partition-derived quantities
(M, Sigma_r_cond, eigenvalues of Phi_11). Output is captured by the
HANDOFF_RETURN_MODELLING_TRACE work and embedded in the §5 numerical
appendix as-is.

Throwaway, scripts/scratch/, not committed long-term.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lifecycle.var import (
    build_nominal_system1_var_config,
    build_nominal_system1_var_config_hardcoded,
    partition_var,
)


CSV = os.path.join(ROOT, "data", "var_dataset.csv")


def _print_matrix(label, M, fmt="{:+.6e}"):
    print(label)
    for r in M:
        print("  " + "  ".join(fmt.format(v) for v in r))


def main():
    print("=" * 80)
    print("VAR estimation reproduction — System I (rtb-as-state, post 2026-05-06)")
    print("=" * 80)
    print(f"CSV: {CSV}  (exists={os.path.exists(CSV)})")
    print()

    # 1. Run the estimation pipeline (CCV-constrained restricted VAR(1)).
    var_config_est, fit_obj, data = build_nominal_system1_var_config(
        csv_path=CSV,
        state_indices=(2, 1, 3, 0),    # cy, spr, rtb, y_1 (default ordering)
        return_indices=(4, 5),
        y_1_index_in_state=3,
        spr_index_in_state=1,
        rtb_index_in_state=2,
        trend="c",
        estimation="restricted",
    )

    var_config_hc = build_nominal_system1_var_config_hardcoded()

    print()
    print("=" * 80)
    print("ELEMENT-WISE COMPARISON (estimated vs hardcoded)")
    print("=" * 80)

    cols = var_config_est["variable_names"]
    n = len(cols)
    print(f"Columns: {cols}")
    print(f"State idx (estimated): {var_config_est['state_indices']}")
    print(f"State idx (hardcoded): {var_config_hc['state_indices']}")
    print(f"Return idx (est):       {var_config_est['return_indices']}")
    print(f"Return idx (hc):        {var_config_hc['return_indices']}")
    print()

    z_bar_est = np.asarray(var_config_est["z_bar"])
    z_bar_hc = np.asarray(var_config_hc["z_bar"])
    print(f"{'col':>5s}  {'estimated':>14s}  {'hardcoded':>14s}  {'abs_diff':>10s}")
    for i, c in enumerate(cols):
        e, h = z_bar_est[i], z_bar_hc[i]
        print(f"{c:>5s}  {e:+14.10e}  {h:+14.10e}  {abs(e-h):10.2e}")
    print()

    Phi_est = np.asarray(var_config_est["Phi"])
    Phi_hc = np.asarray(var_config_hc["Phi"])
    print("Phi (estimated):")
    _print_matrix("", Phi_est, fmt="{:+.6e}")
    print()
    print("Phi (hardcoded):")
    _print_matrix("", Phi_hc, fmt="{:+.6e}")
    print()
    print(f"max |Phi_est - Phi_hc| = {np.max(np.abs(Phi_est - Phi_hc)):.3e}")
    print()

    Omega_est = np.asarray(var_config_est["Omega"])
    Omega_hc = np.asarray(var_config_hc["Omega"])
    print("Omega (estimated):")
    _print_matrix("", Omega_est, fmt="{:+.6e}")
    print()
    print("Omega (hardcoded):")
    _print_matrix("", Omega_hc, fmt="{:+.6e}")
    print()
    print(f"max |Omega_est - Omega_hc| = {np.max(np.abs(Omega_est - Omega_hc)):.3e}")
    print()

    const_est = np.asarray(var_config_est["const"])
    const_hc = np.asarray(var_config_hc["const"])
    print("const = (I-Phi) z_bar  (estimated vs hardcoded):")
    print(f"{'col':>5s}  {'estimated':>14s}  {'hardcoded':>14s}  {'abs_diff':>10s}")
    for i, c in enumerate(cols):
        print(f"{c:>5s}  {const_est[i]:+14.6e}  {const_hc[i]:+14.6e}  {abs(const_est[i]-const_hc[i]):10.2e}")
    print()

    print("=" * 80)
    print("PARTITION-DERIVED QUANTITIES (using hardcoded VAR; estimator output identical)")
    print("=" * 80)
    parts = partition_var(
        Phi_full=Phi_hc, Omega_full=Omega_hc, z_bar=z_bar_hc,
        state_idx=var_config_hc["state_indices"],
        ret_idx=var_config_hc["return_indices"],
        variable_names=cols,
        verbose=False,
    )
    print(f"State names (in state-row order): {parts['state_names']}")
    print(f"Return names (in ret-row order):  {parts['ret_names']}")
    print()
    print("Phi_0_state (state intercepts after partition):")
    print("  " + "  ".join(f"{v:+.6e}" for v in parts["Phi_0_state"]))
    print("Phi_11 (state -> state):")
    _print_matrix("", parts["Phi_11"], fmt="{:+.6e}")
    print()
    print("Phi_0_ret (return intercepts after partition):")
    print("  " + "  ".join(f"{v:+.6e}" for v in parts["Phi_0_ret"]))
    print("Phi_21 (state -> return; this is A_r):")
    _print_matrix("", parts["Phi_21"], fmt="{:+.6e}")
    print()
    print("Sigma_ss (state-innovation covariance):")
    _print_matrix("", parts["Sigma_ss"], fmt="{:+.6e}")
    print()
    print("Sigma_rr (return-innovation covariance, UNCONDITIONAL):")
    _print_matrix("", parts["Sigma_rr"], fmt="{:+.6e}")
    print()
    print("Sigma_rs (return-state cross covariance, return rows × state cols):")
    _print_matrix("", parts["Sigma_rs"], fmt="{:+.6e}")
    print()
    print("M = Sigma_rs @ inv(Sigma_ss)  (state-innovation -> return-mean projection):")
    _print_matrix("", parts["M"], fmt="{:+.6e}")
    print()
    print("Sigma_r_cond = Sigma_rr - M @ Sigma_sr (return cov CONDITIONAL on state innovation):")
    _print_matrix("", parts["Sigma_r_cond"], fmt="{:+.6e}")
    print()

    print("Diagnostics:")
    print(f"  diag(Sigma_rr)       = {np.diag(parts['Sigma_rr'])}")
    print(f"  diag(Sigma_r_cond)   = {np.diag(parts['Sigma_r_cond'])}")
    print(f"  ann std (Sigma_rr)    = {np.sqrt(np.diag(parts['Sigma_rr']))}  (xr, xb)")
    print(f"  ann std (Sigma_cond)  = {np.sqrt(np.diag(parts['Sigma_r_cond']))}  (xr, xb)")
    print(f"  var explained share   = {parts['var_explained_share']}")
    print()

    eigs = np.sort(np.abs(np.linalg.eigvals(parts["Phi_11"])))[::-1]
    print(f"Phi_11 eigenvalue moduli (desc): {eigs}")
    print(f"max |eigenvalue|: {eigs[0]:.6f}  ({'STATIONARY' if eigs[0] < 1.0 else 'NON-STATIONARY'})")
    print()

    # Joint Sigma reordered to (state, return) blocks
    state_idx = np.asarray(var_config_hc["state_indices"], dtype=int)
    ret_idx = np.asarray(var_config_hc["return_indices"], dtype=int)
    perm = np.concatenate([state_idx, ret_idx])
    Sigma_joint = Omega_hc[np.ix_(perm, perm)]
    eig_joint = np.sort(np.linalg.eigvalsh(0.5 * (Sigma_joint + Sigma_joint.T)))
    print("Joint Sigma reordered (state rows then return rows):")
    _print_matrix("", Sigma_joint, fmt="{:+.6e}")
    print(f"\neigvalsh(Sigma_joint): {eig_joint}")
    print(f"all positive? {np.all(eig_joint > 0)}")
    print()

    print("=" * 80)
    print("R^2 per equation (from estimator):")
    for c in cols:
        r2 = var_config_est["equation_r2"][c]
        print(f"  {c:>5s}: R^2 = {r2:.6f}")


if __name__ == "__main__":
    main()
