"""
State distribution diagnostic.

Two checks of whether the VAR's state dynamics make the observed historical
state history "typical" from the agent's perspective. State variable order
is read at runtime from the model's partition (default cy/spr/y_1 since
2026-04-30; legacy y_1/spr/cy under older bundles).

  1. Unconditional fit. Stationary distribution implied by the VAR
     (mean (I-Phi_11)^-1 Phi_0, covariance from discrete Lyapunov).
     For each historical s_t compute the squared Mahalanobis distance
     d^2_t = (s_t - mu_inf)' Sigma_inf^-1 (s_t - mu_inf). Under the
     model d^2 ~ chi^2_3, so E[d^2]=3, 95%-quantile=7.815, 99%=11.345.

  2. Conditional fit. One-step VAR prediction error
     e_t = s_t - (Phi_0 + Phi_11 s_{t-1}); Mahalanobis distance under
     Sigma_ss should also be chi^2_3.
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2

from var import build_nominal_system1_var_config, partition_var

CSV_PATH = "data/var_dataset.csv"
# State variable names; resolved dynamically below from the partition so this
# script keeps working under either ordering (legacy (y_1, spr, cy) or default
# (cy, spr, y_1) after 2026-04-30).  See docs/RETURNS.md sec.5.6.
N_STATE = 3


def _print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    var_config, _, data = build_nominal_system1_var_config(
        csv_path=CSV_PATH,
        estimation="restricted",
    )
    parts = partition_var(
        Phi_full=var_config["Phi"],
        Omega_full=var_config["Omega"],
        z_bar=var_config["z_bar"],
        state_idx=var_config["state_indices"],
        ret_idx=var_config["return_indices"],
        variable_names=var_config["variable_names"],
        verbose=False,
    )

    Phi_0_state = parts["Phi_0_state"]
    Phi_11 = parts["Phi_11"]
    Sigma_ss = parts["Sigma_ss"]

    # State variable names IN MODEL ORDER -- read from the partition so that
    # historical S, model moments, and prediction errors all use the same
    # column ordering.
    STATE_NAMES = tuple(parts["state_names"])

    # ------------------------------------------------------------------
    # Historical state observations
    # ------------------------------------------------------------------
    df = pd.read_csv(CSV_PATH).dropna()
    S = df[list(STATE_NAMES)].to_numpy()
    T = S.shape[0]

    # ------------------------------------------------------------------
    # Stationary moments implied by VAR
    # ------------------------------------------------------------------
    mu_inf = np.linalg.solve(np.eye(N_STATE) - Phi_11, Phi_0_state)
    Sigma_inf = solve_discrete_lyapunov(Phi_11, Sigma_ss)

    # Sample moments for comparison
    mu_emp = S.mean(axis=0)
    Sigma_emp = np.cov(S.T, ddof=1)

    _print_header("MODEL-IMPLIED vs SAMPLE STATIONARY MOMENTS")
    print(f"{'var':>6} {'mu_emp':>12} {'mu_inf':>12} {'std_emp':>12} {'std_inf':>12}")
    for i, name in enumerate(STATE_NAMES):
        print(
            f"{name:>6} {mu_emp[i]:>12.6f} {mu_inf[i]:>12.6f} "
            f"{np.sqrt(Sigma_emp[i,i]):>12.6f} {np.sqrt(Sigma_inf[i,i]):>12.6f}"
        )
    print()
    print("Implied corr matrix (Sigma_inf):")
    Dinv = np.diag(1.0 / np.sqrt(np.diag(Sigma_inf)))
    print(np.round(Dinv @ Sigma_inf @ Dinv, 3))
    print("Sample corr matrix:")
    Dinv_e = np.diag(1.0 / np.sqrt(np.diag(Sigma_emp)))
    print(np.round(Dinv_e @ Sigma_emp @ Dinv_e, 3))

    # ------------------------------------------------------------------
    # Test 1: Unconditional Mahalanobis distance
    # ------------------------------------------------------------------
    Sigma_inf_inv = np.linalg.inv(Sigma_inf)
    diff_uncond = S - mu_inf
    d2_uncond = np.einsum("ti,ij,tj->t", diff_uncond, Sigma_inf_inv, diff_uncond)

    q95, q99 = chi2.ppf([0.95, 0.99], df=N_STATE)
    median_chi2 = chi2.ppf(0.5, df=N_STATE)

    _print_header("TEST 1: UNCONDITIONAL MAHALANOBIS DISTANCE (chi^2_3)")
    print("Theoretical reference:")
    print(f"  E[d^2]            = {N_STATE}")
    print(f"  median            = {median_chi2:.3f}")
    print(f"  95%-quantile      = {q95:.3f}")
    print(f"  99%-quantile      = {q99:.3f}")
    print()
    print("Empirical:")
    print(f"  T (obs)           = {T}")
    print(f"  mean(d^2)         = {d2_uncond.mean():.3f}")
    print(f"  median(d^2)       = {np.median(d2_uncond):.3f}")
    print(f"  max(d^2)          = {d2_uncond.max():.3f}  (year={int(df['year'].iloc[d2_uncond.argmax()])})")
    n95 = int((d2_uncond > q95).sum())
    n99 = int((d2_uncond > q99).sum())
    print(f"  # obs > 95% qtl   = {n95}/{T}  ({100*n95/T:.1f}%, expected ~5%)")
    print(f"  # obs > 99% qtl   = {n99}/{T}  ({100*n99/T:.1f}%, expected ~1%)")
    print()
    # Top 5 most surprising years under the unconditional model
    order = np.argsort(d2_uncond)[::-1][:5]
    print("Top 5 most surprising years (unconditional):")
    print(f"  {'year':>6} {'d^2':>8} {'p_chi2':>10}  state {STATE_NAMES}")
    for idx in order:
        p = float(chi2.sf(d2_uncond[idx], df=N_STATE))
        s = S[idx]
        print(f"  {int(df['year'].iloc[idx]):>6} {d2_uncond[idx]:>8.3f} {p:>10.5f}  ({s[0]:+.4f}, {s[1]:+.4f}, {s[2]:+.4f})")

    # ------------------------------------------------------------------
    # Test 2: One-step conditional Mahalanobis distance
    # ------------------------------------------------------------------
    S_lag = S[:-1]
    S_now = S[1:]
    pred = (Phi_0_state + S_lag @ Phi_11.T)
    err = S_now - pred
    Sigma_ss_inv = np.linalg.inv(Sigma_ss)
    d2_cond = np.einsum("ti,ij,tj->t", err, Sigma_ss_inv, err)

    _print_header("TEST 2: ONE-STEP CONDITIONAL MAHALANOBIS DISTANCE (chi^2_3)")
    print(f"  T (transitions)   = {len(d2_cond)}")
    print(f"  mean(d^2)         = {d2_cond.mean():.3f}    (expect 3.0)")
    print(f"  median(d^2)       = {np.median(d2_cond):.3f}    (expect {median_chi2:.3f})")
    print(f"  max(d^2)          = {d2_cond.max():.3f}  (year={int(df['year'].iloc[1+d2_cond.argmax()])})")
    n95c = int((d2_cond > q95).sum())
    n99c = int((d2_cond > q99).sum())
    print(f"  # obs > 95% qtl   = {n95c}/{len(d2_cond)}  ({100*n95c/len(d2_cond):.1f}%, expected ~5%)")
    print(f"  # obs > 99% qtl   = {n99c}/{len(d2_cond)}  ({100*n99c/len(d2_cond):.1f}%, expected ~1%)")
    print()
    # Per-variable standardized residuals (z-scores) — diagonal of Sigma_ss
    z_cond = err / np.sqrt(np.diag(Sigma_ss))
    print("Standardized one-step innovations (mean / std should be ~0 / ~1):")
    print(f"  {'var':>6} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for i, name in enumerate(STATE_NAMES):
        print(
            f"  {name:>6} {z_cond[:,i].mean():>10.3f} {z_cond[:,i].std(ddof=1):>10.3f} "
            f"{z_cond[:,i].min():>10.3f} {z_cond[:,i].max():>10.3f}"
        )
    print()
    order_c = np.argsort(d2_cond)[::-1][:5]
    print("Top 5 most surprising transitions (conditional):")
    print(f"  {'year':>6} {'d^2':>8} {'p_chi2':>10}  innovation z-scores {STATE_NAMES}")
    for idx in order_c:
        p = float(chi2.sf(d2_cond[idx], df=N_STATE))
        z = z_cond[idx]
        yr = int(df["year"].iloc[idx + 1])
        print(f"  {yr:>6} {d2_cond[idx]:>8.3f} {p:>10.5f}  ({z[0]:+.2f}, {z[1]:+.2f}, {z[2]:+.2f})")


if __name__ == "__main__":
    main()
