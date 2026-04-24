"""
var.py — VAR estimation and partitioning.

Contains:
  - partition_var() — split full VAR into state/return blocks
  - VAR estimation from CSV (restricted and unrestricted)
  - Convenience wrappers for nominal/TIPS systems
  - annualize_var_config() — quarterly → annual compounding (legacy, not used
    when VAR is estimated directly at annual frequency)
  - Hardcoded VAR parameter fallbacks

Dependencies: numpy, pandas (for CSV), statsmodels (for unrestricted VAR)
"""

import numpy as np


# =============================================================================
# VAR PARTITION (GENERIC)
# =============================================================================

def _validate_partition_inputs(Phi_full, Omega_full, z_bar, state_idx, ret_idx):
    n = len(z_bar)
    if Phi_full.shape != (n, n):
        raise ValueError(f"Phi must have shape {(n, n)}, got {Phi_full.shape}")
    if Omega_full.shape != (n, n):
        raise ValueError(f"Omega must have shape {(n, n)}, got {Omega_full.shape}")

    state_idx = np.asarray(state_idx, dtype=int)
    ret_idx = np.asarray(ret_idx, dtype=int)

    if len(state_idx) == 0 or len(ret_idx) == 0:
        raise ValueError("state_idx and ret_idx must both be non-empty")

    all_idx = np.concatenate([state_idx, ret_idx])
    if np.any(all_idx < 0) or np.any(all_idx >= n):
        raise ValueError("state_idx/ret_idx contains out-of-bounds index")

    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("state_idx and ret_idx overlap or contain duplicates")

    if len(np.unique(all_idx)) != n:
        missing = sorted(set(range(n)) - set(all_idx.tolist()))
        raise ValueError(f"state_idx + ret_idx must cover all variables exactly once. Missing: {missing}")

    return state_idx, ret_idx


def partition_var(Phi_full, Omega_full, z_bar, state_idx, ret_idx, variable_names=None, verbose=True):
    """Partition full VAR into state and return blocks using index lists."""
    Phi_full = np.asarray(Phi_full, dtype=float)
    Omega_full = np.asarray(Omega_full, dtype=float)
    z_bar = np.asarray(z_bar, dtype=float)

    state_idx, ret_idx = _validate_partition_inputs(Phi_full, Omega_full, z_bar, state_idx, ret_idx)

    if variable_names is None:
        variable_names = tuple(f"z{i}" for i in range(len(z_bar)))
    else:
        variable_names = tuple(variable_names)
        if len(variable_names) != len(z_bar):
            raise ValueError("variable_names length must match VAR dimension")

    Phi_11 = Phi_full[np.ix_(state_idx, state_idx)]
    Phi_21 = Phi_full[np.ix_(ret_idx, state_idx)]
    Phi_12 = Phi_full[np.ix_(state_idx, ret_idx)]
    Phi_22 = Phi_full[np.ix_(ret_idx, ret_idx)]

    Sigma_ss = Omega_full[np.ix_(state_idx, state_idx)]
    Sigma_rr = Omega_full[np.ix_(ret_idx, ret_idx)]
    Sigma_rs = Omega_full[np.ix_(ret_idx, state_idx)]
    Sigma_sr = Omega_full[np.ix_(state_idx, ret_idx)]

    M = Sigma_rs @ np.linalg.inv(Sigma_ss)
    Sigma_r_cond = Sigma_rr - M @ Sigma_sr

    z_bar_state = z_bar[state_idx]
    z_bar_ret = z_bar[ret_idx]

    # Intercepts: compute from the full VAR then partition.
    # This is exact for both restricted (Phi_12=0, Phi_22=0) and unrestricted VAR.
    Phi_0_full  = (np.eye(len(z_bar)) - Phi_full) @ z_bar
    Phi_0_state = Phi_0_full[state_idx]
    Phi_0_ret   = Phi_0_full[ret_idx]

    Phi_12_norm = np.linalg.norm(Phi_12)
    Phi_22_norm = np.linalg.norm(Phi_22)

    explained_share = 1.0 - np.clip(np.diag(Sigma_r_cond) / np.maximum(np.diag(Sigma_rr), 1e-14), 0.0, 1.0)

    parts = {
        "n_state": len(state_idx),
        "n_ret": len(ret_idx),
        "state_idx": state_idx,
        "ret_idx": ret_idx,
        "state_names": tuple(variable_names[i] for i in state_idx),
        "ret_names": tuple(variable_names[i] for i in ret_idx),
        "z_bar_state": z_bar_state,
        "z_bar_ret": z_bar_ret,
        "Phi_0_state": Phi_0_state,
        "Phi_11": Phi_11,
        "Phi_0_ret": Phi_0_ret,
        "Phi_21": Phi_21,
        "Sigma_ss": Sigma_ss,
        "Sigma_rr": Sigma_rr,
        "Sigma_rs": Sigma_rs,
        "M": M,
        "Sigma_r_cond": Sigma_r_cond,
        "Phi_12_norm": Phi_12_norm,
        "Phi_22_norm": Phi_22_norm,
        "var_explained_share": explained_share,
    }

    if verbose:
        print("=" * 64)
        print("VAR PARTITION SUMMARY")
        print("=" * 64)
        print(f"Full variables: {list(variable_names)}")
        print(f"State variables (on grid): {list(parts['state_names'])}")
        print(f"Return variables (integrated): {list(parts['ret_names'])}")
        print()

        # Stationarity of slow-state sub-VAR
        eigs = np.sort(np.abs(np.linalg.eigvals(Phi_11)))[::-1]
        max_eig = eigs[0]
        stability = "STATIONARY" if max_eig < 1.0 else "*** NON-STATIONARY ***"
        print("Slow-state sub-VAR:")
        print(f"  Phi_11 eigenvalues (|.|): {eigs.round(4).tolist()}")
        print(f"  Max |eigenvalue| = {max_eig:.4f}  --  {stability}")
        print()

        # Restriction check
        print("Restriction check (should be near zero if imposed):")
        print(f"  ||Phi_12|| = {Phi_12_norm:.6e}   (lagged returns -> states)")
        print(f"  ||Phi_22|| = {Phi_22_norm:.6e}   (lagged returns -> returns)")
        print()

        # Return variance explained
        print("Conditional return variance explained by slow-state conditioning:")
        for k, name in enumerate(parts["ret_names"]):
            print(f"  {name}: {100.0 * explained_share[k]:6.2f}%  explained"
                  f"  (residual var = {np.diag(Sigma_r_cond)[k]:.6f})")
        print("=" * 64)

    return parts


# =============================================================================
# VAR ESTIMATION (RESTRICTED OR UNRESTRICTED)
# =============================================================================

def _load_var_dataset(csv_path, columns):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        except Exception:
            pass

    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

    data = df[columns].dropna().astype(float)
    if len(data) < 10:
        raise ValueError("Too few observations after dropna")

    return data


def _compute_r2_per_equation(y_true, y_hat, columns):
    out = {}
    for i, col in enumerate(columns):
        resid = y_true[:, i] - y_hat[:, i]
        sse = float(np.sum(resid ** 2))
        centered = y_true[:, i] - np.mean(y_true[:, i])
        sst = float(np.sum(centered ** 2))
        out[col] = 1.0 - sse / max(sst, 1e-14)
    return out


def _safe_residual_correlation(resid):
    corr = np.corrcoef(resid, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def estimate_var1_from_csv(csv_path, columns, trend="c"):
    """
    Estimate unrestricted VAR(1) from CSV using all lagged variables.

    Returns:
      var_core: dict with z_bar, Phi, Omega, variable_names, const,
                residual_correlation, equation_r2, estimation
      res: statsmodels VARResults
      data: estimation sample DataFrame
    """
    from statsmodels.tsa.api import VAR

    data = _load_var_dataset(csv_path=csv_path, columns=columns)
    res = VAR(data).fit(maxlags=1, ic=None, trend=trend)

    Phi = np.asarray(res.coefs[0], dtype=float)
    Omega = np.asarray(res.sigma_u, dtype=float)

    if "const" in res.params.index:
        const = np.asarray(res.params.loc["const"], dtype=float)
    else:
        const = np.zeros(len(columns), dtype=float)

    # Use sample means as z_bar (CCV convention, footnote 5).
    # OLS intercept + (I-Phi)^{-1} gives a biased stationary mean when
    # state variables have near-unit-root persistence.  Setting z_bar to
    # sample means and recomputing const = (I-Phi) @ z_bar keeps OLS
    # slopes and covariances intact while centering the grid correctly.
    z_bar = data.mean().to_numpy()
    const = (np.eye(len(columns)) - Phi) @ z_bar

    y_true = data.iloc[1:, :].to_numpy()
    y_hat = res.fittedvalues.to_numpy()
    resid = res.resid.to_numpy()

    var_core = {
        "z_bar": z_bar,
        "Phi": Phi,
        "Omega": Omega,
        "variable_names": list(columns),
        "const": const,
        "residual_correlation": _safe_residual_correlation(resid),
        "equation_r2": _compute_r2_per_equation(y_true, y_hat, columns),
        "estimation": "unrestricted",
        "trend": trend,
    }

    return var_core, res, data


def estimate_restricted_var1_from_csv(csv_path, columns, state_indices, trend="c"):
    """
    Estimate restricted VAR(1) where lagged return variables are excluded.

    Restriction: only lagged state variables enter each equation.
    This implies return-lag columns in Phi are zero by construction.

    Returns:
      var_core: dict with z_bar, Phi, Omega, variable_names, const,
                residual_correlation, equation_r2, estimation
      fit_details: dict with coefficient table and residuals
      data: estimation sample DataFrame
    """
    import pandas as pd

    data = _load_var_dataset(csv_path=csv_path, columns=columns)

    state_indices = np.asarray(state_indices, dtype=int)
    n = len(columns)
    if np.any(state_indices < 0) or np.any(state_indices >= n):
        raise ValueError("state_indices contains out-of-bounds index")
    if len(np.unique(state_indices)) != len(state_indices):
        raise ValueError("state_indices contains duplicates")

    state_cols = [columns[i] for i in state_indices]

    Y = data.iloc[1:, :].to_numpy()                 # z_{t+1}
    X_state = data[state_cols].iloc[:-1, :].to_numpy()  # lagged states z_t[state]
    T_eff = Y.shape[0]

    if trend == "c":
        X = np.column_stack([np.ones(T_eff), X_state])
        regressor_names = ["const"] + state_cols
    else:
        X = X_state
        regressor_names = state_cols

    # Multivariate OLS via least squares (equation-by-equation equivalent)
    # coeffs shape: (n_regressors, n_equations)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ coeffs
    resid = Y - Y_hat

    k_params = X.shape[1]
    dof = T_eff - k_params
    if dof <= 0:
        raise ValueError("Not enough observations for restricted VAR estimation")

    Omega = (resid.T @ resid) / dof

    if trend == "c":
        const = coeffs[0, :]
        slope_mat = coeffs[1:, :]   # rows correspond to state_cols
    else:
        const = np.zeros(n)
        slope_mat = coeffs

    Phi = np.zeros((n, n), dtype=float)
    for k, j in enumerate(state_indices):
        Phi[:, j] = slope_mat[k, :]

    # Use sample means as z_bar (CCV convention, footnote 5).
    # See estimate_var1_from_csv docstring for rationale.
    z_bar = data.mean().to_numpy()
    const = (np.eye(n) - Phi) @ z_bar

    coef_table = pd.DataFrame(coeffs, index=regressor_names, columns=columns)

    var_core = {
        "z_bar": z_bar,
        "Phi": Phi,
        "Omega": Omega,
        "variable_names": list(columns),
        "const": const,
        "residual_correlation": _safe_residual_correlation(resid),
        "equation_r2": _compute_r2_per_equation(Y, Y_hat, columns),
        "estimation": "restricted",
        "trend": trend,
        "state_predictor_columns": state_cols,
    }

    fit_details = {
        "coeff_table": coef_table,
        "residuals": resid,
        "n_obs_effective": T_eff,
        "dof": dof,
    }

    return var_core, fit_details, data


def build_var_config_from_dataset(
    csv_path,
    columns,
    state_indices,
    return_indices,
    bill_rate_index_in_state,
    annuity_yield_index_in_state,
    trend="c",
    estimation="restricted",
):
    """
    Estimate VAR(1) and package full var_config for build_model().

    estimation:
      - "restricted": lagged return variables excluded from all equations
      - "unrestricted": standard full VAR(1)
    """
    if estimation == "restricted":
        var_core, fit_obj, data = estimate_restricted_var1_from_csv(
            csv_path=csv_path,
            columns=columns,
            state_indices=state_indices,
            trend=trend,
        )
    elif estimation == "unrestricted":
        var_core, fit_obj, data = estimate_var1_from_csv(
            csv_path=csv_path,
            columns=columns,
            trend=trend,
        )
    else:
        raise ValueError("estimation must be either 'restricted' or 'unrestricted'")

    var_config = {
        **var_core,
        "state_indices": list(state_indices),
        "return_indices": list(return_indices),
        "bill_rate_index_in_state": int(bill_rate_index_in_state),
        "annuity_yield_index_in_state": int(annuity_yield_index_in_state),
    }

    ret_idx_arr = np.asarray(return_indices, dtype=int)
    var_config["max_abs_return_lag_coeff"] = float(np.max(np.abs(var_config["Phi"][:, ret_idx_arr])))

    print("=" * 64)
    print("VAR ESTIMATION SUMMARY")
    print("=" * 64)
    print(f"Estimation mode: {var_core['estimation']}")
    if hasattr(data.index, "min") and hasattr(data.index, "max"):
        try:
            print(f"Sample: {data.index.min().date()} to {data.index.max().date()}")
        except Exception:
            print(f"Sample rows: {len(data)}")
    else:
        print(f"Sample rows: {len(data)}")
    print(f"Columns: {columns}")
    print(f"State indices: {state_indices}")
    print(f"Return indices: {return_indices}")
    print(f"VAR k={len(columns)}, T={len(data) - 1}")
    print(f"Max |Phi[:, return_lag_cols]|: {var_config['max_abs_return_lag_coeff']:.3e}")
    print()
    print("Intercept vector (const):")
    print(np.round(var_config["const"], 6))
    print()
    print("Residual correlation matrix:")
    print(np.round(var_config["residual_correlation"], 4))
    print()
    print("Equation R2:")
    for c in columns:
        print(f"  {c}: {var_config['equation_r2'][c]:.4f}")
    print("=" * 64)

    return var_config, fit_obj, data


def build_tips_system2_var_config(
    csv_path="data/var_dataset.csv",
    state_indices=(0, 3, 4),
    return_indices=(1, 2),
    bill_rate_index_in_state=0,
    annuity_yield_index_in_state=2,
    trend="c",
    estimation="restricted",
):
    """
    Convenience wrapper for System 2: columns = [rtb, xr, xtips, dp, y_real].

    Default uses restricted estimation to align with state/return DP architecture.
    Set estimation="unrestricted" for diagnostic comparison.
    """
    columns = ["rtb", "xr", "xtips", "dp", "y_real"]
    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=state_indices,
        return_indices=return_indices,
        bill_rate_index_in_state=bill_rate_index_in_state,
        annuity_yield_index_in_state=annuity_yield_index_in_state,
        trend=trend,
        estimation=estimation,
    )


def build_nominal_system1_var_config(
    csv_path="data/var_dataset.csv",
    state_indices=(0, 3, 4),
    return_indices=(1, 2),
    bill_rate_index_in_state=0,
    annuity_yield_index_in_state=1,
    trend="c",
    estimation="restricted",
):
    """
    Nominal bond system (System 1).
    columns = [rtb, xr, xb, y_nom, dp]
    States: rtb(0), y_nom(3), dp(4)   Returns: xr(1), xb(2)
    Sample: 1962–2025, T=63 annual observations.

    Data is at ANNUAL frequency — returns are calendar-year sums of quarterly
    log returns, levels are Q4 (end-of-year) values. The VAR is estimated
    directly at annual frequency; no quarterly→annual compounding is needed.

    rtb   = annual real bill return (sum of 4 quarterly rtb)
    xr    = annual excess real stock return (sum of 4 quarterly xr)
    xb    = annual excess nominal bond return (sum of 4 quarterly xb)
    y_nom = 10-year nominal yield, Q4 value (SVENY10 / 100, annual decimal)
    dp    = log dividend-price ratio, Q4 value (log level)

    Partition indices: state_indices=[0,3,4], return_indices=[1,2],
    bill_rate_index_in_state=0.
    """
    columns = ["rtb", "xr", "xb", "y_nom", "dp"]
    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=state_indices,
        return_indices=return_indices,
        bill_rate_index_in_state=bill_rate_index_in_state,
        annuity_yield_index_in_state=annuity_yield_index_in_state,
        trend=trend,
        estimation=estimation,
    )


# =============================================================================
# ANNUAL VAR COMPOUNDING  (quarterly -> annual, for annual DP periods)
# =============================================================================

def annualize_var_config(var_config, h=4):
    """
    Convert a QUARTERLY var_config to an ANNUAL one (h=4 quarters per year).

    The annual DP (ages 25-80, beta=0.96, annual income profile) requires
    annual-frequency VAR parameters.  A quarterly VAR can be exactly
    compounded to annual frequency.

    IMPORTANT: The annual "return" is the SUM of h quarterly returns
    (e.g. R_annual = r_{t+1}+...+r_{t+4}), NOT the h-step-ahead return.
    This changes the return-loading matrix Phi_21 and the intercept c_r.

    Parameters
    ----------
    var_config : dict
        Quarterly var_config from build_nominal_system1_var_config() or
        build_var_config_from_dataset().
    h : int
        Number of quarters per period (default 4 = annual).

    Returns
    -------
    var_config_ann : dict
        Annual var_config ready to pass to build_model().
        z_bar, state_indices, return_indices, variable_names are preserved.

    Annual compounding formulas
    ---------------------------
    Let  n=n_var, s=state indices, r=return indices, h=4.
    Define P_k = Phi_11^k, C_k = sum_{j=0}^{k} P_j  (cumulative sums).

    State dynamics (exact):
        Phi_11_ann = Phi_11^h                                    (state AR)
        c_s_ann    = C_{h-1} @ c_s                               (state intercept)
        Omega_ss_ann = sum_{k=0}^{h-1} P_k @ Omega_ss @ P_k^T   (state innovation cov)

    Cumulative annual return (sum of h quarterly returns):
        Phi_21_ann = Phi_21 @ C_{h-1}                            (return loading)
        c_r_ann    = h*c_r + Phi_21 @ [sum_{k=1}^{h-1} C_{k-1}] @ c_s
                   (constant correction for intermediate state drift)

    Cross-covariance Cov(u_r_annual, u_s_annual):
        Omega_rs_ann = Phi_21 @ [sum_{k=1}^{h-1} C_{k-1} @ Omega_ss @ P_{h-1-k}^T]
                     + Omega_rs @ C_{h-1}^T
        where the first term comes from state noise propagating into cumulative
        returns via intermediate periods, and the second from the contemporaneous
        eps_s / eps_r correlation at each quarter.

    Return innovation variance Var(u_r_annual):
        Omega_rr_ann = Phi_21 @ [sum_{k=1}^{h-1} C_{k-1} @ Omega_ss @ C_{k-1}^T] @ Phi_21^T
                     + h*Omega_rr
                     + Phi_21 @ [sum_{k=1}^{h-1} C_{k-1}] @ Omega_sr
                     + Omega_rs @ [sum_{k=1}^{h-1} C_{k-1}]^T @ Phi_21^T

    Stationary mean z_bar: UNCHANGED (invariant to time aggregation).

    Verification: E[R_ann | z_bar_s, z_bar_s] = h*(c_r + Phi_21 @ z_bar_s)
    i.e. the mean annual return is exactly h times the mean quarterly return.
    """
    Phi_q   = np.asarray(var_config["Phi"],  dtype=float)
    Omega_q = np.asarray(var_config["Omega"], dtype=float)
    c_q     = np.asarray(var_config["const"], dtype=float)
    z_bar   = np.asarray(var_config["z_bar"], dtype=float)

    s_idx = list(var_config["state_indices"])
    r_idx = list(var_config["return_indices"])
    ns, nr = len(s_idx), len(r_idx)

    # Extract quarterly blocks
    Phi_11  = Phi_q[np.ix_(s_idx, s_idx)]
    Phi_21  = Phi_q[np.ix_(r_idx, s_idx)]
    Omega_ss = Omega_q[np.ix_(s_idx, s_idx)]
    Omega_rs = Omega_q[np.ix_(r_idx, s_idx)]
    Omega_rr = Omega_q[np.ix_(r_idx, r_idx)]
    Omega_sr = Omega_rs.T
    c_s = c_q[s_idx]
    c_r = c_q[r_idx]

    # -- Power and cumulative-sum sequences ------------------------------------
    # P[k] = Phi_11^k,  C[k] = sum_{j=0}^{k} P[j]  for k = 0..h-1
    In = np.eye(ns)
    P = [In.copy()]            # P[0] = I
    for _ in range(h - 1):
        P.append(P[-1] @ Phi_11)

    C = [In.copy()]            # C[0] = I = P[0]
    for k in range(1, h):
        C.append(C[k-1] + P[k])

    # -- Annual state dynamics -------------------------------------------------
    Phi_11_ann = P[h-1] @ Phi_11      # Phi_11^h
    c_s_ann    = C[h-1] @ c_s

    Omega_ss_ann = sum(P[k] @ Omega_ss @ P[k].T for k in range(h))

    # -- Annual return mapping -------------------------------------------------
    Phi_21_ann = Phi_21 @ C[h-1]

    # c_r_ann: constant correction for intermediate state drift
    # c_r_ann = h*c_r + Phi_21 @ (sum_{k=1}^{h-1} C[k-1]) @ c_s
    # When evaluated at z_bar_s: gives h*(c_r + Phi_21@z_bar_s) exactly.
    sum_C_for_cr = sum(C[k-1] for k in range(1, h))   # C[0]+C[1]+...+C[h-2]
    c_r_ann = h * c_r + Phi_21 @ sum_C_for_cr @ c_s

    # -- Annual cross-covariance Omega_rs -------------------------------------
    # Omega_rs_ann = Phi_21 @ [sum_{k=1}^{h-1} C[k-1] @ Omega_ss @ P[h-1-k]^T]
    #              + Omega_rs @ C[h-1]^T
    # First term: state-noise path (eps_s_{t+k} -> s_{t+k} -> r_{t+k+..} -> u_r)
    #             & (eps_s_{t+k} -> s_{4(t+1)} via P[h-1-k])
    # Second term: contemporaneous eps_s / eps_r correlation at each quarter
    inner_rs = sum(C[k-1] @ Omega_ss @ P[h-1-k].T for k in range(1, h))
    Omega_rs_ann = Phi_21 @ inner_rs + Omega_rs @ C[h-1].T

    # -- Annual return variance Omega_rr ---------------------------------------
    # Variance from state-path contributions (k=1..h-1):
    inner_rr = sum(C[k-1] @ Omega_ss @ C[k-1].T for k in range(1, h))
    # Cross terms (state eps and return eps within same quarter):
    sum_C_for_cross = sum(C[k-1] for k in range(1, h))   # same as sum_C_for_cr
    Omega_rr_ann = (
        Phi_21 @ inner_rr @ Phi_21.T
        + h * Omega_rr
        + Phi_21 @ sum_C_for_cross @ Omega_sr
        + Omega_rs @ sum_C_for_cross.T @ Phi_21.T
    )

    # -- Assemble full annual Phi, c, Omega ------------------------------------
    n = Phi_q.shape[0]
    Phi_ann   = np.zeros((n, n))
    c_ann     = np.zeros(n)
    Omega_ann = np.zeros((n, n))

    # State block
    for i, si in enumerate(s_idx):
        for j, sj in enumerate(s_idx):
            Phi_ann[si, sj]   = Phi_11_ann[i, j]
            Omega_ann[si, sj] = Omega_ss_ann[i, j]
        c_ann[si] = c_s_ann[i]

    # Return block
    for i, ri in enumerate(r_idx):
        for j, sj in enumerate(s_idx):
            Phi_ann[ri, sj]   = Phi_21_ann[i, j]
            Omega_ann[ri, sj] = Omega_rs_ann[i, j]
            Omega_ann[sj, ri] = Omega_rs_ann[i, j]   # symmetry
        for j, rj in enumerate(r_idx):
            Omega_ann[ri, rj] = Omega_rr_ann[i, j]
        c_ann[ri] = c_r_ann[i]

    # -- Verification ----------------------------------------------------------
    z_bar_s = z_bar[s_idx]
    mean_r_q = c_r + Phi_21 @ z_bar_s
    mean_r_a = c_r_ann + Phi_21_ann @ z_bar_s
    ratio = mean_r_a / mean_r_q if np.all(np.abs(mean_r_q) > 1e-12) else np.ones(nr) * h
    if not np.allclose(ratio, h, atol=1e-8):
        import warnings
        warnings.warn(
            f"annualize_var_config: mean-return ratio differs from h={h}: {ratio}",
            stacklevel=2,
        )

    var_config_ann = dict(var_config)    # shallow copy (scalar fields preserved)
    var_config_ann["Phi"]   = Phi_ann.tolist()
    var_config_ann["const"] = c_ann.tolist()
    var_config_ann["Omega"] = Omega_ann.tolist()
    # z_bar is invariant; state_indices, return_indices, variable_names unchanged
    var_config_ann["_annualized"] = True
    var_config_ann["_periods_per_annual"] = h

    print(f"annualize_var_config: compounded {h} quarters to 1 annual period.")
    print(f"  Phi_11 diagonal (annual): {np.diag(Phi_11_ann)}")
    mean_q_pct = mean_r_q * 100
    mean_a_pct = mean_r_a * 100
    ret_names = [var_config["variable_names"][i] for i in r_idx]
    for i, rn in enumerate(ret_names):
        print(f"  {rn}: quarterly mean={mean_q_pct[i]:.2f}%  annual mean={mean_a_pct[i]:.2f}%  ratio={ratio[i]:.4f}")

    return var_config_ann


# =============================================================================
# HARDCODED VAR PARAMETERS  (fallback if var_dataset.csv is unavailable)
# =============================================================================
# Estimated from var_dataset.csv using build_nominal_system1_var_config()
#
#   System   : Nominal Bond (System 1)
#   Columns  : [rtb, xr, xb, y_nom, dp]
#   States   : rtb(0), y_nom(3), dp(4)   Returns: xr(1), xb(2)
#   Estimation: restricted VAR(1), lagged returns excluded from all equations
#   Omega    : from restricted VAR residuals
#   Frequency: ANNUAL (estimated directly on annual data)
#   Sample   : 1962–2025, T=63 annual observations
#   Data src : GSW feds200628 (NSS params), TB3MS (FRED), CPIAUCSL (FRED),
#              Shiller RTRP + ie_data.xls
#
# Returns are calendar-year sums of quarterly log returns.
# Levels (y_nom, dp) are Q4 (end-of-year) values.
# These parameters are ready for the annual DP solver — no compounding needed.
# =============================================================================

# Variable order: [rtb, xr, xb, y_nom, dp]
_NOM_COLS   = ["rtb", "xr", "xb", "y_nom", "dp"]
_STATE_IDX  = [0, 3, 4]   # rtb, y_nom, dp
_RET_IDX    = [1, 2]      # xr, xb

# --- Unconditional means (sample means, CCV convention) ---
# z_bar = sample mean of each variable over 1962–2025 (T=64).
# Using sample means instead of (I-Phi)^{-1} @ const_OLS avoids
# the amplification of near-unit-root estimation error (see CCV
# footnote 5).  const is then recomputed as (I-Phi) @ z_bar.
_Z_BAR = np.array([
     +6.973252977844e-03,   # rtb   = +0.70%/yr real bill rate
     +5.492004676593e-02,   # xr    = +5.49%/yr excess stock return
     +1.945019508430e-02,   # xb    = +1.95%/yr excess nominal bond return
     +5.777224036114e-02,   # y_nom = 5.78% p.a. 10-year nominal yield (annual decimal)
     -3.670204273186e+00,   # dp    = -3.670    log dividend-price ratio
])

# --- Intercept vector c = (I - Phi) @ z_bar  (annual) ---
_CONST = np.array([
     -8.015300644532e-02,   # rtb
     +6.806375126678e-01,   # xr
     -3.624744795491e-01,   # xb
     +3.901470405114e-02,   # y_nom
     -5.675806782898e-01,   # dp
])

# --- AR(1) coefficient matrix Phi  (annual, restricted: return-lag columns = 0) ---
# Rows = z_{t+1} equations, Cols = z_t predictors
# Phi[i, j] = coefficient on lagged z_j in equation for z_i
# Return columns (1=xr, 2=xb) are zero by restriction.
#
#          L.rtb        L.xr   L.xb    L.y_nom       L.dp
_PHI = np.array([
    [ +4.317806756640e-01,  0.0,  0.0,  +5.533331663342e-01,  -1.420848622420e-02],  # rtb
    [ +9.941864729943e-01,  0.0,  0.0,  -2.554989215210e+00,  +1.321568753455e-01],  # xr
    [ +9.498930261866e-01,  0.0,  0.0,  +1.133431192929e+00,  -8.441491205617e-02],  # xb
    [ -8.366904702718e-02,  0.0,  0.0,  +8.712629906926e-01,  +8.444716115983e-03],  # y_nom
    [ -1.736269954260e+00,  0.0,  0.0,  +2.055355527366e+00,  +8.744087249498e-01],  # dp
])

# --- Residual covariance matrix Omega  (annual) ---
# Diagonal std devs: rtb=1.65%  xr=15.96%  xb=10.49%  y_nom=1.02%  dp=16.82%
#          rtb          xr           xb           y_nom        dp
_OMEGA = np.array([
    [ +2.716807621819e-04,  +5.044381831999e-04,  +1.761932079930e-04,  -2.646935561100e-05,  -5.120640342789e-04],  # rtb
    [ +5.044381831999e-04,  +2.546064446589e-02,  +2.673592303313e-03,  -2.396349328200e-04,  -2.527017286652e-02],  # xr
    [ +1.761932079930e-04,  +2.673592303313e-03,  +1.100365536980e-02,  -1.056337198359e-03,  -1.775978636589e-03],  # xb
    [ -2.646935561100e-05,  -2.396349328200e-04,  -1.056337198359e-03,  +1.040288234819e-04,  +1.595932821758e-04],  # y_nom
    [ -5.120640342789e-04,  -2.527017286652e-02,  -1.775978636589e-03,  +1.595932821758e-04,  +2.830159094619e-02],  # dp
])


def build_nominal_system1_var_config_hardcoded():
    """
    Fallback: return var_config using hardcoded parameter estimates.
    Use when var_dataset.csv is unavailable.
    Identical structure to build_nominal_system1_var_config() output.

    Parameters are at ANNUAL frequency — ready for the annual DP solver.
    """
    print("Using HARDCODED VAR parameters (nominal System 1, annual, T=63).")
    print("  Sample: 1962–2025 (annual)")
    return {
        "z_bar":                  _Z_BAR.copy(),
        "Phi":                    _PHI.copy(),
        "Omega":                  _OMEGA.copy(),
        "const":                  _CONST.copy(),
        "variable_names":         list(_NOM_COLS),
        "state_indices":          list(_STATE_IDX),
        "return_indices":         list(_RET_IDX),
        "bill_rate_index_in_state": 0,
        "annuity_yield_index_in_state": 1,   # y_nom is 2nd state variable [rtb, y_nom, dp]
        "max_abs_return_lag_coeff": 0.0,
        "estimation":             "restricted_hardcoded",
        "trend":                  "c",
        "state_predictor_columns": ["rtb", "y_nom", "dp"],
        "residual_correlation":   None,
        "equation_r2":            None,
    }
