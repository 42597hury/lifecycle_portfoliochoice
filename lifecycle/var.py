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


def _safe_residual_correlation(resid):
    corr = np.corrcoef(resid, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _compute_r2_per_equation_demeaned(Y, Y_hat, columns):
    """R² for demeaned regressions (no intercept, already centred)."""
    out = {}
    for i, col in enumerate(columns):
        sse = float(np.sum((Y[:, i] - Y_hat[:, i]) ** 2))
        sst = float(np.sum(Y[:, i] ** 2))          # demeaned, so TSS = sum(y_tilde^2)
        out[col] = 1.0 - sse / max(sst, 1e-14)
    return out


def estimate_var1_from_csv(csv_path, columns, state_indices=None, trend="c"):
    """
    Estimate unrestricted VAR(1) from CSV using all lagged variables.

    Uses the CCV constrained estimator: z_bar is pinned to the sample mean,
    data is demeaned, Phi is estimated without intercept, and const is
    recovered as (I - Phi) @ z_bar.

    If state_indices is provided, only lagged state columns are used as
    regressors (restricted VAR). Otherwise all columns are used.

    Returns:
      var_core: dict with z_bar, Phi, Omega, variable_names, const,
                residual_correlation, equation_r2, estimation
      fit_details: dict with residuals and diagnostics
      data: estimation sample DataFrame
    """
    import pandas as pd

    data = _load_var_dataset(csv_path=csv_path, columns=columns)
    n = len(columns)

    # 1. Sample mean over ALL rows (the restriction target).
    z_bar = data.mean(axis=0).to_numpy()

    # 2. Demean.
    Z = data.to_numpy() - z_bar          # shape (T, n)
    Y = Z[1:, :]                          # z_tilde_{t+1}, shape (T-1, n)

    if state_indices is not None:
        state_idx = np.asarray(state_indices, dtype=int)
        X = Z[:-1, state_idx]             # demeaned lagged state cols only
    else:
        state_idx = np.arange(n)
        X = Z[:-1, :]                     # all lagged cols

    # 3. OLS WITHOUT intercept (it's pinned by the constraint).
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (n_pred, n)

    Phi = np.zeros((n, n), dtype=float)
    for k, j in enumerate(state_idx):
        Phi[:, j] = coeffs[k, :]
    # Non-predictor columns of Phi remain zero.

    # 4. Recover const so the implied mean equals z_bar by construction.
    const = (np.eye(n) - Phi) @ z_bar

    # Residuals and Omega from the constrained model
    Y_hat = X @ coeffs
    resid = Y - Y_hat
    dof = Y.shape[0] - X.shape[1]
    if dof <= 0:
        raise ValueError("Not enough observations for constrained VAR estimation")
    Omega = (resid.T @ resid) / dof

    estimation_label = "restricted_constrained" if state_indices is not None else "unrestricted_constrained"

    var_core = {
        "z_bar": z_bar,
        "Phi": Phi,
        "Omega": Omega,
        "variable_names": list(columns),
        "const": const,
        "residual_correlation": _safe_residual_correlation(resid),
        "equation_r2": _compute_r2_per_equation_demeaned(Y, Y_hat, columns),
        "estimation": estimation_label,
        "trend": trend,
    }
    if state_indices is not None:
        var_core["state_predictor_columns"] = [columns[i] for i in state_idx]

    fit_details = {
        "residuals": resid,
        "n_obs_effective": Y.shape[0],
        "dof": dof,
    }

    return var_core, fit_details, data


def estimate_restricted_var1_from_csv(csv_path, columns, state_indices, trend="c"):
    """
    Estimate restricted VAR(1) where lagged return variables are excluded.

    Uses the CCV constrained estimator (pins z_bar to sample mean).
    Restriction: only lagged state variables enter each equation.
    This implies return-lag columns in Phi are zero by construction.

    Returns:
      var_core: dict with z_bar, Phi, Omega, variable_names, const,
                residual_correlation, equation_r2, estimation
      fit_details: dict with residuals and diagnostics
      data: estimation sample DataFrame
    """
    return estimate_var1_from_csv(
        csv_path=csv_path,
        columns=columns,
        state_indices=state_indices,
        trend=trend,
    )


def build_var_config_from_dataset(
    csv_path,
    columns,
    state_indices,
    return_indices,
    y_1_index_in_state=3,
    spr_index_in_state=1,
    rtb_index_in_state=2,
    y_1_scalar_fallback=None,
    spr_scalar_fallback=None,
    trend="c",
    estimation="restricted",
):
    """
    Estimate VAR(1) and package full var_config for build_model().

    Uses the CCV constrained estimator (pins z_bar to sample mean).

    estimation:
      - "restricted": lagged return variables excluded from all equations
      - "unrestricted": standard full VAR(1) (still constrained z_bar)

    rtb_index_in_state:
      Position of rtb in the state vector when rtb is on the grid (post the
      rtb-as-state migration). Pass ``None`` when rtb is in the return block
      (legacy 3-state systems and the iid System I).
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
        "y_1_index_in_state": None if y_1_index_in_state is None else int(y_1_index_in_state),
        "spr_index_in_state": None if spr_index_in_state is None else int(spr_index_in_state),
        "rtb_index_in_state": None if rtb_index_in_state is None else int(rtb_index_in_state),
        "y_1_scalar_fallback": None if y_1_scalar_fallback is None else float(y_1_scalar_fallback),
        "spr_scalar_fallback": None if spr_scalar_fallback is None else float(spr_scalar_fallback),
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


def build_nominal_system1_var_config(
    csv_path="data/var_dataset.csv",
    state_indices=(2, 1, 3, 0),
    return_indices=(4, 5),
    y_1_index_in_state=3,
    spr_index_in_state=1,
    rtb_index_in_state=2,
    trend="c",
    estimation="restricted",
):
    """
    Nominal bond system with no riskless asset.
    columns = [y_1, spr, dp, rtb, xr, xb]
    Default state: (dp, spr, rtb, y_1)   Default returns: (xr, xb)

    rtb is in the state block (post rtb-as-state migration). The restriction
    "lagged returns do not predict" applies only to {xr, xb}; lagged rtb is
    freely estimated and captures inflation persistence.

    Default state ordering rationale (dp, spr, rtb, y_1) — same partition slot
    used previously for cy. The orthogonality and refinement-target reasoning
    referenced cy-specific innovation correlations; with dp the structure is
    different but the slot ordering is preserved for continuity. The orthogonality
    properties of dp's innovation should be re-evaluated if axis-by-axis tail
    refinement is being tuned.
      - dp at row 0: replaces cy as the slowest persistent state.
      - y_1 at row 3: M[xb, y_1] dominant entry of M = Sigma_rs @ Sigma_ss^-1.
      - rtb in the middle: avoids adjacent placement of strongly correlated
        (spr, y_1).

    Data is at ANNUAL frequency. The VAR is estimated directly at annual
    frequency using the CCV constrained estimator (z_bar pinned to sample mean).

    y_1  = 1-year nominal short rate, January-of-year value (decimal log)
    spr  = yield spread: y_n (Moody's AAA) - y_1
    dp   = log dividend-price ratio: log(D) - log(P)
    rtb  = real bill return: y_1[t-1] - pi[t]   <-- state variable
    xr   = excess log stock return over nominal bill
    xb   = excess log bond return over nominal bill (CLM constant-duration n=20)

    With the default ordering, state_grid[:, 0] = dp, [:, 1] = spr,
    [:, 2] = rtb, [:, 3] = y_1.

    Migration history: pre-2026-04-30 used (y_1, spr, cy). 2026-04-30 used
    (cy, spr, y_1) with rtb in the return block. 2026-05-06 added rtb-as-state.
    2026-05-07 (this version) replaced cy=-log(CAPE) on 1963-2025 with
    dp=log(D)-log(P) on 1919-2011 (Option A: AAA-throughout, T=92 effective).
    """
    columns = ["y_1", "spr", "dp", "rtb", "xr", "xb"]
    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=state_indices,
        return_indices=return_indices,
        y_1_index_in_state=y_1_index_in_state,
        spr_index_in_state=spr_index_in_state,
        rtb_index_in_state=rtb_index_in_state,
        trend=trend,
        estimation=estimation,
    )


def _read_y1_spread_sample_means(csv_path):
    """Read sample means of y_1 and spread from the VAR dataset."""
    data = _load_var_dataset(csv_path=csv_path, columns=["y_1", "spr"])
    return float(data["y_1"].mean()), float(data["spr"].mean())


def build_no_cy_var_config(
    csv_path="data/var_dataset.csv",
    columns=("y_1", "spr", "rtb", "xr", "xb"),
    state_indices=(2, 1, 0),
    trend="c",
    estimation="restricted",
):
    """Build the System III var_config: rtb, spread, and y_1 predict returns; cy dropped.

    Post rtb-as-state migration: state vector is (rtb, spr, y_1) — rtb at row 0
    (most orthogonal of the three state innovations; mean |rho|=0.385), spr in
    the middle, y_1 at the refinement-target end. Returns are (xr, xb).

    Returns the same tuple shape as the existing baseline builder:
    `(var_config, fit_details, data)`.
    """
    state_idx_list = list(state_indices)
    column_at_state_pos = [columns[i] for i in state_idx_list]
    spr_pos = column_at_state_pos.index("spr")
    y_1_pos = column_at_state_pos.index("y_1")
    rtb_pos = column_at_state_pos.index("rtb")

    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=list(columns),
        state_indices=state_idx_list,
        return_indices=(3, 4),
        y_1_index_in_state=y_1_pos,
        spr_index_in_state=spr_pos,
        rtb_index_in_state=rtb_pos,
        trend=trend,
        estimation=estimation,
    )


def build_rtb_y1_var_config(
    csv_path="data/var_dataset.csv",
    columns=("y_1", "rtb", "xr", "xb"),
    trend="c",
    estimation="restricted",
):
    """Build the System II var_config: rtb and y_1 predict returns.

    Post rtb-as-state migration: state vector is (rtb, y_1) — rtb at row 0
    (more orthogonal of the two state innovations), y_1 at the refinement-target
    end. Returns are (xr, xb). Spread is supplied as a scalar fallback equal to
    its sample mean (used by the bequest annuity factor).

    Returns `(var_config, fit_details, data)`.
    """
    _, spr_mean = _read_y1_spread_sample_means(csv_path)

    columns = list(columns)
    rtb_pos = columns.index("rtb")
    y_1_pos = columns.index("y_1")

    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=(rtb_pos, y_1_pos),  # rtb -> state row 0, y_1 -> row 1
        return_indices=(2, 3),
        y_1_index_in_state=1,
        spr_index_in_state=None,
        rtb_index_in_state=0,
        spr_scalar_fallback=spr_mean,
        trend=trend,
        estimation=estimation,
    )


def build_iid_var_config(
    csv_path="data/var_dataset.csv",
    return_columns=("xr", "xb"),
):
    """Build the System I var_config: iid returns; rtb is iid in the state block.

    Post rtb-as-state migration: rtb cannot live in the return block any more
    because the solver/simulator read rtb from the next-period state vector.
    System I encodes "no predictability" by setting Phi to zero — rtb is then
    an iid draw around its sample mean with variance Sigma_rr[rtb, rtb], which
    matches the iid-returns interpretation.

    State = (rtb,). Returns = (xr, xb). Phi = 0; Omega = sample covariance.
    Spread and y_1 are scalar fallbacks (sample means) for the bequest annuity.

    Returns `(var_config, fit_details, data)`.
    """
    data = _load_var_dataset(
        csv_path=csv_path,
        columns=["y_1", "spr", "rtb", *return_columns],
    )
    full_cols = ["rtb", *return_columns]
    full_data = data[full_cols].to_numpy()
    z_bar_full = full_data.mean(axis=0)
    Sigma_full = np.cov(full_data, rowvar=False, ddof=1)
    y_1_mean, spr_mean = _read_y1_spread_sample_means(csv_path)

    n = len(full_cols)  # rtb + n_ret
    Phi = np.zeros((n, n), dtype=float)
    Omega = np.array(Sigma_full, dtype=float)
    z_bar = np.array(z_bar_full, dtype=float)
    const = (np.eye(n) - Phi) @ z_bar

    var_config = {
        "z_bar": z_bar,
        "Phi": Phi,
        "Omega": Omega,
        "variable_names": list(full_cols),
        "const": const,
        "residual_correlation": None,
        "equation_r2": None,
        "state_indices": [0],            # rtb at state row 0
        "return_indices": list(range(1, n)),
        "y_1_index_in_state": None,      # y_1 not on the grid
        "spr_index_in_state": None,
        "rtb_index_in_state": 0,
        "y_1_scalar_fallback": y_1_mean,
        "spr_scalar_fallback": spr_mean,
        "max_abs_return_lag_coeff": 0.0,
        "estimation": "iid_system_I",
        "trend": "sample_mean_cov",
        "state_predictor_columns": ["rtb"],
    }
    fit_details = {
        "residuals": full_data - z_bar_full,
        "n_obs_effective": len(full_data),
        "dof": len(full_data) - 1,
    }
    return var_config, fit_details, data


# =============================================================================
# HARDCODED VAR PARAMETERS  (fallback if var_dataset.csv is unavailable)
# =============================================================================
# Estimated from var_dataset.csv using build_nominal_system1_var_config()
#
#   System   : Nominal Bond — No Riskless Asset (rtb-as-state, dp variable)
#   Columns  : [y_1, spr, dp, rtb, xr, xb]
#   States   : dp(2), spr(1), rtb(3), y_1(0)   Returns: xr(4), xb(5)
#   Estimation: restricted constrained VAR(1) (CCV w8566 footnote 5)
#               z_bar pinned to sample mean; only return lags (xr, xb)
#               excluded — rtb lags enter freely (inflation persistence
#               channel; Phi[rtb, rtb] ~ +0.47).
#   Omega    : from restricted VAR residuals
#   Frequency: ANNUAL (estimated directly on annual data)
#   Sample   : 1920–2011, T=92 annual observations  (Option A: window
#              shortened from spec's 1871–2011 to honor "Moody's AAA throughout"
#              given AAA starts 1919-01; one year lost to inflation differencing)
#   Data src : chap_26 (Shiller Ch.26 update — P, D, R, CPI Jan-of-year),
#              FRED Moody's AAA (long-bond yield, Jan-of-year)
#
# CCV w8566 reference Table 2 (annual, 1890-1998, T=109): Phi[y, y]=0.92,
# Phi[dp, dp]=0.84, Phi[rtb, rtb]=0.30, Phi[spr, spr]=0.82.
# =============================================================================

# Variable order: [y_1, spr, dp, rtb, xr, xb]
_NOM_COLS   = ["y_1", "spr", "dp", "rtb", "xr", "xb"]
_STATE_IDX  = [2, 1, 3, 0]   # dp, spr, rtb, y_1
_RET_IDX    = [4, 5]         # xr, xb

# --- Unconditional means (= sample means, pinned by CCV constrained estimator) ---
# Estimated 2026-05-07 on data/var_dataset.csv (1920-2011, T=92, Option A).
_Z_BAR = np.array([
    +4.402987766457664570e-02,   # y_1   = +4.40%   nominal short rate
    +1.255104651539200658e-02,   # spr   = +1.26pp  yield spread (AAA - y_1)
    -3.256591312095483914e+00,   # dp    = -3.257   log dividend-price ratio
    +1.641233073311164012e-02,   # rtb   = +1.64%   real bill return
    +5.184427359929520002e-02,   # xr    = +5.18%   excess stock return
    +1.073210114487195208e-02,   # xb    = +1.07%   excess bond return
])

# --- AR(1) coefficient matrix Phi (annual, restricted: only xr/xb lag cols = 0) ---
# Phi[i, j] = coefficient on lagged z_j in equation for z_i.
# Return columns 4 (xr) and 5 (xb) are zero by restriction; column 3 (rtb) is
# freely estimated.
#
#            L.y_1                    L.spr                    L.dp                     L.rtb                    L.xr   L.xb
_PHI = np.array([
    [+9.299060056743475577e-01, +8.871811111637421809e-02, +1.309239276869196859e-03, -4.879196624450553993e-02, 0.0, 0.0],  # y_1
    [+2.344008948027683989e-02, +6.570337478396850450e-01, -3.319193580131267474e-03, +2.526490968066123430e-02, 0.0, 0.0],  # spr
    [-3.853543505103990130e-01, -2.154249268041942500e+00, +9.288388594467132942e-01, -6.009056673164093754e-01, 0.0, 0.0],  # dp
    [+3.378214425295113510e-01, -4.876945057254646887e-01, +8.552260704384884301e-04, +4.716537319880246826e-01, 0.0, 0.0],  # rtb
    [-7.091371152108977283e-02, +3.482547439595059302e+00, +1.162966599772504023e-01, +5.973033322200087779e-03, 0.0, 0.0],  # xr
    [+3.491519517715134246e-01, +3.279032619175778152e+00, +1.821208810550401305e-02, +2.007972763738816169e-01, 0.0, 0.0],  # xb
])

# --- Intercept vector c = (I - Phi) @ z_bar  (annual) ---
_CONST = (np.eye(6) - _PHI) @ _Z_BAR

# --- Residual covariance matrix Omega  (annual, 6x6, full matrix) ---
_OMEGA = np.array([
    [+2.234920362525859695e-04, -1.661887167916548417e-04, -1.844912677731811845e-04, -1.313216553305527885e-04, +3.902042377503587406e-04, -5.253256292926793700e-04],  # y_1
    [-1.661887167916548417e-04, +1.492291942678782772e-04, +2.968638876824434138e-04, +6.420969497128877726e-05, -5.544060097427062255e-04, +1.739181278562723740e-04],  # spr
    [-1.844912677731811845e-04, +2.968638876824434138e-04, +2.073286760974539511e-02, +1.554822952480693467e-04, -2.066659106159631984e-02, -6.617591115692886836e-04],  # dp
    [-1.313216553305527885e-04, +6.420969497128877726e-05, +1.554822952480693467e-04, +1.304450404909156085e-03, -7.527973037346545430e-04, +6.461715739749695131e-04],  # rtb
    [+3.902042377503587406e-04, -5.544060097427062255e-04, -2.066659106159631984e-02, -7.527973037346545430e-04, +3.338076903150290903e-02, +1.467276318990530168e-03],  # xr
    [-5.253256292926793700e-04, +1.739181278562723740e-04, -6.617591115692886836e-04, +6.461715739749695131e-04, +1.467276318990530168e-03, +3.279686436158545451e-03],  # xb
])


def build_nominal_system1_var_config_hardcoded():
    """
    Fallback: return var_config using hardcoded parameter estimates.
    Use when var_dataset.csv is unavailable.
    Identical structure to build_nominal_system1_var_config() output.

    Parameters are at ANNUAL frequency — ready for the annual DP solver.

    Estimated from data/var_dataset.csv (1920-2011, T=92, Option A — see
    docs/CCV_RETURN_IMPLEMENT.md §4.1) under the rtb-as-state restriction.
    State (dp, spr, rtb, y_1); returns (xr, xb).
      max |eig(Phi)| = 0.9493 (stationary)
      R²(y_1)=0.78, R²(spr)=0.44, R²(dp)=0.90, R²(rtb)=0.41,
      R²(xr)=0.13,  R²(xb)=0.41.
    """
    print("Using HARDCODED VAR parameters (nominal System 1, annual, 6-var, dp-state).")
    print("  Estimated from data/var_dataset.csv (1920-2011, T=92, Option A).")
    return {
        "z_bar":                  _Z_BAR.copy(),
        "Phi":                    _PHI.copy(),
        "Omega":                  _OMEGA.copy(),
        "const":                  _CONST.copy(),
        "variable_names":         list(_NOM_COLS),
        "state_indices":          list(_STATE_IDX),
        "return_indices":         list(_RET_IDX),
        "y_1_index_in_state":     3,
        "spr_index_in_state":     1,
        "rtb_index_in_state":     2,
        "y_1_scalar_fallback":    None,
        "spr_scalar_fallback":    None,
        "max_abs_return_lag_coeff": 0.0,
        "estimation":             "restricted_constrained_hardcoded",
        "trend":                  "c",
        "state_predictor_columns": ["dp", "spr", "rtb", "y_1"],
        "residual_correlation":   None,
        "equation_r2":            None,
    }
