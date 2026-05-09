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


def _read_var_sample_means(csv_path, cols):
    """Read sample means of given columns from var_dataset.csv (for scalar fallbacks)."""
    data = _load_var_dataset(csv_path=csv_path, columns=cols)
    return {c: float(data[c].mean()) for c in cols}


# =============================================================================
# REAL-YIELD VAR BUILDERS  (post 2026-05-08 rebuild)
# =============================================================================
# Three systems on the new real-yield CSV with columns
#     [cape, spr, y_1, xr, xb]
#
# All systems use the same restricted constrained estimator:
#   §2.2.r — Phi_1[:, ret_cols] = 0 by construction
#   §2.2.μ — Phi_0 = (I − Phi_1) μ̂_y (mean-pinning via demean+OLS+back-solve)
#
#   Full system   state = (cape, spr, y_1)         3-state    HEADLINE
#   System 2      state = (spr, y_1)               2-state
#   System 1      state = (y_1,)                   1-state
#
# State ordering convention: the slowest persistent state goes first
# (cape), the fast-mean-reverting yield-curve refinement target goes last
# (y_1). System 2 follows the same convention with cape removed.
# =============================================================================

DEFAULT_CSV_PATH = "data/var_dataset.csv"


def build_real_full_var_config(
    csv_path=DEFAULT_CSV_PATH,
    trend="c",
    estimation="restricted",
):
    """HEADLINE: Full real-yield system. State = (cape, spr, y_1), returns = (xr, xb).

    Columns in CSV: [cape, spr, y_1, xr, xb] (positions 0..4).
    State indices: [0, 1, 2] (cape at row 0, spr at row 1, y_1 at row 2).
    Return indices: [3, 4].

    Restrictions imposed by the estimator:
      §2.2.r: lagged xr, xb columns of Phi are zero exactly.
      §2.2.μ: Phi_0 = (I − Phi_1) μ̂_y by mean-pinning.

    Real-yield setup: AR(1)-matched 10-year RLONG baseline. January states use
    inflation information through December t-1. The one-year bill yield
    subtracts the AR(1) one-year inflation forecast; the 10-year Shiller RLONG
    yield subtracts the same AR(1)'s average 10-year forecast. See
    data/build_var_dataset_ar1_10y.py.

    cape = -log(Shiller CAPE), Jan-of-year, from ie_data.xls. Equity predictor.

    rtb_index_in_state is None: the bill is real-risk-free in this setup;
    the solver should read log_R_bill from y_1 directly (handled by the
    solver-rebuild, not this VAR layer).

    Returns `(var_config, fit_details, data)`.
    """
    columns = ["cape", "spr", "y_1", "xr", "xb"]
    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=(0, 1, 2),               # cape, spr, y_1
        return_indices=(3, 4),                 # xr, xb
        y_1_index_in_state=2,                  # y_1 at state row 2
        spr_index_in_state=1,                  # spr at state row 1
        rtb_index_in_state=None,               # bill is real-risk-free; no rtb state
        trend=trend,
        estimation=estimation,
    )


def build_real_system2_var_config(
    csv_path=DEFAULT_CSV_PATH,
    trend="c",
    estimation="restricted",
):
    """System 2 ablation: state = (spr, y_1), returns = (xr, xb). cape dropped.

    Columns loaded from CSV: ["spr", "y_1", "xr", "xb"] (positions 0..3).
    State indices: [0, 1] (spr at row 0, y_1 at row 1).
    Return indices: [2, 3].

    cape is dropped from the VAR. Equity predictability is reduced to whatever
    spr and y_1 capture jointly. Used as an ablation system to study
    portfolio choice when the dp/cape channel for stock-return prediction
    is removed.
    """
    columns = ["spr", "y_1", "xr", "xb"]
    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=(0, 1),                  # spr, y_1
        return_indices=(2, 3),                 # xr, xb
        y_1_index_in_state=1,
        spr_index_in_state=0,
        rtb_index_in_state=None,
        trend=trend,
        estimation=estimation,
    )


def build_real_system1_var_config(
    csv_path=DEFAULT_CSV_PATH,
    trend="c",
    estimation="restricted",
):
    """System 1 ablation: state = (y_1,), returns = (xr, xb). spr and cape dropped.

    Columns loaded: ["y_1", "xr", "xb"] (positions 0..2).
    State indices: [0] (y_1).
    Return indices: [1, 2].

    Most reduced ablation: only the real short rate predicts. spread is
    supplied as a scalar fallback (sample mean) for the bequest annuity
    factor in lifecycle.precompute.
    """
    means = _read_var_sample_means(csv_path, ["spr"])
    columns = ["y_1", "xr", "xb"]
    return build_var_config_from_dataset(
        csv_path=csv_path,
        columns=columns,
        state_indices=(0,),                    # y_1 only
        return_indices=(1, 2),                 # xr, xb
        y_1_index_in_state=0,
        spr_index_in_state=None,               # spr supplied as scalar fallback
        rtb_index_in_state=None,
        spr_scalar_fallback=means["spr"],
        trend=trend,
        estimation=estimation,
    )


# =============================================================================
# HARDCODED FALLBACK PARAMETERS for the FULL system
# =============================================================================
# Estimated from data/var_dataset.csv (1920-2011, T=92) under the AR(1)-matched
# 10-year RLONG real-yield baseline. January t yields use inflation information
# through December t-1. The baseline bond leg is a 10-year constant-duration
# return built from Shiller RLONG.
#
#   Columns         : [cape, spr, y_1, xr, xb]
#   State indices   : [0, 1, 2]    (cape, spr, y_1)
#   Return indices  : [3, 4]       (xr, xb)
#   Restrictions    : §2.2.r (lagged xr, xb -> 0) + §2.2.μ (mean-pinned)
#   Stationarity    : max |eig(Phi)| = 0.9296
#   R2 by equation  : cape=0.80, spr=0.37, y_1=0.65, xr=0.10, xb=0.13
# =============================================================================

_FULL_COLS  = ["cape", "spr", "y_1", "xr", "xb"]
_STATE_IDX  = [0, 1, 2]
_RET_IDX    = [3, 4]

_Z_BAR = np.array([
    -2.727442332889471910e+00,   # cape  = -log(Shiller CAPE)
    +7.180901395220774461e-03,   # spr   = +0.72pp  real spread
    +2.022602699339295929e-02,   # y_1   = +2.02%   real short yield
    +5.184427359929520002e-02,   # xr    = +5.18%   excess stock return
    +6.143380025557598940e-03,   # xb    = +0.61%   excess real bond return
])

# Phi[i, j] = coefficient on lagged z_j in equation for z_i. Restricted: xr, xb columns = 0.
_PHI = np.array([
    [+8.759362478655253748e-01, -5.163931341757732607e-01, +4.441976985185568799e-02, +0.000000000000000000e+00, +0.000000000000000000e+00],  # cape
    [-2.818301196100140519e-03, +6.616657057278459986e-01, +5.591716885498079076e-02, +0.000000000000000000e+00, +0.000000000000000000e+00],  # spr
    [+4.122109969344095112e-03, +2.476827528735099371e-01, +8.798696810275914437e-01, +0.000000000000000000e+00, +0.000000000000000000e+00],  # y_1
    [+1.081430908280127667e-01, +7.102820514848678934e-01, -1.028677026482095247e+00, +0.000000000000000000e+00, +0.000000000000000000e+00],  # xr
    [-1.018710825647135658e-02, +1.504733123895914071e+00, +4.058602057859728163e-01, +0.000000000000000000e+00, +0.000000000000000000e+00],  # xb
])

_CONST = (np.eye(5) - _PHI) @ _Z_BAR

_OMEGA = np.array([
    [+3.740610670869618654e-02, +4.994846251472590632e-04, -3.048967629412476432e-04, -3.426422321715918984e-02, -1.024576436349333534e-03],  # cape
    [+4.994846251472590632e-04, +2.169655764353713395e-04, -2.279620378420310330e-04, -1.735279347423760116e-04, +1.085369725852409600e-04],  # spr
    [-3.048967629412476432e-04, -2.279620378420310330e-04, +3.120208525031241214e-04, +2.267369326126219341e-05, -6.172400383356130239e-04],  # y_1
    [-3.426422321715918984e-02, -1.735279347423760116e-04, +2.267369326126219341e-05, +3.412626603147003029e-02, +8.061635851310839252e-04],  # xr
    [-1.024576436349333534e-03, +1.085369725852409600e-04, -6.172400383356130239e-04, +8.061635851310839252e-04, +3.620976983901966120e-03],  # xb
])


def build_real_full_var_config_hardcoded():
    """Fallback for the full headline system when var_dataset.csv is unavailable.

    AR(1)-matched 10-year RLONG real-yield setup, state = (cape, spr, y_1),
    returns = (xr, xb).
    Estimated from data/var_dataset.csv (1920-2011, T=92).
      max |eig(Phi)| = 0.9296, R2(xb) = 0.129.
    """
    print("Using HARDCODED VAR parameters (FULL real-yield system, 5-var, 1920-2011 T=92).")
    return {
        "z_bar":                  _Z_BAR.copy(),
        "Phi":                    _PHI.copy(),
        "Omega":                  _OMEGA.copy(),
        "const":                  _CONST.copy(),
        "variable_names":         list(_FULL_COLS),
        "state_indices":          list(_STATE_IDX),
        "return_indices":         list(_RET_IDX),
        "y_1_index_in_state":     2,
        "spr_index_in_state":     1,
        "rtb_index_in_state":     None,
        "y_1_scalar_fallback":    None,
        "spr_scalar_fallback":    None,
        "max_abs_return_lag_coeff": 0.0,
        "estimation":             "restricted_constrained_hardcoded",
        "trend":                  "c",
        "state_predictor_columns": ["cape", "spr", "y_1"],
        "residual_correlation":   None,
        "equation_r2":            None,
    }


def zero_bond_excess_mean_var_config(var_config, target_mean=0.0):
    """Return a copy of ``var_config`` with the bond excess-return mean shifted.

    This is an experiment knob, not the headline calibration. It preserves the
    bond-return dynamics around the mean: the ``xb`` row of ``Phi`` and the full
    residual covariance ``Omega`` are left unchanged. Only the unconditional
    mean of ``xb`` is shifted, and ``const = (I - Phi) z_bar`` is recomputed so
    the restricted, mean-pinned VAR remains internally coherent.

    With ``target_mean=0`` the bond has zero unconditional excess log return
    but keeps the same state dependence, volatility, and covariance structure.
    """
    cfg = dict(var_config)
    z_bar = np.asarray(var_config["z_bar"], dtype=float).copy()
    Phi = np.asarray(var_config["Phi"], dtype=float).copy()
    Omega = np.asarray(var_config["Omega"], dtype=float).copy()

    names = list(var_config["variable_names"])
    if "xb" not in names:
        raise ValueError("zero_bond_excess_mean_var_config requires an 'xb' variable")
    xb_idx = names.index("xb")
    old_mean = float(z_bar[xb_idx])
    z_bar[xb_idx] = float(target_mean)

    cfg["z_bar"] = z_bar
    cfg["Phi"] = Phi
    cfg["Omega"] = Omega
    cfg["const"] = (np.eye(len(z_bar)) - Phi) @ z_bar
    cfg["estimation"] = f"{var_config.get('estimation', 'unknown')}_xb_mean_shift"
    cfg["xb_original_mean"] = old_mean
    cfg["xb_target_mean"] = float(target_mean)
    return cfg


def _var_stationary_covariance(Phi, Omega):
    """Stationary covariance of ``z' = const + Phi z + eps``."""
    Phi = np.asarray(Phi, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    k = Phi.shape[0]
    vec = np.linalg.solve(
        np.eye(k * k) - np.kron(Phi, Phi),
        Omega.reshape(k * k, order="F"),
    )
    cov = vec.reshape((k, k), order="F")
    return 0.5 * (cov + cov.T)


def zero_bond_simple_excess_mean_var_config(var_config, target_simple_mean=0.0):
    """Return a copy of ``var_config`` with zero expected simple bond excess return.

    The VAR return variable ``xb`` is a log excess return. Under the local
    lognormal approximation, setting ``E[exp(xb)] - 1`` to ``target_simple_mean``
    requires:

        E[xb] = log(1 + target_simple_mean) - 0.5 * Var(xb).

    The variance is the VAR-implied stationary variance of ``xb``. As with
    ``zero_bond_excess_mean_var_config``, only the mean/intercept changes; the
    ``xb`` row of ``Phi`` and all covariance entries stay unchanged.
    """
    names = list(var_config["variable_names"])
    if "xb" not in names:
        raise ValueError("zero_bond_simple_excess_mean_var_config requires an 'xb' variable")
    xb_idx = names.index("xb")
    cov = _var_stationary_covariance(var_config["Phi"], var_config["Omega"])
    xb_var = float(cov[xb_idx, xb_idx])
    if 1.0 + float(target_simple_mean) <= 0.0:
        raise ValueError("target_simple_mean must be greater than -1")
    target_log_mean = float(np.log1p(target_simple_mean) - 0.5 * xb_var)
    cfg = zero_bond_excess_mean_var_config(var_config, target_mean=target_log_mean)
    cfg["xb_stationary_variance"] = xb_var
    cfg["xb_target_simple_mean"] = float(target_simple_mean)
    cfg["xb_target_log_mean"] = target_log_mean
    return cfg


def build_real_full_var_config_zero_bond_excess_mean(target_mean=0.0):
    """Full real-yield VAR experiment with zero mean bond excess return.

    Same state vector and same ``xb`` dynamics as the hardcoded headline VAR,
    except the unconditional mean of ``xb`` is set to ``target_mean``.
    """
    cfg = build_real_full_var_config_hardcoded()
    out = zero_bond_excess_mean_var_config(cfg, target_mean=target_mean)
    print(
        "Using ZERO-BOND-PREMIUM experiment: "
        f"E[xb] {out['xb_original_mean'] * 100:+.3f}pp -> "
        f"{out['xb_target_mean'] * 100:+.3f}pp; dynamics/covariance unchanged."
    )
    return out


def build_real_full_var_config_zero_bond_simple_excess_mean(target_simple_mean=0.0):
    """Full real-yield VAR experiment with zero simple bond excess return.

    Same state vector and same ``xb`` dynamics as the hardcoded headline VAR,
    except the unconditional mean of ``xb`` is shifted so
    ``E[exp(xb)] - 1 = target_simple_mean`` under the stationary lognormal
    approximation.
    """
    cfg = build_real_full_var_config_hardcoded()
    out = zero_bond_simple_excess_mean_var_config(
        cfg, target_simple_mean=target_simple_mean
    )
    print(
        "Using ZERO-SIMPLE-BOND-PREMIUM experiment: "
        f"E[xb] {out['xb_original_mean'] * 100:+.3f}pp -> "
        f"{out['xb_target_log_mean'] * 100:+.3f}pp, "
        f"E[exp(xb)]-1 -> {out['xb_target_simple_mean'] * 100:+.3f}pp; "
        "dynamics/covariance unchanged."
    )
    return out
