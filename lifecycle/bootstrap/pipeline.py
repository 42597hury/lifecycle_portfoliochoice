"""One bootstrap iteration: resample -> AR(1) -> construction VAR -> EH ->
lambda-loaded returns -> final lifecycle VAR.

All estimators are in-memory copies of the production equivalents from
`build_var_dataset_ar1_10y.py`, `build_var_dataset_real_eh_lambda.py`,
and `lifecycle.var`. The identity test (resample=identity, conditional
AR(1)) reproduces the production CSV outputs to floating-point precision,
which is the contract these copies preserve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .donor_panel import DonorPanel

N_BOND = 10
FINAL_COLUMNS = ("cape", "spr", "y_1", "xr", "xb")
STATE_INDICES = (0, 1, 2)
RETURN_INDICES = (3, 4)
XB_INDEX_IN_FINAL = 4


# =============================================================================
# AR(1) inflation
# =============================================================================

def fit_ar1_inplace(pi: np.ndarray) -> dict[str, float]:
    pi = np.asarray(pi, dtype=float)
    x = pi[:-1]
    y = pi[1:]
    phi = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    intercept = float(y.mean() - phi * x.mean())
    mu = intercept / (1.0 - phi) if abs(1.0 - phi) > 1e-12 else float("nan")
    return {"intercept": intercept, "phi": phi, "mu": mu}


def expected_average_inflation_arr(
    pi_state: np.ndarray, ar1: dict[str, float], n: int
) -> np.ndarray:
    phi = float(ar1["phi"])
    mu = float(ar1["mu"])
    if abs(1.0 - phi) < 1e-12:
        multiplier = 1.0
    else:
        multiplier = phi * (1.0 - phi**n) / (1.0 - phi) / n
    return mu + multiplier * (pi_state - mu)


# =============================================================================
# Construction VAR and EH yield
# =============================================================================

def estimate_construction_var(y_1: np.ndarray, spr_obs: np.ndarray) -> dict:
    z = np.column_stack([y_1, spr_obs]).astype(float)
    z_bar = z.mean(axis=0)
    Z = z - z_bar
    Y_mat, X_mat = Z[1:], Z[:-1]
    coeffs, *_ = np.linalg.lstsq(X_mat, Y_mat, rcond=None)
    Phi = coeffs.T
    resid = Y_mat - X_mat @ coeffs
    dof = Y_mat.shape[0] - X_mat.shape[1]
    Omega = (resid.T @ resid) / dof
    eigvals = np.linalg.eigvals(Phi)
    return {
        "z_bar": z_bar,
        "Phi": Phi,
        "Omega": Omega,
        "max_abs_eig": float(np.max(np.abs(eigvals))),
        "nobs": z.shape[0],
    }


def compute_eh_real_long_yield_arr(
    y_1: np.ndarray, spr_obs: np.ndarray, var_result: dict, n_bond: int = N_BOND
) -> np.ndarray:
    z_bar = var_result["z_bar"]
    Phi = var_result["Phi"]
    k = len(z_bar)
    H1 = np.zeros(k)
    H1[0] = 1.0
    sum_row = np.zeros(k)
    PhiJ = np.eye(k)
    for _ in range(n_bond):
        sum_row += H1 @ PhiJ
        PhiJ = Phi @ PhiJ
    z = np.column_stack([y_1, spr_obs]).astype(float)
    z_demean = z - z_bar
    return z_bar[0] + (1.0 / n_bond) * (z_demean @ sum_row)


# =============================================================================
# CLM bond return
# =============================================================================

def clm_return_from_log_yield_arr(
    y_log: np.ndarray, n: int = N_BOND
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (r_bond, duration), both length len(y_log).

    r_bond[t] = D[t-1]*y_log[t-1] - (D[t-1]-1)*y_log[t] for t>=1.
    r_bond[0] is NaN (no previous yield).
    """
    Y = np.exp(y_log) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 1.0 + Y
        D_raw = (1.0 - g ** (-n)) / (1.0 - g ** (-1))
    D = np.where(np.abs(Y) >= 1e-8, D_raw, float(n))
    r_next = np.full_like(y_log, np.nan)
    r_next[:-1] = D[:-1] * y_log[:-1] - (D[:-1] - 1.0) * y_log[1:]
    r_bond = np.full_like(y_log, np.nan)
    r_bond[1:] = r_next[:-1]
    return r_bond, D


# =============================================================================
# Final lifecycle VAR (restricted and restricted_eh), in-memory
# =============================================================================

def _estimate_restricted_inmemory(
    Z_mat: np.ndarray, state_indices: tuple[int, ...]
) -> dict:
    """Mirror of estimate_restricted_var1_from_csv on an in-memory (T, n) matrix."""
    n = Z_mat.shape[1]
    z_bar = Z_mat.mean(axis=0)
    Z = Z_mat - z_bar
    Y = Z[1:, :]
    state_idx = np.asarray(state_indices, dtype=int)
    X = Z[:-1, state_idx]
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Phi = np.zeros((n, n), dtype=float)
    for k, j in enumerate(state_idx):
        Phi[:, j] = coeffs[k, :]
    const = (np.eye(n) - Phi) @ z_bar
    resid = Y - X @ coeffs
    dof = Y.shape[0] - X.shape[1]
    if dof <= 0:
        raise ValueError("Not enough observations for restricted VAR estimation")
    Omega = (resid.T @ resid) / dof
    return {"z_bar": z_bar, "Phi": Phi, "Omega": Omega, "const": const}


def _estimate_restricted_eh_inmemory(
    Z_mat: np.ndarray, state_indices: tuple[int, ...], xb_index: int
) -> dict:
    """Mirror of estimate_restricted_eh_var1_from_csv on an in-memory (T, n) matrix."""
    n = Z_mat.shape[1]
    z_bar = Z_mat.mean(axis=0)
    z_bar[xb_index] = 0.0
    Z = Z_mat - z_bar
    Y = Z[1:, :]
    state_idx = np.asarray(state_indices, dtype=int)
    X = Z[:-1, state_idx]
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Phi = np.zeros((n, n), dtype=float)
    for k, j in enumerate(state_idx):
        Phi[:, j] = coeffs[k, :]
    Phi[xb_index, :] = 0.0
    const = (np.eye(n) - Phi) @ z_bar
    Y_hat = X @ coeffs
    Y_hat[:, xb_index] = 0.0
    resid = Y - Y_hat
    dof = Y.shape[0] - X.shape[1]
    if dof <= 0:
        raise ValueError("Not enough observations for restricted_eh VAR estimation")
    Omega = (resid.T @ resid) / dof
    return {"z_bar": z_bar, "Phi": Phi, "Omega": Omega, "const": const}


def estimate_final_var(
    Z_mat: np.ndarray, lam: float, state_indices=STATE_INDICES, xb_index=XB_INDEX_IN_FINAL
) -> dict:
    if lam == 0.0:
        return _estimate_restricted_eh_inmemory(Z_mat, state_indices, xb_index)
    return _estimate_restricted_inmemory(Z_mat, state_indices)


# =============================================================================
# One bootstrap iteration
# =============================================================================

@dataclass
class IterationResult:
    draw_id: int
    inflation_mode: str
    block_length_mean: float
    ar1_phi: float
    ar1_mu: float
    construction_max_eig: float
    Phi_R: np.ndarray
    z_bar_R: np.ndarray
    final_results: dict  # lam_val -> per-lambda dict
    final_var_failed: bool
    error: str | None = None


def _per_lambda_block(
    lam: float,
    y_1: np.ndarray,
    y_10_EH: np.ndarray,
    TP_R: np.ndarray,
    pi_info: np.ndarray,
    log_stock_gross: np.ndarray,
    cape_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-lambda final-VAR matrix and return (Z, y_10_lam).

    Z has columns (cape, spr, y_1, xr, xb) and is dropna'd via the leading NaN
    rule (xr and xb need y_1[t-1] and the bond return needs y_10_lam[t+1]).
    """
    y_10_lam = y_10_EH + lam * TP_R
    spr_lam = y_10_lam - y_1
    r_bond, _ = clm_return_from_log_yield_arr(y_10_lam, n=N_BOND)

    y_1_lag = np.concatenate([[np.nan], y_1[:-1]])
    xr = log_stock_gross - pi_info - y_1_lag
    xb = r_bond - y_1_lag

    cape_log = -np.log(cape_raw)

    full = np.column_stack([cape_log, spr_lam, y_1, xr, xb])
    keep = ~np.any(np.isnan(full), axis=1)
    return full[keep], y_10_lam


def run_iteration(
    donor_arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    lambdas: tuple[float, ...],
    inflation_mode: str,
    production_ar1: dict[str, float],
    draw_id: int = 0,
    block_length_mean: float = 8.0,
) -> IterationResult:
    """Run one bootstrap iteration on resampled donor rows `idx`.

    `inflation_mode`:
      - "refit": re-estimate the inflation AR(1) on the resampled pi_info.
      - "conditional": hold the production AR(1) fixed at its full-sample value.
    """
    if inflation_mode not in {"refit", "conditional"}:
        raise ValueError(f"inflation_mode must be 'refit' or 'conditional', got {inflation_mode!r}")

    R = donor_arrays["R"][idx]
    RLONG = donor_arrays["RLONG"][idx]
    CAPE = donor_arrays["CAPE"][idx]
    pi_info = donor_arrays["pi_info"][idx]
    log_stock_gross = donor_arrays["log_stock_gross"][idx]

    if inflation_mode == "refit":
        ar1 = fit_ar1_inplace(pi_info)
    else:
        ar1 = {
            "intercept": float(production_ar1["intercept"]),
            "phi": float(production_ar1["phi"]),
            "mu": float(production_ar1["mu"]),
        }

    if not np.isfinite(ar1["mu"]):
        return IterationResult(
            draw_id=draw_id,
            inflation_mode=inflation_mode,
            block_length_mean=block_length_mean,
            ar1_phi=ar1["phi"],
            ar1_mu=float("nan"),
            construction_max_eig=float("nan"),
            Phi_R=np.full((2, 2), np.nan),
            z_bar_R=np.full(2, np.nan),
            final_results={lam: None for lam in lambdas},
            final_var_failed=True,
            error="ar1_unit_root",
        )

    E_pi_1 = expected_average_inflation_arr(pi_info, ar1, 1)
    E_pi_n = expected_average_inflation_arr(pi_info, ar1, N_BOND)

    y_1_nom = np.log1p(R / 100.0)
    y_n_nom = np.log1p(RLONG / 100.0)
    y_1 = y_1_nom - E_pi_1
    y_n_real = y_n_nom - E_pi_n
    spr_obs = y_n_real - y_1

    try:
        cvar = estimate_construction_var(y_1, spr_obs)
    except Exception as e:
        return IterationResult(
            draw_id=draw_id,
            inflation_mode=inflation_mode,
            block_length_mean=block_length_mean,
            ar1_phi=ar1["phi"],
            ar1_mu=ar1["mu"],
            construction_max_eig=float("nan"),
            Phi_R=np.full((2, 2), np.nan),
            z_bar_R=np.full(2, np.nan),
            final_results={lam: None for lam in lambdas},
            final_var_failed=True,
            error=f"construction_var: {e!r}",
        )

    y_10_EH = compute_eh_real_long_yield_arr(y_1, spr_obs, cvar)
    TP_R = y_n_real - y_10_EH

    final_results: dict[float, dict | None] = {}
    final_var_failed = False
    for lam in lambdas:
        try:
            Z, y_10_lam = _per_lambda_block(
                lam=lam,
                y_1=y_1,
                y_10_EH=y_10_EH,
                TP_R=TP_R,
                pi_info=pi_info,
                log_stock_gross=log_stock_gross,
                cape_raw=CAPE,
            )
            if Z.shape[0] < 10:
                final_results[lam] = None
                final_var_failed = True
                continue
            fvar = estimate_final_var(Z, lam=lam)

            Phi_state = fvar["Phi"][np.ix_(STATE_INDICES, STATE_INDICES)]
            state_max_eig = float(np.max(np.abs(np.linalg.eigvals(Phi_state))))

            xb_col = Z[:, XB_INDEX_IN_FINAL]
            xr_col = Z[:, 3]
            spr_col = Z[:, 1]

            sd_xb = float(np.std(xb_col, ddof=1))
            mean_xb = float(np.mean(xb_col))
            sd_xr = float(np.std(xr_col, ddof=1))
            mean_xr = float(np.mean(xr_col))
            sharpe_xb = (mean_xb + 0.5 * sd_xb * sd_xb) / sd_xb if sd_xb > 0 else float("nan")
            sharpe_xr = (mean_xr + 0.5 * sd_xr * sd_xr) / sd_xr if sd_xr > 0 else float("nan")

            phi_full = fvar["Phi"]
            phi_spr_spr = float(phi_full[1, 1])
            halflife = float("nan")
            if 0.0 < abs(phi_spr_spr) < 1.0:
                halflife = -float(np.log(2.0)) / float(np.log(abs(phi_spr_spr)))

            Omega = fvar["Omega"]
            sd_v_xb = float(np.sqrt(Omega[XB_INDEX_IN_FINAL, XB_INDEX_IN_FINAL]))
            sd_v_xr = float(np.sqrt(Omega[3, 3]))
            sd_v_y1 = float(np.sqrt(Omega[2, 2]))
            sd_v_spr = float(np.sqrt(Omega[1, 1]))
            var_v_y10 = Omega[1, 1] + Omega[2, 2] + 2.0 * Omega[1, 2]
            sd_v_y10 = float(np.sqrt(var_v_y10)) if var_v_y10 > 0 else float("nan")

            def _corr(cov_ij: float, sd_i: float, sd_j: float) -> float:
                if sd_i > 0 and sd_j > 0:
                    return float(cov_ij / (sd_i * sd_j))
                return float("nan")

            corr_vxr_vxb = _corr(Omega[3, 4], sd_v_xr, sd_v_xb)
            cov_vy10_vxb = Omega[1, 4] + Omega[2, 4]
            corr_vy10_vxb = _corr(cov_vy10_vxb, sd_v_y10, sd_v_xb)

            final_results[lam] = {
                "Phi": phi_full,
                "z_bar": fvar["z_bar"],
                "Omega": Omega,
                "T_lam": int(Z.shape[0]),
                "E_spr": float(np.mean(spr_col)),
                "sd_spr": float(np.std(spr_col, ddof=1)),
                "E_xb": mean_xb,
                "sd_xb": sd_xb,
                "Sharpe_xb": sharpe_xb,
                "E_xr": mean_xr,
                "sd_xr": sd_xr,
                "Sharpe_xr": sharpe_xr,
                "Phi_xb_cape": float(phi_full[XB_INDEX_IN_FINAL, 0]),
                "Phi_xb_spr": float(phi_full[XB_INDEX_IN_FINAL, 1]),
                "Phi_xb_y1": float(phi_full[XB_INDEX_IN_FINAL, 2]),
                "Phi_xr_cape": float(phi_full[3, 0]),
                "Phi_xr_spr": float(phi_full[3, 1]),
                "Phi_xr_y1": float(phi_full[3, 2]),
                "Phi_spr_spr": phi_spr_spr,
                "spr_halflife": halflife,
                "state_max_eig": state_max_eig,
                "sd_v_xb": sd_v_xb,
                "corr_vxr_vxb": corr_vxr_vxb,
                "corr_vy10_vxb": corr_vy10_vxb,
            }
        except Exception as e:
            final_results[lam] = None
            final_var_failed = True
            final_results.setdefault("_errors", {})[lam] = repr(e)

    return IterationResult(
        draw_id=draw_id,
        inflation_mode=inflation_mode,
        block_length_mean=block_length_mean,
        ar1_phi=ar1["phi"],
        ar1_mu=ar1["mu"],
        construction_max_eig=cvar["max_abs_eig"],
        Phi_R=cvar["Phi"],
        z_bar_R=cvar["z_bar"],
        final_results=final_results,
        final_var_failed=final_var_failed,
        error=None,
    )


def iteration_to_row(result: IterationResult, lam: float) -> dict:
    fr = result.final_results.get(lam)
    base = {
        "draw_id": result.draw_id,
        "lambda_val": float(lam),
        "inflation_mode": result.inflation_mode,
        "block_length_mean": float(result.block_length_mean),
        "ar1_phi": float(result.ar1_phi),
        "ar1_mu": float(result.ar1_mu),
        "Phi_R_00": float(result.Phi_R[0, 0]),
        "Phi_R_01": float(result.Phi_R[0, 1]),
        "Phi_R_10": float(result.Phi_R[1, 0]),
        "Phi_R_11": float(result.Phi_R[1, 1]),
        "z_bar_R_y1": float(result.z_bar_R[0]),
        "z_bar_R_spr": float(result.z_bar_R[1]),
        "construction_max_eig": float(result.construction_max_eig),
        "final_var_failed": int(fr is None),
    }
    nan_fields = [
        "T_lam", "E_spr", "sd_spr", "E_xb", "sd_xb", "Sharpe_xb",
        "E_xr", "sd_xr", "Sharpe_xr",
        "Phi_xb_cape", "Phi_xb_spr", "Phi_xb_y1",
        "Phi_xr_cape", "Phi_xr_spr", "Phi_xr_y1",
        "Phi_spr_spr", "spr_halflife", "state_max_eig",
        "sd_v_xb", "corr_vxr_vxb", "corr_vy10_vxb",
    ]
    if fr is None:
        for k in nan_fields:
            base[k] = float("nan")
        base["stable_flag"] = 0
        base["Phi_xb_spr_pos"] = 0
        return base
    for k in nan_fields:
        base[k] = fr[k]
    base["stable_flag"] = int(
        np.isfinite(result.construction_max_eig)
        and result.construction_max_eig < 1.0
        and np.isfinite(fr["state_max_eig"])
        and fr["state_max_eig"] < 1.0
    )
    base["Phi_xb_spr_pos"] = int(fr["Phi_xb_spr"] > 0.0)
    return base
