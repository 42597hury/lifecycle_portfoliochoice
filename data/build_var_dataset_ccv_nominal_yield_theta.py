"""Build CCV-style nominal-yield theta real-bond VAR datasets.

This is the cleaner Campbell-Chan-Viceira-style construction discussed in the
thesis notes. It keeps an observed nominal long yield in the construction, but
it removes external inflation-expectation models from the main real-yield step.
The default long yield is Shiller's RLONG series with n=10, but the command
line can switch to Moody's AAA and/or a different constant maturity horizon.

Construction VAR:

    z_t = [rtb_t, xr_t, y_1_nom,t, spr_nom,t, cape_t]

where rtb_t is the ex-post real bill return over [t-1,t], xr_t is stock excess
return over the nominal bill over [t-1,t], and y_1_nom/spr_nom are observed
nominal yield states at t.  The construction drops xb because xb is mechanically
created from the long-yield path after the real yield curve is chosen.

The construction VAR is mean-pinned, unrestricted VAR(1):

    z_{t+1} - z_bar = Phi (z_t - z_bar) + eps_{t+1}.

From this VAR:

    y_1_real,t    = E_t[rtb_{t+1}]
    y_n_real_EH,t = (1/n) sum_{j=1}^{n} E_t[rtb_{t+j}]
    y_n_nom_EH,t  = (1/n) sum_{j=0}^{n-1} E_t[y_1_nom,t+j]

The observed nominal long-yield residual is:

    TP_nom,t = y_n_nom_obs,t - y_n_nom_EH,t.

Under the CCV assumption that the inflation risk premium is zero or constant,
time variation in TP_nom,t is treated as the long-bond premium residual.  For
theta in [0, 1]:

    y_n_real_theta,t = y_n_real_EH,t + theta * TP_nom,t.

For each theta, the script recomputes:

    spr_theta,t = y_n_real_theta,t - y_1_real,t
    r_n_theta,t = n * y_n_real_theta,t-1 - (n-1) * y_n_real_theta,t
    xb_theta,t  = r_n_theta,t - y_1_real,t-1

This follows CCV (NBER 8566) Appendix C exactly. The hypothetical real bill
has yield y_1_real,t = E_t[rtb_{t+1}] and is locally risk-free in real terms:
its time-t yield equals its realized real return over [t, t+1], so the bill's
return next period is y_1_real,t. The hypothetical real long bond is a
zero-coupon claim with one-period log return r_{n,t+1} = n*y_t - (n-1)*y_{t+1}.
Excess return subtracts the bill's return, i.e. the lagged real bill yield.
The construction VAR's first variable rtb is used only for econometric
estimation (so that H_1 * E_t(z_{t+1}) returns expected real-bill returns);
it does not appear in the final excess-return formula.

Outputs are written under data/ccv_nominal_yield_scaling/ for the default
Shiller-10 build, and under source/maturity-specific directories otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_var_dataset_ar1_10y import (
    HERE,
    N_BOND,
    SAMPLE_END,
    SAMPLE_START,
    THESIS,
    load_cape_january,
    load_chap26,
    load_shiller_monthly,
    sharpe_jensen,
)
from build_var_dataset_term_premium_scale import (
    assert_contiguous_year_index,
    constant_maturity_return_from_log_yield,
    theta_label,
)


OUT_DIR = HERE / "ccv_nominal_yield_scaling"
AAA_CSV = THESIS / "AAA.csv"
DEFAULT_THETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
LONG_YIELD_SOURCES = ("shiller", "aaa")
CONSTRUCTION_COLS = ("rtb", "xr", "y_1_nom", "spr_nom", "cape")
FINAL_COLS = ("cape", "spr", "y_1", "xr", "xb")


def default_output_dir(long_yield_source: str, n_bond: int) -> Path:
    """Default output location for one long-yield source/maturity pair."""
    if long_yield_source == "shiller" and int(n_bond) == N_BOND:
        return OUT_DIR
    return HERE / f"ccv_nominal_yield_scaling_{long_yield_source}{int(n_bond)}"


def load_aaa_january() -> pd.Series:
    """Load Moody's AAA January yields in percent."""
    aaa = pd.read_csv(AAA_CSV)
    aaa["observation_date"] = pd.to_datetime(aaa["observation_date"])
    aaa["AAA"] = pd.to_numeric(aaa["AAA"], errors="coerce")
    aaa["year"] = aaa["observation_date"].dt.year
    aaa["month"] = aaa["observation_date"].dt.month
    jan = aaa[aaa["month"] == 1].copy()
    if jan.duplicated("year").any():
        raise ValueError("Multiple January observations per year in AAA.csv")
    return jan.set_index("year")["AAA"].rename("AAA")


def build_nominal_construction_panel(
    *,
    long_yield_source: str = "shiller",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the annual construction VAR panel and raw diagnostics.

    Row t uses returns realized over [t-1,t] and yield states observed at t.
    This mirrors the timing convention in the active dataset builder.
    """
    chap = load_chap26()
    shiller = load_shiller_monthly()
    cape_jan = load_cape_january(shiller)
    long_yield_source = str(long_yield_source).lower()
    if long_yield_source not in LONG_YIELD_SOURCES:
        raise ValueError(
            f"long_yield_source must be one of {LONG_YIELD_SOURCES}, "
            f"got {long_yield_source!r}"
        )

    df = chap.copy()
    df["CAPE"] = cape_jan.reindex(df.index)
    df["pi"] = np.log(df["CPI"] / df["CPI"].shift(1))
    df["y_1_nom"] = np.log1p(df["R"] / 100.0)
    if long_yield_source == "shiller":
        df["long_yield_pct"] = df["RLONG"]
    else:
        df["long_yield_pct"] = load_aaa_january().reindex(df.index)
    df["y_10_nom_obs"] = np.log1p(df["long_yield_pct"] / 100.0)
    df["spr_nom"] = df["y_10_nom_obs"] - df["y_1_nom"]
    df["cape"] = -np.log(df["CAPE"])
    df["long_yield_source"] = long_yield_source

    stock_gross = (df["P"] + df["D"]) / df["P"].shift(1)
    df["xr"] = np.log(stock_gross) - df["y_1_nom"].shift(1)
    df["rtb"] = df["y_1_nom"].shift(1) - df["pi"]

    z_panel = df.loc[:, list(CONSTRUCTION_COLS)].copy()
    diag = df.loc[:, [
        "P", "D", "R", "RLONG", "CPI", "CAPE", "pi",
        "long_yield_pct", "y_1_nom", "y_10_nom_obs", "spr_nom",
        "cape", "rtb", "xr",
    ]].copy()
    diag["long_yield_source"] = long_yield_source
    return z_panel, diag


def estimate_mean_pinned_var(sample: pd.DataFrame) -> dict[str, object]:
    """Mean-pinned unrestricted VAR(1) on all construction variables."""
    if list(sample.columns) != list(CONSTRUCTION_COLS):
        raise ValueError(f"Unexpected construction columns: {list(sample.columns)}")
    if len(sample) < len(CONSTRUCTION_COLS) + 5:
        raise ValueError("Not enough observations for construction VAR.")
    assert_contiguous_year_index(sample.iloc[:, 0], "CCV nominal construction VAR")

    z_bar = sample.mean(axis=0).to_numpy(dtype=float)
    Z = sample.to_numpy(dtype=float) - z_bar
    Y = Z[1:, :]
    X = Z[:-1, :]
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Phi = coeffs.T
    const = (np.eye(len(CONSTRUCTION_COLS)) - Phi) @ z_bar
    resid = Y - X @ coeffs
    dof = Y.shape[0] - X.shape[1]
    if dof <= 0:
        raise ValueError("Not enough degrees of freedom for construction VAR.")
    Sigma = (resid.T @ resid) / dof
    eigvals = np.linalg.eigvals(Phi)
    yhat = X @ coeffs
    r2 = {}
    for j, name in enumerate(CONSTRUCTION_COLS):
        sse = float(np.sum((Y[:, j] - yhat[:, j]) ** 2))
        sst = float(np.sum(Y[:, j] ** 2))
        r2[name] = 1.0 - sse / max(sst, 1e-14)

    return {
        "columns": tuple(CONSTRUCTION_COLS),
        "z_bar": z_bar,
        "Phi": Phi,
        "const": const,
        "Sigma": Sigma,
        "eigvals": eigvals,
        "max_abs_eig": float(np.max(np.abs(eigvals))),
        "r2": r2,
        "nobs": int(len(sample)),
        "sample_start": int(sample.index.min()),
        "sample_end": int(sample.index.max()),
    }


def average_expected_variable(
    full: pd.DataFrame,
    var: dict[str, object],
    variable: str,
    *,
    first_horizon: int,
    count: int,
) -> pd.Series:
    """Average VAR forecasts of one variable over h=first,...,first+count-1."""
    cols = list(var["columns"])
    idx = cols.index(variable)
    Phi = np.asarray(var["Phi"], dtype=float)
    z_bar = np.asarray(var["z_bar"], dtype=float)

    state = full.loc[:, cols].astype(float)
    valid = state.notna().all(axis=1).to_numpy()
    out = np.full(len(state), np.nan, dtype=float)
    if np.any(valid):
        Zc = state.to_numpy()[valid] - z_bar
        acc = np.zeros(int(valid.sum()), dtype=float)
        Phi_pow = np.eye(len(cols))
        last_horizon = first_horizon + count - 1
        for h in range(last_horizon + 1):
            if h >= first_horizon:
                forecast_h = z_bar + Zc @ Phi_pow.T
                acc += forecast_h[:, idx]
            Phi_pow = Phi_pow @ Phi
        out[valid] = acc / count
    return pd.Series(
        out,
        index=full.index,
        name=f"Eavg_{variable}_h{first_horizon}_{last_horizon}",
    )


def build_components(
    *,
    n_bond: int = N_BOND,
    long_yield_source: str = "shiller",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build full-year component table before theta slicing."""
    n_bond = int(n_bond)
    z_panel, raw = build_nominal_construction_panel(
        long_yield_source=long_yield_source,
    )
    sample = z_panel.loc[SAMPLE_START:SAMPLE_END].dropna().astype(float)
    var = estimate_mean_pinned_var(sample)

    comp = raw.join(z_panel, rsuffix="_z")
    comp["y_1_real"] = average_expected_variable(
        z_panel, var, "rtb", first_horizon=1, count=1
    )
    comp["y_10_real_EH"] = average_expected_variable(
        z_panel, var, "rtb", first_horizon=1, count=n_bond
    )
    comp["y_10_nom_EH"] = average_expected_variable(
        z_panel, var, "y_1_nom", first_horizon=0, count=n_bond
    )
    comp["pi_10_VAR"] = comp["y_10_nom_EH"] - comp["y_10_real_EH"]
    comp["term_prem_nom"] = comp["y_10_nom_obs"] - comp["y_10_nom_EH"]
    comp["y_10_real_obs_implied"] = comp["y_10_real_EH"] + comp["term_prem_nom"]
    comp["spr_real_obs_implied"] = comp["y_10_real_obs_implied"] - comp["y_1_real"]
    comp["maturity_n"] = n_bond
    comp["long_yield_source"] = str(long_yield_source).lower()

    return comp, var


def build_theta_dataset(
    components: pd.DataFrame,
    theta: float,
    *,
    n_bond: int = N_BOND,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one theta-specific lifecycle-ready dataset and diagnostics."""
    theta = float(theta)
    n_bond = int(n_bond)
    y_theta = components["y_10_real_EH"] + theta * components["term_prem_nom"]
    r_theta, duration_theta = constant_maturity_return_from_log_yield(
        y_theta, n_bond, zero_coupon=True
    )

    diag = components.copy()
    diag["theta"] = theta
    diag["maturity_n"] = n_bond
    diag["y_10_real_theta"] = y_theta
    diag["spr_theta"] = y_theta - components["y_1_real"]
    diag["r_10_real_theta"] = r_theta
    diag["duration_theta"] = duration_theta
    diag["xb_theta"] = r_theta - components["y_1_real"].shift(1)

    out = pd.DataFrame(
        {
            "cape": components["cape"],
            "spr": diag["spr_theta"],
            "y_1": components["y_1_real"],
            "xr": components["xr"],
            "xb": diag["xb_theta"],
        },
        index=components.index,
    )
    out = out.loc[SAMPLE_START:SAMPLE_END].dropna()
    diag = diag.loc[out.index]
    return out, diag


def verify_theta_dataset(
    theta: float,
    out: pd.DataFrame,
    diag: pd.DataFrame,
) -> list[tuple[str, bool, str]]:
    """Return verification checks for one theta dataset."""
    results = []

    def add(name: str, ok: bool, msg: str) -> None:
        results.append((name, ok, msg))

    expected_t = SAMPLE_END - SAMPLE_START + 1
    add("T", len(out) == expected_t, f"T={len(out)} expected {expected_t}")
    add("columns", list(out.columns) == list(FINAL_COLS), str(list(out.columns)))
    add("no_nan", not out.isna().any().any(), "output has no NaN")

    y_identity = diag["y_10_real_theta"] - (
        diag["y_10_real_EH"] + theta * diag["term_prem_nom"]
    )
    add("theta_yield_identity", float(y_identity.abs().max()) < 1e-14,
        f"max abs={float(y_identity.abs().max()):.3e}")

    spr_identity = out["spr"] - (diag["y_10_real_theta"] - out["y_1"])
    add("spr_identity", float(spr_identity.abs().max()) < 1e-14,
        f"max abs={float(spr_identity.abs().max()):.3e}")

    tp_identity = diag["term_prem_nom"] - (
        diag["y_10_nom_obs"] - diag["y_10_nom_EH"]
    )
    add("nominal_tp_identity", float(tp_identity.abs().max()) < 1e-14,
        f"max abs={float(tp_identity.abs().max()):.3e}")

    pi_identity = diag["pi_10_VAR"] - (
        diag["y_10_nom_EH"] - diag["y_10_real_EH"]
    )
    add("var_inflation_identity", float(pi_identity.abs().max()) < 1e-14,
        f"max abs={float(pi_identity.abs().max()):.3e}")

    xb_identity = out["xb"].iloc[1:] - (
        diag["r_10_real_theta"].iloc[1:] - out["y_1"].shift(1).iloc[1:]
    )
    add("xb_lagged_bill", float(xb_identity.abs().max()) < 1e-14,
        f"max abs={float(xb_identity.abs().max()):.3e}")

    return results


def summarize_theta(theta: float, out: pd.DataFrame, diag: pd.DataFrame) -> dict[str, float]:
    """Moment summary for one theta dataset."""
    xr_j, xr_sd, xr_sh = sharpe_jensen(out["xr"])
    xb_j, xb_sd, xb_sh = sharpe_jensen(out["xb"])
    return {
        "theta": float(theta),
        "mean_y_1": float(out["y_1"].mean()),
        "std_y_1": float(out["y_1"].std(ddof=1)),
        "mean_y_10": float(diag["y_10_real_theta"].mean()),
        "std_y_10": float(diag["y_10_real_theta"].std(ddof=1)),
        "mean_spr": float(out["spr"].mean()),
        "std_spr": float(out["spr"].std(ddof=1)),
        "mean_term_prem_nom_component": float((theta * diag["term_prem_nom"]).mean()),
        "std_term_prem_nom_component": float((theta * diag["term_prem_nom"]).std(ddof=1)),
        "mean_xr": float(out["xr"].mean()),
        "std_xr": float(out["xr"].std(ddof=1)),
        "jensen_xr": float(xr_j),
        "sharpe_xr": float(xr_sh),
        "mean_xb": float(out["xb"].mean()),
        "std_xb": float(xb_sd),
        "jensen_xb": float(xb_j),
        "sharpe_xb": float(xb_sh),
        "mean_duration": float(diag["duration_theta"].mean()),
    }


def construction_var_tables(var: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return coefficient and summary tables for the construction VAR."""
    cols = list(var["columns"])
    Phi = np.asarray(var["Phi"], dtype=float)
    const = np.asarray(var["const"], dtype=float)
    z_bar = np.asarray(var["z_bar"], dtype=float)

    rows = []
    for i, eq in enumerate(cols):
        rows.append({
            "equation": eq,
            "coefficient": "const",
            "value": float(const[i]),
        })
        rows.append({
            "equation": eq,
            "coefficient": "z_bar",
            "value": float(z_bar[i]),
        })
        for j, lag in enumerate(cols):
            rows.append({
                "equation": eq,
                "coefficient": f"lag_{lag}",
                "value": float(Phi[i, j]),
            })

    summary = pd.DataFrame([{
        "sample_start": var["sample_start"],
        "sample_end": var["sample_end"],
        "nobs": var["nobs"],
        "max_abs_eig": var["max_abs_eig"],
        **{f"r2_{k}": v for k, v in var["r2"].items()},
    }])
    return pd.DataFrame(rows), summary


def estimate_final_var_summary(out_dir: Path, thetas: list[float]) -> pd.DataFrame:
    """Estimate the lifecycle final VAR on each theta dataset."""
    repo = HERE.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lifecycle.var import build_real_full_var_config, partition_var

    rows = []
    for theta in thetas:
        label = theta_label(theta)
        csv_path = out_dir / f"var_dataset_theta_{label}.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            cfg, _fit, _data = build_real_full_var_config(csv_path=str(csv_path))
        parts = partition_var(
            Phi_full=np.asarray(cfg["Phi"], dtype=float),
            Omega_full=np.asarray(cfg["Omega"], dtype=float),
            z_bar=np.asarray(cfg["z_bar"], dtype=float),
            state_idx=cfg["state_indices"],
            ret_idx=cfg["return_indices"],
            variable_names=cfg["variable_names"],
            verbose=False,
        )
        Phi_ss = np.asarray(parts["Phi_11"], dtype=float)
        Omega = np.asarray(cfg["Omega"], dtype=float)
        eig = np.linalg.eigvals(Phi_ss)
        sigma_r_cond = np.asarray(parts["Sigma_r_cond"], dtype=float)
        M = np.asarray(parts["M"], dtype=float)
        r2 = cfg["equation_r2"]
        rows.append({
            "theta": float(theta),
            "max_abs_state_eig": float(np.max(np.abs(eig))),
            "min_eig_Omega": float(np.linalg.eigvalsh(Omega).min()),
            "cond_Omega": float(np.linalg.cond(Omega)),
            "max_abs_resid_corr": float(
                np.max(np.abs(np.asarray(cfg["residual_correlation"]) - np.eye(len(FINAL_COLS))))
            ),
            "r2_cape": float(r2["cape"]),
            "r2_spr": float(r2["spr"]),
            "r2_y_1": float(r2["y_1"]),
            "r2_xr": float(r2["xr"]),
            "r2_xb": float(r2["xb"]),
            "sigma_r_cond_xr": float(np.sqrt(sigma_r_cond[0, 0])),
            "sigma_r_cond_xb": float(np.sqrt(sigma_r_cond[1, 1])),
            "M_xb_cape": float(M[1, 0]),
            "M_xb_spr": float(M[1, 1]),
            "M_xb_y_1": float(M[1, 2]),
        })
    return pd.DataFrame(rows)


def write_outputs(
    thetas: list[float],
    *,
    out_dir: Path = OUT_DIR,
    n_bond: int = N_BOND,
    long_yield_source: str = "shiller",
    estimate_final_var: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Build all requested theta datasets and write diagnostics."""
    n_bond = int(n_bond)
    long_yield_source = str(long_yield_source).lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    components, var = build_components(
        n_bond=n_bond,
        long_yield_source=long_yield_source,
    )

    coef_table, var_summary = construction_var_tables(var)
    var_summary["maturity_n"] = n_bond
    var_summary["long_yield_source"] = long_yield_source
    coef_table.to_csv(out_dir / "construction_var_coefficients.csv", index=False)
    var_summary.to_csv(out_dir / "construction_var_summary.csv", index=False)

    component_cols = [
        "rtb", "xr", "cape", "y_1_nom", "spr_nom", "y_10_nom_obs",
        "y_1_real", "y_10_real_EH", "y_10_nom_EH", "pi_10_VAR",
        "term_prem_nom", "y_10_real_obs_implied", "spr_real_obs_implied",
        "long_yield_pct", "maturity_n", "long_yield_source",
    ]
    components.loc[SAMPLE_START:SAMPLE_END, component_cols].to_csv(
        out_dir / "ccv_nominal_components.csv"
    )

    moments = []
    theta_outputs: dict[float, pd.DataFrame] = {}
    all_checks = []
    for theta in thetas:
        out, diag = build_theta_dataset(components, theta, n_bond=n_bond)
        checks = verify_theta_dataset(theta, out, diag)
        all_checks.extend(
            {"theta": theta, "check": name, "ok": ok, "message": msg}
            for name, ok, msg in checks
        )
        if not all(ok for _name, ok, _msg in checks):
            failed = [f"{name}: {msg}" for name, ok, msg in checks if not ok]
            raise RuntimeError(f"theta={theta} verification failed: {failed}")

        label = theta_label(theta)
        out.index.name = "year"
        diag.index.name = "year"
        out.to_csv(out_dir / f"var_dataset_theta_{label}.csv")
        diag.to_csv(out_dir / f"diagnostics_theta_{label}.csv")
        moments.append(summarize_theta(theta, out, diag))
        theta_outputs[theta] = out

    moments_df = pd.DataFrame(moments)
    checks_df = pd.DataFrame(all_checks)
    moments_df.to_csv(out_dir / "ccv_nominal_theta_moments.csv", index=False)
    checks_df.to_csv(out_dir / "ccv_nominal_theta_checks.csv", index=False)

    final_summary = None
    if estimate_final_var:
        final_summary = estimate_final_var_summary(out_dir, thetas)
        final_summary.to_csv(out_dir / "ccv_nominal_final_var_summary.csv", index=False)

    return moments_df, var_summary, final_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CCV-style nominal-yield theta real-bond VAR datasets."
    )
    parser.add_argument(
        "--theta",
        nargs="+",
        type=float,
        default=list(DEFAULT_THETAS),
        help="Theta values to build. Default: 0 0.25 0.5 0.75 1.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: data/ccv_nominal_yield_scaling for "
            "Shiller/10y, otherwise data/ccv_nominal_yield_scaling_<source><n>."
        ),
    )
    parser.add_argument(
        "--long-yield-source",
        choices=LONG_YIELD_SOURCES,
        default="shiller",
        help="Nominal long-yield source. Default: shiller RLONG.",
    )
    parser.add_argument(
        "--n-bond",
        type=int,
        default=N_BOND,
        help="Bond maturity/horizon in years. Default: 10.",
    )
    parser.add_argument(
        "--no-final-var",
        action="store_true",
        help="Skip estimating the final lifecycle VAR summary.",
    )
    args = parser.parse_args()

    thetas = sorted(dict.fromkeys(float(x) for x in args.theta))
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else default_output_dir(args.long_yield_source, args.n_bond)
    )
    moments, construction_summary, final_summary = write_outputs(
        thetas,
        out_dir=out_dir,
        n_bond=args.n_bond,
        long_yield_source=args.long_yield_source,
        estimate_final_var=not args.no_final_var,
    )

    print("=" * 78)
    print("CCV NOMINAL-YIELD THETA REAL-BOND BUILD")
    print("=" * 78)
    print(
        f"Sample: {SAMPLE_START}-{SAMPLE_END}, "
        f"long yield={args.long_yield_source}, maturity n={args.n_bond}, "
        f"thetas={thetas}"
    )
    print("Construction VAR variables:", ", ".join(CONSTRUCTION_COLS))
    row = construction_summary.iloc[0]
    print(
        f"Construction VAR: T={int(row.nobs)}, max |eig|={row.max_abs_eig:.4f}, "
        f"R2(rtb)={row.r2_rtb:.3f}, R2(y_1_nom)={row.r2_y_1_nom:.3f}, "
        f"R2(spr_nom)={row.r2_spr_nom:.3f}"
    )
    print()
    print("Theta moments:")
    print(
        f"  {'theta':>6s} {'E[y1]':>9s} {'E[yN]':>9s} {'E[spr]':>9s} "
        f"{'EJ[xb]':>9s} {'sd[xb]':>9s} {'Sh[xb]':>8s} {'Dur':>7s}"
    )
    for r in moments.itertuples(index=False):
        print(
            f"  {r.theta:6.2f} "
            f"{r.mean_y_1 * 100:+9.3f} "
            f"{r.mean_y_10 * 100:+9.3f} "
            f"{r.mean_spr * 100:+9.3f} "
            f"{r.jensen_xb * 100:+9.3f} "
            f"{r.std_xb * 100:9.3f} "
            f"{r.sharpe_xb:+8.3f} "
            f"{r.mean_duration:7.3f}"
        )
    if final_summary is not None:
        print()
        print("Final lifecycle VAR checks:")
        print(
            f"  {'theta':>6s} {'eig':>7s} {'R2_spr':>7s} {'R2_y1':>7s} "
            f"{'R2_xb':>7s} {'sig_xb|s':>9s} {'M_xb_spr':>10s} {'M_xb_y1':>10s}"
        )
        for r in final_summary.itertuples(index=False):
            print(
                f"  {r.theta:6.2f} "
                f"{r.max_abs_state_eig:7.4f} "
                f"{r.r2_spr:7.3f} "
                f"{r.r2_y_1:7.3f} "
                f"{r.r2_xb:7.3f} "
                f"{r.sigma_r_cond_xb * 100:9.3f} "
                f"{r.M_xb_spr:10.3f} "
                f"{r.M_xb_y_1:10.3f}"
            )
    print()
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
