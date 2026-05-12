"""Compute per-statistic and paired-difference summaries from bootstrap draws."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATS_OF_INTEREST = (
    "E_spr", "sd_spr",
    "E_xb", "sd_xb", "Sharpe_xb",
    "E_xr", "sd_xr", "Sharpe_xr",
    "Phi_xb_cape", "Phi_xb_spr", "Phi_xb_y1",
    "Phi_xr_cape", "Phi_xr_spr", "Phi_xr_y1",
    "Phi_spr_spr", "spr_halflife",
    "state_max_eig",
    "sd_v_xb", "corr_vxr_vxb", "corr_vy10_vxb",
)

PAIRED_STATS = (
    "E_xb", "sd_xb", "Sharpe_xb",
    "Phi_xb_spr", "Phi_xb_y1",
    "corr_vy10_vxb",
)


def summarize_marginals(
    draws: pd.DataFrame,
    point: pd.DataFrame,
    stable_only: bool = False,
) -> pd.DataFrame:
    df = draws.copy()
    if stable_only:
        df = df[df["stable_flag"] == 1]

    rows = []
    for lam in sorted(df["lambda_val"].unique()):
        sub = df[df["lambda_val"] == lam]
        n_total = len(sub)
        sub_ok = sub[sub["final_var_failed"] == 0]
        n_used = len(sub_ok)
        n_explosive = n_total - n_used if not stable_only else int(
            (draws[draws["lambda_val"] == lam]["stable_flag"] == 0).sum()
        )
        for stat in STATS_OF_INTEREST:
            vals = sub_ok[stat].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            point_row = point[(point["lambda_val"] == lam)]
            point_estimate = (
                float(point_row[stat].iloc[0])
                if len(point_row) and stat in point_row.columns and np.isfinite(point_row[stat].iloc[0])
                else float("nan")
            )
            if len(vals) == 0:
                rows.append({
                    "stat": stat,
                    "lambda_val": float(lam),
                    "point_estimate": point_estimate,
                    "boot_mean": float("nan"),
                    "boot_sd": float("nan"),
                    "p05": float("nan"),
                    "p50": float("nan"),
                    "p95": float("nan"),
                    "n_used": 0,
                    "n_explosive_excluded": n_explosive,
                })
                continue
            rows.append({
                "stat": stat,
                "lambda_val": float(lam),
                "point_estimate": point_estimate,
                "boot_mean": float(np.mean(vals)),
                "boot_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "n_used": int(len(vals)),
                "n_explosive_excluded": int(n_explosive),
            })
    return pd.DataFrame(rows)


def summarize_paired(
    draws: pd.DataFrame,
    point: pd.DataFrame,
    pairs: list[tuple[float, float]] | None = None,
    stable_only: bool = False,
) -> pd.DataFrame:
    df = draws.copy()
    if stable_only:
        df = df[df["stable_flag"] == 1]

    lambdas = sorted(df["lambda_val"].unique())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(lambdas) for b in lambdas[:i]]

    pivots = {
        stat: df.pivot_table(index="draw_id", columns="lambda_val", values=stat, aggfunc="first")
        for stat in PAIRED_STATS
    }
    failed_pivot = df.pivot_table(
        index="draw_id", columns="lambda_val", values="final_var_failed", aggfunc="first"
    )

    rows = []
    for stat in PAIRED_STATS:
        piv = pivots[stat]
        point_pivot = (
            point.pivot_table(index="lambda_val", values=stat, aggfunc="first")
            if stat in point.columns else None
        )
        for lam_a, lam_b in pairs:
            if lam_a not in piv.columns or lam_b not in piv.columns:
                continue
            ok = (failed_pivot[lam_a] == 0) & (failed_pivot[lam_b] == 0)
            diff = (piv[lam_a] - piv[lam_b])[ok].dropna().to_numpy(dtype=float)
            point_diff = float("nan")
            if point_pivot is not None and lam_a in point_pivot.index and lam_b in point_pivot.index:
                point_diff = float(point_pivot.loc[lam_a, stat] - point_pivot.loc[lam_b, stat])
            if len(diff) == 0:
                rows.append({
                    "stat": stat,
                    "lambda_a": float(lam_a),
                    "lambda_b": float(lam_b),
                    "point_diff": point_diff,
                    "boot_sd_diff": float("nan"),
                    "p05_diff": float("nan"),
                    "p50_diff": float("nan"),
                    "p95_diff": float("nan"),
                    "frac_diff_pos": float("nan"),
                    "n_paired": 0,
                })
                continue
            rows.append({
                "stat": stat,
                "lambda_a": float(lam_a),
                "lambda_b": float(lam_b),
                "point_diff": point_diff,
                "boot_sd_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan"),
                "p05_diff": float(np.percentile(diff, 5)),
                "p50_diff": float(np.percentile(diff, 50)),
                "p95_diff": float(np.percentile(diff, 95)),
                "frac_diff_pos": float(np.mean(diff > 0.0)),
                "n_paired": int(len(diff)),
            })
    return pd.DataFrame(rows)


def stability_fractions(draws: pd.DataFrame) -> dict:
    out: dict = {}
    out["P_state_eig_lt_1"] = {}
    out["P_construction_eig_lt_1"] = {}
    out["P_Phi_xb_spr_pos"] = {}
    out["P_final_var_failed"] = {}

    for lam in sorted(draws["lambda_val"].unique()):
        sub = draws[draws["lambda_val"] == lam]
        sub_ok = sub[sub["final_var_failed"] == 0]
        n = len(sub_ok)
        if n == 0:
            out["P_state_eig_lt_1"][float(lam)] = float("nan")
            out["P_construction_eig_lt_1"][float(lam)] = float("nan")
            out["P_Phi_xb_spr_pos"][float(lam)] = float("nan")
        else:
            out["P_state_eig_lt_1"][float(lam)] = float((sub_ok["state_max_eig"] < 1.0).mean())
            out["P_construction_eig_lt_1"][float(lam)] = float((sub_ok["construction_max_eig"] < 1.0).mean())
            out["P_Phi_xb_spr_pos"][float(lam)] = float((sub_ok["Phi_xb_spr"] > 0.0).mean())
        out["P_final_var_failed"][float(lam)] = float((sub["final_var_failed"] == 1).mean())

    pivot_E_xb = draws.pivot_table(
        index="draw_id", columns="lambda_val", values="E_xb", aggfunc="first"
    )
    if 1.0 in pivot_E_xb.columns and 0.0 in pivot_E_xb.columns:
        out["P_E_xb_1_gt_0"] = float((pivot_E_xb[1.0] > pivot_E_xb[0.0]).mean())
    pivot_Sh_xb = draws.pivot_table(
        index="draw_id", columns="lambda_val", values="Sharpe_xb", aggfunc="first"
    )
    if 1.0 in pivot_Sh_xb.columns and 0.0 in pivot_Sh_xb.columns:
        out["P_Sharpe_xb_1_gt_0"] = float((pivot_Sh_xb[1.0] > pivot_Sh_xb[0.0]).mean())
    return out


def write_summaries(
    draws_path: Path,
    point_path: Path,
    out_dir: Path,
    paired_pairs: list[tuple[float, float]] | None = None,
) -> dict[str, Path]:
    draws = pd.read_csv(draws_path)
    point = pd.read_csv(point_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    marg = summarize_marginals(draws, point)
    paired = summarize_paired(draws, point, pairs=paired_pairs)
    marg_stable = summarize_marginals(draws, point, stable_only=True)
    paired_stable = summarize_paired(draws, point, pairs=paired_pairs, stable_only=True)
    stab = stability_fractions(draws)

    paths = {
        "summary": out_dir / "summary.csv",
        "summary_paired": out_dir / "summary_paired.csv",
        "summary_stable": out_dir / "summary_stable.csv",
        "summary_paired_stable": out_dir / "summary_paired_stable.csv",
        "stability": out_dir / "stability.csv",
    }
    marg.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["summary_paired"], index=False)
    marg_stable.to_csv(paths["summary_stable"], index=False)
    paired_stable.to_csv(paths["summary_paired_stable"], index=False)

    stab_rows = []
    for key, val in stab.items():
        if isinstance(val, dict):
            for lam, p in val.items():
                stab_rows.append({"flag": key, "lambda_val": lam, "value": p})
        else:
            stab_rows.append({"flag": key, "lambda_val": float("nan"), "value": float(val)})
    pd.DataFrame(stab_rows).to_csv(paths["stability"], index=False)

    return paths
