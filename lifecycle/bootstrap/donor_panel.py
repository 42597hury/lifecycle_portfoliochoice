"""Donor panel: the annual primitive table that the bootstrap resamples.

Wraps `build_var_dataset_ar1_10y.build_full_table()` so that the bootstrap
and the production dataset builder cannot drift.

A donor row at year t carries everything needed to rebuild the full pipeline
from that year's observations:

  R, RLONG          chap_26 short and long nominal yields at Jan year t
  P, D              chap_26 price and dividend at Jan year t
  CAPE              Shiller monthly CAPE sampled at Jan year t
  pi_info           Dec(t-2) -> Dec(t-1) log CPI inflation (= AR(1) state at Jan t)
  log_stock_gross   log((P[t] + D[t]) / P[t-1]) (Jan(t-1) -> Jan(t) log stock gross)

`log_stock_gross` is pre-computed per donor year so that the stock leg keeps
its within-year alignment with `pi_info` (the matching Dec-Dec inflation)
after resampling.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from build_var_dataset_ar1_10y import (  # noqa: E402
    SAMPLE_END,
    SAMPLE_START,
    build_full_table,
)

DONOR_COLUMNS = ("R", "RLONG", "P", "D", "CAPE", "pi_info", "log_stock_gross")


@dataclass(frozen=True)
class DonorPanel:
    """Annual primitive panel for the bootstrap, plus the production AR(1)."""

    table: pd.DataFrame
    sample_start: int
    sample_end: int
    production_ar1: dict

    @property
    def T(self) -> int:
        return len(self.table)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {col: self.table[col].to_numpy(dtype=float) for col in DONOR_COLUMNS}


def build_donor_panel(
    sample_start: int = SAMPLE_START,
    sample_end: int = SAMPLE_END,
) -> DonorPanel:
    df, ar1 = build_full_table()

    log_stock_gross = np.log((df["P"] + df["D"]) / df["P"].shift(1))

    panel = pd.DataFrame(
        {
            "R": df["R"],
            "RLONG": df["RLONG"],
            "P": df["P"],
            "D": df["D"],
            "CAPE": df["CAPE"],
            "pi_info": df["pi_info"],
            "log_stock_gross": log_stock_gross,
        },
        index=df.index,
    )

    panel = panel.loc[sample_start:sample_end].copy()

    if panel.isna().any().any():
        nan_rows = panel[panel.isna().any(axis=1)]
        raise ValueError(
            f"Donor panel has NaN rows in years {list(nan_rows.index)}; "
            f"check chap_26 / Shiller coverage for [{sample_start}, {sample_end}]"
        )

    return DonorPanel(
        table=panel,
        sample_start=int(panel.index.min()),
        sample_end=int(panel.index.max()),
        production_ar1={
            "intercept": float(ar1["intercept"]),
            "phi": float(ar1["phi"]),
            "mu": float(ar1["mu"]),
            "nobs": int(ar1["nobs"]),
        },
    )
