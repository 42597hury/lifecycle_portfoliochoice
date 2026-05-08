"""Inspect inf-horizon newton_failures_per_iter shape across bundles."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PATHS = [
    REPO / "saved_runs/inf_horizon/system_iv_inf_grid_g3_quad3334_ret44_calib1",
    REPO / "saved_runs/inf_horizon/system_iv_inf_grid_g4_quad3334_ret44_calib1",
    REPO / "saved_runs/inf_horizon/system_iv_inf_grid_g5_quad3334_ret44_calib1",
    REPO / "saved_runs/inf_horizon/system_iv_inf_axisbump_run1_sq3333_rq33_calib1",
]


def main() -> None:
    for p in PATHS:
        with open(p / "diagnostics.pkl", "rb") as f:
            d = pickle.load(f)
        fpi = np.asarray(d["newton_failures_per_iter"])
        n_iter = int(d["n_iter"])
        cells_per_iter_proxy = d["newton_iter_histogram"]["n_cells"] / max(n_iter, 1)
        print(f"--- {p.name}")
        print(f"  n_iter={n_iter}  total_failures={int(fpi.sum())}  "
              f"converged={d['converged']}  "
              f"cells_per_iter~{cells_per_iter_proxy:.0f}")
        print(f"  fails @iter 0: {int(fpi[0])}  ({fpi[0]/cells_per_iter_proxy*100:.2f}%)")
        if fpi.size > 5:
            print(f"  fails @iter 5: {int(fpi[5])}  ({fpi[5]/cells_per_iter_proxy*100:.2f}%)")
        if fpi.size > 20:
            print(f"  fails @iter 20: {int(fpi[20])}  ({fpi[20]/cells_per_iter_proxy*100:.2f}%)")
        if fpi.size > 30:
            mid = fpi.size // 2
            print(f"  fails @mid (iter {mid}): {int(fpi[mid])}  ({fpi[mid]/cells_per_iter_proxy*100:.2f}%)")
        print(f"  fails @last (iter {fpi.size-1}): {int(fpi[-1])}  ({fpi[-1]/cells_per_iter_proxy*100:.2f}%)")
        print(f"  argmax fail iter: {int(np.argmax(fpi))}  with {int(fpi.max())} fails")


if __name__ == "__main__":
    main()
