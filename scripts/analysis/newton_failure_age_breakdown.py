"""Print per-age fail-rate breakdown for the post-fix lifecycle bundles."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PATHS = [
    REPO / "saved_runs/ablations/system_i_grid7_nz25_w180_log_calib1",
    REPO / "saved_runs/ablations/system_i_grid7_nz25_w180_bakh_calib1",
    REPO / "saved_runs/ablations/system_i_grid7_nz25_w90_log_calib1",
    REPO / "saved_runs/ablations/system_i_grid7_nz25_w60_log_calib1",
]
LABELS = ["w180_log", "w180_bakh", "w90_log", "w60_log"]


def main() -> None:
    print(
        f"{'age':>4s}  "
        + "  ".join(f"{l:>14s}" for l in LABELS)
    )
    diags = []
    for p in PATHS:
        with open(p / "diagnostics.pkl", "rb") as f:
            diags.append(pickle.load(f))
    afs = [np.asarray(d["age_newton_fail"]) for d in diags]
    # cells per age = n_z * N_state * n_savings (s=0 anchor stripped already)
    # for w180 = 25*7*180=31500; w90 = 15750; w60 = 10500
    n_per = []
    for p in PATHS:
        if "w180" in str(p):
            n_per.append(25 * 7 * 180)
        elif "w90" in str(p):
            n_per.append(25 * 7 * 90)
        elif "w60" in str(p):
            n_per.append(25 * 7 * 60)
        elif "w120" in str(p):
            n_per.append(25 * 7 * 120)
    ages = np.arange(22, 100)
    for i, age in enumerate(ages):
        row = f"{age:>4d}  "
        for af, n in zip(afs, n_per):
            rate = af[i] / n * 100
            row += f"{int(af[i]):>6d} ({rate:>4.1f}%)  "
        print(row)


if __name__ == "__main__":
    main()
