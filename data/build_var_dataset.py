"""
Compatibility entry point for the active VAR dataset.

The active baseline is built by data/build_var_dataset_ar1_10y.py:
AR(1)-matched inflation expectations, Shiller RLONG as the 10-year long yield,
and a 10-year real bond return.
"""
from __future__ import annotations

from build_var_dataset_ar1_10y import main as baseline_main


def main() -> None:
    print("build_var_dataset.py is a compatibility entry point.")
    print("Delegating baseline build to build_var_dataset_ar1_10y.py")
    baseline_main()


if __name__ == "__main__":
    main()
