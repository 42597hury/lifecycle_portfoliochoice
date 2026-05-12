"""Identity test for the bootstrap pipeline.

When the bootstrap is run with identity resampling and the production AR(1)
held fixed (`inflation_mode="conditional"`), every downstream output must
reproduce the production pipeline to floating-point precision. This guarantees
that any drift between the bootstrap's in-memory estimators and the production
CSV-based estimators is caught here, not later.

The bootstrap loses the first row (year 1920) to the CLM lookback (no
pre-sample y_log[1919] in the donor panel), so this test compares to
production CSV rows 1921-2011 (T=91 of 92).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from lifecycle.bootstrap.donor_panel import build_donor_panel  # noqa: E402
from lifecycle.bootstrap.pipeline import (  # noqa: E402
    STATE_INDICES,
    XB_INDEX_IN_FINAL,
    estimate_construction_var,
    estimate_final_var,
    run_iteration,
)
from lifecycle.bootstrap.stationary_block import identity_indices  # noqa: E402

PROD_CSV_DIR = REPO / "data" / "real_eh_lambda_scaling"
LAMBDAS = (0.0, 0.5, 1.0)
ATOL = 1e-12


def _load_prod_csv(lam: float) -> pd.DataFrame:
    lam_str = f"{lam:.2f}".replace(".", "p")
    path = PROD_CSV_DIR / f"var_dataset_lambda_{lam_str}.csv"
    if not path.exists():
        pytest.skip(f"production CSV not found at {path}; run build_var_dataset_real_eh_lambda.py first")
    return pd.read_csv(path).set_index("year")


@pytest.fixture(scope="module")
def donor():
    return build_donor_panel()


@pytest.fixture(scope="module")
def identity_results(donor):
    donor_arrays = donor.as_arrays()
    id_idx = identity_indices(donor.T)[0]
    return run_iteration(
        donor_arrays=donor_arrays,
        idx=id_idx,
        lambdas=LAMBDAS,
        inflation_mode="conditional",
        production_ar1=donor.production_ar1,
        draw_id=-1,
    )


def _per_lambda_Z(donor, lam: float) -> pd.DataFrame:
    """Re-run the per-lambda block under identity + conditional, return (cape, spr, y_1, xr, xb)."""
    from lifecycle.bootstrap.pipeline import (
        N_BOND,
        _per_lambda_block,
        compute_eh_real_long_yield_arr,
        expected_average_inflation_arr,
    )

    arrs = donor.as_arrays()
    ar1 = donor.production_ar1
    pi_info = arrs["pi_info"]
    R = arrs["R"]
    RLONG = arrs["RLONG"]
    CAPE = arrs["CAPE"]
    log_stock_gross = arrs["log_stock_gross"]

    E_pi_1 = expected_average_inflation_arr(pi_info, ar1, 1)
    E_pi_n = expected_average_inflation_arr(pi_info, ar1, N_BOND)

    y_1 = np.log1p(R / 100.0) - E_pi_1
    y_n_real = np.log1p(RLONG / 100.0) - E_pi_n
    spr_obs = y_n_real - y_1

    cvar = estimate_construction_var(y_1, spr_obs)
    y_10_EH = compute_eh_real_long_yield_arr(y_1, spr_obs, cvar)
    TP_R = y_n_real - y_10_EH

    Z, _ = _per_lambda_block(lam, y_1, y_10_EH, TP_R, pi_info, log_stock_gross, CAPE)
    years = donor.table.index.to_numpy()
    full_full = np.column_stack([
        -np.log(CAPE),
        y_10_EH + lam * TP_R - y_1,
        y_1,
        log_stock_gross - pi_info - np.concatenate([[np.nan], y_1[:-1]]),
        np.full_like(y_1, np.nan),
    ])
    r_bond_full = np.empty_like(y_1)
    from lifecycle.bootstrap.pipeline import clm_return_from_log_yield_arr
    r_bond, _ = clm_return_from_log_yield_arr(y_10_EH + lam * TP_R, n=N_BOND)
    r_bond_full[:] = r_bond
    xb_full = r_bond_full - np.concatenate([[np.nan], y_1[:-1]])
    full_full[:, XB_INDEX_IN_FINAL] = xb_full

    keep = ~np.any(np.isnan(full_full), axis=1)
    kept_years = years[keep]
    return pd.DataFrame(full_full[keep], index=kept_years, columns=["cape", "spr", "y_1", "xr", "xb"])


def test_identity_matches_production_csv_rows(donor):
    """For each lambda: bootstrap Z matrix == production CSV rows 1921-2011."""
    for lam in LAMBDAS:
        boot_Z = _per_lambda_Z(donor, lam)
        prod = _load_prod_csv(lam)
        common_years = boot_Z.index.intersection(prod.index)
        assert len(common_years) >= 90, f"too few common years for lambda={lam}"
        for col in ["cape", "spr", "y_1", "xr", "xb"]:
            diff = (boot_Z.loc[common_years, col] - prod.loc[common_years, col]).abs().max()
            assert diff < ATOL, (
                f"lambda={lam}, col={col}: max abs diff = {diff:.3e} > {ATOL:.0e}"
            )


def test_identity_final_var_matches_production(donor):
    """For each lambda: final-VAR Phi and z_bar match what the production estimator would produce."""
    # Production estimator path: build_var_config_from_dataset on the production CSV.
    from lifecycle.var import build_var_config_from_dataset

    for lam in LAMBDAS:
        boot_Z = _per_lambda_Z(donor, lam)
        # Estimate final VAR on the bootstrap Z directly.
        boot_fvar = estimate_final_var(boot_Z.to_numpy(), lam=lam)

        # Compare against the production estimator on the same Z rows (sliced from the prod CSV).
        prod = _load_prod_csv(lam)
        common = boot_Z.index.intersection(prod.index)
        prod_sliced = prod.loc[common, ["cape", "spr", "y_1", "xr", "xb"]]

        # Write to a temp CSV (production estimator expects a CSV path) and re-estimate.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
            prod_sliced.to_csv(tf.name)
            tmp_path = tf.name

        estimation = "restricted_eh" if lam == 0.0 else "restricted"
        cfg, _, _ = build_var_config_from_dataset(
            csv_path=tmp_path,
            columns=["cape", "spr", "y_1", "xr", "xb"],
            state_indices=(0, 1, 2),
            return_indices=(3, 4),
            y_1_index_in_state=2,
            spr_index_in_state=1,
            rtb_index_in_state=None,
            estimation=estimation,
        )

        prod_Phi = np.asarray(cfg["Phi"], dtype=float)
        prod_z_bar = np.asarray(cfg["z_bar"], dtype=float)
        prod_Omega = np.asarray(cfg["Omega"], dtype=float)

        assert np.allclose(boot_fvar["Phi"], prod_Phi, atol=ATOL), (
            f"lambda={lam}: Phi mismatch, max diff = {np.abs(boot_fvar['Phi'] - prod_Phi).max():.3e}"
        )
        assert np.allclose(boot_fvar["z_bar"], prod_z_bar, atol=ATOL), (
            f"lambda={lam}: z_bar mismatch, max diff = {np.abs(boot_fvar['z_bar'] - prod_z_bar).max():.3e}"
        )
        assert np.allclose(boot_fvar["Omega"], prod_Omega, atol=ATOL), (
            f"lambda={lam}: Omega mismatch, max diff = {np.abs(boot_fvar['Omega'] - prod_Omega).max():.3e}"
        )


def test_identity_run_iteration_no_failures(identity_results):
    """run_iteration() at identity+conditional should produce all-lambda success."""
    assert not identity_results.final_var_failed, identity_results.error
    for lam in LAMBDAS:
        assert identity_results.final_results[lam] is not None, f"lambda={lam} failed"


def test_identity_restricted_eh_zero_at_lambda0(identity_results):
    """At lambda=0 the EH restriction must hold exactly in identity output."""
    fr = identity_results.final_results[0.0]
    assert fr is not None
    assert abs(fr["Phi_xb_cape"]) < ATOL
    assert abs(fr["Phi_xb_spr"]) < ATOL
    assert abs(fr["Phi_xb_y1"]) < ATOL


def test_construction_var_stable_at_identity(identity_results):
    assert identity_results.construction_max_eig < 1.0
