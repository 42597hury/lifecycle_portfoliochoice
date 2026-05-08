"""Phase 1 of the real-yields pivot: predictability_ablation rewrite.

Covers:
  * All 3 systems (1, 2, full) load and produce a valid disc config
  * state_grid_sizes length matches len(state_names) for each system
  * Old Roman-numeral codes (I, II, III, IV) explicitly raise
  * Disc-config projection respects the 3-axis template ordering
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lifecycle.model import DiscretizationConfig
from lifecycle.predictability_ablation import (
    DEFAULT_TEMPLATE_STATE_NAMES,
    get_predictability_system_spec,
    prepare_predictability_system,
    project_predictability_disc_config,
)


CSV_PATH = REPO / "data" / "var_dataset.csv"

REQUIRED_REAL_COLUMNS = ("cape", "spr", "y_1", "xr", "xb")


def _real_dataset_ready(path: Path) -> bool:
    """True iff the CSV at ``path`` has the real-yield columns the new VAR expects."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
    except OSError:
        return False
    return all(col in header for col in REQUIRED_REAL_COLUMNS)


def _template_disc():
    """3-axis (cape, spr, y_1) template with deliberately distinct per-axis values."""
    return DiscretizationConfig(
        n_wealth=20,
        n_savings=20,
        state_grid_sizes=(3, 4, 5),     # cape=3, spr=4, y_1=5
        state_n_stds=(2.0, 2.25, 2.5),  # cape=2.0, spr=2.25, y_1=2.5
        n_state_quad_nodes=(3, 3, 5),
        n_z=3,
        n_eps_nodes=3,
        n_eta_nodes=3,
        n_ret_nodes_1d=(3, 3),
    )


# -----------------------------------------------------------------------------
# Code/spec metadata
# -----------------------------------------------------------------------------

def test_default_template_state_names():
    assert DEFAULT_TEMPLATE_STATE_NAMES == ("cape", "spr", "y_1")


@pytest.mark.parametrize(
    "code, expected_state_names",
    [
        ("1", ("y_1",)),
        ("2", ("spr", "y_1")),
        ("full", ("cape", "spr", "y_1")),
    ],
)
def test_spec_state_names(code, expected_state_names):
    spec = get_predictability_system_spec(code)
    assert spec.state_names == expected_state_names
    assert spec.code == code


def test_spec_aliases_resolve_to_canonical_codes():
    assert get_predictability_system_spec("1").code == "1"
    assert get_predictability_system_spec("system_1").code == "1"
    assert get_predictability_system_spec("system_2_real").code == "2"
    assert get_predictability_system_spec("FULL").code == "full"
    assert get_predictability_system_spec("full_system").code == "full"


# -----------------------------------------------------------------------------
# Old-code rejection (handoff hard rule: System I/II/III/IV no longer exist)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("retired", ["I", "II", "III", "IV", "iv", "system_iv", "3"])
def test_old_roman_codes_are_retired(retired):
    with pytest.raises(ValueError, match="retired"):
        get_predictability_system_spec(retired)


# -----------------------------------------------------------------------------
# Disc-config projection (no VAR builder needed)
# -----------------------------------------------------------------------------

def test_project_full_system_is_identity():
    template = _template_disc()
    out = project_predictability_disc_config(
        template, target_state_names=("cape", "spr", "y_1")
    )
    assert out.state_grid_sizes == (3, 4, 5)
    assert out.state_n_stds == (2.0, 2.25, 2.5)
    assert out.n_state_quad_nodes == (3, 3, 5)


def test_project_system_2_drops_cape():
    template = _template_disc()
    out = project_predictability_disc_config(
        template, target_state_names=("spr", "y_1")
    )
    assert out.state_grid_sizes == (4, 5)
    assert out.state_n_stds == (2.25, 2.5)
    assert out.n_state_quad_nodes == (3, 5)


def test_project_system_1_keeps_only_y_1():
    template = _template_disc()
    out = project_predictability_disc_config(
        template, target_state_names=("y_1",)
    )
    assert out.state_grid_sizes == (5,)
    assert out.state_n_stds == (2.5,)
    assert out.n_state_quad_nodes == (5,)


def test_project_rejects_unknown_state_name():
    template = _template_disc()
    with pytest.raises(ValueError, match="missing template axes"):
        project_predictability_disc_config(
            template, target_state_names=("rtb",)  # rtb is gone in the real-yields model
        )


# -----------------------------------------------------------------------------
# End-to-end loading (requires the real-yields VAR + dataset)
# -----------------------------------------------------------------------------

_VAR_AVAILABLE = True
try:
    from lifecycle import var as _var_module
    for _name in (
        "build_real_system1_var_config",
        "build_real_system2_var_config",
        "build_real_full_var_config",
    ):
        if not hasattr(_var_module, _name):
            _VAR_AVAILABLE = False
            break
except Exception:
    _VAR_AVAILABLE = False

_var_skip = pytest.mark.skipif(
    not _VAR_AVAILABLE or not _real_dataset_ready(CSV_PATH),
    reason=(
        "real-yields VAR builder or its real-yield CSV "
        f"(needs columns {REQUIRED_REAL_COLUMNS}) not yet committed"
    ),
)


@_var_skip
@pytest.mark.parametrize(
    "code, expected_state_names",
    [
        ("1", ("y_1",)),
        ("2", ("spr", "y_1")),
        ("full", ("cape", "spr", "y_1")),
    ],
)
def test_prepare_predictability_system_loads(code, expected_state_names):
    template = _template_disc()
    meta = prepare_predictability_system(
        code,
        csv_path=str(CSV_PATH),
        disc_config_template=template,
    )
    assert meta["state_names"] == expected_state_names
    assert meta["disc_config"].state_grid_sizes is not None
    assert len(meta["disc_config"].state_grid_sizes) == len(expected_state_names)
    assert len(meta["disc_config"].n_state_quad_nodes) == len(expected_state_names)
    assert meta["system_code"] == code
