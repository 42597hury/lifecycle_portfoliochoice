from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, NamedTuple

from lifecycle.model import DiscretizationConfig


DEFAULT_TEMPLATE_STATE_NAMES = ("cape", "spr", "y_1")


class PredictabilitySystemSpec(NamedTuple):
    code: str
    label: str
    name: str
    description: str
    builder_name: str
    state_names: tuple[str, ...]


_SYSTEM_SPECS: dict[str, PredictabilitySystemSpec] = {
    "1": PredictabilitySystemSpec(
        code="1",
        label="system_1_real",
        name="real bill yield only",
        description=(
            "Single-axis real-yields state on y_1 (real bill yield). "
            "No spread or CAPE. Bill return is state-determined; "
            "stock and bond returns load on y_1 alone."
        ),
        builder_name="build_real_system1_var_config",
        state_names=("y_1",),
    ),
    "2": PredictabilitySystemSpec(
        code="2",
        label="system_2_real",
        name="bill yield plus spread",
        description=(
            "Two-axis real-yields state on (spr, y_1) with y_1 at the "
            "K-bump end. Adds the term-spread channel on top of System 1; "
            "no CAPE."
        ),
        builder_name="build_real_system2_var_config",
        state_names=("spr", "y_1"),
    ),
    "full": PredictabilitySystemSpec(
        code="full",
        label="full_system_real",
        name="full real-yields system",
        description=(
            "Three-axis real-yields state on (cape, spr, y_1) with y_1 at "
            "the K-bump end. CAPE drives stock-return predictability; "
            "the bond predictor uses (y_1, spr); the bill is state-"
            "determined."
        ),
        builder_name="build_real_full_var_config",
        state_names=("cape", "spr", "y_1"),
    ),
}


_SYSTEM_ALIASES = {
    # System 1 (real bill yield only)
    "1": "1",
    "system_1": "1",
    "system_1_real": "1",
    "real_bill": "1",
    "y_1": "1",
    "y1": "1",
    # System 2 (bill + spread)
    "2": "2",
    "system_2": "2",
    "system_2_real": "2",
    "bill_spr": "2",
    "spr_y_1": "2",
    "spr_y1": "2",
    # Full system (cape + spr + y_1)
    "full": "full",
    "system_full": "full",
    "full_system": "full",
    "full_system_real": "full",
    "all": "full",
    "cape_spr_y_1": "full",
    "cape_spr_y1": "full",
}


# Old (pre-pivot) system codes that are explicitly rejected. The 4-axis
# nominal model with Roman-numeral systems (I/II/III/IV) was deleted in
# the real-yields pivot; calling code that still passes these is a bug
# that must be fixed at the call site rather than silently re-routed to
# a same-numbered new-model system that has different state variables.
_RETIRED_SYSTEM_CODES = frozenset({
    "i", "system_i", "iid", "iid_returns", "no_predictability",
    "ii", "system_ii", "rtb_y1", "rtb_y_1",
    "iii", "system_iii", "no_cy", "rtb_spr_y1", "rtb_spread_y1", "rate_side",
    "iv", "system_iv", "baseline", "full_var", "full_predictability",
    # 3 (System III) was a 4-axis nominal sub-system — System 3 does not
    # exist in the real-yields pivot.
    "3", "system_3",
    # 4 was an alias for System IV. The new "full" code is "full".
    "4", "system_4",
})


def _normalize_system_code(system: str | int) -> str:
    if isinstance(system, int):
        key = str(system)
    else:
        key = str(system).strip().lower()
        key = key.replace("-", "_").replace(" ", "_").replace("/", "_")
        while "__" in key:
            key = key.replace("__", "_")

    if key in _RETIRED_SYSTEM_CODES:
        raise ValueError(
            f"Predictability system {system!r} was retired in the real-yields "
            "pivot. The new model has three systems: '1' (y_1 only), "
            "'2' (spr + y_1), 'full' (cape + spr + y_1). Update the call site "
            "explicitly — old and new same-numbered systems have different "
            "state variables and are not compatible."
        )

    code = _SYSTEM_ALIASES.get(key)
    if code is None:
        raise ValueError(
            f"Unknown predictability system {system!r}. Choose one of "
            "'1', '2', 'full'."
        )
    return code


def get_predictability_system_spec(system: str | int) -> PredictabilitySystemSpec:
    """Return the canonical metadata for System 1 / 2 / Full."""
    return _SYSTEM_SPECS[_normalize_system_code(system)]


def _resolve_builder(builder_name: str) -> Callable[..., tuple[dict[str, Any], Any, Any]]:
    """Late-bind a VAR builder from lifecycle.var.

    Resolved at call time rather than import time so this module imports
    cleanly while lifecycle/var.py is mid-rewrite for the real-yields pivot.
    """
    var_module = importlib.import_module("lifecycle.var")
    try:
        return getattr(var_module, builder_name)
    except AttributeError as exc:
        raise ImportError(
            f"lifecycle.var has no builder named {builder_name!r}. "
            "The real-yields VAR config must export "
            "build_real_system1_var_config, build_real_system2_var_config, "
            "and build_real_full_var_config."
        ) from exc


def _project_axis_tuple(
    values: Any,
    template_state_names: tuple[str, ...],
    target_state_names: tuple[str, ...],
    field_name: str,
) -> tuple[Any, ...]:
    try:
        values_tuple = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must be a sequence aligned with {template_state_names}, "
            f"got {values!r}"
        ) from exc

    if len(values_tuple) != len(template_state_names):
        raise ValueError(
            f"{field_name} must have length {len(template_state_names)} to match "
            f"template_state_names={template_state_names}, got {values_tuple}"
        )

    axis_lookup = dict(zip(template_state_names, values_tuple))
    missing = [name for name in target_state_names if name not in axis_lookup]
    if missing:
        raise ValueError(
            f"{field_name} cannot be projected onto {target_state_names}; "
            f"missing template axes {missing}"
        )

    return tuple(axis_lookup[name] for name in target_state_names)


def _project_axis_setting(
    value: Any,
    template_state_names: tuple[str, ...],
    target_state_names: tuple[str, ...],
    field_name: str,
) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (int, float)):
        return value
    return _project_axis_tuple(value, template_state_names, target_state_names, field_name)


def project_predictability_disc_config(
    disc_config_template: DiscretizationConfig,
    target_state_names: tuple[str, ...],
    *,
    template_state_names: tuple[str, ...] = DEFAULT_TEMPLATE_STATE_NAMES,
) -> DiscretizationConfig:
    """Project a baseline 3-axis discretization config onto a sub-system.

    All three real-yields systems are persistent (no iid axis), so projection
    is uniform: pick the matching axis settings out of the 3-axis template
    by name. The returned config has ``state_grid_sizes``, ``state_n_stds``,
    and ``n_state_quad_nodes`` of length ``len(target_state_names)``.
    """
    return disc_config_template._replace(
        state_grid_sizes=_project_axis_tuple(
            disc_config_template.state_grid_sizes,
            template_state_names,
            target_state_names,
            "state_grid_sizes",
        ),
        state_n_stds=_project_axis_setting(
            disc_config_template.state_n_stds,
            template_state_names,
            target_state_names,
            "state_n_stds",
        ),
        n_state_quad_nodes=_project_axis_setting(
            disc_config_template.n_state_quad_nodes,
            template_state_names,
            target_state_names,
            "n_state_quad_nodes",
        ),
    )


def _call_builder(
    builder: Callable[..., tuple[dict[str, Any], Any, Any]],
    *,
    csv_path: str,
    builder_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    sig = inspect.signature(builder)
    unsupported = sorted(
        key for key in builder_kwargs.keys() if key not in sig.parameters
    )
    if unsupported:
        raise TypeError(
            f"{builder.__name__} does not accept builder kwargs {unsupported}"
        )
    return builder(csv_path=csv_path, **builder_kwargs)


def prepare_predictability_system(
    system: str | int,
    *,
    csv_path: str,
    disc_config_template: DiscretizationConfig,
    template_state_names: tuple[str, ...] = DEFAULT_TEMPLATE_STATE_NAMES,
    builder_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the selected real-yields VAR and the matching discretization config.

    The 3-axis ``disc_config_template`` stays user-owned; this helper only
    projects its state-axis settings onto the selected System 1 / 2 / Full.
    """
    spec = get_predictability_system_spec(system)
    builder = _resolve_builder(spec.builder_name)
    var_config, var_res, var_data = _call_builder(
        builder,
        csv_path=csv_path,
        builder_kwargs={} if builder_kwargs is None else dict(builder_kwargs),
    )

    variable_names = tuple(str(name) for name in var_config["variable_names"])
    state_indices = tuple(int(i) for i in var_config["state_indices"])
    observed_state_names = tuple(variable_names[i] for i in state_indices)
    if observed_state_names != spec.state_names:
        raise ValueError(
            f"{builder.__name__} returned state_names={observed_state_names}, "
            f"expected {spec.state_names}"
        )

    disc_config = project_predictability_disc_config(
        disc_config_template,
        observed_state_names,
        template_state_names=template_state_names,
    )

    return {
        "system_code": spec.code,
        "system_label": spec.label,
        "system_name": spec.name,
        "system_title": (
            "Full System (real)" if spec.code == "full"
            else f"System {spec.code} (real, {spec.name})"
        ),
        "system_description": spec.description,
        "state_names": observed_state_names,
        "var_builder_name": builder.__name__,
        "var_config": var_config,
        "var_res": var_res,
        "var_data": var_data,
        "disc_config": disc_config,
    }
