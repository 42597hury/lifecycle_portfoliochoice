from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple

from lifecycle.model import DiscretizationConfig
from lifecycle.var import (
    build_iid_var_config,
    build_no_cy_var_config,
    build_nominal_system1_var_config,
    build_rtb_y1_var_config,
)


DEFAULT_TEMPLATE_STATE_NAMES = ("cy", "spr", "rtb", "y_1")


class PredictabilitySystemSpec(NamedTuple):
    code: str
    label: str
    name: str
    description: str
    builder: Callable[..., tuple[dict[str, Any], Any, Any]]
    state_names: tuple[str, ...]


_SYSTEM_SPECS: dict[str, PredictabilitySystemSpec] = {
    "I": PredictabilitySystemSpec(
        code="I",
        label="system_i_iid",
        name="iid returns",
        description="No return predictability; rtb is iid in the single-axis state, returns are iid.",
        builder=build_iid_var_config,
        state_names=("rtb",),
    ),
    "II": PredictabilitySystemSpec(
        code="II",
        label="system_ii_rtb_y1",
        name="rtb plus y_1",
        description="Inflation-persistence channel (rtb) plus the short rate; state is ordered (rtb, y_1).",
        builder=build_rtb_y1_var_config,
        state_names=("rtb", "y_1"),
    ),
    "III": PredictabilitySystemSpec(
        code="III",
        label="system_iii_rtb_spr_y1",
        name="rtb plus spread plus y_1",
        description="Rate-side predictability with inflation persistence; cy removed; state ordered (rtb, spr, y_1).",
        builder=build_no_cy_var_config,
        state_names=("rtb", "spr", "y_1"),
    ),
    "IV": PredictabilitySystemSpec(
        code="IV",
        label="system_iv_full_var",
        name="full VAR baseline",
        description="Baseline lifecycle model with the full (cy, spr, rtb, y_1) state vector.",
        builder=build_nominal_system1_var_config,
        state_names=("cy", "spr", "rtb", "y_1"),
    ),
}


_SYSTEM_ALIASES = {
    "i": "I",
    "1": "I",
    "system_i": "I",
    "system_1": "I",
    "iid": "I",
    "iid_returns": "I",
    "no_predictability": "I",
    "ii": "II",
    "2": "II",
    "system_ii": "II",
    "system_2": "II",
    "rtb_y1": "II",
    "rtb_y_1": "II",
    "iii": "III",
    "3": "III",
    "system_iii": "III",
    "system_3": "III",
    "no_cy": "III",
    "rtb_spr_y1": "III",
    "rtb_spread_y1": "III",
    "rate_side": "III",
    "iv": "IV",
    "4": "IV",
    "system_iv": "IV",
    "system_4": "IV",
    "baseline": "IV",
    "full": "IV",
    "full_var": "IV",
    "full_predictability": "IV",
}


def _normalize_system_code(system: str | int) -> str:
    if isinstance(system, int):
        key = str(system)
    else:
        key = str(system).strip().lower()
        key = key.replace("-", "_").replace(" ", "_").replace("/", "_")
        while "__" in key:
            key = key.replace("__", "_")

    code = _SYSTEM_ALIASES.get(key)
    if code is None:
        raise ValueError(
            "Unknown predictability system "
            f"{system!r}. Choose one of I, II, III, IV."
        )
    return code


def get_predictability_system_spec(system: str | int) -> PredictabilitySystemSpec:
    """Return the canonical metadata for System I/II/III/IV."""
    return _SYSTEM_SPECS[_normalize_system_code(system)]


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
    """Project a baseline discretization config onto a lower-dimensional state vector."""
    if target_state_names == ("rtb",):
        # System I: rtb is iid in a single-axis state. Use a small grid since
        # there's no persistence; the axis still needs >1 node so the solver
        # can read the rtb realisation off the state vector.
        rtb_size = _project_axis_setting(
            disc_config_template.state_grid_sizes,
            template_state_names,
            ("rtb",),
            "state_grid_sizes",
        )
        rtb_n_stds = _project_axis_setting(
            disc_config_template.state_n_stds,
            template_state_names,
            ("rtb",),
            "state_n_stds",
        )
        rtb_quad = _project_axis_setting(
            disc_config_template.n_state_quad_nodes,
            template_state_names,
            ("rtb",),
            "n_state_quad_nodes",
        )
        return disc_config_template._replace(
            state_grid_sizes=rtb_size,
            state_n_stds=rtb_n_stds,
            n_state_quad_nodes=rtb_quad,
        )

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
    """
    Build the selected ablation VAR and the matching discretization config.

    The baseline 3D `disc_config_template` stays user-owned; this helper only
    projects its state-axis settings onto the selected System I/II/III/IV.
    """
    spec = get_predictability_system_spec(system)
    var_config, var_res, var_data = _call_builder(
        spec.builder,
        csv_path=csv_path,
        builder_kwargs={} if builder_kwargs is None else dict(builder_kwargs),
    )

    variable_names = tuple(str(name) for name in var_config["variable_names"])
    state_indices = tuple(int(i) for i in var_config["state_indices"])
    observed_state_names = tuple(variable_names[i] for i in state_indices)
    if observed_state_names != spec.state_names:
        raise ValueError(
            f"{spec.builder.__name__} returned state_names={observed_state_names}, "
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
        "system_title": f"System {spec.code} ({spec.name})",
        "system_description": spec.description,
        "state_names": observed_state_names,
        "var_builder_name": spec.builder.__name__,
        "var_config": var_config,
        "var_res": var_res,
        "var_data": var_data,
        "disc_config": disc_config,
    }
