from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulator.config import SimulationConfig
from src.simulator.model import (
    BreakdownSpec,
    FabModel,
    ProcessTime,
    ProductRoute,
    PMSpec,
    ReleasePlan,
    ReleaseSpec,
    RouteStep,
    SetupRule,
    ToolGroupSpec,
    TransportRule,
)


def load_model(config: SimulationConfig) -> FabModel:
    dataset_dir = config.dataset_path()
    routes = _load_routes(dataset_dir, config.route_file_glob)
    toolgroups = _apply_dataclass_overrides(
        _load_toolgroups(dataset_dir / config.toolgroups_file),
        config.toolgroup_overrides,
        "tool group",
    )
    pm_specs = _apply_dataclass_overrides(
        _load_pm_specs(dataset_dir / config.pm_file),
        config.pm_overrides,
        "PM spec",
    )
    breakdown_specs = _apply_dataclass_overrides(
        _load_breakdown_specs(dataset_dir / config.breakdown_file),
        config.breakdown_overrides,
        "breakdown spec",
    )
    setup_rules = _apply_dataclass_overrides(
        _load_setup_rules(dataset_dir / config.setups_file),
        config.setup_overrides,
        "setup rule",
    )
    releases = _load_releases(
        dataset_dir / config.release_file,
        config.release_scenario,
        config.max_lots,
    )
    start_datetime = min(release.start_date for release in releases)
    release_plan = ReleasePlan(releases=releases, start_datetime=start_datetime)
    transport = _apply_transport_override(
        _load_transport(dataset_dir / config.transport_file),
        config.transport_override,
    )
    _validate_references(routes, toolgroups, releases, pm_specs, breakdown_specs)
    return FabModel(
        routes=routes,
        toolgroup_specs=toolgroups,
        pm_specs=pm_specs,
        breakdown_specs=breakdown_specs,
        setup_rules=setup_rules,
        release_plan=release_plan,
        transport=transport,
        start_datetime=start_datetime,
    )


def _load_routes(dataset_dir: Path, route_file_glob: str) -> dict[str, ProductRoute]:
    route_paths = sorted(
        dataset_dir.glob(route_file_glob),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not route_paths:
        raise FileNotFoundError(f"No route CSV files found in {dataset_dir}")
    routes: dict[str, ProductRoute] = {}
    for path in route_paths:
        df = pd.read_csv(path)
        steps = [
            RouteStep(
                route=_text(row["ROUTE"]),
                step_number=int(row["STEP"]),
                description=_text(row["STEP DESCRIPTION"]),
                area=_text(row["AREA"]),
                toolgroup=_text(row["TOOLGROUP"]),
                processing_unit=_text(row["PROCESSING UNIT"]),
                process_time=ProcessTime(
                    distribution=_optional_text(row.get("PROCESSINGTIME DISTRIBUTION")),
                    mean=_float(row.get("MEAN"), default=0.0),
                    offset=_float(row.get("OFFSET"), default=0.0),
                    units=_text(row.get("PT UNITS", "min")),
                ),
                cascading_interval=_optional_float(row.get("CASCADING INTERVAL")),
                batch_minimum=_optional_float(row.get("BATCH MINIMUM")),
                batch_maximum=_optional_float(row.get("BATCH MAXIMUM")),
                setup=_optional_text(row.get("SETUP")),
                setup_time=_optional_float(row.get("SETUP TIME")),
                setup_units=_optional_text(row.get("ST UNITS")),
                sampling_probability=_optional_float(row.get("PROCESSING PROBABILITY in % (Sampling)")),
                rework_probability=_optional_float(row.get("REWORK PROBABILITY in %")),
                step_for_rework=_optional_int(row.get("STEP FOR REWORK")),
                cqt_start_step=_optional_int(row.get("STEP FOR CRITICAL QUEUE TIME")),
                cqt=_optional_float(row.get("CQT")),
                cqt_units=_optional_text(row.get("CQTUNITS")),
            )
            for row in df.to_dict("records")
        ]
        steps = sorted(steps, key=lambda step: step.step_number)
        route_name = steps[0].route
        routes[route_name] = ProductRoute(
            name=route_name,
            product_name=route_name.replace("Route_", ""),
            steps=tuple(steps),
        )
    return routes


def _load_toolgroups(path: Path) -> dict[str, ToolGroupSpec]:
    df = pd.read_csv(path)
    specs = {}
    for row in df.to_dict("records"):
        spec = ToolGroupSpec(
            area=_text(row["AREA"]),
            name=_text(row["TOOLGROUP"]),
            number_of_tools=int(row["NUMBER OF TOOLS"]),
            loading_time=_float(row.get("LOADINGTIME"), default=0.0),
            unloading_time=_float(row.get("UNLOADINGTIME"), default=0.0),
            dispatching=_text(row.get("DISPATCHING", "")),
            ranking_1=_optional_text(row.get("Ranking 1")),
            ranking_2=_optional_text(row.get("Ranking 2")),
            ranking_3=_optional_text(row.get("Ranking 3")),
        )
        specs[spec.name] = spec
    return specs


def _load_pm_specs(path: Path) -> dict[str, PMSpec]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    specs = {}
    for row in df.to_dict("records"):
        spec = PMSpec(
            name=_text(row["PM EVENT NAME"]),
            valid_for_type=_text(row["PM EVENT VALID FOR TYPE"]),
            type_name=_text(row["TYPE NAME"]),
            pm_type=_text(row["PM TYPE"]),
            mean_time_before_pm=_float(row["MTBeforePM"]),
            mean_time_before_pm_units=_text(row["MTBPM UNITS"]),
            repair_distribution=_text(row["TTR DISTRIBUTION"]),
            repair_mean=_float(row["MEAN"]),
            repair_offset=_float(row["OFFSET"]),
            repair_units=_text(row["TTR UNITS"]),
            first_one_distribution=_text(row["FIRST ONE AT DISTRIBUTION"]),
            first_one_at=_float(row["FOA"]),
            first_one_units=_text(row["FOA UNITS"]),
        )
        specs[spec.name] = spec
    return specs


def _load_breakdown_specs(path: Path) -> dict[str, BreakdownSpec]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    specs = {}
    for row in df.to_dict("records"):
        spec = BreakdownSpec(
            name=_text(row["DOWN EVENT NAME"]),
            valid_for_type=_text(row["DOWN EVENT VALID FOR TYPE"]),
            type_name=_text(row["TYPE NAME"]),
            down_type=_text(row["DOWN TYPE"]),
            ttf_distribution=_text(row["TTF DISTRIBUTION"]),
            mean_time_to_failure=_float(row["MTTF"]),
            mean_time_to_failure_units=_text(row["MTTF UNITS"]),
            repair_distribution=_text(row["TTR DISTRIBUTION"]),
            mean_time_to_repair=_float(row["MTTR"]),
            repair_units=_text(row["MTTR UNITS"]),
            first_one_distribution=_text(row["FIRST ONE AT DISTRIBUTION"]),
            first_one_at=_float(row["FOA"]),
            first_one_units=_text(row["FOA UNITS"]),
        )
        specs[spec.name] = spec
    return specs


def _load_setup_rules(path: Path) -> dict[str, SetupRule]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rules = {}
    for index, row in enumerate(df.to_dict("records"), start=1):
        current_setup = _text(row["CURRENT SETUP"])
        new_setup = _text(row["NEW SETUP"])
        setup_group = _optional_text(row.get("SETUP GROUP NAME"))
        key = setup_group or f"{current_setup}->{new_setup}"
        if key in rules:
            key = f"{key}#{index}"
        rules[key] = SetupRule(
            key=key,
            setup_group_name=setup_group,
            current_setup=current_setup,
            new_setup=new_setup,
            setup_time=_float(row["SETUP TIME"]),
            setup_units=_text(row["ST UNITS"]),
            minimal_number_of_runs=_optional_float(row.get("MINMAL NUMBER OF RUNS")),
        )
    return rules


def _load_releases(
    path: Path,
    release_scenario: str | None,
    max_lots: int | None,
) -> tuple[ReleaseSpec, ...]:
    columns = pd.read_csv(path, nrows=0).columns
    required_columns = [
        "PRODUCT NAME",
        "ROUTE NAME",
        "LOT NAME/TYPE",
        "PRIORITY",
        "WAFERS PER LOT",
        "START DATE",
    ]
    missing = sorted(set(required_columns) - set(columns))
    if missing:
        raise ValueError(f"Release file is missing required columns: {missing}")
    optional_columns = [
        column
        for column in ("SUPERHOTLOT", "DUE DATE", "Release Scenario")
        if column in columns
    ]
    usecols = required_columns + optional_columns
    dtype = {
        "PRODUCT NAME": "string",
        "ROUTE NAME": "string",
        "LOT NAME/TYPE": "string",
        "PRIORITY": "int16",
        "WAFERS PER LOT": "int16",
    }
    if "SUPERHOTLOT" in usecols:
        dtype["SUPERHOTLOT"] = "string"
    if "Release Scenario" in usecols:
        dtype["Release Scenario"] = "string"
    parse_dates = ["START DATE"]
    if "DUE DATE" in usecols:
        parse_dates.append("DUE DATE")
    nrows = max_lots if release_scenario is None else None
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        nrows=nrows,
    )
    if release_scenario is not None and "Release Scenario" in df.columns:
        df = df[df["Release Scenario"] == release_scenario]
        if max_lots is not None:
            df = df.iloc[:max_lots]
    if df.empty:
        raise ValueError(f"No releases loaded from {path}")
    if "SUPERHOTLOT" not in df.columns:
        df["SUPERHOTLOT"] = "no"
    if "DUE DATE" not in df.columns:
        df["DUE DATE"] = pd.NaT
    if "Release Scenario" not in df.columns:
        df["Release Scenario"] = pd.NA
    releases = tuple(
        ReleaseSpec(
            product_name=str(product_name).strip(),
            route_name=str(route_name).strip(),
            lot_name=str(lot_name).strip(),
            priority=int(priority),
            super_hot_lot=str(super_hot_lot).strip().lower() == "yes",
            wafers_per_lot=int(wafers_per_lot),
            start_date=start_date.to_pydatetime(),
            due_date=None if pd.isna(due_date) else due_date.to_pydatetime(),
            release_scenario=_optional_text(release_scenario_value),
        )
        for (
            product_name,
            route_name,
            lot_name,
            priority,
            wafers_per_lot,
            start_date,
            super_hot_lot,
            due_date,
            release_scenario_value,
        ) in df[
            [
                "PRODUCT NAME",
                "ROUTE NAME",
                "LOT NAME/TYPE",
                "PRIORITY",
                "WAFERS PER LOT",
                "START DATE",
                "SUPERHOTLOT",
                "DUE DATE",
                "Release Scenario",
            ]
        ].itertuples(index=False, name=None)
    )
    return tuple(sorted(releases, key=lambda release: release.start_date))


def _load_transport(path: Path) -> TransportRule | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return TransportRule(
        from_location=_text(row["FROM LOCATION"]),
        to_location=_text(row["TO LOCATION"]),
        distribution=_text(row["TRANSPORTTIME DISTRIBUTION"]),
        mean=_float(row["MEAN"]),
        offset=_float(row["OFFSET"]),
        units=_text(row["TT UNITS"]),
    )


def _apply_dataclass_overrides(
    items: dict[str, Any],
    overrides: dict[str, dict[str, Any]] | None,
    label: str,
) -> dict[str, Any]:
    if not overrides:
        return items
    updated = dict(items)
    for key, values in overrides.items():
        if key not in updated:
            raise ValueError(f"Cannot override unknown {label}: {key}")
        updated[key] = _replace_dataclass(updated[key], values, f"{label} {key}")
    return updated


def _apply_transport_override(
    transport: TransportRule | None,
    override: dict[str, Any] | None,
) -> TransportRule | None:
    if override is None:
        return transport
    if transport is None:
        return TransportRule(**override)
    return _replace_dataclass(transport, override, "transport")


def _replace_dataclass(instance: Any, values: dict[str, Any], label: str) -> Any:
    allowed = {field.name for field in fields(instance)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown override keys for {label}: {', '.join(unknown)}")
    return replace(instance, **values)


def _validate_references(
    routes: dict[str, ProductRoute],
    toolgroups: dict[str, ToolGroupSpec],
    releases: tuple[ReleaseSpec, ...],
    pm_specs: dict[str, PMSpec],
    breakdown_specs: dict[str, BreakdownSpec],
) -> None:
    missing_toolgroups = sorted(
        {
            step.toolgroup
            for route in routes.values()
            for step in route.steps
            if step.toolgroup not in toolgroups
        }
    )
    if missing_toolgroups:
        raise ValueError(f"Routes reference unknown tool groups: {missing_toolgroups}")
    missing_routes = sorted({release.route_name for release in releases if release.route_name not in routes})
    if missing_routes:
        raise ValueError(f"Releases reference unknown routes: {missing_routes}")
    missing_pm_toolgroups = sorted(
        {
            spec.type_name
            for spec in pm_specs.values()
            if spec.valid_for_type == "toolgroup" and spec.type_name not in toolgroups
        }
    )
    if missing_pm_toolgroups:
        raise ValueError(f"PM specs reference unknown tool groups: {missing_pm_toolgroups}")
    valid_areas = {toolgroup.area for toolgroup in toolgroups.values()}
    missing_breakdown_areas = sorted(
        {
            spec.type_name
            for spec in breakdown_specs.values()
            if spec.valid_for_type == "area" and spec.type_name not in valid_areas
        }
    )
    if missing_breakdown_areas:
        raise ValueError(f"Breakdown specs reference unknown areas: {missing_breakdown_areas}")


def _text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    return float(value)


def _optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(float(value))


def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)
