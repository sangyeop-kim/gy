from datetime import datetime
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulator.config import SimulationConfig
from src.simulator.domain import (
    BreakdownEvent,
    FabModel,
    PMEvent,
    ProcessTime,
    Release,
    Route,
    RouteStep,
    SetupRule,
    ToolGroup,
    TransportRule,
)


def load_model(config: SimulationConfig) -> FabModel:
    dataset_dir = config.dataset_path()
    routes = _load_routes(dataset_dir, config.route_file_glob)
    toolgroups = _load_toolgroups(dataset_dir / config.toolgroups_file)
    toolgroups = _apply_toolgroup_overrides(toolgroups, config.toolgroup_overrides)
    pm_events = _load_pm_events(dataset_dir / config.pm_file)
    pm_events = _apply_dataclass_overrides(pm_events, config.pm_overrides, "PM event")
    breakdown_events = _load_breakdown_events(dataset_dir / config.breakdown_file)
    breakdown_events = _apply_dataclass_overrides(
        breakdown_events,
        config.breakdown_overrides,
        "breakdown event",
    )
    setup_rules = _load_setup_rules(dataset_dir / config.setups_file)
    setup_rules = _apply_dataclass_overrides(
        setup_rules, config.setup_overrides, "setup rule"
    )
    releases = _load_releases(
        dataset_dir / config.release_file,
        config.release_scenario,
        config.max_lots,
    )
    transport = _load_transport(dataset_dir / config.transport_file)
    transport = _apply_transport_override(transport, config.transport_override)

    _validate_references(routes, toolgroups, releases, pm_events, breakdown_events)
    start_datetime = min(release.start_date for release in releases)
    return FabModel(
        routes=routes,
        toolgroups=toolgroups,
        pm_events=pm_events,
        breakdown_events=breakdown_events,
        setup_rules=setup_rules,
        releases=releases,
        transport=transport,
        start_datetime=start_datetime,
    )


def _load_routes(dataset_dir: Path, route_file_glob: str) -> dict[str, Route]:
    routes: dict[str, Route] = {}
    route_paths = sorted(
        dataset_dir.glob(route_file_glob),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not route_paths:
        raise FileNotFoundError(f"No route CSV files found in {dataset_dir}")

    for path in route_paths:
        df = pd.read_csv(path)
        steps: list[RouteStep] = []
        for row in df.to_dict("records"):
            step = RouteStep(
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
                sampling_probability=_optional_float(
                    row.get("PROCESSING PROBABILITY in % (Sampling)")
                ),
                rework_probability=_optional_float(row.get("REWORK PROBABILITY in %")),
                step_for_rework=_optional_int(row.get("STEP FOR REWORK")),
                cqt_start_step=_optional_int(row.get("STEP FOR CRITICAL QUEUE TIME")),
                cqt=_optional_float(row.get("CQT")),
                cqt_units=_optional_text(row.get("CQTUNITS")),
            )
            steps.append(step)

        route_name = steps[0].route
        product_name = route_name.replace("Route_", "")
        routes[route_name] = Route(
            name=route_name,
            product_name=product_name,
            steps=tuple(sorted(steps, key=lambda step: step.step_number)),
        )
    return routes


def _load_toolgroups(path: Path) -> dict[str, ToolGroup]:
    df = pd.read_csv(path)
    toolgroups: dict[str, ToolGroup] = {}
    for row in df.to_dict("records"):
        toolgroup = ToolGroup(
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
        toolgroups[toolgroup.name] = toolgroup
    return toolgroups


def _load_pm_events(path: Path) -> dict[str, PMEvent]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    events: dict[str, PMEvent] = {}
    for row in df.to_dict("records"):
        event = PMEvent(
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
        events[event.name] = event
    return events


def _load_breakdown_events(path: Path) -> dict[str, BreakdownEvent]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    events: dict[str, BreakdownEvent] = {}
    for row in df.to_dict("records"):
        event = BreakdownEvent(
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
        events[event.name] = event
    return events


def _load_setup_rules(path: Path) -> dict[str, SetupRule]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rules: dict[str, SetupRule] = {}
    for index, row in enumerate(df.to_dict("records"), start=1):
        current_setup = _text(row["CURRENT SETUP"])
        new_setup = _text(row["NEW SETUP"])
        setup_group = _optional_text(row.get("SETUP GROUP NAME"))
        key = setup_group or f"{current_setup}->{new_setup}"
        if key in rules:
            key = f"{key}#{index}"
        rule = SetupRule(
            key=key,
            setup_group_name=setup_group,
            current_setup=current_setup,
            new_setup=new_setup,
            setup_time=_float(row["SETUP TIME"]),
            setup_units=_text(row["ST UNITS"]),
            minimal_number_of_runs=_optional_float(row.get("MINMAL NUMBER OF RUNS")),
        )
        rules[rule.key] = rule
    return rules


def _load_releases(
    path: Path,
    release_scenario: str | None,
    max_lots: int | None,
) -> tuple[Release, ...]:
    # When filtering by scenario we must read all rows first, then slice
    nrows = max_lots if release_scenario is None else None
    df = pd.read_csv(path, low_memory=False, nrows=nrows)

    if release_scenario is not None and "Release Scenario" in df.columns:
        df = df[df["Release Scenario"] == release_scenario]
        if max_lots is not None:
            df = df.iloc[:max_lots]

    if df.empty:
        raise ValueError(f"No releases loaded from {path}")

    # Vectorized datetime parsing — much faster than calling pd.to_datetime per row
    start_dates: list[datetime] = pd.to_datetime(df["START DATE"]).dt.to_pydatetime().tolist()

    if "DUE DATE" in df.columns:
        due_parsed = pd.to_datetime(df["DUE DATE"], errors="coerce")
        due_dates: list[datetime | None] = [
            None if pd.isna(v) else v.to_pydatetime() for v in due_parsed
        ]
    else:
        due_dates = [None] * len(df)

    # Extract columns as lists up-front to avoid per-row DataFrame access
    product_names = df["PRODUCT NAME"].astype(str).str.strip().tolist()
    route_names = df["ROUTE NAME"].astype(str).str.strip().tolist()
    lot_names = df["LOT NAME/TYPE"].astype(str).str.strip().tolist()
    priorities = df["PRIORITY"].tolist()
    wafers = df["WAFERS PER LOT"].tolist()
    super_hot = (
        (df["SUPERHOTLOT"].astype(str).str.strip().str.lower() == "yes").tolist()
        if "SUPERHOTLOT" in df.columns
        else [False] * len(df)
    )
    scenarios: list[str | None] = (
        [_optional_text(v) for v in df["Release Scenario"]]
        if "Release Scenario" in df.columns
        else [None] * len(df)
    )

    releases = [
        Release(
            product_name=product_names[i],
            route_name=route_names[i],
            lot_name=lot_names[i],
            priority=int(priorities[i]),
            super_hot_lot=super_hot[i],
            wafers_per_lot=int(wafers[i]),
            start_date=start_dates[i],
            due_date=due_dates[i],
            release_scenario=scenarios[i],
        )
        for i in range(len(df))
    ]

    return tuple(sorted(releases, key=lambda r: r.start_date))


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


def _apply_toolgroup_overrides(
    toolgroups: dict[str, ToolGroup],
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, ToolGroup]:
    return _apply_dataclass_overrides(toolgroups, overrides, "tool group")


def _apply_transport_override(
    transport: TransportRule | None,
    override: dict[str, Any] | None,
) -> TransportRule | None:
    if override is None:
        return transport
    if transport is None:
        required = {"from_location", "to_location", "distribution", "mean", "offset", "units"}
        missing = sorted(required - set(override))
        if missing:
            raise ValueError(
                "transport_override must include these keys when no transport file "
                f"is loaded: {', '.join(missing)}"
            )
        return TransportRule(**override)
    return _replace_dataclass(transport, override, "transport")


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


def _replace_dataclass(instance: Any, values: dict[str, Any], label: str) -> Any:
    allowed = {field.name for field in fields(instance)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown override keys for {label}: {', '.join(unknown)}")
    return replace(instance, **values)


def _validate_references(
    routes: dict[str, Route],
    toolgroups: dict[str, ToolGroup],
    releases: tuple[Release, ...],
    pm_events: dict[str, PMEvent],
    breakdown_events: dict[str, BreakdownEvent],
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

    missing_routes = sorted(
        {release.route_name for release in releases if release.route_name not in routes}
    )
    if missing_routes:
        raise ValueError(f"Releases reference unknown routes: {missing_routes}")

    missing_pm_toolgroups = sorted(
        {
            event.type_name
            for event in pm_events.values()
            if event.valid_for_type == "toolgroup" and event.type_name not in toolgroups
        }
    )
    if missing_pm_toolgroups:
        raise ValueError(f"PM events reference unknown tool groups: {missing_pm_toolgroups}")

    valid_areas = {toolgroup.area for toolgroup in toolgroups.values()}
    missing_breakdown_areas = sorted(
        {
            event.type_name
            for event in breakdown_events.values()
            if event.valid_for_type == "area" and event.type_name not in valid_areas
        }
    )
    if missing_breakdown_areas:
        raise ValueError(
            f"Breakdown events reference unknown areas: {missing_breakdown_areas}"
        )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def _optional_datetime(value: Any) -> datetime | None:
    if _is_missing(value):
        return None
    return _parse_datetime(value)


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
