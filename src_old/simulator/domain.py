from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ProcessTime:
    distribution: str | None
    mean: float
    offset: float
    units: str


@dataclass(frozen=True)
class RouteStep:
    route: str
    step_number: int
    description: str
    area: str
    toolgroup: str
    processing_unit: str
    process_time: ProcessTime
    cascading_interval: float | None = None
    batch_minimum: float | None = None
    batch_maximum: float | None = None
    setup: str | None = None
    setup_time: float | None = None
    setup_units: str | None = None
    sampling_probability: float | None = None
    rework_probability: float | None = None
    step_for_rework: int | None = None
    cqt_start_step: int | None = None
    cqt: float | None = None
    cqt_units: str | None = None


@dataclass(frozen=True)
class Route:
    name: str
    product_name: str
    steps: tuple[RouteStep, ...]

    def remaining_nominal_minutes(self, step_index: int) -> float:
        return sum(step.process_time.mean for step in self.steps[step_index:])


@dataclass(frozen=True)
class ToolGroup:
    area: str
    name: str
    number_of_tools: int
    loading_time: float
    unloading_time: float
    dispatching: str
    ranking_1: str | None = None
    ranking_2: str | None = None
    ranking_3: str | None = None


@dataclass(frozen=True)
class TransportRule:
    from_location: str
    to_location: str
    distribution: str
    mean: float
    offset: float
    units: str


@dataclass(frozen=True)
class PMEvent:
    name: str
    valid_for_type: str
    type_name: str
    pm_type: str
    mean_time_before_pm: float
    mean_time_before_pm_units: str
    repair_distribution: str
    repair_mean: float
    repair_offset: float
    repair_units: str
    first_one_distribution: str
    first_one_at: float
    first_one_units: str


@dataclass(frozen=True)
class BreakdownEvent:
    name: str
    valid_for_type: str
    type_name: str
    down_type: str
    ttf_distribution: str
    mean_time_to_failure: float
    mean_time_to_failure_units: str
    repair_distribution: str
    mean_time_to_repair: float
    repair_units: str
    first_one_distribution: str
    first_one_at: float
    first_one_units: str


@dataclass(frozen=True)
class SetupRule:
    key: str
    setup_group_name: str | None
    current_setup: str
    new_setup: str
    setup_time: float
    setup_units: str
    minimal_number_of_runs: float | None = None


@dataclass(frozen=True)
class Release:
    product_name: str
    route_name: str
    lot_name: str
    priority: int
    super_hot_lot: bool
    wafers_per_lot: int
    start_date: datetime
    due_date: datetime | None
    release_scenario: str | None


@dataclass(frozen=True)
class FabModel:
    """Fixed input data loaded from CSV files.

    These fields represent the immutable part of an experiment: products,
    routes, tool groups, and release definitions from SMT2020.
    """

    routes: Mapping[str, Route]
    toolgroups: Mapping[str, ToolGroup]
    pm_events: Mapping[str, PMEvent]
    breakdown_events: Mapping[str, BreakdownEvent]
    setup_rules: Mapping[str, SetupRule]
    releases: tuple[Release, ...]
    transport: TransportRule | None
    start_datetime: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", MappingProxyType(dict(self.routes)))
        object.__setattr__(self, "toolgroups", MappingProxyType(dict(self.toolgroups)))
        object.__setattr__(self, "pm_events", MappingProxyType(dict(self.pm_events)))
        object.__setattr__(
            self,
            "breakdown_events",
            MappingProxyType(dict(self.breakdown_events)),
        )
        object.__setattr__(self, "setup_rules", MappingProxyType(dict(self.setup_rules)))


@dataclass
class Lot:
    id: int
    name: str
    product_name: str
    route: Route
    priority: int
    super_hot_lot: bool
    wafers_per_lot: int
    release_time: float
    due_time: float | None
    step_index: int = 0
    completed_time: float | None = None
    current_toolgroup: str | None = None
    history: list[tuple[int, str, float, float]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.step_index >= len(self.route.steps)

    @property
    def current_step(self) -> RouteStep:
        return self.route.steps[self.step_index]
