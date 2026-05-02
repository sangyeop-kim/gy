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
class ProductRoute:
    name: str
    product_name: str
    steps: tuple[RouteStep, ...]

    def remaining_nominal_minutes(self, step_index: int) -> float:
        return sum(step.process_time.mean for step in self.steps[step_index:])


Route = ProductRoute


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
class TransportRule:
    from_location: str
    to_location: str
    distribution: str
    mean: float
    offset: float
    units: str


@dataclass(frozen=True)
class ReleaseSpec:
    product_name: str
    route_name: str
    lot_name: str
    priority: int
    super_hot_lot: bool
    wafers_per_lot: int
    start_date: datetime
    due_date: datetime | None
    release_scenario: str | None


Release = ReleaseSpec


@dataclass(frozen=True)
class ReleasePlan:
    releases: tuple[ReleaseSpec, ...]
    start_datetime: datetime

    def release_time_minutes(self, release: ReleaseSpec) -> float:
        return (release.start_date - self.start_datetime).total_seconds() / 60.0

    def due_time_minutes(self, release: ReleaseSpec) -> float | None:
        if release.due_date is None:
            return None
        return (release.due_date - self.start_datetime).total_seconds() / 60.0


@dataclass(frozen=True)
class ToolGroupSpec:
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
class PMSpec:
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

    @property
    def is_counter_based(self) -> bool:
        return self.mean_time_before_pm_units.lower() == "wafer"


PMEvent = PMSpec


@dataclass(frozen=True)
class BreakdownSpec:
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


BreakdownEvent = BreakdownSpec


@dataclass
class Tool:
    id: str
    index: int
    group_name: str
    setup_state: str | None = None
    busy: bool = False
    busy_until: float | None = None
    current_lot_ids: tuple[int, ...] = ()
    down_reasons: set[str] = field(default_factory=set)
    calendar_pm_rules: tuple[PMSpec, ...] = ()
    counter_pm_rules: tuple[PMSpec, ...] = ()
    breakdown_rules: tuple[BreakdownSpec, ...] = ()
    pm_wafer_counter: dict[str, float] = field(default_factory=dict)
    active_pm: set[str] = field(default_factory=set)

    @property
    def down(self) -> bool:
        return bool(self.down_reasons)

    @property
    def idle(self) -> bool:
        return not self.busy and not self.down

    def start(self, lot_ids: tuple[int, ...], completion_time: float) -> None:
        self.busy = True
        self.busy_until = completion_time
        self.current_lot_ids = lot_ids

    def finish(self) -> tuple[int, ...]:
        lot_ids = self.current_lot_ids
        self.busy = False
        self.busy_until = None
        self.current_lot_ids = ()
        return lot_ids

    def mark_down(self, reason: str) -> None:
        self.down_reasons.add(reason)

    def clear_down(self, reason: str) -> None:
        self.down_reasons.discard(reason)


@dataclass
class ToolGroup:
    spec: ToolGroupSpec
    tools: list[Tool]
    waiting_lot_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_spec(
        cls,
        spec: ToolGroupSpec,
        calendar_pm_rules: tuple[PMSpec, ...],
        counter_pm_rules: tuple[PMSpec, ...],
        breakdown_rules: tuple[BreakdownSpec, ...],
    ) -> "ToolGroup":
        tools = [
            Tool(
                id=f"{spec.name}#{index}",
                index=index,
                group_name=spec.name,
                calendar_pm_rules=calendar_pm_rules,
                counter_pm_rules=counter_pm_rules,
                breakdown_rules=breakdown_rules,
            )
            for index in range(spec.number_of_tools)
        ]
        return cls(spec=spec, tools=tools)

    @property
    def idle_tools(self) -> list[Tool]:
        return [tool for tool in self.tools if tool.idle]

    def tool(self, index: int) -> Tool:
        return self.tools[index]

    def add_waiting_lot(self, lot_id: int) -> None:
        self.waiting_lot_ids.append(lot_id)

    def remove_waiting_lot(self, lot_id: int) -> None:
        self.waiting_lot_ids.remove(lot_id)


ToolGroupSpecAlias = ToolGroupSpec


@dataclass
class Lot:
    id: int
    name: str
    product_name: str
    route: ProductRoute
    priority: int
    super_hot_lot: bool
    wafers_per_lot: int
    release_time: float
    due_time: float | None
    step_index: int = 0
    completed_time: float | None = None
    current_toolgroup: str | None = None
    waiting_since: float | None = None
    history: list[tuple[int, str, int, float, float]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.step_index >= len(self.route.steps)

    @property
    def current_step(self) -> RouteStep:
        return self.route.steps[self.step_index]


@dataclass(frozen=True)
class FabModel:
    routes: Mapping[str, ProductRoute]
    toolgroup_specs: Mapping[str, ToolGroupSpec]
    pm_specs: Mapping[str, PMSpec]
    breakdown_specs: Mapping[str, BreakdownSpec]
    setup_rules: Mapping[str, SetupRule]
    release_plan: ReleasePlan
    transport: TransportRule | None
    start_datetime: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", MappingProxyType(dict(self.routes)))
        object.__setattr__(self, "toolgroup_specs", MappingProxyType(dict(self.toolgroup_specs)))
        object.__setattr__(self, "pm_specs", MappingProxyType(dict(self.pm_specs)))
        object.__setattr__(self, "breakdown_specs", MappingProxyType(dict(self.breakdown_specs)))
        object.__setattr__(self, "setup_rules", MappingProxyType(dict(self.setup_rules)))

    @property
    def toolgroups(self) -> Mapping[str, ToolGroupSpec]:
        return self.toolgroup_specs

    @property
    def pm_events(self) -> Mapping[str, PMSpec]:
        return self.pm_specs

    @property
    def breakdown_events(self) -> Mapping[str, BreakdownSpec]:
        return self.breakdown_specs

    @property
    def releases(self) -> tuple[ReleaseSpec, ...]:
        return self.release_plan.releases
