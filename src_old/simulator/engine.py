from dataclasses import dataclass, field
from datetime import datetime
import heapq
from itertools import count

from src.simulator.config import SimulationConfig
from src.simulator.domain import BreakdownEvent, FabModel, Lot, PMEvent, Release, RouteStep, ToolGroup
from src.simulator.utils import RandomStream, SimulationSummary, select_lot, summarize_lots


MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = 24.0 * MINUTES_PER_HOUR


@dataclass(order=True)
class Event:
    time: float
    sequence: int
    type: str = field(compare=False)
    lot_id: int = field(compare=False)
    toolgroup: str | None = field(default=None, compare=False)
    batch_id: int | None = field(default=None, compare=False)
    payload: str | None = field(default=None, compare=False)
    tool_index: int | None = field(default=None, compare=False)
    downtime_duration: float | None = field(default=None, compare=False)


@dataclass
class ToolRuntimeState:
    index: int
    busy: bool = False
    busy_until: float | None = None
    setup_state: str | None = None
    down_reasons: set[str] = field(default_factory=set)
    pm_wafer_counter: dict[str, float] = field(default_factory=dict)
    pm_active: set[str] = field(default_factory=set)
    calendar_pm_rules: tuple[PMEvent, ...] = ()
    counter_pm_rules: tuple[PMEvent, ...] = ()
    breakdown_rules: tuple[BreakdownEvent, ...] = ()

    @property
    def down(self) -> bool:
        return bool(self.down_reasons)

    @property
    def idle(self) -> bool:
        return not self.busy and not self.down


@dataclass
class ToolGroupState:
    spec: ToolGroup
    queue: list[int] = field(default_factory=list)
    tools: list[ToolRuntimeState] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools = [
                ToolRuntimeState(index=tool_index)
                for tool_index in range(self.spec.number_of_tools)
            ]

    def tool(self, tool_index: int) -> ToolRuntimeState:
        return self.tools[tool_index]

    @property
    def busy(self) -> int:
        return sum(tool.busy for tool in self.tools)

    @property
    def unavailable(self) -> int:
        return sum(tool.down for tool in self.tools)

    @property
    def idle_capacity(self) -> int:
        return len(self.idle_tool_indices)

    @property
    def idle_tool_indices(self) -> list[int]:
        return [tool.index for tool in self.tools if tool.idle]


@dataclass(frozen=True)
class SimulationResult:
    summary: SimulationSummary
    lots: tuple[Lot, ...]
    event_log: tuple[dict[str, float | int | str | None], ...]


class Simulator:
    """Discrete-event simulator for SMT2020 dataset 2."""

    def __init__(self, model: FabModel, config: SimulationConfig) -> None:
        self.model = model
        self.config = config
        self.random = RandomStream(config.random_seed)
        self.now = 0.0
        self._sequence = count()
        self._events: list[Event] = []
        self._lots: dict[int, Lot] = {}
        self._event_log: list[dict[str, float | int | str | None]] = []
        self._batch_sequence = count(1)
        self._batches: dict[int, tuple[int, ...]] = {}
        self._cqt_open: dict[tuple[int, int], tuple[float, float]] = {}
        self._cqt_lateness: list[float] = []
        self._calendar_pm_by_toolgroup = self._index_calendar_pm_by_toolgroup()
        self._counter_pm_by_toolgroup = self._index_counter_pm_by_toolgroup()
        self._breakdowns_by_toolgroup = self._index_breakdowns_by_toolgroup()
        self._setup_times = self._index_setup_times()
        self._toolgroups = {
            name: ToolGroupState(spec=toolgroup)
            for name, toolgroup in model.toolgroups.items()
        }
        self._attach_tool_rules()
        self._defer_dispatch = False
        self._pending_dispatch_toolgroups: set[str] = set()

    def run(self) -> SimulationResult:
        self._schedule_releases()
        self._schedule_initial_calendar_pm()
        self._schedule_initial_breakdowns()
        while self._events:
            event = heapq.heappop(self._events)
            if self.config.until_minutes is not None and event.time > self.config.until_minutes:
                break
            self.now = event.time
            self._process_current_time_events(event)
            self._dispatch_pending_toolgroups()
            if self._all_released_lots_complete():
                break

        lots = [self._lots[lot_id] for lot_id in sorted(self._lots)]
        summary = summarize_lots(lots, self.config.warmup_minutes, self._cqt_lateness)
        return SimulationResult(
            summary=summary,
            lots=tuple(lots),
            event_log=tuple(self._event_log),
        )

    def _process_current_time_events(self, first_event: Event) -> None:
        self._defer_dispatch = True
        self._pending_dispatch_toolgroups.clear()
        self._process_event(first_event)
        while self._events and self._events[0].time == self.now:
            self._process_event(heapq.heappop(self._events))
        self._defer_dispatch = False

    def _process_event(self, event: Event) -> None:
        if event.type == "release":
            self._handle_release(event.lot_id)
        elif event.type == "arrive":
            self._handle_arrival(event.lot_id)
        elif event.type == "complete":
            if event.toolgroup is None or event.tool_index is None:
                raise ValueError("Completion event is missing a tool group")
            self._handle_completion(event.lot_id, event.toolgroup, event.tool_index)
        elif event.type == "complete_batch":
            if event.toolgroup is None or event.batch_id is None or event.tool_index is None:
                raise ValueError("Batch completion event is missing data")
            self._handle_batch_completion(event.batch_id, event.toolgroup, event.tool_index)
        elif event.type == "downtime_start":
            if (
                event.toolgroup is None
                or event.payload is None
                or event.tool_index is None
                or event.downtime_duration is None
            ):
                raise ValueError("Downtime start event is missing data")
            self._handle_downtime_start(
                event.toolgroup,
                event.tool_index,
                event.payload,
                event.downtime_duration,
            )
        elif event.type == "downtime_end":
            if event.toolgroup is None or event.payload is None or event.tool_index is None:
                raise ValueError("Downtime end event is missing data")
            self._handle_downtime_end(event.toolgroup, event.tool_index, event.payload)
        else:
            raise ValueError(f"Unknown event type: {event.type}")

    def _request_dispatch(self, state: ToolGroupState) -> None:
        if self._defer_dispatch:
            self._pending_dispatch_toolgroups.add(state.spec.name)
        else:
            self._try_dispatch(state)

    def _dispatch_pending_toolgroups(self) -> None:
        while self._pending_dispatch_toolgroups:
            toolgroup_names = sorted(self._pending_dispatch_toolgroups)
            self._pending_dispatch_toolgroups.clear()
            for toolgroup_name in toolgroup_names:
                self._try_dispatch(self._toolgroups[toolgroup_name])

    def _all_released_lots_complete(self) -> bool:
        return bool(self._lots) and all(lot.completed_time is not None for lot in self._lots.values())

    def _schedule_releases(self) -> None:
        releases = self.model.releases
        if self.config.max_lots is not None:
            releases = releases[: self.config.max_lots]

        for lot_id, release in enumerate(releases, start=1):
            lot = self._lot_from_release(lot_id, release)
            self._lots[lot_id] = lot
            self._push("release", lot.release_time, lot_id)

    def _lot_from_release(self, lot_id: int, release: Release) -> Lot:
        route = self.model.routes[release.route_name]
        return Lot(
            id=lot_id,
            name=release.lot_name,
            product_name=release.product_name,
            route=route,
            priority=release.priority,
            super_hot_lot=release.super_hot_lot,
            wafers_per_lot=release.wafers_per_lot,
            release_time=self._minutes_from_start(release.start_date),
            due_time=self._minutes_from_start(release.due_date) if release.due_date else None,
        )

    def _handle_release(self, lot_id: int) -> None:
        self._log("release", lot_id)
        self._push("arrive", self.now, lot_id)

    def _handle_arrival(self, lot_id: int) -> None:
        lot = self._lots[lot_id]
        self._skip_unsampled_steps(lot)
        if lot.is_complete:
            lot.completed_time = self.now
            self._log("complete_lot", lot_id)
            return

        state = self._toolgroups[lot.current_step.toolgroup]
        lot.current_toolgroup = state.spec.name
        state.queue.append(lot_id)
        self._log("arrive", lot_id, state.spec.name)
        self._check_cqt_end(lot)
        self._request_dispatch(state)

    def _try_dispatch(self, state: ToolGroupState) -> None:
        while state.idle_capacity > 0 and state.queue:
            candidates = [self._lots[lot_id] for lot_id in state.queue]
            lot = select_lot(candidates, self.now, self.config.dispatching_rule)
            if self._is_batch_step(lot.current_step):
                batch_lots = self._select_batch_lots(state, lot)
                if batch_lots is None:
                    return
                self._start_batch(state, batch_lots)
            else:
                state.queue.remove(lot.id)
                self._start_single_lot(state, lot)

    def _start_single_lot(self, state: ToolGroupState, lot: Lot) -> None:
        tool_index = self._select_tool_index(state, lot.current_step)
        tool = state.tool(tool_index)
        tool.busy = True
        setup_minutes = self._setup_duration(state, lot.current_step, tool_index)
        process_minutes = self._step_duration(lot.current_step, lot, batch_lots=(lot,))
        completion_time = self.now + setup_minutes + process_minutes
        tool.busy_until = completion_time
        lot.history.append((lot.current_step.step_number, state.spec.name, self.now, completion_time))
        self._log("start", lot.id, state.spec.name, tool_index=tool_index)
        if setup_minutes > 0:
            self._log("setup", lot.id, state.spec.name, tool_index=tool_index)
        self._push("complete", completion_time, lot.id, state.spec.name, tool_index=tool_index)

    def _start_batch(self, state: ToolGroupState, lots: list[Lot]) -> None:
        for lot in lots:
            state.queue.remove(lot.id)
        lead_lot = lots[0]
        tool_index = self._select_tool_index(state, lead_lot.current_step)
        tool = state.tool(tool_index)
        tool.busy = True
        setup_minutes = self._setup_duration(state, lead_lot.current_step, tool_index)
        process_minutes = self._step_duration(lead_lot.current_step, lead_lot, batch_lots=tuple(lots))
        completion_time = self.now + setup_minutes + process_minutes
        tool.busy_until = completion_time
        batch_id = next(self._batch_sequence)
        self._batches[batch_id] = tuple(lot.id for lot in lots)
        for lot in lots:
            lot.history.append((lot.current_step.step_number, state.spec.name, self.now, completion_time))
            self._log("start_batch", lot.id, state.spec.name, tool_index=tool_index)
        if setup_minutes > 0:
            self._log("setup", lead_lot.id, state.spec.name, tool_index=tool_index)
        self._push(
            "complete_batch",
            completion_time,
            lead_lot.id,
            state.spec.name,
            batch_id=batch_id,
            tool_index=tool_index,
        )

    def _handle_completion(self, lot_id: int, toolgroup_name: str, tool_index: int) -> None:
        lot = self._lots[lot_id]
        state = self._toolgroups[toolgroup_name]
        tool = state.tool(tool_index)
        tool.busy = False
        tool.busy_until = None
        self._log("complete_step", lot_id, toolgroup_name, tool_index=tool_index)
        self._after_step_completion(lot, state, tool_index, lot.wafers_per_lot)
        self._request_dispatch(state)

    def _handle_batch_completion(self, batch_id: int, toolgroup_name: str, tool_index: int) -> None:
        state = self._toolgroups[toolgroup_name]
        tool = state.tool(tool_index)
        tool.busy = False
        tool.busy_until = None
        lot_ids = self._batches.pop(batch_id)
        batch_wafers = sum(self._lots[lot_id].wafers_per_lot for lot_id in lot_ids)
        self._record_counter_pm_usage(state, tool_index, batch_wafers, lot_ids[0])
        for lot_id in lot_ids:
            lot = self._lots[lot_id]
            self._log("complete_batch_step", lot_id, toolgroup_name, tool_index=tool_index)
            self._after_step_completion(
                lot,
                state,
                tool_index,
                wafers_for_counter_pm=0,
            )
        self._request_dispatch(state)

    def _after_step_completion(
        self,
        lot: Lot,
        state: ToolGroupState,
        tool_index: int,
        wafers_for_counter_pm: int,
    ) -> None:
        if wafers_for_counter_pm > 0:
            self._record_counter_pm_usage(state, tool_index, wafers_for_counter_pm, lot.id)
        self._check_cqt_start(lot)
        next_step_index = self._next_step_index(lot)
        lot.step_index = next_step_index
        lot.current_toolgroup = None

        if lot.is_complete:
            lot.completed_time = self.now
            self._log("complete_lot", lot.id)
        else:
            arrival_time = self.now + self._transport_duration()
            self._push("arrive", arrival_time, lot.id)

    def _next_step_index(self, lot: Lot) -> int:
        step = lot.current_step
        if self.config.enable_rework and self.random.bernoulli_percent(step.rework_probability):
            if step.step_for_rework is not None:
                return max(0, step.step_for_rework - 1)

        return lot.step_index + 1

    def _skip_unsampled_steps(self, lot: Lot) -> None:
        if not self.config.enable_sampling:
            return
        while not lot.is_complete:
            probability = lot.current_step.sampling_probability
            if probability is None or self.random.bernoulli_percent(probability):
                return
            self._log("skip_sampling_step", lot.id, lot.current_step.toolgroup)
            lot.step_index += 1

    def _step_duration(
        self,
        step: RouteStep,
        lot: Lot,
        batch_lots: tuple[Lot, ...],
    ) -> float:
        process = step.process_time
        if self.config.sample_process_times and process.distribution == "uniform":
            base = self.random.uniform_around(process.mean, process.offset)
        else:
            base = process.mean

        unit = step.processing_unit.lower()
        if unit == "wafer" and self.config.wafer_time_mode == "per_wafer":
            if self.config.enable_cascading and step.cascading_interval is not None:
                base += max(0, lot.wafers_per_lot - 1) * step.cascading_interval
            else:
                base *= lot.wafers_per_lot
        elif unit == "batch" and self.config.enable_batching:
            batch_wafers = sum(batch_lot.wafers_per_lot for batch_lot in batch_lots)
            if self.config.enable_cascading and step.cascading_interval is not None:
                base += max(0, batch_wafers - 1) * step.cascading_interval

        toolgroup = self.model.toolgroups[step.toolgroup]
        return max(0.0, base + toolgroup.loading_time + toolgroup.unloading_time)

    def _transport_duration(self) -> float:
        if not self.config.enable_transport or self.model.transport is None:
            return 0.0
        transport = self.model.transport
        if self.config.sample_transport_times and transport.distribution == "uniform":
            return max(0.0, self.random.uniform_around(transport.mean, transport.offset))
        return max(0.0, transport.mean)

    def _push(
        self,
        event_type: str,
        event_time: float,
        lot_id: int,
        toolgroup: str | None = None,
        batch_id: int | None = None,
        payload: str | None = None,
        tool_index: int | None = None,
        downtime_duration: float | None = None,
    ) -> None:
        heapq.heappush(
            self._events,
            Event(
                time=event_time,
                sequence=next(self._sequence),
                type=event_type,
                lot_id=lot_id,
                toolgroup=toolgroup,
                batch_id=batch_id,
                payload=payload,
                tool_index=tool_index,
                downtime_duration=downtime_duration,
            ),
        )

    def _minutes_from_start(self, value: datetime) -> float:
        return (value - self.model.start_datetime).total_seconds() / 60.0

    def _log(
        self,
        event_type: str,
        lot_id: int,
        toolgroup: str | None = None,
        tool_index: int | None = None,
    ) -> None:
        if not self.config.write_event_log:
            return
        lot = self._lots[lot_id]
        self._event_log.append(
            {
                "time": self.now,
                "event": event_type,
                "lot_id": lot_id,
                "lot_name": lot.name,
                "product_name": lot.product_name,
                "step_index": lot.step_index,
                "toolgroup": toolgroup,
                "tool_index": tool_index,
                "payload": None,
            }
        )

    def _is_batch_step(self, step: RouteStep) -> bool:
        return (
            self.config.enable_batching
            and step.processing_unit.lower() == "batch"
            and step.batch_minimum is not None
        )

    def _select_batch_lots(self, state: ToolGroupState, lead_lot: Lot) -> list[Lot] | None:
        step = lead_lot.current_step
        minimum = step.batch_minimum or 0.0
        maximum = step.batch_maximum or float("inf")
        compatible = [
            self._lots[lot_id]
            for lot_id in state.queue
            if self._lots[lot_id].current_step == step
        ]
        ordered = []
        remaining = compatible
        while remaining and sum(lot.wafers_per_lot for lot in ordered) < maximum:
            selected = select_lot(remaining, self.now, self.config.dispatching_rule)
            if sum(lot.wafers_per_lot for lot in ordered) + selected.wafers_per_lot > maximum:
                break
            ordered.append(selected)
            remaining = [lot for lot in remaining if lot.id != selected.id]

        wafers = sum(lot.wafers_per_lot for lot in ordered)
        if wafers >= minimum or self._no_more_lot_arrivals_expected():
            return ordered
        return None

    def _no_more_lot_arrivals_expected(self) -> bool:
        return not any(event.type in {"release", "arrive"} for event in self._events)

    def _select_tool_index(self, state: ToolGroupState, step: RouteStep) -> int:
        idle = state.idle_tool_indices
        if not idle:
            raise ValueError(f"No idle tool available in {state.spec.name}")
        return min(idle, key=lambda index: self._setup_duration_preview(state, step, index))

    def _setup_duration_preview(
        self,
        state: ToolGroupState,
        step: RouteStep,
        tool_index: int,
    ) -> float:
        if step.setup is None:
            return 0.0
        current = state.tool(tool_index).setup_state
        if current is None or current == step.setup:
            return 0.0
        rule_time = self._setup_times.get((current, step.setup))
        if rule_time is not None:
            return rule_time
        if step.setup_time is not None:
            return self._to_minutes(step.setup_time, step.setup_units)
        return 0.0

    def _setup_duration(self, state: ToolGroupState, step: RouteStep, tool_index: int) -> float:
        duration = self._setup_duration_preview(state, step, tool_index)
        if step.setup is not None:
            state.tool(tool_index).setup_state = step.setup
        return duration

    def _check_cqt_start(self, lot: Lot) -> None:
        step = lot.current_step
        if step.cqt_start_step is None or step.cqt is None:
            return
        limit = self._to_minutes(step.cqt, step.cqt_units)
        self._cqt_open[(lot.id, step.cqt_start_step)] = (self.now, limit)
        self._log("cqt_start", lot.id, step.toolgroup)

    def _check_cqt_end(self, lot: Lot) -> None:
        key = (lot.id, lot.current_step.step_number)
        if key not in self._cqt_open:
            return
        start, limit = self._cqt_open.pop(key)
        elapsed = self.now - start
        if elapsed > limit:
            self._cqt_lateness.append(elapsed - limit)
            self._log("cqt_violation", lot.id, lot.current_step.toolgroup)
        else:
            self._log("cqt_pass", lot.id, lot.current_step.toolgroup)

    def _handle_downtime_start(
        self,
        toolgroup_name: str,
        tool_index: int,
        payload: str,
        downtime_duration: float,
    ) -> None:
        state = self._toolgroups[toolgroup_name]
        tool = state.tool(tool_index)
        if tool.busy:
            self._push(
                "downtime_start",
                tool.busy_until if tool.busy_until is not None else self.now,
                0,
                toolgroup_name,
                payload=payload,
                tool_index=tool_index,
                downtime_duration=downtime_duration,
            )
            return

        parts = payload.split(":", 1)
        kind = parts[0]
        event_name = parts[1] if len(parts) > 1 else payload
        if kind == "pm_counter" and event_name in tool.pm_active:
            return
        if kind == "pm_counter":
            tool.pm_active.add(event_name)
        tool.down_reasons.add(payload)
        self._log_tool_event("downtime_start", toolgroup_name, tool_index, payload)
        self._push(
            "downtime_end",
            self.now + downtime_duration,
            0,
            toolgroup_name,
            payload=payload,
            tool_index=tool_index,
        )

    def _handle_downtime_end(
        self,
        toolgroup_name: str,
        tool_index: int,
        payload: str,
    ) -> None:
        state = self._toolgroups[toolgroup_name]
        tool = state.tool(tool_index)
        parts = payload.split(":", 1)
        kind = parts[0]
        event_name = parts[1] if len(parts) > 1 else payload
        tool.down_reasons.discard(payload)
        if kind == "pm_counter":
            tool.pm_active.discard(event_name)
        self._log_tool_event("downtime_end", toolgroup_name, tool_index, payload)
        self._request_dispatch(state)
        if kind == "pm_calendar":
            pm_event = self.model.pm_events.get(event_name)
            if pm_event is not None:
                self._schedule_next_calendar_pm(pm_event, toolgroup_name, tool_index)
        elif kind == "breakdown":
            breakdown = self.model.breakdown_events.get(event_name)
            if breakdown is not None:
                self._schedule_next_breakdown(breakdown, toolgroup_name, tool_index)

    def _record_counter_pm_usage(
        self,
        state: ToolGroupState,
        tool_index: int,
        wafers_processed: int,
        trigger_lot_id: int,
    ) -> None:
        if not self.config.enable_pm:
            return
        tool = state.tool(tool_index)
        for event in tool.counter_pm_rules:
            if event.name in tool.pm_active:
                continue
            counter = tool.pm_wafer_counter.get(event.name, 0.0) + wafers_processed
            threshold = max(1.0, event.mean_time_before_pm)
            if counter >= threshold:
                tool.pm_wafer_counter[event.name] = counter - threshold
                duration = self._sample_repair_duration(event.repair_distribution, event.repair_mean, event.repair_offset, event.repair_units)
                payload = f"pm_counter:{event.name}"
                self._push(
                    "downtime_start",
                    self.now,
                    trigger_lot_id,
                    state.spec.name,
                    payload=payload,
                    tool_index=tool_index,
                    downtime_duration=duration,
                )
            else:
                tool.pm_wafer_counter[event.name] = counter

    def _schedule_initial_calendar_pm(self) -> None:
        if not self.config.enable_pm:
            return
        for toolgroup_name, state in self._toolgroups.items():
            for tool in state.tools:
                for event in tool.calendar_pm_rules:
                    first_at = self._to_minutes(event.first_one_at, event.first_one_units)
                    self._schedule_calendar_pm_window(
                        event,
                        toolgroup_name,
                        tool.index,
                        first_at,
                    )

    def _schedule_next_calendar_pm(
        self,
        event: PMEvent,
        toolgroup_name: str,
        tool_index: int,
    ) -> None:
        interval = self._to_minutes(event.mean_time_before_pm, event.mean_time_before_pm_units)
        self._schedule_calendar_pm_window(event, toolgroup_name, tool_index, self.now + interval)

    def _schedule_calendar_pm_window(
        self,
        event: PMEvent,
        toolgroup_name: str,
        tool_index: int,
        start: float,
    ) -> None:
        duration = self._sample_repair_duration(
            event.repair_distribution,
            event.repair_mean,
            event.repair_offset,
            event.repair_units,
        )
        payload = f"pm_calendar:{event.name}"
        self._push(
            "downtime_start",
            start,
            0,
            toolgroup_name,
            payload=payload,
            tool_index=tool_index,
            downtime_duration=duration,
        )

    def _schedule_initial_breakdowns(self) -> None:
        if not self.config.enable_breakdowns:
            return
        for toolgroup_name, state in self._toolgroups.items():
            for tool in state.tools:
                for breakdown in tool.breakdown_rules:
                    self._schedule_next_breakdown(
                        breakdown,
                        toolgroup_name,
                        tool.index,
                        from_start=True,
                    )

    def _schedule_next_breakdown(
        self,
        breakdown: BreakdownEvent,
        toolgroup_name: str,
        tool_index: int,
        from_start: bool = False,
    ) -> None:
        if from_start:
            delay = self._sample_distribution(
                breakdown.first_one_distribution,
                breakdown.first_one_at,
                0.0,
                breakdown.first_one_units,
            )
            start = delay
        else:
            delay = self._sample_distribution(
                breakdown.ttf_distribution,
                breakdown.mean_time_to_failure,
                0.0,
                breakdown.mean_time_to_failure_units,
            )
            start = self.now + delay
        duration = self._sample_distribution(
            breakdown.repair_distribution,
            breakdown.mean_time_to_repair,
            0.0,
            breakdown.repair_units,
        )
        payload = f"breakdown:{breakdown.name}"
        self._push(
            "downtime_start",
            start,
            0,
            toolgroup_name,
            payload=payload,
            tool_index=tool_index,
            downtime_duration=duration,
        )

    def _index_calendar_pm_by_toolgroup(self) -> dict[str, tuple[PMEvent, ...]]:
        indexed: dict[str, list[PMEvent]] = {}
        for event in self.model.pm_events.values():
            if event.valid_for_type == "toolgroup" and event.mean_time_before_pm_units.lower() != "wafer":
                indexed.setdefault(event.type_name, []).append(event)
        return {key: tuple(value) for key, value in indexed.items()}

    def _index_counter_pm_by_toolgroup(self) -> dict[str, tuple[PMEvent, ...]]:
        indexed: dict[str, list[PMEvent]] = {}
        for event in self.model.pm_events.values():
            if event.valid_for_type == "toolgroup" and event.mean_time_before_pm_units.lower() == "wafer":
                indexed.setdefault(event.type_name, []).append(event)
        return {key: tuple(value) for key, value in indexed.items()}

    def _index_breakdowns_by_toolgroup(self) -> dict[str, tuple[BreakdownEvent, ...]]:
        indexed: dict[str, list[BreakdownEvent]] = {
            toolgroup.name: []
            for toolgroup in self.model.toolgroups.values()
        }
        for event in self.model.breakdown_events.values():
            valid_for = event.valid_for_type.lower()
            if valid_for == "area":
                for toolgroup in self.model.toolgroups.values():
                    if toolgroup.area == event.type_name:
                        indexed[toolgroup.name].append(event)
            elif valid_for == "toolgroup" and event.type_name in indexed:
                indexed[event.type_name].append(event)
        return {key: tuple(value) for key, value in indexed.items()}

    def _attach_tool_rules(self) -> None:
        for toolgroup_name, state in self._toolgroups.items():
            calendar_pm = self._calendar_pm_by_toolgroup.get(toolgroup_name, ())
            counter_pm = self._counter_pm_by_toolgroup.get(toolgroup_name, ())
            breakdowns = self._breakdowns_by_toolgroup.get(toolgroup_name, ())
            for tool in state.tools:
                tool.calendar_pm_rules = calendar_pm
                tool.counter_pm_rules = counter_pm
                tool.breakdown_rules = breakdowns

    def _index_setup_times(self) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for rule in self.model.setup_rules.values():
            result[(rule.current_setup, rule.new_setup)] = self._to_minutes(
                rule.setup_time,
                rule.setup_units,
            )
        return result

    def _sample_repair_duration(
        self,
        distribution: str,
        mean: float,
        offset: float,
        units: str,
    ) -> float:
        return self._sample_distribution(distribution, mean, offset, units)

    def _sample_distribution(
        self,
        distribution: str | None,
        mean: float,
        offset: float,
        units: str | None,
    ) -> float:
        distribution = (distribution or "constant").lower()
        mean_minutes = self._to_minutes(mean, units)
        offset_minutes = self._to_minutes(offset, units)
        if distribution == "uniform":
            return max(0.0, self.random.uniform_around(mean_minutes, offset_minutes))
        if distribution == "exponential":
            return max(0.0, self.random.exponential(mean_minutes))
        return max(0.0, mean_minutes)

    def _to_minutes(self, value: float, units: str | None) -> float:
        units = (units or "min").lower()
        if units in {"min", "minute", "minutes"}:
            return value
        if units in {"hr", "hour", "hours"}:
            return value * MINUTES_PER_HOUR
        if units in {"day", "days"}:
            return value * MINUTES_PER_DAY
        if units in {"wafer", "wafers"}:
            return value
        return value

    def _log_tool_event(
        self,
        event_type: str,
        toolgroup: str,
        tool_index: int,
        payload: str,
    ) -> None:
        if not self.config.write_event_log:
            return
        self._event_log.append(
            {
                "time": self.now,
                "event": event_type,
                "lot_id": 0,
                "lot_name": None,
                "product_name": None,
                "step_index": None,
                "toolgroup": toolgroup,
                "tool_index": tool_index,
                "payload": payload,
            }
        )
