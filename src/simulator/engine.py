from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from src.simulator.config import SimulationConfig
from src.simulator.model import BreakdownSpec, FabModel, Lot, PMSpec, RouteStep, Tool, ToolGroup
from src.simulator.policies import DispatchPolicy
from src.simulator.runtime import Event, EventCalendar, RandomStream, SimulationSummary, summarize_lots


MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = 24.0 * MINUTES_PER_HOUR


@dataclass(frozen=True)
class SimulationResult:
    summary: SimulationSummary
    lots: tuple[Lot, ...]
    event_log: tuple[dict[str, float | int | str | None], ...]


class Simulator:
    """Discrete-event simulator assembled from release, route, tool, and policy objects."""

    def __init__(self, model: FabModel, config: SimulationConfig) -> None:
        self.model = model
        self.config = config
        self.random = RandomStream(config.random_seed)
        self.dispatch_policy = DispatchPolicy(config.dispatching_rule)
        self.now = 0.0
        self.calendar = EventCalendar()
        self.lots: dict[int, Lot] = {}
        self.toolgroups = self._build_toolgroups()
        self.toolgroup_dispatch_policies = self._build_toolgroup_dispatch_policies()
        self.event_log: list[dict[str, float | int | str | None]] = []
        self._next_lot_id = 1
        self._cqt_open: dict[tuple[int, int], tuple[float, float]] = {}
        self._cqt_lateness: list[float] = []
        self._cqt_checks = 0
        self.cqt_lateness_by_toolgroup: dict[str, list[float]] = defaultdict(list)
        self.cqt_counts_by_lot: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._pending_dispatch_toolgroups: set[str] = set()
        self._defer_dispatch = False
        self._completed_lots_count = 0
        self._started_operations_count = 0
        self._completed_operations_count = 0
        self._event_counts: dict[str, int] = {}
        self.blocked_reason: str | None = None
        self._arrival_times: dict[tuple[int, int, str], float] = {}
        self.queue_waits_by_toolgroup: dict[str, list[float]] = defaultdict(list)
        self.process_times_by_toolgroup: dict[str, list[float]] = defaultdict(list)
        self.process_times_by_tool: dict[tuple[str, int], list[float]] = defaultdict(list)
        self.downtime_counts_by_toolgroup: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.downtime_counts_by_tool: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cqt_counts_by_toolgroup: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._setup_times = {
            (rule.current_setup, rule.new_setup): self._to_minutes(rule.setup_time, rule.setup_units)
            for rule in model.setup_rules.values()
        }

    def run(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_events: int = 10000,
    ) -> SimulationResult:
        self._schedule_releases()
        self._schedule_initial_tool_events()
        processed_events = 0
        next_progress_at = max(1, progress_interval_events)
        self._emit_progress(progress_callback, processed_events, "scheduled")
        while self.calendar.has_events():
            event = self.calendar.pop()
            if self.config.until_minutes is not None and event.time > self.config.until_minutes:
                break
            self.now = event.time
            processed_events += self._process_same_time_events(event)
            self._dispatch_pending_toolgroups()
            if self._is_blocked_waiting_state():
                self.blocked_reason = "waiting_lots_cannot_satisfy_dispatch_constraints"
                self._emit_progress(progress_callback, processed_events, "blocked")
                break
            if progress_callback is not None and processed_events >= next_progress_at:
                self._emit_progress(progress_callback, processed_events, event.kind)
                while next_progress_at <= processed_events:
                    next_progress_at += max(1, progress_interval_events)
            if self._all_released_lots_complete():
                break
        self._emit_progress(progress_callback, processed_events, "finished")
        lots = [self.lots[lot_id] for lot_id in sorted(self.lots)]
        return SimulationResult(
            summary=summarize_lots(
                lots,
                self.config.warmup_minutes,
                self._cqt_lateness,
                self._cqt_checks,
            ),
            lots=tuple(lots),
            event_log=tuple(self.event_log),
        )

    def _build_toolgroups(self) -> dict[str, ToolGroup]:
        calendar_pm = self._pm_rules_by_toolgroup(counter_based=False)
        counter_pm = self._pm_rules_by_toolgroup(counter_based=True)
        breakdowns = self._breakdown_rules_by_toolgroup()
        return {
            name: ToolGroup.from_spec(
                spec,
                calendar_pm_rules=calendar_pm.get(name, ()),
                counter_pm_rules=counter_pm.get(name, ()),
                breakdown_rules=breakdowns.get(name, ()),
            )
            for name, spec in self.model.toolgroup_specs.items()
        }

    def _build_toolgroup_dispatch_policies(self) -> dict[str, DispatchPolicy]:
        overrides = self.config.toolgroup_dispatching_rules or {}
        unknown = sorted(set(overrides) - set(self.model.toolgroup_specs))
        if unknown:
            raise ValueError(f"Dispatching rule override references unknown tool groups: {unknown}")
        return {
            name: DispatchPolicy(overrides.get(name, self.config.dispatching_rule))
            for name in self.model.toolgroup_specs
        }

    def _pm_rules_by_toolgroup(self, counter_based: bool) -> dict[str, tuple[PMSpec, ...]]:
        indexed: dict[str, list[PMSpec]] = {}
        for spec in self.model.pm_specs.values():
            if spec.valid_for_type == "toolgroup" and spec.is_counter_based == counter_based:
                indexed.setdefault(spec.type_name, []).append(spec)
        return {key: tuple(value) for key, value in indexed.items()}

    def _breakdown_rules_by_toolgroup(self) -> dict[str, tuple[BreakdownSpec, ...]]:
        indexed: dict[str, list[BreakdownSpec]] = {
            name: [] for name in self.model.toolgroup_specs
        }
        for spec in self.model.breakdown_specs.values():
            valid_for = spec.valid_for_type.lower()
            if valid_for == "area":
                for toolgroup in self.model.toolgroup_specs.values():
                    if toolgroup.area == spec.type_name:
                        indexed[toolgroup.name].append(spec)
            elif valid_for == "toolgroup" and spec.type_name in indexed:
                indexed[spec.type_name].append(spec)
        return {key: tuple(value) for key, value in indexed.items()}

    def _schedule_releases(self) -> None:
        for release in self.model.release_plan.releases:
            lot_id = self._next_lot_id
            self._next_lot_id += 1
            lot = Lot(
                id=lot_id,
                name=release.lot_name,
                product_name=release.product_name,
                route=self.model.routes[release.route_name],
                priority=release.priority,
                super_hot_lot=release.super_hot_lot,
                wafers_per_lot=release.wafers_per_lot,
                release_time=self.model.release_plan.release_time_minutes(release),
                due_time=self.model.release_plan.due_time_minutes(release),
            )
            self.lots[lot_id] = lot
            self.calendar.push(lot.release_time, "lot_release", target=lot_id)

    def _schedule_initial_tool_events(self) -> None:
        for toolgroup in self.toolgroups.values():
            for tool in toolgroup.tools:
                if self.config.enable_pm:
                    for pm in tool.calendar_pm_rules:
                        first_at = self._to_minutes(pm.first_one_at, pm.first_one_units)
                        self._schedule_calendar_pm(tool, pm, first_at)
                if self.config.enable_breakdowns:
                    for breakdown in tool.breakdown_rules:
                        self._schedule_breakdown(tool, breakdown, from_start=True)

    def _process_same_time_events(self, first_event: Event) -> int:
        processed_events = 1
        self._defer_dispatch = True
        self._pending_dispatch_toolgroups.clear()
        self._process_event(first_event)
        while self.calendar.has_event_at(self.now):
            for event in self.calendar.pop_same_time(self.now):
                self._process_event(event)
                processed_events += 1
        self._defer_dispatch = False
        return processed_events

    def _process_event(self, event: Event) -> None:
        self._event_counts[event.kind] = self._event_counts.get(event.kind, 0) + 1
        if event.kind == "lot_release":
            self._handle_release(int(event.target))
        elif event.kind == "lot_arrival":
            self._handle_arrival(int(event.target))
        elif event.kind == "operation_complete":
            toolgroup_name, tool_index = event.target
            self._handle_operation_complete(toolgroup_name, int(tool_index), tuple(event.data["lot_ids"]))
        elif event.kind == "downtime_start":
            toolgroup_name, tool_index = event.target
            self._handle_downtime_start(
                toolgroup_name,
                int(tool_index),
                str(event.data["reason"]),
                float(event.data["duration"]),
            )
        elif event.kind == "downtime_end":
            toolgroup_name, tool_index = event.target
            self._handle_downtime_end(toolgroup_name, int(tool_index), str(event.data["reason"]))
        else:
            raise ValueError(f"Unknown event kind: {event.kind}")

    def _handle_release(self, lot_id: int) -> None:
        self._log_lot("release", lot_id)
        self.calendar.push(self.now, "lot_arrival", target=lot_id)

    def _handle_arrival(self, lot_id: int) -> None:
        lot = self.lots[lot_id]
        self._skip_unsampled_steps(lot)
        if lot.is_complete:
            lot.completed_time = self.now
            self._completed_lots_count += 1
            self._log_lot("complete_lot", lot_id)
            return
        toolgroup = self.toolgroups[lot.current_step.toolgroup]
        lot.current_toolgroup = toolgroup.spec.name
        lot.waiting_since = self.now
        self._arrival_times[(lot.id, lot.step_index, toolgroup.spec.name)] = self.now
        toolgroup.add_waiting_lot(lot_id)
        self._log_lot("arrive", lot_id, toolgroup.spec.name)
        self._check_cqt_end(lot)
        self._request_dispatch(toolgroup.spec.name)

    def _request_dispatch(self, toolgroup_name: str) -> None:
        if self._defer_dispatch:
            self._pending_dispatch_toolgroups.add(toolgroup_name)
        else:
            self._dispatch_toolgroup(self.toolgroups[toolgroup_name])

    def _dispatch_pending_toolgroups(self) -> None:
        while self._pending_dispatch_toolgroups:
            names = sorted(self._pending_dispatch_toolgroups)
            self._pending_dispatch_toolgroups.clear()
            for name in names:
                self._dispatch_toolgroup(self.toolgroups[name])

    def _dispatch_toolgroup(self, toolgroup: ToolGroup) -> None:
        policy = self.toolgroup_dispatch_policies[toolgroup.spec.name]
        while toolgroup.idle_tools and toolgroup.waiting_lot_ids:
            deferred_lot_ids: set[int] = set()
            lots: list[Lot] | None = None
            waiting_lots = [self.lots[lot_id] for lot_id in toolgroup.waiting_lot_ids]
            while toolgroup.idle_tools:
                candidates = (
                    waiting_lots
                    if not deferred_lot_ids
                    else [lot for lot in waiting_lots if lot.id not in deferred_lot_ids]
                )
                if not candidates:
                    return
                lead_lot = policy.select_lot(candidates, self.now)
                tool = self._select_tool(toolgroup, lead_lot.current_step)
                if self._is_batch_step(toolgroup, lead_lot.current_step):
                    lots = self._select_batch_lots(toolgroup, lead_lot, tool, policy)
                    if lots is None:
                        deferred_lot_ids.add(lead_lot.id)
                        continue
                else:
                    lots = [lead_lot]
                break
            if not lots:
                return
            self._start_operation(toolgroup, tool, lots)

    def _select_tool(self, toolgroup: ToolGroup, step: RouteStep) -> Tool:
        return min(
            toolgroup.idle_tools,
            key=lambda tool: (self._setup_duration_preview(tool, step), tool.index),
        )

    def _start_operation(self, toolgroup: ToolGroup, tool: Tool, lots: list[Lot]) -> None:
        for lot in lots:
            toolgroup.remove_waiting_lot(lot.id)
            lot.waiting_since = None
        lead_lot = lots[0]
        setup_minutes = self._setup_duration(tool, lead_lot.current_step)
        process_minutes = self._step_duration(toolgroup, lead_lot.current_step, lead_lot, tuple(lots))
        completion_time = self.now + setup_minutes + process_minutes
        tool.start(tuple(lot.id for lot in lots), completion_time)
        self._started_operations_count += len(lots)
        observed_process_time = completion_time - self.now
        event_name = "start_batch" if len(lots) > 1 else "start"
        for lot in lots:
            arrival_key = (lot.id, lot.step_index, toolgroup.spec.name)
            arrival_time = self._arrival_times.pop(arrival_key, lot.waiting_since or self.now)
            self.queue_waits_by_toolgroup[toolgroup.spec.name].append(self.now - arrival_time)
            self.process_times_by_toolgroup[toolgroup.spec.name].append(observed_process_time)
            self.process_times_by_tool[(toolgroup.spec.name, tool.index)].append(observed_process_time)
            lot.history.append((lot.current_step.step_number, toolgroup.spec.name, tool.index, self.now, completion_time))
            self._log_lot(event_name, lot.id, toolgroup.spec.name, tool.index)
        if setup_minutes > 0:
            self._log_lot("setup", lead_lot.id, toolgroup.spec.name, tool.index)
        self.calendar.push(
            completion_time,
            "operation_complete",
            target=(toolgroup.spec.name, tool.index),
            lot_ids=tuple(lot.id for lot in lots),
        )

    def _handle_operation_complete(self, toolgroup_name: str, tool_index: int, lot_ids: tuple[int, ...]) -> None:
        toolgroup = self.toolgroups[toolgroup_name]
        tool = toolgroup.tool(tool_index)
        tool.finish()
        self._completed_operations_count += len(lot_ids)
        total_wafers = sum(self.lots[lot_id].wafers_per_lot for lot_id in lot_ids)
        self._record_counter_pm_usage(tool, total_wafers, lot_ids[0])
        event_name = "complete_batch_step" if len(lot_ids) > 1 else "complete_step"
        for lot_id in lot_ids:
            lot = self.lots[lot_id]
            self._log_lot(event_name, lot_id, toolgroup_name, tool_index)
            self._after_step_completion(lot)
        self._request_dispatch(toolgroup_name)

    def _after_step_completion(self, lot: Lot) -> None:
        self._check_cqt_start(lot)
        from_toolgroup = lot.current_step.toolgroup
        next_step_index = self._next_step_index(lot)
        lot.step_index = next_step_index
        lot.current_toolgroup = None
        if lot.is_complete:
            lot.completed_time = self.now
            self._completed_lots_count += 1
            self._log_lot("complete_lot", lot.id)
        else:
            to_toolgroup = lot.current_step.toolgroup
            self.calendar.push(
                self.now + self._transport_duration(from_toolgroup, to_toolgroup),
                "lot_arrival",
                target=lot.id,
            )

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
            self._log_lot("skip_sampling_step", lot.id, lot.current_step.toolgroup)
            lot.step_index += 1

    def _select_batch_lots(
        self,
        toolgroup: ToolGroup,
        lead_lot: Lot,
        tool: Tool,
        policy: DispatchPolicy,
    ) -> list[Lot] | None:
        step = lead_lot.current_step
        maximum = step.batch_maximum or float("inf")
        minimum = step.batch_minimum or 0.0
        compatible = [
            self.lots[lot_id]
            for lot_id in toolgroup.waiting_lot_ids
            if self._batch_compatible(toolgroup, lead_lot, self.lots[lot_id])
        ]
        selected: list[Lot] = []
        remaining = compatible
        while remaining and self._batch_quantity(toolgroup, selected) < maximum:
            lot = policy.select_lot(remaining, self.now, tool)
            if self._batch_quantity(toolgroup, [*selected, lot]) > maximum:
                break
            selected.append(lot)
            remaining = [item for item in remaining if item.id != lot.id]
        if self._batch_quantity(toolgroup, selected) >= minimum:
            return selected
        return None

    def _is_blocked_waiting_state(self) -> bool:
        if not self.lots or self._completed_lots_count >= len(self.lots):
            return False
        if self.calendar.has_productive_events():
            return False
        if any(tool.busy for toolgroup in self.toolgroups.values() for tool in toolgroup.tools):
            return False
        if not any(toolgroup.waiting_lot_ids for toolgroup in self.toolgroups.values()):
            return False
        return not any(
            self._toolgroup_has_dispatchable_waiting_work(toolgroup)
            for toolgroup in self.toolgroups.values()
        )

    def _toolgroup_has_dispatchable_waiting_work(self, toolgroup: ToolGroup) -> bool:
        if not toolgroup.idle_tools or not toolgroup.waiting_lot_ids:
            return False
        for lot_id in toolgroup.waiting_lot_ids:
            lot = self.lots[lot_id]
            if not self._is_batch_step(toolgroup, lot.current_step):
                return True
            if self._batch_minimum_is_available(toolgroup, lot):
                return True
        return False

    def _batch_minimum_is_available(self, toolgroup: ToolGroup, lead_lot: Lot) -> bool:
        step = lead_lot.current_step
        minimum = step.batch_minimum or 0.0
        maximum = step.batch_maximum or float("inf")
        quantity = 0
        for lot_id in toolgroup.waiting_lot_ids:
            lot = self.lots[lot_id]
            if self._batch_compatible(toolgroup, lead_lot, lot):
                lot_quantity = self._batch_lot_quantity(toolgroup, lot)
                if lot_quantity <= maximum:
                    quantity += lot_quantity
            if quantity >= minimum:
                return True
        return minimum <= 0

    def _is_batch_step(self, toolgroup: ToolGroup, step: RouteStep) -> bool:
        return (
            self.config.enable_batching
            and toolgroup.spec.batching_tool
            and step.processing_unit.lower() == "batch"
            and step.batch_minimum is not None
        )

    def _batch_compatible(self, toolgroup: ToolGroup, lead_lot: Lot, candidate: Lot) -> bool:
        if candidate.current_step != lead_lot.current_step:
            return False
        criterion = (toolgroup.spec.batch_criterion or "").strip().lower()
        if criterion == "same product and same step":
            return candidate.product_name == lead_lot.product_name
        return True

    def _batch_quantity(self, toolgroup: ToolGroup, lots: list[Lot]) -> float:
        return sum(self._batch_lot_quantity(toolgroup, lot) for lot in lots)

    def _batch_lot_quantity(self, toolgroup: ToolGroup, lot: Lot) -> float:
        unit = (toolgroup.spec.batching_unit or "wafer").strip().lower()
        if unit in {"lot", "lots"}:
            return 1.0
        return float(lot.wafers_per_lot)

    def _step_duration(self, toolgroup: ToolGroup, step: RouteStep, lot: Lot, batch_lots: tuple[Lot, ...]) -> float:
        process = step.process_time
        if self.config.sample_process_times and (process.distribution or "").lower() == "uniform":
            base = self.random.uniform_around(process.mean, process.offset)
        else:
            base = process.mean
        unit = step.processing_unit.lower()
        if unit == "wafer" and self.config.wafer_time_mode == "per_wafer":
            if self._uses_cascading(toolgroup, step):
                base += max(0, lot.wafers_per_lot - 1) * step.cascading_interval
            else:
                base *= lot.wafers_per_lot
        elif unit == "batch" and self._is_batch_step(toolgroup, step):
            batch_wafers = sum(batch_lot.wafers_per_lot for batch_lot in batch_lots)
            if self._uses_cascading(toolgroup, step):
                base += max(0, batch_wafers - 1) * step.cascading_interval
        spec = self.model.toolgroup_specs[step.toolgroup]
        return max(0.0, base + spec.loading_time + spec.unloading_time)

    def _uses_cascading(self, toolgroup: ToolGroup, step: RouteStep) -> bool:
        return (
            self.config.enable_cascading
            and toolgroup.spec.cascading_tool
            and step.cascading_interval is not None
        )

    def _setup_duration_preview(self, tool: Tool, step: RouteStep) -> float:
        if step.setup is None:
            return 0.0
        current = tool.setup_state
        if current is None or current == step.setup:
            return 0.0
        if (current, step.setup) in self._setup_times:
            return self._setup_times[(current, step.setup)]
        if step.setup_time is not None:
            return self._to_minutes(step.setup_time, step.setup_units)
        return 0.0

    def _setup_duration(self, tool: Tool, step: RouteStep) -> float:
        duration = self._setup_duration_preview(tool, step)
        if step.setup is not None:
            tool.setup_state = step.setup
        return duration

    def _transport_duration(self, from_toolgroup: str, to_toolgroup: str) -> float:
        transport = self.model.transport
        if not self.config.enable_transport or transport is None:
            return 0.0
        # The SMT2020 file has one Fab-to-Fab rule, so every step-to-step transfer uses it.
        _ = (from_toolgroup, to_toolgroup)
        if self.config.sample_transport_times and transport.distribution.lower() == "uniform":
            return max(0.0, self.random.uniform_around(transport.mean, transport.offset))
        return max(0.0, transport.mean)

    def _record_counter_pm_usage(self, tool: Tool, wafers_processed: int, trigger_lot_id: int) -> None:
        if not self.config.enable_pm:
            return
        for pm in tool.counter_pm_rules:
            if pm.name in tool.active_pm:
                continue
            counter = tool.pm_wafer_counter.get(pm.name, 0.0) + wafers_processed
            threshold = max(1.0, pm.mean_time_before_pm)
            if counter >= threshold:
                tool.pm_wafer_counter[pm.name] = counter - threshold
                tool.active_pm.add(pm.name)
                duration = self._sample_distribution(pm.repair_distribution, pm.repair_mean, pm.repair_offset, pm.repair_units)
                self.calendar.push(
                    self.now,
                    "downtime_start",
                    target=(tool.group_name, tool.index),
                    reason=f"pm_counter:{pm.name}",
                    duration=duration,
                    trigger_lot_id=trigger_lot_id,
                )
            else:
                tool.pm_wafer_counter[pm.name] = counter

    def _schedule_calendar_pm(self, tool: Tool, pm: PMSpec, start: float) -> None:
        duration = self._sample_distribution(pm.repair_distribution, pm.repair_mean, pm.repair_offset, pm.repair_units)
        self.calendar.push(
            start,
            "downtime_start",
            target=(tool.group_name, tool.index),
            reason=f"pm_calendar:{pm.name}",
            duration=duration,
        )

    def _schedule_breakdown(self, tool: Tool, breakdown: BreakdownSpec, from_start: bool) -> None:
        if from_start:
            start = self._sample_distribution(
                breakdown.first_one_distribution,
                breakdown.first_one_at,
                0.0,
                breakdown.first_one_units,
            )
        else:
            start = self.now + self._sample_distribution(
                breakdown.ttf_distribution,
                breakdown.mean_time_to_failure,
                0.0,
                breakdown.mean_time_to_failure_units,
            )
        duration = self._sample_distribution(
            breakdown.repair_distribution,
            breakdown.mean_time_to_repair,
            0.0,
            breakdown.repair_units,
        )
        self.calendar.push(
            start,
            "downtime_start",
            target=(tool.group_name, tool.index),
            reason=f"breakdown:{breakdown.name}",
            duration=duration,
        )

    def _handle_downtime_start(
        self,
        toolgroup_name: str,
        tool_index: int,
        reason: str,
        duration: float,
    ) -> None:
        tool = self.toolgroups[toolgroup_name].tool(tool_index)
        if tool.busy and tool.busy_until is not None and tool.busy_until > self.now:
            self.calendar.push(
                tool.busy_until,
                "downtime_start",
                target=(toolgroup_name, tool_index),
                reason=reason,
                duration=duration,
            )
            return
        tool.mark_down(reason)
        self._record_downtime_metric(toolgroup_name, tool_index, reason)
        self._log_tool("downtime_start", toolgroup_name, tool_index, reason)
        self.calendar.push(
            self.now + duration,
            "downtime_end",
            target=(toolgroup_name, tool_index),
            reason=reason,
        )

    def _handle_downtime_end(self, toolgroup_name: str, tool_index: int, reason: str) -> None:
        tool = self.toolgroups[toolgroup_name].tool(tool_index)
        tool.clear_down(reason)
        kind, name = self._split_reason(reason)
        if kind == "pm_counter":
            tool.active_pm.discard(name)
        self._log_tool("downtime_end", toolgroup_name, tool_index, reason)
        self._request_dispatch(toolgroup_name)
        if kind == "pm_calendar":
            pm = self.model.pm_specs.get(name)
            if pm is not None:
                interval = self._to_minutes(pm.mean_time_before_pm, pm.mean_time_before_pm_units)
                self._schedule_calendar_pm(tool, pm, self.now + interval)
        elif kind == "breakdown":
            breakdown = self.model.breakdown_specs.get(name)
            if breakdown is not None:
                self._schedule_breakdown(tool, breakdown, from_start=False)

    def _split_reason(self, reason: str) -> tuple[str, str]:
        parts = reason.split(":", 1)
        if len(parts) == 1:
            return parts[0], parts[0]
        return parts[0], parts[1]

    def _check_cqt_start(self, lot: Lot) -> None:
        step = lot.current_step
        if step.cqt_start_step is None or step.cqt is None:
            return
        self._cqt_open[(lot.id, step.cqt_start_step)] = (
            self.now,
            self._to_minutes(step.cqt, step.cqt_units),
        )
        self._log_lot("cqt_start", lot.id, step.toolgroup)

    def _check_cqt_end(self, lot: Lot) -> None:
        key = (lot.id, lot.current_step.step_number)
        if key not in self._cqt_open:
            return
        start, limit = self._cqt_open.pop(key)
        elapsed = self.now - start
        self._cqt_checks += 1
        if elapsed > limit:
            lateness = elapsed - limit
            self._cqt_lateness.append(lateness)
            self.cqt_lateness_by_toolgroup[lot.current_step.toolgroup].append(lateness)
            self.cqt_counts_by_toolgroup[lot.current_step.toolgroup]["violations"] += 1
            self.cqt_counts_by_lot[lot.id]["violations"] += 1
            self._log_lot("cqt_violation", lot.id, lot.current_step.toolgroup)
        else:
            self.cqt_counts_by_toolgroup[lot.current_step.toolgroup]["passes"] += 1
            self.cqt_counts_by_lot[lot.id]["passes"] += 1
            self._log_lot("cqt_pass", lot.id, lot.current_step.toolgroup)

    def _record_downtime_metric(self, toolgroup_name: str, tool_index: int, reason: str) -> None:
        kind, _ = self._split_reason(reason)
        category = "pm" if kind.startswith("pm_") else kind
        self.downtime_counts_by_toolgroup[toolgroup_name]["downtime_starts"] += 1
        self.downtime_counts_by_toolgroup[toolgroup_name][f"{category}_starts"] += 1
        tool_key = (toolgroup_name, tool_index)
        self.downtime_counts_by_tool[tool_key]["downtime_starts"] += 1
        self.downtime_counts_by_tool[tool_key][f"{category}_starts"] += 1

    def _all_released_lots_complete(self) -> bool:
        return bool(self.lots) and self._completed_lots_count >= len(self.lots)

    def _emit_progress(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        processed_events: int,
        phase: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "phase": phase,
                "processed_events": processed_events,
                "sim_time": self.now,
                "sim_datetime": self._sim_datetime().isoformat(),
                "total_lots": len(self.lots),
                "released_lots": len(self.lots),
                "completed_lots": self._completed_lots_count,
                "started_operations": self._started_operations_count,
                "completed_operations": self._completed_operations_count,
                "waiting_lots": sum(len(toolgroup.waiting_lot_ids) for toolgroup in self.toolgroups.values()),
                "busy_tools": sum(1 for toolgroup in self.toolgroups.values() for tool in toolgroup.tools if tool.busy),
                "idle_tools": sum(1 for toolgroup in self.toolgroups.values() for tool in toolgroup.tools if tool.idle),
                "down_tools": sum(1 for toolgroup in self.toolgroups.values() for tool in toolgroup.tools if tool.down),
                "pending_events": len(self.calendar),
                "event_log_rows": len(self.event_log),
                "event_counts": dict(self._event_counts),
                "blocked_reason": self.blocked_reason,
            }
        )

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

    def _log_lot(
        self,
        event: str,
        lot_id: int,
        toolgroup: str | None = None,
        tool_index: int | None = None,
    ) -> None:
        if not self.config.write_event_log:
            return
        lot = self.lots[lot_id]
        self.event_log.append(
            {
                "time": self.now,
                "datetime": self._sim_datetime().isoformat(),
                "event": event,
                "lot_id": lot_id,
                "lot_name": lot.name,
                "product_name": lot.product_name,
                "priority": lot.priority,
                "super_hot_lot": lot.super_hot_lot,
                "step_index": lot.step_index,
                "toolgroup": toolgroup,
                "tool_index": tool_index,
                "payload": None,
            }
        )

    def _log_tool(self, event: str, toolgroup: str, tool_index: int, payload: str) -> None:
        if not self.config.write_event_log:
            return
        self.event_log.append(
            {
                "time": self.now,
                "datetime": self._sim_datetime().isoformat(),
                "event": event,
                "lot_id": 0,
                "lot_name": None,
                "product_name": None,
                "priority": None,
                "super_hot_lot": None,
                "step_index": None,
                "toolgroup": toolgroup,
                "tool_index": tool_index,
                "payload": payload,
            }
        )

    def _sim_datetime(self):
        return self.model.start_datetime + timedelta(minutes=self.now)
