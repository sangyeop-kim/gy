from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.simulator.engine import MINUTES_PER_DAY
from src.simulator.model import Lot, Tool, ToolGroup


FEATURE_NAMES = (
    "now_days",
    "queue_lots_100",
    "idle_tool_ratio",
    "busy_tool_ratio",
    "down_tool_ratio",
    "priority_30",
    "super_hot",
    "wait_age_days",
    "release_age_days",
    "due_in_days_30",
    "slack_days_30",
    "critical_ratio_10",
    "current_step_days",
    "remaining_days_30",
    "route_progress",
    "wafers_25",
    "setup_preview_days",
    "is_batch_step",
    "batch_min_250",
    "batch_max_250",
    "compatible_batch_qty_250",
    "batch_min_satisfied",
    "cqt_remaining_days_7",
)


@dataclass(frozen=True)
class DispatchFeatureEncoder:
    """Build fixed-size candidate features for variable-length dispatch actions."""

    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def size(self) -> int:
        return len(FEATURE_NAMES)

    def encode_candidates(
        self,
        simulator,
        toolgroup: ToolGroup,
        candidates: tuple[Lot, ...],
        tool: Tool | None = None,
    ) -> np.ndarray:
        return np.asarray(
            [self.encode_lot(simulator, toolgroup, lot, candidates, tool) for lot in candidates],
            dtype=np.float32,
        )

    def encode_lot(
        self,
        simulator,
        toolgroup: ToolGroup,
        lot: Lot,
        candidates: tuple[Lot, ...],
        tool: Tool | None = None,
    ) -> list[float]:
        now = simulator.now
        step = lot.current_step
        tools = toolgroup.tools
        num_tools = max(1, len(tools))
        idle_ratio = len(toolgroup.idle_tools) / num_tools
        busy_ratio = sum(1 for item in tools if item.busy) / num_tools
        down_ratio = sum(1 for item in tools if item.down) / num_tools
        wait_age = max(0.0, now - (lot.waiting_since if lot.waiting_since is not None else lot.release_time))
        release_age = max(0.0, now - lot.release_time)
        remaining = max(lot.route.remaining_nominal_minutes(lot.step_index), 0.0)
        current = self._current_step_nominal_minutes(lot)
        due_in = (lot.due_time - now) if lot.due_time is not None else 30.0 * MINUTES_PER_DAY
        slack = due_in - remaining if lot.due_time is not None else 30.0 * MINUTES_PER_DAY
        critical_ratio = due_in / max(remaining, 1.0) if lot.due_time is not None else 10.0
        setup_preview = self._setup_preview(simulator, toolgroup, step, tool)
        batch_qty = self._compatible_batch_quantity(simulator, toolgroup, lot, candidates)
        batch_min = step.batch_minimum or 0.0
        batch_max = step.batch_maximum or 0.0
        cqt_remaining = self._cqt_remaining(simulator, lot)
        return [
            self._scale(now, 30.0 * MINUTES_PER_DAY),
            self._scale(len(toolgroup.waiting_lot_ids), 100.0),
            idle_ratio,
            busy_ratio,
            down_ratio,
            self._scale(lot.priority, 30.0),
            1.0 if lot.super_hot_lot else 0.0,
            self._scale(wait_age, MINUTES_PER_DAY),
            self._scale(release_age, 30.0 * MINUTES_PER_DAY),
            self._clip(due_in / (30.0 * MINUTES_PER_DAY), -1.0, 1.0),
            self._clip(slack / (30.0 * MINUTES_PER_DAY), -1.0, 1.0),
            self._clip(critical_ratio / 10.0, -1.0, 1.0),
            self._scale(current, MINUTES_PER_DAY),
            self._scale(remaining, 30.0 * MINUTES_PER_DAY),
            self._scale(lot.step_index, max(1, len(lot.route.steps))),
            self._scale(lot.wafers_per_lot, 25.0),
            self._scale(setup_preview, MINUTES_PER_DAY),
            1.0 if step.processing_unit.lower() == "batch" else 0.0,
            self._scale(batch_min, 250.0),
            self._scale(batch_max, 250.0),
            self._scale(batch_qty, 250.0),
            1.0 if batch_qty >= batch_min else 0.0,
            self._clip(cqt_remaining / (7.0 * MINUTES_PER_DAY), -1.0, 1.0),
        ]

    def _setup_preview(self, simulator, toolgroup: ToolGroup, step, tool: Tool | None) -> float:
        if tool is not None:
            return simulator._setup_duration_preview(tool, step)
        if not toolgroup.idle_tools:
            return 0.0
        return min(simulator._setup_duration_preview(item, step) for item in toolgroup.idle_tools)

    def _compatible_batch_quantity(self, simulator, toolgroup: ToolGroup, lead_lot: Lot, candidates: tuple[Lot, ...]) -> float:
        if not simulator._is_batch_step(toolgroup, lead_lot.current_step):
            return float(lead_lot.wafers_per_lot)
        compatible = [
            candidate for candidate in candidates
            if simulator._batch_compatible(toolgroup, lead_lot, candidate)
        ]
        return simulator._batch_quantity(toolgroup, compatible)

    def _cqt_remaining(self, simulator, lot: Lot) -> float:
        key = (lot.id, lot.current_step.step_number)
        if key not in simulator._cqt_open:
            return 7.0 * MINUTES_PER_DAY
        start, limit = simulator._cqt_open[key]
        return start + limit - simulator.now

    def _current_step_nominal_minutes(self, lot: Lot) -> float:
        if lot.is_complete:
            return 0.0
        step = lot.current_step
        base = step.process_time.mean
        if step.processing_unit.lower() == "wafer":
            base *= lot.wafers_per_lot
        return max(base, 0.0)

    def _scale(self, value: float, denominator: float) -> float:
        if denominator <= 0 or not math.isfinite(float(value)):
            return 0.0
        return self._clip(float(value) / denominator, -1.0, 1.0)

    def _clip(self, value: float, lower: float, upper: float) -> float:
        if not math.isfinite(float(value)):
            return 0.0
        return max(lower, min(upper, float(value)))
