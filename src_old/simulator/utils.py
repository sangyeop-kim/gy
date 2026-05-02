from dataclasses import dataclass
from collections.abc import Iterable
import random

from src.simulator.domain import Lot


SUPPORTED_DISPATCHING_RULES = (
    "fifo",
    "lifo",
    "spt",
    "lpt",
    "srpt",
    "lrpt",
    "edd",
    "least_slack",
    "slack_per_remaining_step",
    "critical_ratio",
    "priority_fifo",
    "priority_spt",
    "priority_edd",
    "priority_least_slack",
    "priority_cr_fifo",
)


class RandomStream:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def uniform_around(self, mean: float, offset: float) -> float:
        if offset <= 0:
            return mean
        return self._rng.uniform(mean - offset, mean + offset)

    def exponential(self, mean: float) -> float:
        if mean <= 0:
            return 0.0
        return self._rng.expovariate(1.0 / mean)

    def bernoulli_percent(self, probability: float | None) -> bool:
        if probability is None or probability <= 0:
            return False
        if probability >= 100:
            return True
        return self._rng.random() < probability / 100.0


def select_lot(candidates: Iterable[Lot], now: float, rule: str) -> Lot:
    lots = list(candidates)
    if not lots:
        raise ValueError("Cannot dispatch from an empty queue")

    if rule == "fifo":
        return min(lots, key=lambda lot: (lot.release_time, lot.id))
    if rule == "lifo":
        return max(lots, key=lambda lot: (lot.release_time, lot.id))
    if rule == "spt":
        return min(lots, key=lambda lot: (_current_step_nominal_minutes(lot), lot.release_time, lot.id))
    if rule == "lpt":
        return max(lots, key=lambda lot: (_current_step_nominal_minutes(lot), -lot.release_time, -lot.id))
    if rule == "srpt":
        return min(lots, key=lambda lot: (_remaining_nominal_minutes(lot), lot.release_time, lot.id))
    if rule == "lrpt":
        return max(lots, key=lambda lot: (_remaining_nominal_minutes(lot), -lot.release_time, -lot.id))
    if rule == "edd":
        return min(lots, key=lambda lot: (_due_time(lot), lot.release_time, lot.id))
    if rule == "least_slack":
        return min(lots, key=lambda lot: (_slack(lot, now), lot.release_time, lot.id))
    if rule == "slack_per_remaining_step":
        return min(lots, key=lambda lot: (_slack_per_remaining_step(lot, now), lot.release_time, lot.id))
    if rule == "priority_fifo":
        return min(lots, key=lambda lot: (-lot.priority, lot.release_time, lot.id))
    if rule == "priority_spt":
        return min(
            lots,
            key=lambda lot: (
                -lot.priority,
                _current_step_nominal_minutes(lot),
                lot.release_time,
                lot.id,
            ),
        )
    if rule == "priority_edd":
        return min(
            lots,
            key=lambda lot: (
                -lot.priority,
                _due_time(lot),
                lot.release_time,
                lot.id,
            ),
        )
    if rule == "priority_least_slack":
        return min(
            lots,
            key=lambda lot: (
                -lot.priority,
                _slack(lot, now),
                lot.release_time,
                lot.id,
            ),
        )
    if rule == "critical_ratio":
        return min(lots, key=lambda lot: (_critical_ratio(lot, now), lot.release_time, lot.id))
    if rule == "priority_cr_fifo":
        return min(
            lots,
            key=lambda lot: (
                -lot.priority,
                _critical_ratio(lot, now),
                lot.release_time,
                lot.id,
            ),
        )

    raise ValueError(f"Unsupported dispatching rule: {rule}")


def _current_step_nominal_minutes(lot: Lot) -> float:
    if lot.is_complete:
        return 0.0
    step = lot.current_step
    base = step.process_time.mean
    if step.processing_unit.lower() == "wafer":
        base *= lot.wafers_per_lot
    return max(base, 0.0)


def _remaining_nominal_minutes(lot: Lot) -> float:
    return max(lot.route.remaining_nominal_minutes(lot.step_index), 0.0)


def _due_time(lot: Lot) -> float:
    return lot.due_time if lot.due_time is not None else float("inf")


def _critical_ratio(lot: Lot, now: float) -> float:
    if lot.due_time is None:
        return float("inf")
    remaining = max(_remaining_nominal_minutes(lot), 1.0)
    return (lot.due_time - now) / remaining


def _slack(lot: Lot, now: float) -> float:
    if lot.due_time is None:
        return float("inf")
    return lot.due_time - now - _remaining_nominal_minutes(lot)


def _slack_per_remaining_step(lot: Lot, now: float) -> float:
    remaining_steps = max(len(lot.route.steps) - lot.step_index, 1)
    return _slack(lot, now) / remaining_steps


@dataclass(frozen=True)
class SimulationSummary:
    released_lots: int
    completed_lots: int
    throughput_lots: int
    average_cycle_time_minutes: float | None
    average_tardiness_minutes: float | None
    max_tardiness_minutes: float | None
    tardy_lots: int
    on_time_ratio: float | None
    first_release_time: float | None
    last_completion_time: float | None
    cqt_violations: int = 0
    average_cqt_lateness_minutes: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "released_lots": self.released_lots,
            "completed_lots": self.completed_lots,
            "throughput_lots": self.throughput_lots,
            "average_cycle_time_minutes": self.average_cycle_time_minutes,
            "average_tardiness_minutes": self.average_tardiness_minutes,
            "max_tardiness_minutes": self.max_tardiness_minutes,
            "tardy_lots": self.tardy_lots,
            "on_time_ratio": self.on_time_ratio,
            "first_release_time": self.first_release_time,
            "last_completion_time": self.last_completion_time,
            "cqt_violations": self.cqt_violations,
            "average_cqt_lateness_minutes": self.average_cqt_lateness_minutes,
        }


def summarize_lots(
    lots: list[Lot],
    warmup_minutes: float = 0.0,
    cqt_lateness: list[float] | None = None,
) -> SimulationSummary:
    completed = [
        lot
        for lot in lots
        if lot.completed_time is not None and lot.release_time >= warmup_minutes
    ]
    if completed:
        cycle_times = [lot.completed_time - lot.release_time for lot in completed]
        due_lots = [lot for lot in completed if lot.due_time is not None]
        tardiness = [max(0.0, lot.completed_time - lot.due_time) for lot in due_lots]
        on_time = [lot.completed_time <= lot.due_time for lot in due_lots]
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        avg_tardiness = sum(tardiness) / len(tardiness) if tardiness else None
        max_tardiness = max(tardiness) if tardiness else None
        tardy_lots = sum(lateness > 0 for lateness in tardiness)
        on_time_ratio = sum(on_time) / len(on_time) if on_time else None
        last_completion = max(lot.completed_time for lot in completed)
    else:
        avg_cycle_time = None
        avg_tardiness = None
        max_tardiness = None
        tardy_lots = 0
        on_time_ratio = None
        last_completion = None

    released_after_warmup = [lot for lot in lots if lot.release_time >= warmup_minutes]
    first_release = min((lot.release_time for lot in lots), default=None)
    cqt_lateness = cqt_lateness or []
    return SimulationSummary(
        released_lots=len(released_after_warmup),
        completed_lots=len(completed),
        throughput_lots=len(completed),
        average_cycle_time_minutes=avg_cycle_time,
        average_tardiness_minutes=avg_tardiness,
        max_tardiness_minutes=max_tardiness,
        tardy_lots=tardy_lots,
        on_time_ratio=on_time_ratio,
        first_release_time=first_release,
        last_completion_time=last_completion,
        cqt_violations=len(cqt_lateness),
        average_cqt_lateness_minutes=(
            sum(cqt_lateness) / len(cqt_lateness) if cqt_lateness else None
        ),
    )
