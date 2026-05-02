from dataclasses import dataclass, field
import heapq
from itertools import count
import random

from src.simulator.model import Lot

PRODUCTIVE_EVENT_KINDS = {"lot_release", "lot_arrival", "operation_complete"}


@dataclass(order=True)
class Event:
    time: float
    sequence: int
    kind: str = field(compare=False)
    target: str | int | tuple | None = field(default=None, compare=False)
    data: dict = field(default_factory=dict, compare=False)


class EventCalendar:
    def __init__(self) -> None:
        self._sequence = count()
        self._events: list[Event] = []
        self._productive_events = 0

    def push(
        self,
        time: float,
        kind: str,
        target: str | int | tuple | None = None,
        **data,
    ) -> None:
        event = Event(
            time=time,
            sequence=next(self._sequence),
            kind=kind,
            target=target,
            data=data,
        )
        heapq.heappush(
            self._events,
            event,
        )
        if kind in PRODUCTIVE_EVENT_KINDS:
            self._productive_events += 1

    def pop(self) -> Event:
        event = heapq.heappop(self._events)
        if event.kind in PRODUCTIVE_EVENT_KINDS:
            self._productive_events -= 1
        return event

    def pop_same_time(self, time: float) -> list[Event]:
        events = []
        while self._events and self._events[0].time == time:
            event = heapq.heappop(self._events)
            if event.kind in PRODUCTIVE_EVENT_KINDS:
                self._productive_events -= 1
            events.append(event)
        return events

    def has_events(self) -> bool:
        return bool(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def has_event_at(self, time: float) -> bool:
        return bool(self._events) and self._events[0].time == time

    def has_pending_arrivals_or_releases(self) -> bool:
        return any(event.kind in {"lot_release", "lot_arrival"} for event in self._events)

    def has_productive_events(self) -> bool:
        return self._productive_events > 0


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
    max_cqt_lateness_minutes: float | None = None
    cqt_checks: int = 0
    cqt_violation_rate: float | None = None

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
            "max_cqt_lateness_minutes": self.max_cqt_lateness_minutes,
            "cqt_checks": self.cqt_checks,
            "cqt_violation_rate": self.cqt_violation_rate,
        }


def summarize_lots(
    lots: list[Lot],
    warmup_minutes: float = 0.0,
    cqt_lateness: list[float] | None = None,
    cqt_checks: int = 0,
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
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        avg_tardiness = sum(tardiness) / len(tardiness) if tardiness else None
        max_tardiness = max(tardiness) if tardiness else None
        tardy_lots = sum(value > 0 for value in tardiness)
        on_time_ratio = (
            sum(lot.completed_time <= lot.due_time for lot in due_lots) / len(due_lots)
            if due_lots
            else None
        )
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
        max_cqt_lateness_minutes=max(cqt_lateness) if cqt_lateness else None,
        cqt_checks=cqt_checks,
        cqt_violation_rate=(len(cqt_lateness) / cqt_checks if cqt_checks else None),
    )
