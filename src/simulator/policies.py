from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.simulator.model import Lot, Tool


@dataclass(frozen=True)
class DispatchContext:
    now: float
    tool: Tool | None = None


DispatchRule = Callable[[tuple[Lot, ...], DispatchContext], Lot]
DISPATCH_RULES: dict[str, DispatchRule] = {}


def register_dispatch_rule(name: str) -> Callable[[DispatchRule], DispatchRule]:
    """Register a lot selection rule used when an idle tool chooses work."""

    def decorator(func: DispatchRule) -> DispatchRule:
        if name in DISPATCH_RULES:
            raise ValueError(f"Dispatching rule already registered: {name}")
        DISPATCH_RULES[name] = func
        return func

    return decorator


class DispatchPolicy:
    def __init__(self, rule: str) -> None:
        if rule not in DISPATCH_RULES:
            raise ValueError(
                f"Unsupported dispatching rule: {rule}. "
                f"Available rules: {', '.join(SUPPORTED_DISPATCHING_RULES)}"
            )
        self.rule = rule

    def select_lot(self, waiting_lots: Iterable[Lot], now: float, tool: Tool | None = None) -> Lot:
        lots = tuple(waiting_lots)
        if not lots:
            raise ValueError("Cannot dispatch from an empty waiting list")
        return DISPATCH_RULES[self.rule](lots, DispatchContext(now=now, tool=tool))


@register_dispatch_rule("fifo")
def _fifo(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_waiting_since(lot), lot.release_time, lot.id))


@register_dispatch_rule("lifo")
def _lifo(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return max(lots, key=lambda lot: (_waiting_since(lot), lot.release_time, lot.id))


@register_dispatch_rule("spt")
def _spt(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_current_step_nominal_minutes(lot), _waiting_since(lot), lot.id))


@register_dispatch_rule("lpt")
def _lpt(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return max(lots, key=lambda lot: (_current_step_nominal_minutes(lot), -_waiting_since(lot), -lot.id))


@register_dispatch_rule("srpt")
def _srpt(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_remaining_nominal_minutes(lot), _waiting_since(lot), lot.id))


@register_dispatch_rule("lrpt")
def _lrpt(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return max(lots, key=lambda lot: (_remaining_nominal_minutes(lot), -_waiting_since(lot), -lot.id))


@register_dispatch_rule("edd")
def _edd(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_due_time(lot), _waiting_since(lot), lot.id))


@register_dispatch_rule("least_slack")
def _least_slack(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_slack(lot, context.now), _waiting_since(lot), lot.id))


@register_dispatch_rule("slack_per_remaining_step")
def _slack_per_remaining_step_rule(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_slack_per_remaining_step(lot, context.now), _waiting_since(lot), lot.id))


@register_dispatch_rule("critical_ratio")
def _critical_ratio_rule(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (_critical_ratio(lot, context.now), _waiting_since(lot), lot.id))


@register_dispatch_rule("priority_fifo")
def _priority_fifo(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (*_dispatch_priority(lot), _waiting_since(lot), lot.id))


@register_dispatch_rule("priority_spt")
def _priority_spt(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(
        lots,
        key=lambda lot: (*_dispatch_priority(lot), _current_step_nominal_minutes(lot), _waiting_since(lot), lot.id),
    )


@register_dispatch_rule("priority_edd")
def _priority_edd(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (*_dispatch_priority(lot), _due_time(lot), _waiting_since(lot), lot.id))


@register_dispatch_rule("priority_least_slack")
def _priority_least_slack(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(lots, key=lambda lot: (*_dispatch_priority(lot), _slack(lot, context.now), _waiting_since(lot), lot.id))


@register_dispatch_rule("priority_cr_fifo")
def _priority_cr_fifo(lots: tuple[Lot, ...], context: DispatchContext) -> Lot:
    return min(
        lots,
        key=lambda lot: (*_dispatch_priority(lot), _critical_ratio(lot, context.now), _waiting_since(lot), lot.id),
    )


SUPPORTED_DISPATCHING_RULES = tuple(DISPATCH_RULES)


def _waiting_since(lot: Lot) -> float:
    return lot.waiting_since if lot.waiting_since is not None else lot.release_time


def _dispatch_priority(lot: Lot) -> tuple[int, int]:
    return (0 if lot.super_hot_lot else 1, -lot.priority)


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
    return (lot.due_time - now) / max(_remaining_nominal_minutes(lot), 1.0)


def _slack(lot: Lot, now: float) -> float:
    if lot.due_time is None:
        return float("inf")
    return lot.due_time - now - _remaining_nominal_minutes(lot)


def _slack_per_remaining_step(lot: Lot, now: float) -> float:
    remaining_steps = max(len(lot.route.steps) - lot.step_index, 1)
    return _slack(lot, now) / remaining_steps
