from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Iterable

import pandas as pd

from src.simulator.config import SimulationConfig
from src.simulator.engine import Simulator
from src.simulator.io import load_model
from src.simulator.model import Lot


def run_policy_comparison(
    base_config: SimulationConfig,
    policies: Iterable[str],
    *,
    max_lots: int | None = None,
    output_dir: str | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run the same scenario under multiple dispatching policies.

    Returns:
        overall_df: one row per policy.
        toolgroup_df: one row per policy and toolgroup.
        lot_tables: policy -> lot-level result table.
        event_tables: policy -> event log table.
    """

    overall_rows = []
    toolgroup_rows = []
    lot_tables: dict[str, pd.DataFrame] = {}
    event_tables: dict[str, pd.DataFrame] = {}

    for policy in policies:
        config = base_config.with_overrides(
            max_lots=max_lots,
            output_dir=output_dir or base_config.output_dir,
            random_seed=random_seed if random_seed is not None else base_config.random_seed,
            dispatching_rule=policy,
            write_event_log=True,
        )
        started_at = perf_counter()
        model = load_model(config)
        simulator = Simulator(model, config)
        result = simulator.run()
        runtime_seconds = perf_counter() - started_at

        lots_df = add_lot_cqt_metrics(lots_to_frame(result.lots), simulator)
        event_df = pd.DataFrame(result.event_log)
        overall = result.summary.as_dict()
        overall.update(
            {
                "policy": policy,
                "runtime_seconds": runtime_seconds,
                "completed_ratio": (
                    result.summary.completed_lots / result.summary.released_lots
                    if result.summary.released_lots
                    else None
                ),
            }
        )
        overall_rows.append(overall)

        tg_df = toolgroup_metrics_from_simulator(simulator)
        if not tg_df.empty:
            tg_df.insert(0, "policy", policy)
            toolgroup_rows.extend(tg_df.to_dict("records"))

        lot_tables[policy] = lots_df
        event_tables[policy] = event_df

    overall_df = pd.DataFrame(overall_rows).set_index("policy")
    toolgroup_df = pd.DataFrame(toolgroup_rows)
    return overall_df, toolgroup_df, lot_tables, event_tables


def lots_to_frame(lots: Iterable[Lot]) -> pd.DataFrame:
    rows = []
    for lot in lots:
        tardiness = (
            max(0.0, lot.completed_time - lot.due_time)
            if lot.completed_time is not None and lot.due_time is not None
            else None
        )
        rows.append(
            {
                "lot_id": lot.id,
                "lot_name": lot.name,
                "product_name": lot.product_name,
                "priority": lot.priority,
                "super_hot_lot": lot.super_hot_lot,
                "release_time": lot.release_time,
                "due_time": lot.due_time,
                "completed_time": lot.completed_time,
                "cycle_time": (
                    lot.completed_time - lot.release_time
                    if lot.completed_time is not None
                    else None
                ),
                "completed": lot.completed_time is not None,
                "tardy": tardiness is not None and tardiness > 0,
                "tardiness": tardiness,
            }
        )
    return pd.DataFrame(rows)


def add_lot_cqt_metrics(lots_df: pd.DataFrame, simulator: Simulator) -> pd.DataFrame:
    if lots_df.empty:
        return lots_df
    cqt_df = pd.DataFrame(
        [
            {
                "lot_id": lot_id,
                "cqt_passes": int(counts.get("passes", 0)),
                "cqt_violations": int(counts.get("violations", 0)),
            }
            for lot_id, counts in simulator.cqt_counts_by_lot.items()
        ]
    )
    if cqt_df.empty:
        lots_df = lots_df.copy()
        lots_df["cqt_passes"] = 0
        lots_df["cqt_violations"] = 0
        lots_df["has_cqt_violation"] = False
        return lots_df
    merged = lots_df.merge(cqt_df, on="lot_id", how="left")
    merged["cqt_passes"] = merged["cqt_passes"].fillna(0).astype(int)
    merged["cqt_violations"] = merged["cqt_violations"].fillna(0).astype(int)
    merged["has_cqt_violation"] = merged["cqt_violations"] > 0
    return merged


def product_metrics(lots_df: pd.DataFrame) -> pd.DataFrame:
    if lots_df.empty:
        return pd.DataFrame(columns=["product_name"])
    return (
        lots_df.groupby("product_name")
        .agg(
            released_lots=("lot_id", "count"),
            completed_lots=("completed", "sum"),
            super_hot_lots=("super_hot_lot", "sum"),
            tardy_lots=("tardy", "sum"),
            average_cycle_time_minutes=("cycle_time", "mean"),
            average_tardiness_minutes=("tardiness", "mean"),
            max_tardiness_minutes=("tardiness", "max"),
        )
        .reset_index()
        .assign(
            completed_ratio=lambda df: df["completed_lots"] / df["released_lots"],
            tardy_ratio=lambda df: df["tardy_lots"] / df["released_lots"],
        )
    )


def snapshot_metrics(lots: Iterable[Lot], now: float) -> dict[str, float | int | None]:
    lot_list = list(lots)
    released = [lot for lot in lot_list if lot.release_time <= now]
    not_released = [lot for lot in lot_list if lot.release_time > now]
    completed = [
        lot
        for lot in released
        if lot.completed_time is not None and lot.completed_time <= now
    ]
    in_system = [
        lot
        for lot in released
        if lot.completed_time is None or lot.completed_time > now
    ]
    waiting = [lot for lot in in_system if lot.waiting_since is not None]
    completed_due = [lot for lot in completed if lot.due_time is not None]
    due_by_now = [lot for lot in released if lot.due_time is not None and lot.due_time <= now]
    completed_on_time = [
        lot
        for lot in completed_due
        if lot.completed_time is not None and lot.completed_time <= lot.due_time
    ]
    completed_tardy = [
        lot
        for lot in completed_due
        if lot.completed_time is not None and lot.completed_time > lot.due_time
    ]
    overdue_in_system = [
        lot
        for lot in in_system
        if lot.due_time is not None and lot.due_time < now
    ]
    cycle_times = [lot.completed_time - lot.release_time for lot in completed]
    tardiness = [
        max(0.0, lot.completed_time - lot.due_time)
        for lot in completed_due
        if lot.completed_time is not None
    ]
    return {
        "snapshot_time": now,
        "total_lots_loaded": len(lot_list),
        "released_lots_by_snapshot": len(released),
        "not_yet_released_lots": len(not_released),
        "completed_lots_by_snapshot": len(completed),
        "in_system_lots": len(in_system),
        "waiting_lots": len(waiting),
        "processing_lots_estimate": max(0, len(in_system) - len(waiting)),
        "due_by_snapshot_lots": len(due_by_now),
        "completed_on_time_lots": len(completed_on_time),
        "completed_tardy_lots": len(completed_tardy),
        "overdue_unfinished_lots": len(overdue_in_system),
        "service_level_completed_due": (
            len(completed_on_time) / len(completed_due) if completed_due else None
        ),
        "completion_ratio_released": (
            len(completed) / len(released) if released else None
        ),
        "average_cycle_time_minutes": (
            sum(cycle_times) / len(cycle_times) if cycle_times else None
        ),
        "average_tardiness_minutes": (
            sum(tardiness) / len(tardiness) if tardiness else None
        ),
        "max_tardiness_minutes": max(tardiness) if tardiness else None,
    }


def lot_snapshot_frame(lots: Iterable[Lot], now: float) -> pd.DataFrame:
    rows = []
    for lot in lots:
        released = lot.release_time <= now
        completed = lot.completed_time is not None and lot.completed_time <= now
        overdue = released and not completed and lot.due_time is not None and lot.due_time < now
        rows.append(
            {
                "lot_id": lot.id,
                "lot_name": lot.name,
                "product_name": lot.product_name,
                "priority": lot.priority,
                "super_hot_lot": lot.super_hot_lot,
                "release_time": lot.release_time,
                "due_time": lot.due_time,
                "completed_time": lot.completed_time,
                "released_by_snapshot": released,
                "completed_by_snapshot": completed,
                "in_system_at_snapshot": released and not completed,
                "waiting_at_snapshot": released and not completed and lot.waiting_since is not None,
                "overdue_unfinished_at_snapshot": overdue,
                "completed_tardy": (
                    completed and lot.due_time is not None and lot.completed_time > lot.due_time
                ),
                "step_index_at_snapshot": lot.step_index if released and not completed else None,
                "current_toolgroup_at_snapshot": lot.current_toolgroup if released and not completed else None,
            }
        )
    return pd.DataFrame(rows)


def waiting_lots_to_frame(simulator: Simulator) -> pd.DataFrame:
    rows = []
    compatible_wafers: dict[tuple[str, int], int] = {}
    compatible_lots: dict[tuple[str, int], int] = {}
    for toolgroup in simulator.toolgroups.values():
        for lot_id in toolgroup.waiting_lot_ids:
            lot = simulator.lots[lot_id]
            key = (toolgroup.spec.name, id(lot.current_step))
            compatible_wafers[key] = compatible_wafers.get(key, 0) + lot.wafers_per_lot
            compatible_lots[key] = compatible_lots.get(key, 0) + 1
    for toolgroup in simulator.toolgroups.values():
        for lot_id in toolgroup.waiting_lot_ids:
            lot = simulator.lots[lot_id]
            step = lot.current_step
            key = (toolgroup.spec.name, id(step))
            rows.append(
                {
                    "lot_id": lot.id,
                    "lot_name": lot.name,
                    "product_name": lot.product_name,
                    "priority": lot.priority,
                    "super_hot_lot": lot.super_hot_lot,
                    "toolgroup": toolgroup.spec.name,
                    "route": step.route,
                    "step_number": step.step_number,
                    "step_description": step.description,
                    "processing_unit": step.processing_unit,
                    "batch_minimum": step.batch_minimum,
                    "batch_maximum": step.batch_maximum,
                    "wafers_per_lot": lot.wafers_per_lot,
                    "compatible_waiting_lots": compatible_lots[key],
                    "compatible_waiting_wafers": compatible_wafers[key],
                    "batch_minimum_satisfied": (
                        compatible_wafers[key] >= step.batch_minimum
                        if step.batch_minimum is not None
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def waiting_step_summary(waiting_df: pd.DataFrame) -> pd.DataFrame:
    if waiting_df.empty:
        return pd.DataFrame()
    return (
        waiting_df.groupby(
            [
                "toolgroup",
                "route",
                "step_number",
                "step_description",
                "processing_unit",
                "batch_minimum",
                "batch_maximum",
            ],
            dropna=False,
        )
        .agg(
            waiting_lots=("lot_id", "count"),
            waiting_wafers=("wafers_per_lot", "sum"),
            super_hot_lots=("super_hot_lot", "sum"),
            max_priority=("priority", "max"),
            batch_minimum_satisfied=("batch_minimum_satisfied", "max"),
        )
        .reset_index()
        .sort_values(["waiting_wafers", "waiting_lots"], ascending=[False, False])
    )


def toolgroup_metrics(lots: Iterable[Lot], event_df: pd.DataFrame) -> pd.DataFrame:
    visit_df = lot_toolgroup_visits(lots)
    wait_df = queue_wait_metrics(event_df)
    step_df = step_process_metrics(event_df)
    event_metrics_df = toolgroup_event_metrics(event_df)

    frames = [visit_df, wait_df, step_df, event_metrics_df]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="toolgroup", how="outer")
    return merged.fillna(
        {
            "visited_lots": 0,
            "tardy_lots": 0,
            "completed_steps": 0,
            "started_steps": 0,
            "cqt_violations": 0,
            "cqt_passes": 0,
            "cqt_checks": 0,
            "downtime_starts": 0,
            "breakdown_starts": 0,
            "pm_starts": 0,
        }
    )


def toolgroup_metrics_from_simulator(simulator: Simulator) -> pd.DataFrame:
    visit_df = lot_toolgroup_visits(simulator.lots.values())
    wait_df = _toolgroup_wait_frame(simulator)
    process_df = _toolgroup_process_frame(simulator)
    event_df = _toolgroup_runtime_event_frame(simulator)
    frames = [frame for frame in (visit_df, wait_df, process_df, event_df) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="toolgroup", how="outer")
    return merged.fillna(
        {
            "visited_lots": 0,
            "tardy_lots": 0,
            "completed_steps": 0,
            "started_steps": 0,
            "cqt_violations": 0,
            "cqt_passes": 0,
            "downtime_starts": 0,
            "breakdown_starts": 0,
            "pm_starts": 0,
        }
    )


def tool_metrics_from_simulator(simulator: Simulator) -> pd.DataFrame:
    process_df = _tool_process_frame(simulator)
    event_df = _tool_runtime_event_frame(simulator)
    frames = [frame for frame in (process_df, event_df) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["toolgroup", "tool_index"], how="outer")
    return merged.fillna(
        {
            "started_steps": 0,
            "completed_steps": 0,
            "downtime_starts": 0,
            "breakdown_starts": 0,
            "pm_starts": 0,
        }
    )


def _toolgroup_wait_frame(simulator: Simulator) -> pd.DataFrame:
    rows = []
    for toolgroup, waits in simulator.queue_waits_by_toolgroup.items():
        series = pd.Series(waits, dtype="float64")
        rows.append(
            {
                "toolgroup": toolgroup,
                "started_steps": len(waits),
                "avg_queue_wait": series.mean(),
                "p90_queue_wait": series.quantile(0.9),
                "max_queue_wait": series.max(),
            }
        )
    return pd.DataFrame(rows)


def _toolgroup_process_frame(simulator: Simulator) -> pd.DataFrame:
    rows = []
    for toolgroup, times in simulator.process_times_by_toolgroup.items():
        series = pd.Series(times, dtype="float64")
        rows.append(
            {
                "toolgroup": toolgroup,
                "completed_steps": len(times),
                "avg_process_time": series.mean(),
                "p90_process_time": series.quantile(0.9),
                "max_process_time": series.max(),
            }
        )
    return pd.DataFrame(rows)


def _toolgroup_runtime_event_frame(simulator: Simulator) -> pd.DataFrame:
    toolgroups = (
        set(simulator.downtime_counts_by_toolgroup)
        | set(simulator.cqt_counts_by_toolgroup)
    )
    rows = []
    for toolgroup in toolgroups:
        downtime = simulator.downtime_counts_by_toolgroup.get(toolgroup, {})
        cqt = simulator.cqt_counts_by_toolgroup.get(toolgroup, {})
        lateness = simulator.cqt_lateness_by_toolgroup.get(toolgroup, [])
        cqt_checks = int(cqt.get("violations", 0) + cqt.get("passes", 0))
        rows.append(
            {
                "toolgroup": toolgroup,
                "cqt_violations": int(cqt.get("violations", 0)),
                "cqt_passes": int(cqt.get("passes", 0)),
                "cqt_checks": cqt_checks,
                "cqt_violation_rate": (
                    int(cqt.get("violations", 0)) / cqt_checks if cqt_checks else None
                ),
                "average_cqt_lateness_minutes": (
                    sum(lateness) / len(lateness) if lateness else None
                ),
                "max_cqt_lateness_minutes": max(lateness) if lateness else None,
                "downtime_starts": int(downtime.get("downtime_starts", 0)),
                "breakdown_starts": int(downtime.get("breakdown_starts", 0)),
                "pm_starts": int(downtime.get("pm_starts", 0)),
            }
        )
    return pd.DataFrame(rows)


def _tool_process_frame(simulator: Simulator) -> pd.DataFrame:
    rows = []
    for (toolgroup, tool_index), times in simulator.process_times_by_tool.items():
        series = pd.Series(times, dtype="float64")
        rows.append(
            {
                "toolgroup": toolgroup,
                "tool_index": tool_index,
                "started_steps": len(times),
                "completed_steps": len(times),
                "avg_process_time": series.mean(),
                "p90_process_time": series.quantile(0.9),
                "max_process_time": series.max(),
            }
        )
    return pd.DataFrame(rows)


def _tool_runtime_event_frame(simulator: Simulator) -> pd.DataFrame:
    rows = []
    for (toolgroup, tool_index), downtime in simulator.downtime_counts_by_tool.items():
        rows.append(
            {
                "toolgroup": toolgroup,
                "tool_index": tool_index,
                "downtime_starts": int(downtime.get("downtime_starts", 0)),
                "breakdown_starts": int(downtime.get("breakdown_starts", 0)),
                "pm_starts": int(downtime.get("pm_starts", 0)),
            }
        )
    return pd.DataFrame(rows)


def lot_toolgroup_visits(lots: Iterable[Lot]) -> pd.DataFrame:
    rows = []
    for lot in lots:
        tardy = (
            lot.completed_time is not None
            and lot.due_time is not None
            and lot.completed_time > lot.due_time
        )
        visited = {history[1] for history in lot.history}
        for toolgroup in visited:
            rows.append({"toolgroup": toolgroup, "lot_id": lot.id, "tardy": tardy})
    if not rows:
        return pd.DataFrame(columns=["toolgroup"])
    df = pd.DataFrame(rows)
    metrics = (
        df.groupby("toolgroup")
        .agg(
            visited_lots=("lot_id", "nunique"),
            tardy_lots=("tardy", "sum"),
        )
        .reset_index()
    )
    metrics["tardy_visit_rate"] = metrics["tardy_lots"] / metrics["visited_lots"]
    return metrics


def queue_wait_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["toolgroup"])
    arrivals = event_df[event_df["event"] == "arrive"][
        ["lot_id", "step_index", "toolgroup", "time"]
    ].rename(columns={"time": "arrive_time"})
    starts = event_df[event_df["event"].isin(["start", "start_batch"])][
        ["lot_id", "step_index", "toolgroup", "time"]
    ].rename(columns={"time": "start_time"})
    if arrivals.empty or starts.empty:
        return pd.DataFrame(columns=["toolgroup"])
    merged = arrivals.merge(
        starts,
        on=["lot_id", "step_index", "toolgroup"],
        how="inner",
    )
    merged["queue_wait"] = merged["start_time"] - merged["arrive_time"]
    return (
        merged.groupby("toolgroup")
        .agg(
            started_steps=("lot_id", "count"),
            avg_queue_wait=("queue_wait", "mean"),
            p90_queue_wait=("queue_wait", lambda values: values.quantile(0.9)),
            max_queue_wait=("queue_wait", "max"),
        )
        .reset_index()
    )


def step_process_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["toolgroup"])
    starts = event_df[event_df["event"].isin(["start", "start_batch"])][
        ["lot_id", "step_index", "toolgroup", "tool_index", "time"]
    ].rename(columns={"time": "start_time"})
    completes = event_df[event_df["event"].isin(["complete_step", "complete_batch_step"])][
        ["lot_id", "step_index", "toolgroup", "tool_index", "time"]
    ].rename(columns={"time": "complete_time"})
    if starts.empty or completes.empty:
        return pd.DataFrame(columns=["toolgroup"])
    merged = starts.merge(
        completes,
        on=["lot_id", "step_index", "toolgroup", "tool_index"],
        how="inner",
    )
    merged["process_time_observed"] = merged["complete_time"] - merged["start_time"]
    return (
        merged.groupby("toolgroup")
        .agg(
            completed_steps=("lot_id", "count"),
            avg_process_time=("process_time_observed", "mean"),
            p90_process_time=("process_time_observed", lambda values: values.quantile(0.9)),
            max_process_time=("process_time_observed", "max"),
        )
        .reset_index()
    )


def toolgroup_event_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty or "toolgroup" not in event_df.columns:
        return pd.DataFrame(columns=["toolgroup"])
    rows = []
    for toolgroup, group in event_df.dropna(subset=["toolgroup"]).groupby("toolgroup"):
        events = group["event"]
        payload = group["payload"].fillna("").astype(str) if "payload" in group.columns else pd.Series([], dtype=str)
        rows.append(
            {
                "toolgroup": toolgroup,
                "cqt_violations": int((events == "cqt_violation").sum()),
                "cqt_passes": int((events == "cqt_pass").sum()),
                "downtime_starts": int((events == "downtime_start").sum()),
                "breakdown_starts": int(((events == "downtime_start") & payload.str.startswith("breakdown:")).sum()),
                "pm_starts": int(((events == "downtime_start") & payload.str.startswith("pm_")).sum()),
            }
        )
    return pd.DataFrame(rows)


def tool_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty or "tool_index" not in event_df.columns:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    step_df = tool_step_metrics(event_df)
    event_df = tool_event_metrics(event_df)
    frames = [frame for frame in (step_df, event_df) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["toolgroup", "tool_index"], how="outer")
    return merged.fillna(
        {
            "started_steps": 0,
            "completed_steps": 0,
            "downtime_starts": 0,
            "breakdown_starts": 0,
            "pm_starts": 0,
        }
    )


def tool_step_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    starts = event_df[event_df["event"].isin(["start", "start_batch"])][
        ["lot_id", "step_index", "toolgroup", "tool_index", "time"]
    ].rename(columns={"time": "start_time"})
    completes = event_df[event_df["event"].isin(["complete_step", "complete_batch_step"])][
        ["lot_id", "step_index", "toolgroup", "tool_index", "time"]
    ].rename(columns={"time": "complete_time"})
    if starts.empty:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    if completes.empty:
        return (
            starts.groupby(["toolgroup", "tool_index"])
            .agg(started_steps=("lot_id", "count"))
            .reset_index()
        )
    merged = starts.merge(
        completes,
        on=["lot_id", "step_index", "toolgroup", "tool_index"],
        how="left",
    )
    merged["process_time_observed"] = merged["complete_time"] - merged["start_time"]
    return (
        merged.groupby(["toolgroup", "tool_index"])
        .agg(
            started_steps=("lot_id", "count"),
            completed_steps=("complete_time", "count"),
            avg_process_time=("process_time_observed", "mean"),
            p90_process_time=("process_time_observed", lambda values: values.quantile(0.9)),
            max_process_time=("process_time_observed", "max"),
        )
        .reset_index()
    )


def tool_event_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    tool_events = event_df.dropna(subset=["toolgroup", "tool_index"])
    if tool_events.empty:
        return pd.DataFrame(columns=["toolgroup", "tool_index"])
    rows = []
    for (toolgroup, tool_index), group in tool_events.groupby(["toolgroup", "tool_index"]):
        events = group["event"]
        payload = group["payload"].fillna("").astype(str) if "payload" in group.columns else pd.Series([], dtype=str)
        rows.append(
            {
                "toolgroup": toolgroup,
                "tool_index": int(tool_index),
                "downtime_starts": int((events == "downtime_start").sum()),
                "breakdown_starts": int(((events == "downtime_start") & payload.str.startswith("breakdown:")).sum()),
                "pm_starts": int(((events == "downtime_start") & payload.str.startswith("pm_")).sum()),
            }
        )
    return pd.DataFrame(rows)


def rank_policies(overall_df: pd.DataFrame) -> pd.DataFrame:
    lower_is_better = [
        "average_cycle_time_minutes",
        "average_tardiness_minutes",
        "max_tardiness_minutes",
        "tardy_lots",
        "cqt_violations",
        "last_completion_time",
    ]
    ranks = pd.DataFrame(index=overall_df.index)
    for column in lower_is_better:
        if column in overall_df.columns:
            ranks[f"{column}_rank"] = overall_df[column].rank(method="min", ascending=True)
    if "on_time_ratio" in overall_df.columns:
        ranks["on_time_ratio_rank"] = overall_df["on_time_ratio"].rank(method="min", ascending=False)
    rank_columns = [column for column in ranks.columns if column.endswith("_rank")]
    ranks["mean_rank"] = ranks[rank_columns].mean(axis=1)
    return ranks.sort_values("mean_rank")
