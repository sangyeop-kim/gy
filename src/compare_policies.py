from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.simulator import SimulationConfig, Simulator, load_model
from src.simulator.analysis import (
    lots_to_frame,
    add_lot_cqt_metrics,
    lot_snapshot_frame,
    product_metrics,
    rank_policies,
    snapshot_metrics,
    tool_metrics_from_simulator,
    toolgroup_metrics_from_simulator,
    waiting_lots_to_frame,
    waiting_step_summary,
)
from src.simulator.policies import SUPPORTED_DISPATCHING_RULES


DEFAULT_POLICIES = (
    "fifo",
    "spt",
    "edd",
    "critical_ratio",
    "priority_cr_fifo",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dispatching policies at fab, product, toolgroup, tool, lot, and event levels."
    )
    parser.add_argument("--config", default="configs/default_simulation.json")
    parser.add_argument("--output-dir", default="outputs/policy_comparison")
    parser.add_argument(
        "--max-lots",
        type=_optional_int,
        default=None,
        help="Number of release lots to simulate. Use 'none' to run the full release file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--until-minutes",
        type=float,
        default=None,
        help="Stop the simulation at this many minutes from the release file start datetime.",
    )
    parser.add_argument(
        "--until-date",
        default="2018-04-01T00:00:00",
        help="Stop at an absolute date/datetime, e.g. 2018-03-01 or 2018-03-01T00:00:00.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        help="Policy names to compare. Use 'all' to run every registered policy.",
    )
    parser.add_argument(
        "--save-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-policy event logs. Disable with --no-save-events for smaller outputs.",
    )
    parser.add_argument(
        "--save-lots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-policy lot result tables. Disable with --no-save-lots for smaller outputs.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print simulator progress inside each policy run.",
    )
    parser.add_argument(
        "--progress-interval-events",
        type=int,
        default=50000,
        help="Print simulator progress every N processed events.",
    )
    args = parser.parse_args()

    policies = _resolve_policies(args.policies)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_lots is None:
        print(
            "WARNING: --max-lots none loads the full release file. "
            "This can take a long time for policy comparisons.",
            flush=True,
        )

    base_config = SimulationConfig.from_json(args.config)
    if args.seed is not None:
        base_config = base_config.with_overrides(random_seed=args.seed)

    overall_rows = []
    snapshot_rows = []
    product_rows = []
    toolgroup_rows = []
    tool_rows = []
    total_started_at = perf_counter()

    total = len(policies)
    for index, policy in enumerate(policies, start=1):
        policy_started_at = perf_counter()
        print("", flush=True)
        print("=" * 96, flush=True)
        print(
            f"[{index}/{total}] policy={policy} max_lots={args.max_lots} start",
            flush=True,
        )
        print("=" * 96, flush=True)
        config = base_config.with_overrides(
            max_lots=args.max_lots,
            output_dir=str(output_dir),
            dispatching_rule=policy,
            write_event_log=args.save_events,
        )
        _print_step(index, total, policy, "load_model", policy_started_at)
        initial_model = load_model(config)
        until_minutes = _resolve_until_minutes(
            initial_model.start_datetime,
            args.until_minutes,
            args.until_date,
        )
        if until_minutes is not None:
            config = config.with_overrides(until_minutes=until_minutes)
            model = load_model(config)
        else:
            model = initial_model
        _print_step(
            index, total, policy, f"loaded releases={len(model.releases)}", policy_started_at
        )
        _print_step(index, total, policy, "simulate", policy_started_at)
        simulator = Simulator(model, config)
        result = simulator.run(
            progress_callback=(
                _make_progress_callback(index, total, policy, policy_started_at)
                if args.progress
                else None
            ),
            progress_interval_events=args.progress_interval_events,
        )
        _print_step(
            index,
            total,
            policy,
            f"simulated completed={result.summary.completed_lots}/{result.summary.released_lots}",
            policy_started_at,
        )

        _print_step(index, total, policy, "aggregate", policy_started_at)
        lots_df = add_lot_cqt_metrics(lots_to_frame(result.lots), simulator)
        event_df = pd.DataFrame(result.event_log) if args.save_events else pd.DataFrame()

        overall = result.summary.as_dict()
        overall.update(
            {
                "policy": policy,
                "blocked_reason": simulator.blocked_reason,
                "completed_ratio": (
                    result.summary.completed_lots / result.summary.released_lots
                    if result.summary.released_lots
                    else None
                ),
            }
        )
        overall_rows.append(overall)
        snapshot = snapshot_metrics(result.lots, simulator.now)
        snapshot.update(
            {
                "policy": policy,
                "snapshot_datetime": (
                    model.start_datetime + pd.to_timedelta(simulator.now, unit="m")
                ).isoformat(),
                "blocked_reason": simulator.blocked_reason,
            }
        )
        snapshot_rows.append(snapshot)

        products = product_metrics(lots_df)
        if not products.empty:
            products.insert(0, "policy", policy)
            product_rows.extend(products.to_dict("records"))

        toolgroups = toolgroup_metrics_from_simulator(simulator)
        if not toolgroups.empty:
            toolgroups.insert(0, "policy", policy)
            toolgroup_rows.extend(toolgroups.to_dict("records"))

        tools = tool_metrics_from_simulator(simulator)
        if not tools.empty:
            tools.insert(0, "policy", policy)
            tool_rows.extend(tools.to_dict("records"))

        if args.save_lots:
            _print_step(index, total, policy, "save lots", policy_started_at)
            lots_df.to_csv(output_dir / f"lots_{policy}.csv", index=False)
            lot_snapshot_frame(result.lots, simulator.now).to_csv(
                output_dir / f"snapshot_lots_{policy}.csv",
                index=False,
            )
        if args.save_events:
            _print_step(index, total, policy, "save events", policy_started_at)
            event_df.to_csv(output_dir / f"events_{policy}.csv", index=False)
        if simulator.blocked_reason is not None:
            _print_step(
                index, total, policy, "save blocked waiting diagnostics", policy_started_at
            )
            waiting_df = waiting_lots_to_frame(simulator)
            waiting_df.to_csv(output_dir / f"blocked_waiting_lots_{policy}.csv", index=False)
            waiting_step_summary(waiting_df).to_csv(
                output_dir / f"blocked_waiting_steps_{policy}.csv",
                index=False,
            )
        _print_step(
            index,
            total,
            policy,
            f"done elapsed={_format_seconds(perf_counter() - policy_started_at)}",
            policy_started_at,
        )
        print("-" * 96, flush=True)

    print("writing summary tables", flush=True)
    overall_df = pd.DataFrame(overall_rows).set_index("policy")
    snapshot_df = pd.DataFrame(snapshot_rows).set_index("policy")
    product_df = pd.DataFrame(product_rows)
    toolgroup_df = pd.DataFrame(toolgroup_rows)
    tool_df = pd.DataFrame(tool_rows)
    rank_df = rank_policies(overall_df)

    overall_df.to_csv(output_dir / "overall_policy_summary.csv")
    snapshot_df.to_csv(output_dir / "snapshot_policy_summary.csv")
    product_df.to_csv(output_dir / "product_policy_summary.csv", index=False)
    toolgroup_df.to_csv(output_dir / "toolgroup_policy_summary.csv", index=False)
    tool_df.to_csv(output_dir / "tool_policy_summary.csv", index=False)
    rank_df.to_csv(output_dir / "policy_rank.csv")

    metadata = {
        "config": args.config,
        "output_dir": str(output_dir),
        "max_lots": args.max_lots,
        "seed": args.seed if args.seed is not None else base_config.random_seed,
        "policies": policies,
        "save_events": args.save_events,
        "save_lots": args.save_lots,
        "progress": args.progress,
        "progress_interval_events": args.progress_interval_events,
        "until_minutes": args.until_minutes,
        "until_date": args.until_date,
        "outputs": {
            "overall": "overall_policy_summary.csv",
            "snapshot": "snapshot_policy_summary.csv",
            "product": "product_policy_summary.csv",
            "toolgroup": "toolgroup_policy_summary.csv",
            "tool": "tool_policy_summary.csv",
            "rank": "policy_rank.csv",
            "lots": "lots_<policy>.csv" if args.save_lots else None,
            "snapshot_lots": "snapshot_lots_<policy>.csv" if args.save_lots else None,
            "events": "events_<policy>.csv" if args.save_events else None,
            "blocked_waiting_lots": "blocked_waiting_lots_<policy>.csv",
            "blocked_waiting_steps": "blocked_waiting_steps_<policy>.csv",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("policy ranking")
    print(rank_df[["mean_rank"]].to_string())
    print()
    print(f"total elapsed {_format_seconds(perf_counter() - total_started_at)}")
    print(f"wrote {output_dir}")


def _resolve_policies(values: list[str]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(SUPPORTED_DISPATCHING_RULES)
    unknown = sorted(set(values) - set(SUPPORTED_DISPATCHING_RULES))
    if unknown:
        raise ValueError(
            f"Unknown policies: {unknown}. Available policies: {list(SUPPORTED_DISPATCHING_RULES)}"
        )
    return values


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "all"}:
        return None
    return int(value)


def _resolve_until_minutes(
    start_datetime: datetime,
    until_minutes: float | None,
    until_date: str | None,
) -> float | None:
    if until_minutes is not None and until_date is not None:
        raise ValueError("Use only one of --until-minutes or --until-date")
    if until_minutes is not None:
        return until_minutes
    if until_date is None:
        return None
    target = datetime.fromisoformat(until_date)
    return (target - start_datetime).total_seconds() / 60.0


def _print_step(index: int, total: int, policy: str, message: str, started_at: float) -> None:
    print(
        f"[{index}/{total}] policy={policy} {message} "
        f"(elapsed {_format_seconds(perf_counter() - started_at)})",
        flush=True,
    )


def _make_progress_callback(index: int, total: int, policy: str, started_at: float):
    def callback(progress: dict) -> None:
        completed = progress["completed_lots"]
        total_lots = progress["total_lots"]
        ratio = completed / total_lots if total_lots else 0.0
        print(
            f"[{index}/{total}] policy={policy} sim "
            f"phase={progress['phase']} "
            f"events={progress['processed_events']:,} "
            f"sim_time={progress['sim_time']:.1f}min "
            f"date={progress['sim_datetime']} "
            f"ops={progress['completed_operations']:,}/{progress['started_operations']:,} "
            f"completed={completed:,}/{total_lots:,} ({ratio:.1%}) "
            f"waiting={progress['waiting_lots']:,} "
            f"tools busy/idle/down={progress['busy_tools']:,}/{progress['idle_tools']:,}/{progress['down_tools']:,} "
            f"pending_events={progress['pending_events']:,} "
            f"kinds={_format_event_counts(progress['event_counts'])} "
            f"event_log_rows={progress['event_log_rows']:,} "
            f"blocked={progress['blocked_reason'] or '-'} "
            f"(elapsed {_format_seconds(perf_counter() - started_at)})",
            flush=True,
        )

    return callback


def _format_event_counts(event_counts: dict) -> str:
    if not event_counts:
        return "-"
    keys = [
        "operation_complete",
        "lot_arrival",
        "downtime_start",
        "downtime_end",
        "lot_release",
    ]
    parts = [f"{key}={event_counts[key]:,}" for key in keys if key in event_counts]
    return ",".join(parts) if parts else "-"


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rest:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rest:.0f}s"


if __name__ == "__main__":
    main()
