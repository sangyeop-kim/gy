from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.rl.dqn import DQNAgent
from src.rl.features import DispatchFeatureEncoder
from src.rl.selector import RLDispatchSelector
from src.simulator import SimulationConfig, Simulator, load_model
from src.simulator.analysis import (
    add_lot_cqt_metrics,
    lots_to_frame,
    product_metrics,
    snapshot_metrics,
    tool_metrics_from_simulator,
    toolgroup_metrics_from_simulator,
    waiting_lots_to_frame,
    waiting_step_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulator with a trained DQN dispatch checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint produced by train_dqn.py")
    parser.add_argument("--config", default="configs/default_simulation.json")
    parser.add_argument("--output-dir", default="outputs/rl_dqn_simulation")
    parser.add_argument("--max-lots", type=_optional_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--fallback-policy",
        default="priority_cr_fifo",
        help="Only used if a non-RL fallback is needed by simulator plumbing.",
    )
    parser.add_argument("--write-event-log", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = DispatchFeatureEncoder()
    agent = DQNAgent.load(args.checkpoint, device=args.device)
    config = SimulationConfig.from_json(args.config).with_overrides(
        max_lots=args.max_lots,
        random_seed=args.seed,
        dispatching_rule=args.fallback_policy,
        write_event_log=args.write_event_log,
        output_dir=str(output_dir),
    )
    model = load_model(config)
    selector = RLDispatchSelector(agent, encoder, explore=False, train_online=False)
    simulator = Simulator(model, config, dispatch_selector=selector)
    result = simulator.run()
    lots_df = add_lot_cqt_metrics(lots_to_frame(result.lots), simulator)
    event_df = pd.DataFrame(result.event_log)

    summary = result.summary.as_dict()
    summary.update(
        {
            "policy": "rl_dqn",
            "checkpoint": str(Path(args.checkpoint)),
            "device": str(agent.device),
            "seed": args.seed,
            "blocked_reason": simulator.blocked_reason,
            "completed_ratio": (
                result.summary.completed_lots / result.summary.released_lots
                if result.summary.released_lots
                else None
            ),
            "decisions": selector.decisions,
            "rewards": selector.rewards,
            "simulation_time_minutes": simulator.now,
            "elapsed_seconds": perf_counter() - started,
            **_recent_reward_components(selector),
        }
    )
    snapshot = snapshot_metrics(result.lots, simulator.now)
    snapshot.update({"policy": "rl_dqn", "blocked_reason": simulator.blocked_reason})

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame([snapshot]).to_csv(output_dir / "snapshot_summary.csv", index=False)
    lots_df.to_csv(output_dir / "lots.csv", index=False)
    product_metrics(lots_df).to_csv(output_dir / "product_summary.csv", index=False)
    toolgroup_metrics_from_simulator(simulator).to_csv(output_dir / "toolgroup_summary.csv", index=False)
    tool_metrics_from_simulator(simulator).to_csv(output_dir / "tool_summary.csv", index=False)
    if args.write_event_log:
        event_df.to_csv(output_dir / "events.csv", index=False)
    if simulator.blocked_reason is not None:
        waiting_df = waiting_lots_to_frame(simulator)
        waiting_df.to_csv(output_dir / "blocked_waiting_lots.csv", index=False)
        waiting_step_summary(waiting_df).to_csv(output_dir / "blocked_waiting_steps.csv", index=False)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


def _recent_reward_components(selector: RLDispatchSelector, window: int = 1000) -> dict[str, float | None]:
    if not selector.reward_components:
        return {}
    recent = selector.reward_components[-window:]
    keys = recent[0].keys()
    return {
        f"reward_component_{key}": sum(item[key] for item in recent) / len(recent)
        for key in keys
    }


def _optional_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


if __name__ == "__main__":
    main()
