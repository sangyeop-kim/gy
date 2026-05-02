from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.rl.dqn import DQNAgent, DQNConfig
from src.rl.features import DispatchFeatureEncoder
from src.rl.selector import RLDispatchSelector
from src.simulator import SimulationConfig, Simulator, load_model
from src.simulator.analysis import add_lot_cqt_metrics, lots_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small DQN dispatch policy on the SMT2020 simulator.")
    parser.add_argument("--config", default="configs/default_simulation.json")
    parser.add_argument("--output-dir", default="outputs/rl_dqn")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-lots", type=int, default=200)
    parser.add_argument("--eval-max-lots", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--fallback-policy", default="priority_cr_fifo")
    parser.add_argument("--fallback-probability", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = DispatchFeatureEncoder()
    agent = DQNAgent(
        DQNConfig(
            input_dim=encoder.size,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
            device=args.device,
        )
    )
    base_config = SimulationConfig.from_json(args.config)
    episode_rows = []
    started = perf_counter()
    for episode in range(1, args.episodes + 1):
        episode_seed = args.seed + episode - 1
        config = base_config.with_overrides(
            max_lots=args.max_lots,
            random_seed=episode_seed,
            dispatching_rule=args.fallback_policy,
            write_event_log=False,
        )
        model = load_model(config)
        selector = RLDispatchSelector(
            agent,
            encoder,
            explore=True,
            train_online=True,
            fallback_probability=args.fallback_probability,
        )
        simulator = Simulator(model, config, dispatch_selector=selector)
        result = simulator.run()
        avg_loss = sum(selector.losses[-100:]) / min(100, len(selector.losses)) if selector.losses else None
        row = {
            "episode": episode,
            "seed": episode_seed,
            "released_lots": result.summary.released_lots,
            "completed_lots": result.summary.completed_lots,
            "completed_ratio": (
                result.summary.completed_lots / result.summary.released_lots
                if result.summary.released_lots
                else None
            ),
            "average_cycle_time_minutes": result.summary.average_cycle_time_minutes,
            "average_tardiness_minutes": result.summary.average_tardiness_minutes,
            "tardy_lots": result.summary.tardy_lots,
            "on_time_ratio": result.summary.on_time_ratio,
            "cqt_violations": result.summary.cqt_violations,
            "blocked_reason": simulator.blocked_reason,
            "decisions": selector.decisions,
            "rewards": selector.rewards,
            "replay_size": len(agent.replay),
            "training_steps": agent.training_steps,
            "epsilon": agent.epsilon,
            "device": str(agent.device),
            "avg_recent_loss": avg_loss,
            **_recent_reward_components(selector),
            "elapsed_seconds": perf_counter() - started,
        }
        episode_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    agent_path = output_dir / "dqn_agent.pkl"
    agent.save(str(agent_path))
    pd.DataFrame(episode_rows).to_csv(output_dir / "training_episodes.csv", index=False)
    (output_dir / "feature_names.json").write_text(
        json.dumps(list(encoder.feature_names()), indent=2),
        encoding="utf-8",
    )
    eval_row = evaluate(agent, encoder, base_config, args)
    pd.DataFrame([eval_row]).to_csv(output_dir / "evaluation.csv", index=False)
    print("evaluation")
    print(json.dumps(eval_row, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


def evaluate(agent: DQNAgent, encoder: DispatchFeatureEncoder, base_config: SimulationConfig, args) -> dict:
    config = base_config.with_overrides(
        max_lots=args.eval_max_lots,
        random_seed=args.seed,
        dispatching_rule=args.fallback_policy,
        write_event_log=False,
    )
    model = load_model(config)
    selector = RLDispatchSelector(agent, encoder, explore=False, train_online=False)
    simulator = Simulator(model, config, dispatch_selector=selector)
    result = simulator.run()
    lots_df = add_lot_cqt_metrics(lots_to_frame(result.lots), simulator)
    return {
        "policy": "rl_dqn",
        "released_lots": result.summary.released_lots,
        "completed_lots": result.summary.completed_lots,
        "completed_ratio": (
            result.summary.completed_lots / result.summary.released_lots
            if result.summary.released_lots
            else None
        ),
        "average_cycle_time_minutes": result.summary.average_cycle_time_minutes,
        "average_tardiness_minutes": result.summary.average_tardiness_minutes,
        "tardy_lots": result.summary.tardy_lots,
        "on_time_ratio": result.summary.on_time_ratio,
        "cqt_violations": result.summary.cqt_violations,
        "blocked_reason": simulator.blocked_reason,
        "decisions": selector.decisions,
        "completed_super_hot_lots": int(lots_df.loc[lots_df["completed"], "super_hot_lot"].sum()) if not lots_df.empty else 0,
        **_recent_reward_components(selector),
    }


def _recent_reward_components(selector: RLDispatchSelector, window: int = 1000) -> dict[str, float | None]:
    if not selector.reward_components:
        return {}
    recent = selector.reward_components[-window:]
    keys = recent[0].keys()
    return {
        f"reward_component_{key}": sum(item[key] for item in recent) / len(recent)
        for key in keys
    }


if __name__ == "__main__":
    main()
