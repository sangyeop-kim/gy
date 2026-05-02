import argparse
import json
from pathlib import Path

import pandas as pd

from src.simulator.config import SimulationConfig
from src.simulator.data_loader import load_model
from src.simulator.engine import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SMT2020 dataset 2 simulator.")
    parser.add_argument(
        "--config",
        default="configs/default_simulation.json",
        help="Path to a JSON simulation config.",
    )
    parser.add_argument("--dataset-dir", help="Override dataset directory.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--max-lots", type=int, help="Override maximum released lots.")
    parser.add_argument("--until-minutes", type=float, help="Stop simulation after this time.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--write-event-log",
        action="store_true",
        help="Write an event-level CSV log. This can be large.",
    )
    args = parser.parse_args()

    config = SimulationConfig.from_json(args.config).with_overrides(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_lots=args.max_lots,
        until_minutes=args.until_minutes,
        random_seed=args.seed,
        write_event_log=True if args.write_event_log else None,
    )

    model = load_model(config)
    result = Simulator(model, config).run()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result.summary.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lots_path = output_dir / "lots.csv"
    pd.DataFrame(
        [
            {
                "lot_id": lot.id,
                "lot_name": lot.name,
                "product_name": lot.product_name,
                "priority": lot.priority,
                "release_time": lot.release_time,
                "due_time": lot.due_time,
                "completed_time": lot.completed_time,
                "cycle_time": (
                    lot.completed_time - lot.release_time
                    if lot.completed_time is not None
                    else None
                ),
                "completed": lot.completed_time is not None,
            }
            for lot in result.lots
        ]
    ).to_csv(lots_path, index=False)

    if config.write_event_log:
        pd.DataFrame(result.event_log).to_csv(output_dir / "event_log.csv", index=False)

    print(json.dumps(result.summary.as_dict(), indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {lots_path}")


if __name__ == "__main__":
    main()
