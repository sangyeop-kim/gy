from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    dataset_dir: str = "dataset"
    route_file_glob: str = "Route_Product_*.csv"
    toolgroups_file: str = "Toolgroups.csv"
    pm_file: str = "PM.csv"
    breakdown_file: str = "Breakdown.csv"
    setups_file: str = "Setups.csv"
    transport_file: str = "Transport.csv"
    release_file: str = "Lotrelease - variable due dates.csv"
    output_dir: str = "outputs"
    random_seed: int = 42
    max_lots: int | None = 1000
    until_minutes: float | None = None
    warmup_minutes: float = 0.0
    release_scenario: str | None = None
    dispatching_rule: str = "priority_cr_fifo"
    toolgroup_dispatching_rules: dict[str, str] | None = None
    sample_process_times: bool = True
    sample_transport_times: bool = True
    wafer_time_mode: str = "per_wafer"
    enable_sampling: bool = True
    enable_rework: bool = True
    enable_transport: bool = True
    enable_batching: bool = True
    enable_cascading: bool = True
    enable_pm: bool = True
    enable_breakdowns: bool = True
    write_event_log: bool = False
    toolgroup_overrides: dict[str, dict[str, Any]] | None = None
    transport_override: dict[str, Any] | None = None
    pm_overrides: dict[str, dict[str, Any]] | None = None
    breakdown_overrides: dict[str, dict[str, Any]] | None = None
    setup_overrides: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> "SimulationConfig":
        allowed = {field.name for field in fields(self)}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        return replace(self, **overrides)

    def dataset_path(self) -> Path:
        return Path(self.dataset_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)
