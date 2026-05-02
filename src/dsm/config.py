"""Hyperparameters for DSM construction, clustering, and visualization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WeightsConfig:
    """Per-product weight assignment.

    method: 'uniform' | 'release_count' | 'wafer_count' | 'wspw'
    custom: optional dict {product_name: weight}; takes precedence when set.
    """

    method: str = "uniform"
    release_path: str | None = None
    custom: dict[str, float] | None = None


@dataclass(frozen=True)
class EdgesConfig:
    """Per-edge weight knobs.

    Each edge's weight = product_weight * decay^(lag-1) * type_factor, summed across
    products / step transitions. Type factors:
      - sequence : sequence_factor  (default 1.0)
      - cqt      : cqt_boost        (default 3.0)
      - rework   : rework_boost * (rework_probability / 100)  (default boost 1.0)

    Set rework_boost=0 to drop rework edges entirely; set sequence_factor=0 to keep
    only constraint edges.
    """

    window: int = 5
    decay: float = 0.35
    sequence_factor: float = 1.0
    cqt_boost: float = 3.0
    rework_boost: float = 1.0
    include_self_edges: bool = False
    exclude_same_area: bool = False


@dataclass(frozen=True)
class MatrixConfig:
    normalize: str = "minmax"  # only minmax for now
    threshold: float = 0.05    # binary cutoff on normalized weight


@dataclass(frozen=True)
class MclParams:
    expansion: int = 3
    inflation: float = 1.2
    self_loop: float = 5.0
    max_iter: int = 200
    tol: float = 1e-7


@dataclass(frozen=True)
class LouvainParams:
    resolution: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class ClusteringConfig:
    """Single clustering method per run.

    method: 'hierarchical' | 'mcl' | 'directed_louvain'
    """

    method: str = "directed_louvain"
    n_clusters: int = 8                     # used only by hierarchical
    hierarchical_linkage: str = "ward"
    mcl: MclParams = field(default_factory=MclParams)
    louvain: LouvainParams = field(default_factory=LouvainParams)


@dataclass(frozen=True)
class DsmConfig:
    """Top-level DSM configuration.

    `simulation_config` points at a SimulationConfig JSON used only to load the route model
    (max_lots is forced to 1 — release information is not used in DSM construction).
    """

    simulation_config: str = "configs/default_simulation.json"
    output_dir: str = "outputs/dsm_analysis"
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    edges: EdgesConfig = field(default_factory=EdgesConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualize: bool = True
    top_n_visual: int | None = None  # None = show all nodes

    @classmethod
    def from_json(cls, path: str | Path) -> DsmConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DsmConfig:
        weights = WeightsConfig(**data.get("weights", {}))
        edges = EdgesConfig(**data.get("edges", {}))
        matrix = MatrixConfig(**data.get("matrix", {}))

        clust_raw = dict(data.get("clustering", {}))
        if "mcl" in clust_raw and isinstance(clust_raw["mcl"], dict):
            clust_raw["mcl"] = MclParams(**clust_raw["mcl"])
        if "louvain" in clust_raw and isinstance(clust_raw["louvain"], dict):
            clust_raw["louvain"] = LouvainParams(**clust_raw["louvain"])
        clustering = ClusteringConfig(**clust_raw)

        top_level = {k: v for k, v in data.items()
                     if k not in {"weights", "edges", "matrix", "clustering"}}
        return cls(weights=weights, edges=edges, matrix=matrix,
                   clustering=clustering, **top_level)

    def with_overrides(self, **overrides: Any) -> DsmConfig:
        return replace(self, **overrides)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
