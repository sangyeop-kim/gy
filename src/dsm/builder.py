"""Build DSM matrices (raw / normalized / binary) from a route model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.dsm.config import DsmConfig, EdgesConfig, MatrixConfig, WeightsConfig
from src.simulator import SimulationConfig, load_model


def compute_product_weights(weights_cfg: WeightsConfig,
                            dataset_dir: Path,
                            release_file: str) -> dict[str, float] | None:
    """Return product_name -> weight mapping. None means uniform (1.0 for all).

    method:
      - 'uniform'        : None
      - 'release_count'  : count of release rows per product
      - 'wafer_count'    : sum of WAFERS PER LOT (* LOTS PER RELEASE if column exists)
      - 'wspw'           : wafers-per-week (requires RELEASE INTERVAL column,
                           falls back to wafer_count otherwise)

    `weights_cfg.custom`, if set, takes precedence and is normalized to mean 1.0.
    """
    if weights_cfg.custom:
        vals = pd.Series(weights_cfg.custom, dtype=float)
        return (vals / vals.mean()).to_dict()

    if weights_cfg.method == "uniform":
        return None

    release_path = Path(weights_cfg.release_path) if weights_cfg.release_path else dataset_dir / release_file
    rel = pd.read_csv(release_path, low_memory=False)

    def _wafer_sum() -> pd.Series:
        wafers = rel["WAFERS PER LOT"].astype(float)
        if "LOTS PER RELEASE" in rel.columns:
            wafers = wafers * rel["LOTS PER RELEASE"].astype(float)
        return wafers.groupby(rel["PRODUCT NAME"]).sum()

    if weights_cfg.method == "release_count":
        agg = rel.groupby("PRODUCT NAME").size().astype(float)
    elif weights_cfg.method == "wafer_count":
        agg = _wafer_sum()
    elif weights_cfg.method == "wspw":
        if "RELEASE INTERVAL" not in rel.columns:
            agg = _wafer_sum()
        else:
            unit_to_min = {"min": 1.0, "hour": 60.0, "day": 1440.0, "week": 10080.0}
            interval_min = (rel["RELEASE INTERVAL"].astype(float)
                            * rel["R UNITS"].str.lower().map(unit_to_min))
            wspw = (rel["WAFERS PER LOT"].astype(float)
                    * rel["LOTS PER RELEASE"].astype(float)
                    * (10080.0 / interval_min))
            agg = wspw.groupby(rel["PRODUCT NAME"]).sum()
    else:
        raise ValueError(f"unknown weights.method: {weights_cfg.method}")

    return (agg / agg.mean()).to_dict()


def build_sequence_edges(model, edges_cfg: EdgesConfig,
                         product_weights: dict[str, float] | None) -> pd.DataFrame:
    rows = []
    for route in model.routes.values():
        pw = 1.0 if product_weights is None else product_weights.get(route.product_name, 1.0)
        steps = list(route.steps)
        for i, src in enumerate(steps):
            for lag in range(1, edges_cfg.window + 1):
                j = i + lag
                if j >= len(steps):
                    break
                dst = steps[j]
                if not edges_cfg.include_self_edges and src.toolgroup == dst.toolgroup:
                    continue
                if edges_cfg.exclude_same_area and src.area == dst.area:
                    continue
                rows.append({
                    "edge_type": "sequence",
                    "route": route.name,
                    "product_name": route.product_name,
                    "src_toolgroup": src.toolgroup,
                    "dst_toolgroup": dst.toolgroup,
                    "src_area": src.area,
                    "dst_area": dst.area,
                    "src_step_number": src.step_number,
                    "dst_step_number": dst.step_number,
                    "lag": lag,
                    "edge_weight": pw * edges_cfg.sequence_factor * (edges_cfg.decay ** (lag - 1)),
                })
    return pd.DataFrame(rows)


def build_constraint_edges(model, edges_cfg: EdgesConfig,
                           product_weights: dict[str, float] | None) -> pd.DataFrame:
    rows = []
    for route in model.routes.values():
        pw = 1.0 if product_weights is None else product_weights.get(route.product_name, 1.0)
        by_step = {step.step_number: step for step in route.steps}
        for src in route.steps:
            if src.cqt_start_step is not None and src.cqt_start_step in by_step:
                dst = by_step[src.cqt_start_step]
                if (src.toolgroup != dst.toolgroup
                        and not (edges_cfg.exclude_same_area and src.area == dst.area)):
                    rows.append({
                        "edge_type": "cqt",
                        "route": route.name,
                        "product_name": route.product_name,
                        "src_toolgroup": src.toolgroup,
                        "dst_toolgroup": dst.toolgroup,
                        "src_area": src.area,
                        "dst_area": dst.area,
                        "src_step_number": src.step_number,
                        "dst_step_number": dst.step_number,
                        "lag": None,
                        "edge_weight": pw * edges_cfg.cqt_boost,
                    })
            if src.rework_probability is not None and src.step_for_rework is not None:
                dst = by_step.get(src.step_for_rework)
                if (dst is not None and src.toolgroup != dst.toolgroup
                        and not (edges_cfg.exclude_same_area and src.area == dst.area)):
                    rows.append({
                        "edge_type": "rework",
                        "route": route.name,
                        "product_name": route.product_name,
                        "src_toolgroup": src.toolgroup,
                        "dst_toolgroup": dst.toolgroup,
                        "src_area": src.area,
                        "dst_area": dst.area,
                        "src_step_number": src.step_number,
                        "dst_step_number": dst.step_number,
                        "lag": None,
                        "edge_weight": pw * edges_cfg.rework_boost * (src.rework_probability / 100.0),
                    })
    return pd.DataFrame(rows)


def aggregate_edges(all_edges: pd.DataFrame) -> pd.DataFrame:
    """Per (src, dst) pair: total weight + per-type breakdown."""
    if all_edges.empty:
        return pd.DataFrame(columns=["src_toolgroup", "dst_toolgroup", "edge_weight"])

    def _sum_of_type(s, t):
        return s[all_edges.loc[s.index, "edge_type"].eq(t)].sum()

    return (
        all_edges.groupby(["src_toolgroup", "dst_toolgroup"])
        .agg(
            edge_weight=("edge_weight", "sum"),
            sequence_weight=("edge_weight", lambda s: _sum_of_type(s, "sequence")),
            cqt_weight=("edge_weight", lambda s: _sum_of_type(s, "cqt")),
            rework_weight=("edge_weight", lambda s: _sum_of_type(s, "rework")),
            edge_count=("edge_weight", "count"),
            n_products=("route", "nunique"),
        )
        .reset_index()
        .sort_values("edge_weight", ascending=False)
    )


def edges_to_matrix(edge_summary: pd.DataFrame) -> pd.DataFrame:
    if edge_summary.empty:
        return pd.DataFrame()
    matrix = edge_summary.pivot_table(
        index="src_toolgroup", columns="dst_toolgroup",
        values="edge_weight", aggfunc="sum", fill_value=0.0,
    )
    nodes = sorted(set(matrix.index) | set(matrix.columns))
    return matrix.reindex(index=nodes, columns=nodes, fill_value=0.0)


def normalize_matrix(matrix: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    if method != "minmax":
        raise ValueError(f"unsupported normalize method: {method}")
    vmin, vmax = matrix.values.min(), matrix.values.max()
    if vmax > vmin:
        return (matrix - vmin) / (vmax - vmin)
    return matrix.copy()


def make_binary(norm_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return (norm_matrix >= threshold).astype(int)


@dataclass
class DsmArtifacts:
    edges: pd.DataFrame              # full edge list (per route step)
    edge_summary: pd.DataFrame       # aggregated per (src, dst)
    raw: pd.DataFrame                # raw weight matrix
    normalized: pd.DataFrame         # min-max normalized [0, 1]
    binary: pd.DataFrame             # threshold-applied 0/1 matrix
    product_weights: dict[str, float] | None
    threshold: float


def build_dsm(config: DsmConfig) -> DsmArtifacts:
    """End-to-end DSM construction from a DsmConfig."""
    sim_cfg = SimulationConfig.from_json(config.simulation_config).with_overrides(max_lots=1)
    model = load_model(sim_cfg)

    product_weights = compute_product_weights(
        config.weights, Path(sim_cfg.dataset_dir), sim_cfg.release_file,
    )

    seq = build_sequence_edges(model, config.edges, product_weights)
    con = build_constraint_edges(model, config.edges, product_weights)
    all_edges = pd.concat([seq, con], ignore_index=True) if not con.empty else seq
    summary = aggregate_edges(all_edges)
    raw = edges_to_matrix(summary)
    norm = normalize_matrix(raw, method=config.matrix.normalize)
    binary = make_binary(norm, config.matrix.threshold)
    return DsmArtifacts(
        edges=all_edges,
        edge_summary=summary,
        raw=raw,
        normalized=norm,
        binary=binary,
        product_weights=product_weights,
        threshold=config.matrix.threshold,
    )
