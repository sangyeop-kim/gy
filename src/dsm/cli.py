"""CLI: build a DSM from a JSON config and write artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace as dc_replace
from pathlib import Path

import pandas as pd

from src.dsm.builder import build_dsm
from src.dsm.clustering import run_clustering
from src.dsm.config import DsmConfig
from src.dsm.visualize import save_clustered_dsm


def _fmt_num(x: float) -> str:
    """Compact stringification for folder names: 0.05 -> '0.05', 3.0 -> '3', 1.2 -> '1.2'."""
    if float(x).is_integer():
        return str(int(x))
    return f"{x:g}"


def hyperparam_subdir(config: DsmConfig) -> str:
    """Build a deterministic subdirectory name from key hyperparameters.

    Format:
        {method}__w-{weights}__t{thr}__seq{sf}_cqt{cqt}_rwk{rwk}__win{w}_dec{dec}{__no-area}
    """
    e = config.edges
    parts = [
        config.clustering.method,
        f"w-{config.weights.method}",
        f"t{_fmt_num(config.matrix.threshold)}",
        f"seq{_fmt_num(e.sequence_factor)}_cqt{_fmt_num(e.cqt_boost)}_rwk{_fmt_num(e.rework_boost)}",
        f"win{e.window}_dec{_fmt_num(e.decay)}",
    ]
    if e.exclude_same_area:
        parts.append("no-area")
    return "__".join(parts)


def _save_artifacts(artifacts, result, config: DsmConfig, output_dir: Path) -> list[Path]:
    """Write the minimal set of files needed for downstream rule construction:

    - dsm_network.csv      : binary directed edges (src, dst). "what connects to what"
    - dsm_clusters.csv     : toolgroup -> cluster id. "what is grouped with what"
    - dsm_config_used.json : full hyperparameters used (reproducibility / folder identity)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    bm = artifacts.binary
    nodes = list(bm.index)
    edges = [(nodes[i], nodes[j])
             for i in range(len(nodes)) for j in range(len(nodes))
             if i != j and bm.values[i, j]]
    network_csv = output_dir / "dsm_network.csv"
    pd.DataFrame(edges, columns=["src", "dst"]).to_csv(network_csv, index=False)
    paths.append(network_csv)

    cluster_csv = output_dir / "dsm_clusters.csv"
    pd.DataFrame({
        "toolgroup": nodes,
        "cluster": result.labels,
    }).to_csv(cluster_csv, index=False)
    paths.append(cluster_csv)

    config_echo = output_dir / "dsm_config_used.json"
    config_dict = config.as_dict()
    config_dict["_run_summary"] = {
        "method": result.method,
        "n_clusters": result.n_clusters,
        "directed_modularity": result.modularity,
        "binary_edges": len(edges),
        "n_nodes": len(nodes),
    }
    config_echo.write_text(json.dumps(config_dict, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    paths.append(config_echo)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DSM (raw / normalized / binary) and run clustering. "
                    "All --override flags fall back to the JSON config value if not given; "
                    "the JSON config in turn falls back to the dataclass defaults shown here. "
                    "If --output-dir is omitted, a hyperparameter-encoded subdirectory is "
                    "auto-created under the config's output_dir."
    )
    parser.add_argument("--config", default="configs/default_dsm.json",
                        help="path to DSM JSON config (default: configs/default_dsm.json)")
    parser.add_argument("--output-dir", default=None,
                        help="explicit output dir; if omitted, "
                             "auto subdir under config.output_dir (default: outputs/dsm_analysis)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override matrix.threshold (default: 0.05)")
    parser.add_argument("--weights-method", default=None,
                        choices=["uniform", "release_count", "wafer_count", "wspw"],
                        help="override weights.method (default: uniform)")
    parser.add_argument("--method", default=None,
                        choices=["hierarchical", "mcl", "directed_louvain"],
                        help="override clustering.method (default: directed_louvain)")
    parser.add_argument("--sequence-factor", type=float, default=None,
                        help="override edges.sequence_factor (default: 1.0)")
    parser.add_argument("--cqt-boost", type=float, default=None,
                        help="override edges.cqt_boost (default: 3.0)")
    parser.add_argument("--rework-boost", type=float, default=None,
                        help="override edges.rework_boost; rework weight = "
                             "rework_boost * (rework_probability / 100). "
                             "0 to drop rework edges (default: 1.0)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="skip writing the PNG (default: visualize=True)")
    args = parser.parse_args()

    config = DsmConfig.from_json(args.config)

    # ---- overrides ----
    if args.threshold is not None:
        config = config.with_overrides(matrix=dc_replace(config.matrix, threshold=args.threshold))

    edge_overrides = {k: v for k, v in {
        "cqt_boost": args.cqt_boost,
        "rework_boost": args.rework_boost,
        "sequence_factor": args.sequence_factor,
    }.items() if v is not None}
    if edge_overrides:
        config = config.with_overrides(edges=dc_replace(config.edges, **edge_overrides))

    if args.weights_method:
        config = config.with_overrides(
            weights=dc_replace(config.weights, method=args.weights_method))
    if args.method:
        config = config.with_overrides(
            clustering=dc_replace(config.clustering, method=args.method))
    if args.no_visualize:
        config = config.with_overrides(visualize=False)

    # ---- resolve output dir ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.output_dir) / hyperparam_subdir(config)

    # ---- run ----
    print(f"[dsm] building DSM from {config.simulation_config}")
    artifacts = build_dsm(config)
    print(f"[dsm] matrix shape={artifacts.binary.shape}, "
          f"non-zero raw={int((artifacts.raw.values > 0).sum())}, "
          f"binary edges={int(artifacts.binary.values.sum())} "
          f"@ threshold={config.matrix.threshold}")

    print(f"[dsm] clustering: {config.clustering.method}")
    result = run_clustering(artifacts.normalized, artifacts.binary, config.clustering)
    print(f"  -> k={result.n_clusters}  dir-modularity={result.modularity:.3f}")

    saved = _save_artifacts(artifacts, result, config, output_dir)

    if config.visualize:
        png = output_dir / "dsm_clusters.png"
        save_clustered_dsm(
            artifacts.binary, result, save_path=str(png),
            title=f"{result.method}  (k={result.n_clusters}, Q={result.modularity:.3f})  "
                  f"threshold={config.matrix.threshold}",
            top_n=config.top_n_visual,
        )
        saved.append(png)

    print(f"[dsm] wrote {len(saved)} files to {output_dir}:")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
