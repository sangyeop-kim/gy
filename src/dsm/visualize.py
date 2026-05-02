"""Binary DSM and clustered DSM visualization."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.dsm.clustering import ClusteringResult


def _select_view_nodes(binary_matrix: pd.DataFrame,
                       ordered_nodes: list[str],
                       top_n: int | None) -> list[str]:
    if top_n is None or top_n >= len(ordered_nodes):
        return list(ordered_nodes)
    strength = binary_matrix.sum(axis=0) + binary_matrix.sum(axis=1)
    keep = set(strength.sort_values(ascending=False).head(top_n).index)
    return [n for n in ordered_nodes if n in keep]


def plot_clustered_dsm(binary_matrix: pd.DataFrame,
                       result: ClusteringResult,
                       title: str | None = None,
                       top_n: int | None = None,
                       ax=None):
    """Plot binary DSM reordered by cluster, with cluster boundary boxes and labels."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    label_by_node = dict(zip(binary_matrix.index, result.labels))
    view_nodes = _select_view_nodes(binary_matrix, result.ordered_nodes, top_n)
    sub = binary_matrix.loc[view_nodes, view_nodes]
    view_labels = [label_by_node[n] for n in view_nodes]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(sub.values, aspect="auto",
              cmap=mcolors.ListedColormap(["white", "black"]),
              vmin=0, vmax=1)

    n = len(view_nodes)
    starts = [0]
    for k in range(1, n):
        if view_labels[k] != view_labels[k - 1]:
            starts.append(k)
    starts.append(n)
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(starts) - 1)))
    for idx in range(len(starts) - 1):
        s, e = starts[idx], starts[idx + 1]
        rect = plt.Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                             fill=False, edgecolor=palette[idx], linewidth=2.0)
        ax.add_patch(rect)
        ax.text(s - 0.3, s - 0.3, f"C{view_labels[s]}",
                fontsize=7, color=palette[idx], fontweight="bold",
                va="bottom", ha="left")

    title = title or f"{result.method} (k={result.n_clusters}, Q={result.modularity:.3f})"
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(n))
    ax.set_xticklabels(view_nodes, rotation=90, fontsize=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(view_nodes, fontsize=4)
    ax.set_xlabel("destination toolgroup")
    ax.set_ylabel("source toolgroup")
    return ax


def save_clustered_dsm(binary_matrix: pd.DataFrame,
                       result: ClusteringResult,
                       save_path: str,
                       title: str | None = None,
                       top_n: int | None = None):
    """Render a single clustered DSM and save to disk."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_clustered_dsm(binary_matrix, result, title=title, top_n=top_n, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
