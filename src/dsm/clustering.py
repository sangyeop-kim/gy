"""Directed-aware clustering methods for DSM analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.dsm.config import ClusteringConfig, LouvainParams, MclParams


@dataclass
class ClusteringResult:
    method: str
    ordered_nodes: list[str]
    labels: np.ndarray  # cluster id per node, in original matrix index order
    n_clusters: int
    modularity: float


def _node_features(weight_matrix: pd.DataFrame) -> np.ndarray:
    feats = np.hstack([weight_matrix.values, weight_matrix.values.T])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms


def cluster_hierarchical(weight_matrix: pd.DataFrame,
                         n_clusters: int = 8,
                         method: str = "ward") -> tuple[list[str], np.ndarray]:
    """Hierarchical clustering on (in+out) cosine distance."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import pdist

    nodes = list(weight_matrix.index)
    feats = _node_features(weight_matrix)
    Z = linkage(pdist(feats, metric="cosine"), method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    leaf_order = leaves_list(Z)
    ordered = [nodes[i] for i in leaf_order]
    return ordered, labels


def cluster_mcl(weight_matrix: pd.DataFrame,
                params: MclParams = MclParams()) -> tuple[list[str], np.ndarray]:
    """Markov clustering on column-stochastic transition matrix (directed flow)."""
    nodes = list(weight_matrix.index)
    M = weight_matrix.values.astype(float).copy()
    np.fill_diagonal(M, M.diagonal() + params.self_loop)

    col_sum = M.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    M = M / col_sum

    for _ in range(params.max_iter):
        M_prev = M
        M = np.linalg.matrix_power(M, params.expansion)
        M = np.power(M, params.inflation)
        col_sum = M.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        M = M / col_sum
        if np.abs(M - M_prev).sum() < params.tol:
            break

    attractors = np.where(M.sum(axis=1) > 1e-6)[0]
    cluster_of = -np.ones(len(nodes), dtype=int)
    for cid, attr in enumerate(attractors, start=1):
        for member in np.where(M[attr] > 1e-6)[0]:
            if cluster_of[member] == -1:
                cluster_of[member] = cid
    next_id = (cluster_of.max() if cluster_of.max() > 0 else 0) + 1
    for i in range(len(nodes)):
        if cluster_of[i] == -1:
            cluster_of[i] = next_id
            next_id += 1

    unique = sorted(set(cluster_of.tolist()))
    remap = {old: new for new, old in enumerate(unique, start=1)}
    labels = np.array([remap[c] for c in cluster_of])

    order = sorted(range(len(nodes)), key=lambda i: (labels[i], nodes[i]))
    ordered = [nodes[i] for i in order]
    return ordered, labels


def cluster_directed_louvain(binary_matrix: pd.DataFrame,
                             params: LouvainParams = LouvainParams()) -> tuple[list[str], np.ndarray]:
    """Directed Louvain (Leicht-Newman directed modularity via networkx)."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    nodes = list(binary_matrix.index)
    bm = binary_matrix.values
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j and bm[i, j]:
                G.add_edge(nodes[i], nodes[j])
    communities = louvain_communities(G, resolution=params.resolution, seed=params.seed)
    label_by_node = {n: cid for cid, comm in enumerate(communities, start=1) for n in comm}
    labels = np.array([label_by_node[n] for n in nodes])
    order = sorted(range(len(nodes)), key=lambda i: (labels[i], nodes[i]))
    ordered = [nodes[i] for i in order]
    return ordered, labels


def modularity_score(binary_matrix: pd.DataFrame,
                     labels: np.ndarray,
                     directed: bool = True) -> float:
    """Directed (Leicht-Newman) or undirected modularity of a labeling."""
    import networkx as nx
    from networkx.algorithms.community import modularity

    nodes = list(binary_matrix.index)
    bm = binary_matrix.values
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    if directed:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j and bm[i, j]:
                    G.add_edge(nodes[i], nodes[j])
    else:
        sym = ((bm + bm.T) > 0).astype(int)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sym[i, j]:
                    G.add_edge(nodes[i], nodes[j])
    if G.number_of_edges() == 0:
        return float("nan")
    label_by_node = dict(zip(nodes, labels))
    groups: dict[int, set[str]] = {}
    for n, label in label_by_node.items():
        groups.setdefault(label, set()).add(n)
    return modularity(G, list(groups.values()))


def run_clustering(normalized: pd.DataFrame,
                   binary: pd.DataFrame,
                   cfg: ClusteringConfig) -> ClusteringResult:
    """Run the single clustering method named in cfg.method."""
    if cfg.method == "hierarchical":
        ordered, labels = cluster_hierarchical(
            normalized, n_clusters=cfg.n_clusters, method=cfg.hierarchical_linkage,
        )
    elif cfg.method == "mcl":
        ordered, labels = cluster_mcl(normalized, params=cfg.mcl)
    elif cfg.method == "directed_louvain":
        ordered, labels = cluster_directed_louvain(binary, params=cfg.louvain)
    else:
        raise ValueError(f"unknown clustering method: {cfg.method}")
    q = modularity_score(binary, labels, directed=True)
    return ClusteringResult(
        method=cfg.method, ordered_nodes=ordered, labels=labels,
        n_clusters=int(len(set(labels))), modularity=float(q),
    )
