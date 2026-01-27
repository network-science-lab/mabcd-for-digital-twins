from typing import Any

import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score

from mfdt.config_finder.cr_helpers import get_communities


def partitions_correlation(
    graph_1: nx.Graph, 
    graph_2: nx.Graph,
    graph_1_partitions: list[set[Any]] | list[frozenset[Any]] | None = None,
    graph_2_partitions: list[set[Any]] | list[frozenset[Any]] | None = None,
    seed: int | None = 42,
) -> float:
    """Get AMI of communities detected for two graphs spanned on the same set of vertices."""
    if set(graph_1.nodes) != set(graph_2.nodes):
        raise ValueError("Graphs must have identical node sets.")

    # obtain partitions in each graph
    if not graph_1_partitions:
        graph_1_partitions = get_communities(graph_1, seed)

    if not graph_2_partitions:
        graph_2_partitions = get_communities(graph_2, seed)

    # create dict keyed by nodes' ids, valued by array with partitions they're assigned into
    nodes_partitions = {node: [] for node in graph_1.nodes}
    for community_label, community_set in enumerate(graph_1_partitions):
        for node in community_set:
            nodes_partitions[node].append(community_label)
    for community_label, community_set in enumerate(graph_2_partitions):
        for node in community_set:
            nodes_partitions[node].append(community_label)

    # convert into two tables of indices accepted by sklearn
    partition_1_idcs, partition_2_idcs = [], []
    for node, (partition_1_idx, partition_2_idx) in nodes_partitions.items():
        partition_1_idcs.append(partition_1_idx)
        partition_2_idcs.append(partition_2_idx)

    # compute AMI and return it
    return float(adjusted_mutual_info_score(partition_1_idcs, partition_2_idcs))


def edges_r(graph_1: nx.Graph, graph_2: nx.Graph) -> float | None:
    g1_edges = set(graph_1.edges)
    g2_edges = set(graph_2.edges)
    if min(len(g1_edges), len(g2_edges)) == 0:
           return None
    return len(g1_edges.intersection(g2_edges)) / min(len(g1_edges), len(g2_edges))
