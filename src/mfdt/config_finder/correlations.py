from typing import Any

import networkx as nx
from scipy.stats import kendalltau
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
            if (
                node not in nodes_partitions
            ):  # if communities for graph before destructive alignment were provided
                continue
            nodes_partitions[node].append(community_label)
    for community_label, community_set in enumerate(graph_2_partitions):
        for node in community_set:
            if node not in nodes_partitions:
                continue
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


def _degree_seq_ordered_by_labels(
    graph: nx.Graph,
    nodes_to_labels: dict[Any, int],
) -> list[int]:
    """Get degree sequence ordered by nodes' labels."""
    labels_degree_seq = [(nodes_to_labels[n], d) for n, d in graph.degree()]
    return [d for _, d in sorted(labels_degree_seq, key=lambda x: x[0])]


def degrees_correlation(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    nodes_to_labels: dict[Any, int],
) -> float:
    """Get kendall tau correlation of degrees for two graphs spanned on the same set of vertices."""
    if set(graph_1.nodes) != set(graph_2.nodes):
        raise ValueError("Graphs must have identical node sets.")

    # obtain ranked degree sequences
    ranked_deg_seq_1 = _degree_seq_ordered_by_labels(graph_1, nodes_to_labels)
    ranked_deg_seq_2 = _degree_seq_ordered_by_labels(graph_2, nodes_to_labels)

    # compute kendall tau correlation and return it
    return float(
        kendalltau(
            ranked_deg_seq_1,
            ranked_deg_seq_2,
            nan_policy="raise",
            variant="b",
        ).correlation
    )
