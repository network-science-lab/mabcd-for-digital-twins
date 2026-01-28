"""Correlations of various metrics between layers."""

from typing import Any

import networkx as nx
import network_diffusion as nd
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import adjusted_mutual_info_score

from mfdt.correlations import cr_helpers


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
        graph_1_partitions = cr_helpers.get_communities(graph_1, seed)

    if not graph_2_partitions:
        graph_2_partitions = cr_helpers.get_communities(graph_2, seed)

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


def get_partitions_cor(net: nd.MultilayerNetwork, partitions: dict[str, list[set]]) -> pd.DataFrame:
    """Get correlation (AMI) matrix for partitions."""
    partitions_cor_raw = []
    for la_name, lb_name in cr_helpers.prepare_layer_pairs(list(net.layers.keys())):
        aligned_layers = cr_helpers.align_layers(net, la_name, lb_name, "destructive")
        partitions_stat = partitions_correlation(
            aligned_layers[la_name],
            aligned_layers[lb_name],
            graph_1_partitions=partitions[la_name],
            graph_2_partitions=partitions[lb_name],
            seed=42,
        )
        partitions_cor_raw.append({(la_name, lb_name): partitions_stat})
    partitions_cor_df = cr_helpers.create_correlation_matrix(partitions_cor_raw)
    return partitions_cor_df.round(3).fillna(0.0)


def edges_r(graph_1: nx.Graph, graph_2: nx.Graph) -> float | None:
    g1_edges = set(graph_1.edges)
    g2_edges = set(graph_2.edges)
    if min(len(g1_edges), len(g2_edges)) == 0:
        return None
    return len(g1_edges.intersection(g2_edges)) / min(len(g1_edges), len(g2_edges))


def get_edges_cor(net: nd.MultilayerNetwork) -> pd.DataFrame:
    """Get correlation matrix for edges (R)."""
    edges_cor_raw = []
    for la_name, lb_name in cr_helpers.prepare_layer_pairs(list(net.layers.keys())):
        aligned_layers = cr_helpers.align_layers(net, la_name, lb_name, "destructive")
        edges_stat = edges_r(aligned_layers[la_name], aligned_layers[lb_name])
        edges_cor_raw.append({(la_name, lb_name): edges_stat})
    edges_cor_df = cr_helpers.create_correlation_matrix(edges_cor_raw)
    return edges_cor_df.round(3).fillna(0.0)


def degrees_correlation(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    nodes_to_labels: dict[Any, int],
) -> float:
    """Get kendall tau correlation of degrees for two graphs spanned on the same set of vertices."""
    if set(graph_1.nodes) != set(graph_2.nodes):
        raise ValueError("Graphs must have identical node sets.")

    # obtain ranked degree sequences
    ranked_deg_seq_1 = cr_helpers._degree_seq_ordered_by_labels(graph_1, nodes_to_labels)
    ranked_deg_seq_2 = cr_helpers._degree_seq_ordered_by_labels(graph_2, nodes_to_labels)

    # compute kendall tau correlation and return it
    return float(
        kendalltau(
            ranked_deg_seq_1,
            ranked_deg_seq_2,
            nan_policy="raise",
            variant="b",
        ).correlation  # FIXME!
    )


def get_degrees_cor(net: nd.MultilayerNetwork) -> pd.DataFrame:
    """Get correlation (Kendall tau) matrix for degrees."""
    nodes_to_labels = cr_helpers._label_nodes_by_total_degree(net)
    degrees_cor_raw = []
    for la_name, lb_name in cr_helpers.prepare_layer_pairs(list(net.layers.keys())):
        aligned_layers = cr_helpers.align_layers(net, la_name, lb_name, "destructive")
        degrees_stat = degrees_correlation(
            aligned_layers[la_name], aligned_layers[lb_name], nodes_to_labels
        )
        degrees_cor_raw.append({(la_name, lb_name): degrees_stat})
    degrees_cor_df = cr_helpers.create_correlation_matrix(degrees_cor_raw)
    return degrees_cor_df.round(3).fillna(0.0)
