"""Evaluate how well the parameters of mABCD have been found for the given network."""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any
from scipy.stats import kstest

from mfdt.config_finder.cr_helpers import get_communities
from mfdt.params_handler import Network, load_networks, create_out_dir
from mfdt.config_finder.basic_finder import (
    get_edges_cor,
    get_partitions_cor,
    get_degrees_cor,
    _avg_partitions_noise,
)


def divergence_R_edges_correlation(original: Network, twin: Network, **kwargs) -> np.float64:
    """
    Calculate the divergence score for edges correlation matrices R of the original and twin networks.
    """
    edges_cor_mat_orig = get_edges_cor(original.n_graph_nx)
    edges_cor_mat_twin = get_edges_cor(twin.n_graph_nx)
    l = len(original.n_graph_nx.layers.keys())
    rss = ((edges_cor_mat_orig - edges_cor_mat_twin) ** 2).values.sum()
    return np.sqrt(rss / (l * (l - 1)))


def divergence_tau_degrees_correlation(original: Network, twin: Network, **kwargs) -> np.float64:
    """
    Calculate the divergence score for degree correlation matrices of the original and twin networks.
    """
    degrees_cor_mat_orig = get_degrees_cor(original.n_graph_nx)
    degrees_cor_mat_twin = get_degrees_cor(twin.n_graph_nx)
    l = len(original.n_graph_nx.layers.keys())
    rss = ((degrees_cor_mat_orig - degrees_cor_mat_twin) ** 2).values.sum()
    return np.sqrt(rss / (4 * l * (l - 1)))


def divergence_r_communities_correlation(
    original: Network,
    twin: Network,
    original_communities: dict[str, list[set[int]]],
    twin_communities: dict[str, list[set[int]]],
) -> np.float64:
    """
    Calculate the divergence score for communities correlation matrices of the original and twin networks.
    """
    edges_cor_mat_orig = get_partitions_cor(original.n_graph_nx, original_communities)
    edges_cor_mat_twin = get_partitions_cor(twin.n_graph_nx, twin_communities)
    l = len(original.n_graph_nx.layers.keys())
    rss = ((edges_cor_mat_orig - edges_cor_mat_twin) ** 2).values.sum()
    return np.sqrt(rss / (l * (l - 1)))


def divergence_gamma_degree_distribution(original: Network, twin: Network, **kwargs) -> np.float64:
    """
    Calculate the divergence score for degree distributions of the original and twin networks.
    """
    ks_distances = 0.0
    for l_name, l_graph in original.n_graph_nx.layers.items():
        twin_deg_seq = [d for _, d in twin.n_graph_nx.layers[l_name].degree()]
        orig_deg_seq = [d for _, d in l_graph.degree()]
        ks_distances += kstest(orig_deg_seq, twin_deg_seq).statistic
    l = len(original.n_graph_nx.layers.keys())
    return ks_distances / l


def divergence_beta_community_sizes_distribution(
    original: Network,
    twin: Network,
    original_communities: dict[str, list[set[int]]],
    twin_communities: dict[str, list[set[int]]],
) -> np.float64:
    """
    Calculate the divergence score for community size distributions of the original and twin networks.
    """
    ks_distances = 0.0
    for l_name, _ in original.n_graph_nx.layers.items():
        twin_com_sizes_seq = [len(com) for com in twin_communities[l_name]]
        orig_com_sizes_seq = [len(com) for com in original_communities[l_name]]
        ks_distances += kstest(orig_com_sizes_seq, twin_com_sizes_seq).statistic
    l = len(twin.n_graph_nx.layers.keys())
    return ks_distances / l


def divergence_xi_intercommunity_noise(
    original: Network,
    twin: Network,
    original_communities: dict[str, list[set[int]]],
    twin_communities: dict[str, list[set[int]]],
) -> np.float64:
    """
    Calculate the divergence score for intercommunity noise of the original and twin networks.
    """
    xi_rss = 0.0
    for l_name, _ in original.n_graph_nx.layers.items():
        xi_original = _avg_partitions_noise(
            original.n_graph_nx.layers[l_name], original_communities[l_name]
        )
        twin_l_graph = twin.n_graph_nx.layers[l_name]
        xi_twin = _avg_partitions_noise(twin_l_graph, twin_communities[l_name])
        xi_rss += (xi_original - xi_twin) ** 2
    l = len(original.n_graph_nx.layers.keys())
    return np.sqrt(xi_rss / l)
