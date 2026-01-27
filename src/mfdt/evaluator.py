"""Evaluate how well the parameters of mABCD have been found for the given network."""

import json
import pandas as pd
import numpy as np
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


def divergence_R_edges_correlation(
    original: Network, twin: Network, **kwargs
) -> np.float64:
    """
    Calculate the divergence score for edges correlation matrices R of the original and twin networks.
    """
    edges_cor_mat_orig = get_edges_cor(original.n_graph_nx)
    edges_cor_mat_twin = get_edges_cor(twin.n_graph_nx)
    l = len(original.n_graph_nx.layers.keys())
    rss = ((edges_cor_mat_orig - edges_cor_mat_twin) ** 2).values.sum()
    return np.sqrt(rss / (l * (l - 1)))


def divergence_tau_degrees_correlation(
    original: Network, twin: Network, **kwargs
) -> np.float64:
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


def divergence_gamma_degree_distribution(
    original: Network, twin: Network, **kwargs
) -> np.float64:
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
    l = len(original.n_graph_nx.layers.keys())
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


divergencies_calculators = {
    "R_edges_correlation": divergence_R_edges_correlation,
    "tau_degrees_correlation": divergence_tau_degrees_correlation,
    "r_communities_correlation": divergence_r_communities_correlation,
    "gamma_degree_distribution": divergence_gamma_degree_distribution,
    "beta_community_sizes_distribution": divergence_beta_community_sizes_distribution,
    "xi_intercommunity_noise": divergence_xi_intercommunity_noise,
}


def compute_error(
    original_network: Network,
    original_communities: dict[str, list[set[int]]],
    twin_network: Network,
    twin_communities: dict[str, list[set[int]]],
    divergencies: list[str],
) -> dict[str, Any]:
    """Estimate configuration for given network."""
    print(
        original_network.n_type,
        original_network.n_name,
        original_network.n_graph_nx.get_actors_num(),
        original_network.n_graph_nx.get_layer_names(),
    )
    print(
        twin_network.n_type,
        twin_network.n_name,
        twin_network.n_graph_nx.get_actors_num(),
        twin_network.n_graph_nx.get_layer_names(),
    )
    # print([len(lc) for ln, lc in original_communities.items()])
    errors = {
        div: divergencies_calculators[div](
            original_network,
            twin_network,
            original_communities=original_communities,
            twin_communities=twin_communities,
        )
        for div in divergencies
    }
    return errors


def get_original_network(on_path: str, lm_path: str) -> Network:
    """Load MLN as usual, but rename its layers to match mABCD twins."""
    original_network = load_networks(networks=[on_path], device="cpu")[0]
    with open(lm_path, "r", encoding="utf-8") as f:
        layer_map = json.load(f)
    original_network.n_graph_nx.layers = {
        layer_map[ln]: lg for ln, lg in original_network.n_graph_nx.layers.items()
    }
    return original_network


def get_communities_all_layers(
    net: Network,
) -> dict[str, list[set[Any]] | list[frozenset[Any]]]:
    """Cluster the network to use resulting partitions in evaluation."""
    return {
        l_name: get_communities(l_graph)
        for l_name, l_graph in net.n_graph_nx.layers.items()
    }


def run_experiments(config: dict[str, Any]) -> None:

    out_dir = create_out_dir(config["evaluator"]["out_dir"])
    rng_seed = config["run"]["rng_seed"]
    divergencies = config["evaluator"]["divergencies"]
    original_network = get_original_network(
        config["original_network"], config["layer_map"]
    )
    twin_networks = load_networks(networks=config["twin_networks"], device="cpu")
    original_communities = get_communities_all_layers(original_network)

    print("Starting evaluation of the estimated configuration...")
    t_errors = {}
    for twin in twin_networks:
        twin_communities = get_communities_all_layers(twin)
        t_error = compute_error(
            original_network=original_network,
            original_communities=original_communities,
            twin_network=twin,
            twin_communities=twin_communities,
            divergencies=divergencies,
        )
        t_error[twin.n_name] = t_error
    # create dataframe from this dict
    df_errors = pd.DataFrame.from_dict(t_errors, orient="index")
    # add an entry with averaged errors over all the twins

    # save errors into the output directory

    print(f"Estimated configs saved.")
