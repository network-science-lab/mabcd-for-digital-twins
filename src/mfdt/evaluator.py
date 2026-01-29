"""Evaluate how well the parameters of mABCD have been found for the given network."""

import json
import pandas as pd
import numpy as np
from typing import Any

from mfdt.correlations.cr_helpers import get_communities
from mfdt.divergences import (
    divergence_R_edges_correlation,
    divergence_beta_community_sizes_distribution,
    divergence_gamma_degree_distribution,
    divergence_r_communities_correlation,
    divergence_tau_degrees_correlation,
    divergence_xi_intercommunity_noise,
)
from mfdt.params_handler import Network, load_networks, create_out_dir


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
    print(f"Evaluating twin network: {twin_network.n_name}...")
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
    rng_seed: int | None = None,
) -> dict[str, list[set[Any]] | list[frozenset[Any]]]:
    """Cluster the network to use resulting partitions in evaluation."""
    return {
        l_name: get_communities(l_graph, rng_seed)
        for l_name, l_graph in net.n_graph_nx.layers.items()
    }


def run_experiments(config: dict[str, Any]) -> None:
    out_dir = create_out_dir(config["evaluator"]["out_dir"])
    rng_seed = config["run"]["rng_seed"]
    divergencies = config["evaluator"]["divergencies"]
    original_network = get_original_network(config["original_network"], config["layer_map"])
    twin_networks = load_networks(networks=config["twin_networks"], device="cpu")
    original_communities = get_communities_all_layers(original_network, rng_seed=rng_seed)

    print("Starting evaluation of the estimated configuration...")
    t_errors = {}
    for twin in twin_networks:
        twin_communities = get_communities_all_layers(twin, rng_seed=rng_seed)
        t_error = compute_error(
            original_network=original_network,
            original_communities=original_communities,
            twin_network=twin,
            twin_communities=twin_communities,
            divergencies=divergencies,
        )
        t_errors[twin.n_name] = t_error
    # create dataframe from this dict
    df_errors = pd.DataFrame.from_dict(t_errors, orient="index")
    # add an entry with averaged errors over all the twins
    df_errors["mean_divergence"] = np.mean(df_errors, axis=1)
    mean_div = np.mean(df_errors, axis=0)
    std_div = np.std(df_errors, axis=0)
    df_errors.loc["Mean"] = mean_div
    df_errors.loc["Std"] = std_div
    # save errors into the output directory
    df_errors.to_csv(f"{out_dir}/divergence_scores.csv", index_label="graph")
    print(f"Estimated configs saved.")
