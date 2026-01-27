"""Evaulate how well the parameters of mABCD have been found for the given network."""

import json
from typing import Any

from mfdt.config_finder.cr_helpers import get_communities
from mfdt.params_handler import Network, load_networks, create_out_dir


def compute_error(
    original_network: Network,
    original_communities: dict[str, list[set[int]]],
    twin_network: Network,
    divergencies: list[str],
) -> dict[str, Any]:
    """Estimate configuration for given network."""
    # TODO: Lukasz - it's your part, below it's just dummy code
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
    print([len(lc) for ln, lc in original_communities.items()])
    return {"": None}


def get_original_network(on_path: str, lm_path: str) -> Network:
    """Load MLN as usual, but rename its layers to match mABCD twins."""
    original_network = load_networks(networks=[on_path], device="cpu")[0]
    with open(lm_path, "r", encoding="utf-8") as f:
        layer_map = json.load(f)
    original_network.n_graph_nx.layers =  {
        layer_map[ln]: lg for ln, lg in original_network.n_graph_nx.layers.items()
    }
    return original_network


def get_original_communities(on: Network) -> dict[str, list[set[Any]]]:
    """Cluster the original network to use resulting partitions in evaluation."""
    return {l_name: get_communities(l_graph) for l_name, l_graph in on.n_graph_nx.layers.items()}


def run_experiments(config: dict[str, Any]) -> None:

    out_dir = create_out_dir(config["evaluator"]["out_dir"])
    rng_seed = config["run"]["rng_seed"]
    divergencies = config["evaluator"]["divergencies"]
    original_network = get_original_network(config["original_network"], config["layer_map"])
    twin_networks = load_networks(networks=config["twin_networks"], device="cpu")
    original_communities = get_original_communities(original_network)

    print("Starting evaluation of the estimated configuration...")
    t_errors = {}
    for twin in twin_networks:
        t_error = compute_error(
            original_network=original_network,
            original_communities=original_communities,
            twin_network=twin,
            divergencies=divergencies,
        )
        t_error[twin.n_name] = t_error
    # create dataframe from this dict
    # add an entry with averaged errors over all the twins
    # save errors into the output directory
    print(f"Estimated configs saved.")
