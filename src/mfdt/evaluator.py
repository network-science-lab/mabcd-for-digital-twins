"""Evaulate how well the parameters of mABCD have been found for the given network."""

from typing import Any

from mfdt.params_handler import load_networks, create_out_dir


def run_experiments(config: dict[str, Any]) -> None:

    original_network = load_networks(networks=[config["network"]], device="cpu")[0]
    twins = load_networks(networks=config["twins"], device="cpu")
    out_dir = create_out_dir(config["evaluator"]["out_dir"])
    # rng_seed = config["run"]["rng_seed"]

    print("Starting evaluation of the estimated configuration...")
    for t in twins:
        print(t.n_type, t.n_name, t.n_graph_nx.get_actors_num(), t.n_graph_nx.get_layer_names())
    print(f"Estimated configs saved.")
