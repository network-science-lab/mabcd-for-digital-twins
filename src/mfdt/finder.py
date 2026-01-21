"""Infer configuration model for real networks."""

import json
from typing import Any
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from mfdt.params_handler import Network, load_networks, create_out_dir
from mfdt.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig, BaseMLNConfig
from mfdt.config_finder.config_model import get_edges_cor, get_layer_params


def estimate_config(network: Network, target_dir: Path) -> None:
    """Estimate configuration for given network and save it as a yaml file."""

    out_dir = create_out_dir(target_dir / network.n_type)

    l_map = {l_name: l_idx for l_idx, l_name in enumerate(sorted(network.n_graph_nx.layers), 1)}
    json.dump(l_map, open(out_dir / f"{network.n_name}_lmap.json", "w", encoding="utf-8"))

    n = network.n_graph_nx.get_actors_num()

    # infer edges' correlation matrix
    edges_cor = get_edges_cor(net=network.n_graph_nx)
    edges_cor = edges_cor.rename(l_map, axis=0)
    edges_cor = edges_cor.rename(l_map, axis=1)

    # infer layers' parameters
    layers_par = get_layer_params(net=network.n_graph_nx)
    layers_par = layers_par.rename(l_map, axis=0)

    # debugging
    # edges_cor_path = out_dir / f"{network.n_name}_edges.csv"
    # edges_cor.to_csv(edges_cor_path)
    # layers_par_path = out_dir / f"{network.n_name}_layers.csv"
    # layers_par.to_csv(layers_par_path, index=False)

    # save estimated config
    est_config = BaseMLNConfig(n=n, edges_cor=edges_cor, layer_params=layers_par)
    with open(out_dir / f"{network.n_name}_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(est_config.to_yaml(), f, sort_keys=False, indent=4)

    # test generation of a network from the inferred config
    # with open(out_dir / f"{network.n_name}_config.yaml", "r", encoding="utf-8") as f:
    #     dict_config = yaml.safe_load(f)
    # dict_config["seed"] = 42
    # dict_config["d_max_iter"] = 1000
    # dict_config["c_max_iter"] = 1000
    # dict_config["t"] = 100
    # dict_config["eps"] = 0.01
    # dict_config["d"] = 2
    # dict_config["edges_filename"] = str(out_dir / f"{network.n_name}-twin_edges.dat")
    # dict_config["communities_filename"] = str(out_dir / f"{network.n_name}-twin_communities.dat")
    # mln_config = MLNConfig.from_yaml(dict_config)
    # MLNABCDGraphGenerator()(config=mln_config)
    # warnings.warn("The approximation error is not implemented yet")


def run_experiments(config: dict[str, Any]) -> None:

    nets = load_networks(networks=config["networks"], device="cpu")
    out_dir = create_out_dir(config["finder"]["out_dir"])

    p_bar = tqdm(np.arange(len(nets)), desc="", leave=False, colour="green")
    for net_idx in p_bar:
        net = nets[net_idx]
        p_bar.set_description_str(net.rich_name)
        estimate_config(network=net, target_dir=out_dir)

