"""Infer configuration model for the real networks."""

import warnings
from pathlib import Path

import yaml

from src.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig, BaseMLNConfig
from src.mln_abcd.config_finder import config_model
from src.loaders.net_loader import load_network
from src.utils import set_rng_seed
import json


NETWORKS = [
    # ("bigreal", "arxiv_netscience_coauthorship"),
    ("smallreal", "aucs"),
    # ("bigreal", "cannes"),  # TODO: compute it on the server - it's too heavy
    # ("smallreal", "ckm_physicians"),
    # ("smallreal", "eu_transportation"),  # TODO: too big errors for this network!
    # ("smallreal", "l2_course_net_1"),
    # ("smallreal", "lazega"),
    # ("bigreal", "timik1q2009"),
    # ("smallreal", "toy_network"),
]

RNG_SEED = 42

OUT_DIR = Path(__file__).parent.parent / "data/nets_properties/configuration_model2"


def main(net_type: str, net_name: str, out_dir: Path) -> None:

    ref_net = load_network(net_type, net_name)[(net_type, net_name)]
    layers_mapping = {l_name: l_idx for l_idx, l_name in enumerate(sorted(ref_net.layers), 1)}
    json.dump(layers_mapping, open(out_dir / f"{net_name}_lmap.json", "w"))

    n = ref_net.get_actors_num()

    # infer edges' correlation matrix
    edges_cor = config_model.get_edges_cor(net=ref_net)
    edges_cor = edges_cor.rename(layers_mapping, axis=0)
    edges_cor = edges_cor.rename(layers_mapping, axis=1)

    # infer layers' parameters
    layers_par = config_model.get_layer_params(net=ref_net)
    layers_par = layers_par.rename(layers_mapping, axis=0)

    # debugging
    # edges_cor_path = out_dir / f"{net_name}_edges.csv"
    # edges_cor.to_csv(edges_cor_path)
    # layers_par_path = out_dir / f"{net_name}_layers.csv"
    # layers_par.to_csv(layers_par_path, index=False)

    # save estimated config
    est_config = BaseMLNConfig(n=n, edges_cor=edges_cor, layer_params=layers_par)
    with open(out_dir / f"{net_name}_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(est_config.to_yaml(), f, sort_keys=False, indent=4)

    # test generation of a network from the inferred config
    with open(out_dir / f"{net_name}_config.yaml", "r", encoding="utf-8") as f:
        dict_config = yaml.safe_load(f)
    dict_config["seed"] = RNG_SEED
    dict_config["d_max_iter"] = 1000
    dict_config["c_max_iter"] = 1000
    dict_config["t"] = 100
    dict_config["eps"] = 0.01
    dict_config["d"] = 2
    dict_config["edges_filename"] = str(out_dir / f"{net_name}-twin_edges.dat")
    dict_config["communities_filename"] = str(out_dir / f"{net_name}-twin_communities.dat")
    mln_config = MLNConfig.from_yaml(dict_config)
    MLNABCDGraphGenerator()(config=mln_config)
    warnings.warn("The approximation error is not implemented yet") # TODO: address it!


if __name__ == "__main__":

    set_rng_seed(seed=RNG_SEED)
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    for (net_type, net_name) in NETWORKS:
        print(net_type, net_name)
        main(net_type, net_name, OUT_DIR)
