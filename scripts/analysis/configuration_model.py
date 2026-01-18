"""Infer configuration model for the real networks."""

import tempfile
import warnings
from pathlib import Path

from src.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig
from src.mln_abcd.config_finder import config_model
from src.loaders.net_loader import load_network
from src.utils import set_rng_seed


NETWORKS = [
    # ("bigreal", "arxiv_netscience_coauthorship"),
    ("smallreal", "aucs"),
    # ("bigreal", "cannes"),  # TODO: compute it on the server - it's too heavy
    # ("smallreal", "ckm_physicians"),
    # ("smallreal", "eu_transportation"),  # TODO: too big errors for this network!
    # ("smallreal", "l2_course_net_1"),
    ("smallreal", "lazega"),
    # ("bigreal", "timik1q2009"),
    ("smallreal", "toy_network"),
]

RNG_SEED = 42

OUT_DIR = Path(__file__).parent.parent.parent / "data/nets_properties/configuration_model2"


if __name__ == "__main__":

    set_rng_seed(seed=RNG_SEED)
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    for (net_type, net_name) in NETWORKS:
        print(net_type, net_name)
        ref_net = load_network(net_type, net_name)[(net_type, net_name)]
        layers_mapping = {l_name: l_idx for l_idx, l_name in enumerate(sorted(ref_net.layers), 1)}

        # infer edges' correlation matrix
        edges_cor_path = OUT_DIR / f"{net_name}_edges.csv"
        edges_cor = config_model.get_edges_cor(net=ref_net)
        print(edges_cor)
        edges_cor = edges_cor.rename(layers_mapping, axis=0)
        edges_cor = edges_cor.rename(layers_mapping, axis=1)
        edges_cor.to_csv(edges_cor_path)

        # infer layers' parameters
        layers_par_path = OUT_DIR / f"{net_name}_layers.csv"
        layers_par = config_model.get_layer_params(net=ref_net)
        print(layers_par)
        layers_par = layers_par.rename(layers_mapping, axis=0)
        layers_par.to_csv(layers_par_path, index=False)

        # test if they can be used
        n = ref_net.get_actors_num()
        edges_cor=MLNConfig.get_edges_cor(str(edges_cor_path))
        layer_params=MLNConfig.get_layer_params(n, str(layers_par_path))
        with tempfile.TemporaryDirectory() as temp_dir:
            mln_config = MLNConfig(
                seed=RNG_SEED,
                n=n,
                edges_cor=edges_cor,
                layer_params=layer_params,
                d_max_iter=1000,
                c_max_iter=1000,
                t=100,
                eps=0.01,
                d=2,
                edges_filename=f"{temp_dir}/edges.dat",
                communities_filename=f"{temp_dir}/communities.dat",
            )
            MLNABCDGraphGenerator()(config=mln_config)

            # compute approximation error
            warnings.warn("The approximation error is not implemented yet") # TODO: address it!
