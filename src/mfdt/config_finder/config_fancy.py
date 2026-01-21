"""Fancy methods for inferring mABCD configuration parameters."""

import networkx as nx
import network_diffusion as nd
import pandas as pd

from mfdt.config_finder import correlations
from mfdt.config_finder.config_model import (
    get_beta_s_S_xi,
    get_edges_cor,
    get_gamma_delta_Delta,
    get_q, get_tau,
)
from mfdt.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig, BaseMLNConfig


def get_r_fancy(net: nd.MultilayerNetwork, seed: int | None = None) -> dict[str, float]:
    """
    Get correlations between partitions.
    
    Note, that due to impossibility to reverse the process of creating partitions by MLNABCD, this
    function only approximates the correlations as follows. It takes the first (alphabetically)
    layer of the network as a reference. Then it uses it to compute correlations with other layers.
    """
    net = net.to_multiplex()[0]
    layer_names = sorted(list(net.layers))

    ref_layer = net[layer_names[0]]
    ref_partitions = nx.community.louvain_communities(ref_layer, seed=seed)

    r = {}
    for l_name in layer_names:
        ami = correlations.partitions_correlation(
            graph_1=ref_layer,
            graph_2=net[l_name],
            graph_1_partitions=ref_partitions,
            seed=seed,
        )
        r[l_name] = ami
    
    return r


def get_layer_params_fancy(net: nd.MultilayerNetwork, seed: int | None = None) -> pd.DataFrame:
    """Infer layers' parameters used by MLNABCD for a given network."""
    q, gamma_delta_Delta, beta_s_S_xi = {}, {}, {}

    nb_actors = net.get_actors_num()
    for l_name, l_graph in net.layers.items():
        q[l_name] = get_q(l_graph, nb_actors)
        gamma_delta_Delta[l_name] = get_gamma_delta_Delta(l_graph)
        beta_s_S_xi[l_name] = get_beta_s_S_xi(l_graph)

    tau = get_tau(net, alpha=None)
    r = get_r_fancy(net, seed=seed)

    params_dict = {
        l_name: {
            **{"q": q[l_name]},
            **{"tau": tau[l_name]},
            **{"r": r[l_name]},
            **gamma_delta_Delta[l_name],
            **beta_s_S_xi[l_name],
        }
        for l_name in net.layers
    }
    params_df = pd.DataFrame(params_dict).T.sort_index()
    return params_df.round(3).replace(0.0, 0.001)


def estimate_config_fancy(net: nd.MultilayerNetwork) -> tuple[dict[str, int], BaseMLNConfig]:
    """Estimate configuration for given network in the barbarian way."""
    l_map = {l_name: l_idx for l_idx, l_name in enumerate(sorted(net.layers), 1)}
    n = net.get_actors_num()
    print("Fancy....")
    edges_cor = get_edges_cor(net=net)
    edges_cor = edges_cor.rename(l_map, axis=0)
    edges_cor = edges_cor.rename(l_map, axis=1)
    layers_par = get_layer_params_fancy(net=net)
    layers_par = layers_par.rename(l_map, axis=0)
    est_config = BaseMLNConfig(n=n, edges_cor=edges_cor, layer_params=layers_par)
    return l_map, est_config
