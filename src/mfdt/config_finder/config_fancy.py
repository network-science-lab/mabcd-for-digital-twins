"""Fancy methods for inferring mABCD configuration parameters."""

import networkx as nx
import network_diffusion as nd
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from mfdt.config_finder import correlations, helpers
from mfdt.config_finder.config_model import (
    get_beta_s_S_xi,
    get_edges_cor,
    get_gamma_delta_Delta,
    get_q,
    get_tau,
    get_r,
)
from mfdt.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig, BaseMLNConfig


def get_comm_ami(net: nd.MultilayerNetwork, seed: int | None = None) -> pd.DataFrame:
    """Get interlatyer 'correlations' (i.e. AMI) between partitions."""
    net = net.to_multiplex()[0]
    part_cor_raw = []
    for la_name, lb_name in helpers.prepare_layer_pairs(list(net.layers.keys())):
        aligned_layers = helpers.align_layers(net, la_name, lb_name, "destructive")
        part_ami = correlations.partitions_correlation(
            aligned_layers[la_name],
            aligned_layers[lb_name], 
            seed=seed,
        )
        part_cor_raw.append({(la_name, lb_name): part_ami})
    part_cor_df = helpers.create_correlation_matrix(part_cor_raw)
    return part_cor_df  # .round(3).fillna(0.0)


def get_r_fancy(net: nd.MultilayerNetwork, seed: int | None = None) -> dict[str, float]:
    """Get correlations between partitions."""
    comm_ami = get_comm_ami(net, seed)
    print(comm_ami)

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
    get_r_fancy(net, seed=seed)
    r = get_r(net, seed=seed)

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
