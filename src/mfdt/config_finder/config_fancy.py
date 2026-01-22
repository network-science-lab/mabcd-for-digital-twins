"""Fancy methods for inferring mABCD configuration parameters."""

import networkx as nx
import network_diffusion as nd
import numpy as np
import pandas as pd

from mfdt.config_finder import correlations, helpers
from mfdt.config_finder.config_model import (
    get_beta_s_S_xi,
    get_edges_cor,
    get_gamma_delta_Delta,
    get_q,
    get_tau,
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


def frobenius_norm(comm_ami: pd.DataFrame) -> float:
    """Get Frobenius norm from the inter-layer community 'correlation' matrix."""
    ca_arr = comm_ami.to_numpy()
    return np.linalg.norm(ca_arr, ord="fro").item()


def estimate_all_but_r(net: nd.MultilayerNetwork) -> tuple[dict[str, int], BaseMLNConfig]:
    """Estimate configuration for given network in the barbarian way."""
    l_map = {l_name: l_idx for l_idx, l_name in enumerate(sorted(net.layers), 1)}
    n = net.get_actors_num()
    edges_cor = get_edges_cor(net=net)

    q, gamma_delta_Delta, beta_s_S_xi = {}, {}, {}
    for l_name, l_graph in net.layers.items():
        q[l_name] = get_q(l_graph, n)
        gamma_delta_Delta[l_name] = get_gamma_delta_Delta(l_graph)
        beta_s_S_xi[l_name] = get_beta_s_S_xi(l_graph)

    tau = get_tau(net, alpha=None)

    params_dict = {
        l_name: {
            **{"q": q[l_name]},
            **{"tau": tau[l_name]},
            **{"r": [0.5] * len(net.layers)},
            **gamma_delta_Delta[l_name],
            **beta_s_S_xi[l_name],
        }
        for l_name in net.layers
    }
    params_df = pd.DataFrame(params_dict).T.sort_index()
    layers_par = params_df.round(3).replace(0.0, 0.001)

    edges_cor = edges_cor.rename(l_map, axis=0)
    edges_cor = edges_cor.rename(l_map, axis=1)
    layers_par = layers_par.rename(l_map, axis=0)
    est_config = BaseMLNConfig(n=n, edges_cor=edges_cor, layer_params=layers_par)

    return l_map, est_config


"""
get all parameters but r  # OK
get community matrix: A  # OK
foolishly guess r  # OK

for number of optimisation steps:
    generate twin
    get community matrix: A'
    compute distance between A and A'
    according to the distance update r

return r value that preserves the smallest distance
"""


def estimate_config_fancy(
    net: nd.MultilayerNetwork,
    seed: int | None = None,
) -> tuple[dict[str, int], BaseMLNConfig]:
    """Estimate configuration for given network in the barbarian way."""
    return estimate_all_but_r(net)
