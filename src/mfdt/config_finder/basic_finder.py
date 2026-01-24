"""Functions to infer configuration parameters of the existing network."""

from typing import Any

import juliacall  # this is added to silent a warning raised by importing both torch an juliacall
import networkx as nx
import network_diffusion as nd
import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import kendalltau

from mfdt.config_finder import correlations, cr_helpers
from mfdt.mln_abcd.julia_wrapper import BaseMLNConfig


def get_q(net: nx.Graph, num_actors: int) -> float:
    """Get fraction of active nodes."""
    return len(net.nodes) / num_actors


def get_tau(net: nd.MultilayerNetwork, alpha: float | None = 0.05) -> dict[str, float]:
    """
    Get correlations between node labels and their degrees.

    Note, that due to multilaterance of the network, the routine first converts labels of nodes
    from the first (alphabetically) layer, in a way to maximise the correlation (a node with the
    maximal degree has assigned the highest ID). Then it uses these labels in computations of the
    correlations in remaining layers. In addition it considers only nodes with a positive degree.  
    """
    net = net.to_multiplex()[0]
    layer_names = sorted(list(net.layers))

    degree_sequence  = cr_helpers.get_degree_sequence(net).T
    degree_sequence = degree_sequence.sort_index().sort_values(by=layer_names[0], ascending=False)
    actors_map = {id: idx for idx, id in enumerate(list(degree_sequence.index)[::-1])}
    degree_sequence = degree_sequence.rename(index=actors_map)

    tau = {}
    for l_name in layer_names:
        l_ds = degree_sequence[l_name][degree_sequence[l_name] > 0]
        statistic, pvalue = kendalltau(
            x=l_ds.index.to_list(),
            y=l_ds.to_list(),
            nan_policy="raise",
            variant="b",
        )
        tau[l_name] = statistic.item() if (not alpha or pvalue < alpha) else 0.0
    
    return tau


def get_r(net: nd.MultilayerNetwork, seed: int | None = None) -> dict[str, float]:
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


def _fit_exponent_powerlaw(raw_data: list[int] | list[float]) -> float:
    results = powerlaw.Fit(raw_data, discrete=True, verbose=False)
    return results.alpha


def get_gamma_delta_Delta(net: nx.Graph, cap_estimates: bool = False) -> dict[str, float]:
    """Get powerlaw exponent and min/max degree for a given layer."""
    degrees = [d for _, d in net.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees) if not cap_estimates else max(min(degrees), 5)
    return {
        "gamma": _fit_exponent_powerlaw(degrees),
        "delta": min_degree / len(net.nodes),
        "Delta": max_degree / len(net.nodes),
    }


def _avg_partitions_noise(
    net: nx.Graph, partitions: list[set[Any]] | list[frozenset[Any]]
) -> float:
    """
    The noise is fraction of edges inside partitions to number of all edges in the graph.
    
    xi = 0 -> all communities are separated, xi = 1 -> no distinctive communities.
    """
    all_edges = len(net.edges)
    internal_edges = 0
    for partition in partitions:
        sub_net = net.subgraph(partition)
        internal_edges += len(sub_net.edges)
    return (all_edges - internal_edges) / all_edges


def get_beta_s_S_xi(net: nx.Graph, cap_estimates: bool = False) -> dict[str, float]:
    """Get powerlaw exponent and min/max community size for a given layer."""
    # partitions = nx.community.louvain_communities(net)
    partitions = nx.community.greedy_modularity_communities(net)
    partitions_sizes = [len(part) for part in partitions]
    min_ps = min(partitions_sizes) if not cap_estimates else max(min(partitions_sizes), 30)
    return {
        "beta": _fit_exponent_powerlaw(partitions_sizes),
        "s": min_ps / len(net.nodes),
        "S": max(partitions_sizes) / len(net.nodes),
        "xi": _avg_partitions_noise(net, partitions),
    }


def get_edges_cor(net: nd.MultilayerNetwork) -> pd.DataFrame:
    """Get correlation matrix for edges."""
    edges_cor_raw = []
    for la_name, lb_name in cr_helpers.prepare_layer_pairs(list(net.layers.keys())):
        aligned_layers = cr_helpers.align_layers(net, la_name, lb_name, "destructive")
        edges_stat = correlations.edges_r(aligned_layers[la_name], aligned_layers[lb_name])
        edges_cor_raw.append({(la_name, lb_name): edges_stat})
    edges_cor_df = cr_helpers.create_correlation_matrix(edges_cor_raw)
    return edges_cor_df.round(3).fillna(0.0)


def get_layer_params(net: nd.MultilayerNetwork, seed: int | None = None) -> pd.DataFrame:
    """Infer layers' parameters used by MLNABCD for a given network."""
    q, gamma_delta_Delta, beta_s_S_xi = {}, {}, {}

    nb_actors = net.get_actors_num()
    for l_name, l_graph in net.layers.items():
        q[l_name] = get_q(l_graph, nb_actors)
        gamma_delta_Delta[l_name] = get_gamma_delta_Delta(l_graph)
        beta_s_S_xi[l_name] = get_beta_s_S_xi(l_graph)

    tau = get_tau(net, alpha=None)
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


def estimate_config_rudimentarly(
    net: nd.MultilayerNetwork,
    seed: int | None = None,
) -> tuple[dict[str, int], BaseMLNConfig]:
    """Estimate configuration for given network in the barbarian way."""
    l_map = {l_name: l_idx for l_idx, l_name in enumerate(sorted(net.layers), 1)}
    n = net.get_actors_num()
    edges_cor = get_edges_cor(net=net)
    edges_cor = edges_cor.rename(l_map, axis=0)
    edges_cor = edges_cor.rename(l_map, axis=1)
    layers_par = get_layer_params(net=net, seed=seed)
    layers_par = layers_par.rename(l_map, axis=0)
    est_config = BaseMLNConfig(n=n, edges_cor=edges_cor, layer_params=layers_par)
    return l_map, est_config
