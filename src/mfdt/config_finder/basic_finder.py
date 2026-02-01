"""Functions to infer configuration parameters of the existing network."""

from typing import Any

import juliacall  # this is added to silent a warning raised by importing both torch an juliacall
import network_diffusion as nd
import networkx as nx
import pandas as pd
import powerlaw
from scipy.stats import kendalltau

from mfdt.correlations import correlations, cr_helpers
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

    degree_sequence = cr_helpers.get_degree_sequence(net).T
    degree_sequence["sum"] = degree_sequence.sum(axis=1)
    degree_sequence = degree_sequence.sort_index().sort_values(by="sum", ascending=False)
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
        tau[l_name] = statistic.item() if (not alpha or pvalue < alpha) else 0.0  # FIXME!

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
    ref_partitions = cr_helpers.get_communities(ref_layer, seed=seed)

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
    results = powerlaw.Fit(
        data=raw_data,
        discrete=True,
        verbose=False,
        # fit_method="KS",  # uncomment to use Kolmogorov-Smirnov test
        xmin_distribution="power_law",
    )
    return results.alpha


def get_gamma_delta_Delta(net: nx.Graph, cap_estimates: bool = False) -> dict[str, float]:
    """Get powerlaw exponent and min/max degree for a given layer."""
    degrees = [d for _, d in net.degree()]  # FIXME!
    max_degree = max(degrees)
    min_degree = min(degrees) if not cap_estimates else max(min(degrees), 5)
    return {
        "gamma": _fit_exponent_powerlaw(degrees),
        "delta": min_degree / len(net.nodes),
        "Delta": max_degree / len(net.nodes),
    }


def avg_partitions_noise(net: nx.Graph, partitions: list[set[Any]] | list[frozenset[Any]]) -> float:
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
    partitions = cr_helpers.get_communities(net)
    partitions_sizes = [len(part) for part in partitions]
    min_ps = min(partitions_sizes) if not cap_estimates else max(min(partitions_sizes), 10)
    return {
        "beta": _fit_exponent_powerlaw(partitions_sizes),
        "s": min_ps / len(net.nodes),
        "S": max(partitions_sizes) / len(net.nodes),
        "xi": avg_partitions_noise(net, partitions),
    }


def get_layer_params(net: nd.MultilayerNetwork, seed: int | None = None) -> pd.DataFrame:
    """Infer layers' parameters used by MLNABCD for a given network."""
    q, gamma_delta_Delta, beta_s_S_xi = {}, {}, {}

    nb_actors = net.get_actors_num()
    for l_name, l_graph in net.layers.items():
        q[l_name] = get_q(l_graph, nb_actors)
        gamma_delta_Delta[l_name] = get_gamma_delta_Delta(l_graph, cap_estimates=True)
        beta_s_S_xi[l_name] = get_beta_s_S_xi(l_graph, cap_estimates=True)

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
) -> tuple[dict[str, str], BaseMLNConfig]:
    """Estimate configuration for given network in the barbarian way."""
    l_map = {l_name: str(l_idx) for l_idx, l_name in enumerate(sorted(net.layers), 1)}
    n = net.get_actors_num()
    edges_cor = correlations.get_edges_cor(net=net)
    edges_cor = edges_cor.rename(l_map, axis=0)
    edges_cor = edges_cor.rename(l_map, axis=1)
    layers_par = get_layer_params(net=net, seed=seed)
    layers_par = layers_par.rename(l_map, axis=0)
    est_config = BaseMLNConfig(n=n, edges_cor=edges_cor, layer_params=layers_par)
    return l_map, est_config
