from copy import deepcopy
from itertools import combinations
from typing import Literal

import networkx as nx
import network_diffusion as nd
import numpy as np

import pandas as pd


def align_layers(
    net: nd.MultilayerNetwork,
    l1_name: str,
    l2_name: str,
    method: Literal["destructive", "additive"],
) -> dict[str, nx.Graph]:
    """
    Align set of nodes in the given two layers of a multilayer network.

    :param net: _description_
    :param l1_name: _description_
    :param l2_name: _description_
    :param method: there are two options:
        `additive` - target set of nodes is union of nodes in two layers
        `destructive` - target set of nodes is intersection of nodes in two layers
    :return: _description_
    """
    l1 = deepcopy(net[l1_name])
    l1_nodes = set(l1.nodes)

    l2 = deepcopy(net[l2_name])
    l2_nodes = set(l2.nodes)

    if method == "additive":
        correct_nodes = l1_nodes.union(l2_nodes)
    elif method == "destructive":
        correct_nodes = l1_nodes.intersection(l2_nodes)
    else:
        raise ValueError("Unknown alignment method!")

    l1.remove_nodes_from(l1_nodes.difference(correct_nodes))
    l1.add_nodes_from(correct_nodes.difference(l1_nodes))

    l2.remove_nodes_from(l2_nodes.difference(correct_nodes))
    l2.add_nodes_from(correct_nodes.difference(l2_nodes))

    return {l1_name: l1, l2_name: l2}


def get_degree_sequence(net: nd.MultilayerNetwork) -> pd.DataFrame:
    net_degrees = {}
    for l_name, l_graph in net.layers.items():
        net_degrees[l_name] = dict(l_graph.degree())
    return pd.DataFrame(net_degrees).T


def prepare_layer_pairs(entities: list[str]) -> list[tuple[str, str]]:
    return list(combinations(entities, 2))


def _get_col_names(raw_statistics: list[dict[tuple[str, str], float]]) -> list[str]:
    """Get names of compared entities in `raw_statistics`."""
    l_names = []
    for record in raw_statistics:
        for (l1_name, l2_name), _ in record.items():
            l_names.append(l1_name)
            l_names.append(l2_name)
    return list(set(l_names))


def create_correlation_matrix(raw_statistics: list[dict[tuple[str, str], float]]) -> pd.DataFrame:
    """Create correlation matrix that can be plotted as a heatmap."""
    col_names = _get_col_names(raw_statistics)
    matrix = pd.DataFrame(index=sorted(col_names), columns=sorted(col_names), data=np.nan)
    for record in raw_statistics:
        for (la_name, lb_name), statistic in record.items():
            matrix.loc[la_name, lb_name] = statistic
            matrix.loc[lb_name, la_name] = statistic
    for l_name in col_names:
        matrix.loc[l_name, l_name] = 1.0
    return matrix
