"""A loader for big real multilayer networks stored in the dataset."""

from pathlib import Path

import pandas as pd
import network_diffusion as nd
import networkx as nx

from src.loaders.constants import (
    MLN_RAW_DATA_PATH,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    ARXIV_NETSCIENCE_COAUTHORSHIP_MATH,
    CANNES,
    TIMIK1Q2009,
    WILDCARD_ALL,
    return_some_layers
)


def get_ddm_network(
    layernames_path: Path,
    edgelist_path: Path,
    weighted: bool,
    digraph: bool,
) -> nd.MultilayerNetwork:
    # read mapping of layer IDs to their names
    with open(layernames_path, encoding="utf-8") as file:
        layer_names = file.readlines()
    layer_names = [ln.rstrip('\n').split(" ") for ln in layer_names]
    layer_names = {ln[0]: ln[1] for ln in layer_names}
    
    # read the edgelist and create containers for the layers
    df = pd.read_csv(
        edgelist_path,
        names=["layer_id", "node_1", "node_2", "weight"],
        sep=" "
    )
    net_ids_dict = {
        l_name: nx.DiGraph() if digraph else nx.Graph()
        for l_name in list(df["layer_id"].unique())
    }

    # populate network with edges
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        if weighted:
            attrs = {"weight": row["weight"]}
        else:
            attrs = {}
        net_ids_dict[row["layer_id"]].add_edge(row["node_1"], row["node_2"], **attrs)
    
    # rename layers
    net_names_dict = {
        layer_names[str(layer_id)]: layer_graph
        for layer_id, layer_graph in net_ids_dict.items()
    }

    # create multilater network from edges
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_names_dict.keys()), network_list=list(net_names_dict.values())
    )


@return_some_layers
def get_arxiv_network() -> nd.MultilayerNetwork:
    root_path = Path(f"{MLN_RAW_DATA_PATH}/arxiv_netscience_coauthorship/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "arxiv_netscience_layers.txt",
        edgelist_path=root_path / "arxiv_netscience_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_cannes_network() -> nd.MultilayerNetwork:
    root_path = Path(f"{MLN_RAW_DATA_PATH}/cannes_2013_social/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "Cannes2013_layers.txt",
        edgelist_path=root_path / "Cannes2013_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_timik1q2009_network() -> nd.MultilayerNetwork:
    layer_graphs = []
    layer_names = []
    for i in Path(f"{MLN_RAW_DATA_PATH}/timik1q2009").glob("*.csv"):
        layer_names.append(i.stem)
        layer_graphs.append(nx.from_pandas_edgelist(pd.read_csv(i)))
    return nd.MultilayerNetwork.from_nx_layers(layer_graphs, layer_names)


def load_big_real(net_name: str) -> dict[str, nd.MultilayerNetwork]:
    loaded_nets = {}
    if net_name == ARXIV_NETSCIENCE_COAUTHORSHIP or net_name == WILDCARD_ALL:
        loaded_nets[ARXIV_NETSCIENCE_COAUTHORSHIP] = get_arxiv_network()
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH or net_name == WILDCARD_ALL:
        loaded_nets[ARXIV_NETSCIENCE_COAUTHORSHIP_MATH] = get_arxiv_network(["math.OC"])
    elif net_name == CANNES or net_name == WILDCARD_ALL:
        loaded_nets[CANNES] = get_cannes_network()
    elif net_name == TIMIK1Q2009 or net_name == WILDCARD_ALL:
        loaded_nets[TIMIK1Q2009] = get_timik1q2009_network()
    return loaded_nets
