"""A loader for small real multilayer networks stored in the dataset."""

import pandas as pd
import network_diffusion as nd
import networkx as nx

from src.loaders.constants import (
    MLN_RAW_DATA_PATH,
    AUCS,
    CKM_PHYSICIANS,
    EU_TRANSPORTATION,
    EU_TRANSPORT_KLM,
    FMRI74,
    L2_COURSE_NET_1,
    L2_COURSE_NET_2,
    L2_COURSE_NET_3,
    LAZEGA,
    TOY_NETWORK,
    WILDCARD_ALL,
    return_some_layers
)
from src.loaders.fmri74 import read_fmri74


def _network_from_pandas(path: str) -> nd.MultilayerNetwork:
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in df["layer"].unique()}
    for _, row in df.iterrows():
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_dict.keys()), network_list=list(net_dict.values())
    )


def get_aucs_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_real/aucs.mpx")


def get_ckm_physicians_network() -> nd.MultilayerNetwork:
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/CKM-Physicians-Innovation_4NoNature.edges"
    )


@return_some_layers
def get_eu_transportation_network() -> nd.MultilayerNetwork:
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/EUAirTransportation_multiplex_4NoNature.edges"
    )


def get_lazega_network() -> nd.MultilayerNetwork:
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )


def load_small_real(net_name: str) -> dict[str, nd.MultilayerNetwork]:
    loaded_nets = {}
    if net_name == AUCS or net_name == WILDCARD_ALL:
        loaded_nets[AUCS] = get_aucs_network()
    elif net_name == CKM_PHYSICIANS or net_name == WILDCARD_ALL:
        loaded_nets[CKM_PHYSICIANS] = get_ckm_physicians_network()
    elif net_name == EU_TRANSPORTATION or net_name == WILDCARD_ALL:
        loaded_nets[EU_TRANSPORTATION] = get_eu_transportation_network()
    elif net_name == EU_TRANSPORT_KLM or net_name == WILDCARD_ALL:
        loaded_nets[EU_TRANSPORT_KLM] = get_eu_transportation_network(["KLM"])
    elif net_name == FMRI74 or net_name == WILDCARD_ALL:
        loaded_nets[FMRI74] = read_fmri74(
            network_dir=f"{MLN_RAW_DATA_PATH}/CONTROL_fmt",
            binary=True,
            thresh=0.5,
        )
    elif net_name == L2_COURSE_NET_1 or net_name == WILDCARD_ALL:
        loaded_nets[L2_COURSE_NET_1] = nd.nets.get_l2_course_net(
            node_features=True,
            edge_features=True,
            directed=False,
        ).snaps[0]
    elif net_name == L2_COURSE_NET_2 or net_name == WILDCARD_ALL:
        loaded_nets[L2_COURSE_NET_2] = nd.nets.get_l2_course_net(
            node_features=True,
            edge_features=True,
            directed=False,
        ).snaps[1]
    elif net_name == L2_COURSE_NET_3 or net_name == WILDCARD_ALL:
        loaded_nets[L2_COURSE_NET_3] = nd.nets.get_l2_course_net(
            node_features=True,
            edge_features=True,
            directed=False,
        ).snaps[2]
    elif net_name == LAZEGA or net_name == WILDCARD_ALL:
        loaded_nets[LAZEGA] = get_lazega_network()
    elif net_name == TOY_NETWORK or net_name == WILDCARD_ALL:
        loaded_nets[TOY_NETWORK] = nd.nets.get_toy_network_piotr()
    return loaded_nets
