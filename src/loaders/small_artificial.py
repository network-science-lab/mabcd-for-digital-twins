"""A loader for small artificial multilayer networks stored in the dataset."""

import network_diffusion as nd

from src.loaders.constants import (
    MLN_RAW_DATA_PATH,
    ER1,
    ER2,
    ER3,
    ER5,
    SF1,
    SF2,
    SF3,
    SF5,
    WILDCARD_ALL,
    return_some_layers
)


def get_er2_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/er_2.mpx")


def get_er3_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/er_3.mpx")


@return_some_layers
def get_er5_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/er_5.mpx")


def get_sf2_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/sf_2.mpx")


def get_sf3_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/sf_3.mpx")


@return_some_layers
def get_sf5_network() -> nd.MultilayerNetwork:
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/sf_5.mpx")


def load_small_artificial(net_name: str) -> dict[str, nd.MultilayerNetwork]:
    loaded_nets = {}
    if net_name == ER1 or net_name == WILDCARD_ALL:
        loaded_nets[ER1] = get_er5_network(["l2"])
    elif net_name == ER2 or net_name == WILDCARD_ALL:
        loaded_nets[ER2] = get_er2_network()
    elif net_name == ER3 or net_name == WILDCARD_ALL:
        loaded_nets[ER3] = get_er3_network()
    elif net_name == ER5 or net_name == WILDCARD_ALL:
        loaded_nets[ER5] = get_er5_network()
    elif net_name == SF1 or net_name == WILDCARD_ALL:
        loaded_nets[SF1] = get_sf5_network(["l3"])
    elif net_name == SF2 or net_name == WILDCARD_ALL:
        loaded_nets[SF2] = get_sf2_network()
    elif net_name == SF3 or net_name == WILDCARD_ALL:
        loaded_nets[SF3] = get_sf3_network()
    elif net_name == SF5 or net_name == WILDCARD_ALL:
        loaded_nets[SF5] = get_sf5_network()
    return loaded_nets
