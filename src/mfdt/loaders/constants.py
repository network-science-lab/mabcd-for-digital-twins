"""A script with strings used across module in form of variables."""

# TODO: revise this file once the dataset is decided upon

from functools import wraps
from pathlib import Path
from typing import Callable

import network_diffusion as nd

MLN_RAW_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/networks"

# network names
ARXIV_NETSCIENCE_COAUTHORSHIP = "arxiv_netscience_coauthorship"
ARXIV_NETSCIENCE_COAUTHORSHIP_MATH = "arxiv_netscience_coauthorship_math.oc"
AUCS = "aucs"
CANNES = "cannes"
CKM_PHYSICIANS = "ckm_physicians"
EU_TRANSPORTATION = "eu_transportation"
EU_TRANSPORT_KLM = "eu_transport_klm"
ER1 = "er1"
ER2 = "er2"
ER3 = "er3"
ER5 = "er5"
FMRI74 = "fmri74"
L2_COURSE_NET_1 = "l2_course_net_1"
L2_COURSE_NET_2 = "l2_course_net_2"
L2_COURSE_NET_3 = "l2_course_net_3"
LAZEGA = "lazega"
SF1 = "sf1"
SF2 = "sf2"
SF3 = "sf3"
SF5 = "sf5"
TIMIK1Q2009 = "timik1q2009"
TOY_NETWORK = "toy_network"
FREEBASE = "Freebase"

MLNABCD_PREFIX = "mlnabcd"

SEPARATOR = "^"
WILDCARD_ALL = "*"


def return_some_layers(get_network_func: Callable) -> Callable:
    """Decorator for network loader to filter out a multilayer network to return only one layer."""

    @wraps(get_network_func)
    def wrapper(layer_slice=None):
        net = get_network_func()
        if not layer_slice:
            return net
        l_graphs = [net.layers[layer] for layer in layer_slice]
        return nd.MultilayerNetwork.from_nx_layers(l_graphs, layer_slice)

    return wrapper
