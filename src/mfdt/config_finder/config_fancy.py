"""Fancy methods for inferring mABCD configuration parameters."""

import tempfile
from pathlib import Path
from typing import Callable

import network_diffusion as nd
import numpy as np
import pandas as pd
import skopt

from mfdt.config_finder import correlations, helpers
from mfdt.config_finder.config_model import (
    get_beta_s_S_xi,
    get_edges_cor,
    get_gamma_delta_Delta,
    get_q,
    get_tau,
)
from mfdt.config_finder.figures import plot_optimisation_process
from mfdt.loaders.net_loader import _prepare_network
from mfdt.mln_abcd.julia_reader import load_edgelist
from mfdt.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig, BaseMLNConfig


def get_comm_ami(net: nd.MultilayerNetwork, seed: int | None = None) -> np.ndarray:
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
    part_cor_df = helpers.create_correlation_matrix(part_cor_raw)  # .round(3).fillna(0.0)
    return part_cor_df.to_numpy()


def frobenius_norm(comm_ami: np.ndarray) -> float:
    """Get Frobenius norm from the inter-layer community 'correlation' matrix."""
    return np.linalg.norm(comm_ami, ord="fro").item()


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
            **{"r": None},
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


def prepare_objective(
    fixed_mabcd_params: BaseMLNConfig,
    A: np.ndarray,
    r_space: list[skopt.space.Real],
    nb_twins: int = 5,
    rng_seed: int | None = None,
) -> Callable:
    """
    Prepare the objective function to estimate r for given real network.

    :param fixed_mabcd_params: found parameters of mABCD that are going to be fixed 
    :param A: found interlayer community 'correlation' matrix
    :param r_space: decision variables with their ranges (i.e., r)
    :param nb_twins: numer of times to create candidate twin network to reduce noise
    :param rng_seed: seed set of RNG
    :return: objective function to be optimised
    """
    fixed_dict = fixed_mabcd_params.to_yaml()
    fixed_dict["d_max_iter"] = 1000
    fixed_dict["c_max_iter"] = 1000
    fixed_dict["t"] = 100
    fixed_dict["eps"] = 0.05
    fixed_dict["d"] = 2
    seed_generator = np.random.default_rng(seed=rng_seed)

    @skopt.utils.use_named_args(dimensions=r_space)
    def objective(**r_dict) -> float:

        # prepare fully fledged candidate configuraiton to evaluate
        candidate_config = fixed_dict.copy()
        candidate_config["layer_params"]["r"] = [r_dict[f"r_{i}"] for i in range(len(A))]
        candidate_config["seed"] = int(seed_generator.random() * 279)

        # repreat twinning process n times to reduce noise
        A_primes = []
        for _ in range(nb_twins):

            # create a twin network (no short way to get rid of this tempdir)
            with tempfile.TemporaryDirectory() as tmpdir:
                fixed_dict["edges_filename"] = str(Path(tmpdir) / "_edges.dat")
                fixed_dict["communities_filename"] = str(Path(tmpdir) / "_communities.dat")
                mln_config = MLNConfig.from_yaml(fixed_dict)
                MLNABCDGraphGenerator()(config=mln_config)
                twin = _prepare_network(load_edgelist(Path(tmpdir) / "_edges.dat"))

            # compute correlation matrix and append it to the sample
            A_prime_n = get_comm_ami(net=twin, seed=rng_seed)
            A_primes.append(A_prime_n)

        # average the sample, compute distance from the real network and return it as a loss
        A_prime = np.mean(A_primes, axis=0)
        loss = np.abs(frobenius_norm(A_prime) - frobenius_norm(A))
        print(loss, r_dict)
        return loss
    
    return objective


def estimate_config_fancy(
    net: nd.MultilayerNetwork,
    seed: int | None = None,
) -> tuple[dict[str, int], BaseMLNConfig]:
    """
    Estimate configuration for given network using optimisation mechanisms.

    - get all parameters but `r` and `tau`
    - get community matrix: `A` (a surrogate for these values we can estimate)
    - for number of optimisation steps:
        - generate twin
        - get community matrix: `A'`
        - compute distance between `A` and `A'`
        - according to the distance update `r` and `tau`
    - preserve `r` and `tau` that preserves the smallest distance
    """
    l_map, fixed_mabcd_params = estimate_all_but_r(net)
    A = get_comm_ami(net, seed)
    r_space = [
        skopt.space.Real(0.0, 1.0, name=f"r_{i}")
        for i in range(A.shape[0])
    ]
    objective = prepare_objective(fixed_mabcd_params, A, r_space, rng_seed=seed)
    result = skopt.gp_minimize(
        func=objective,
        dimensions=r_space,
        n_calls=20,
        noise="gaussian",
        random_state=seed,
        n_jobs=5,
    )
    plot_optimisation_process(result, Path("optim_2.png"))
    fixed_mabcd_params.layer_params["r"] = result.x
    return l_map, fixed_mabcd_params
