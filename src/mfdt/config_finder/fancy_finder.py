"""Fancy methods for inferring mABCD configuration parameters."""

import copy
from pathlib import Path
from typing import Callable

import network_diffusion as nd
import numpy as np
import skopt
import yaml

from mfdt.config_finder.ff_figures import plot_optimisation_process
from mfdt.config_finder.ff_helpers import (
    SerialOptimizeResult,
    convert_result,
    estimate_fixed_params,
    get_comm_ami,
    get_decision_space,
    get_stacked_A_element_variance,
    get_criterium,
    prepare_log_dir,
)
from mfdt.loaders.net_loader import _prepare_network
from mfdt.mln_abcd.julia_reader import load_edgelist
from mfdt.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig, BaseMLNConfig


def prepare_objective(
    fixed_mabcd_params: BaseMLNConfig,
    A: np.ndarray,
    decision_space: list[skopt.space.Real],
    criterium: Callable,
    nb_twins: int,
    d_max_iter: int,
    c_max_iter: int,
    t: int,
    eps: float,
    d: int,
    rng_seed: int | None = None,
    out_dir: Path | None = None,
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
    fixed_dict["d_max_iter"] = d_max_iter
    fixed_dict["c_max_iter"] = c_max_iter
    fixed_dict["t"] = t
    fixed_dict["eps"] = eps
    fixed_dict["d"] = d
    seed_generator = np.random.default_rng(seed=rng_seed)
    out_dir_server = prepare_log_dir(out_dir)

    @skopt.utils.use_named_args(dimensions=decision_space)
    def objective(**decision_vars) -> float:
        # prepare fully fledged candidate configuraiton to evaluate
        candidate_config = copy.deepcopy(fixed_dict)
        candidate_config["seed"] = int(seed_generator.random() * 279)
        candidate_config["edges_filename"] = "eval_edges.dat"
        candidate_config["communities_filename"] = "eval_communities.dat"

        # update evaluated values or r and tau depending on if the'yre fixed or decision vars
        r_list = [decision_vars.get(f"r_{i}") for i in range(len(A))]
        if None not in r_list:
            candidate_config["layer_params"]["r"] = r_list
        tau_list = [decision_vars.get(f"tau_{i}") for i in range(len(A))]
        if None not in tau_list:
            candidate_config["layer_params"]["tau"] = tau_list

        # repreat twinning process n times to reduce noise
        A_primes = []
        with out_dir_server() as tmpdir:
            # create config according to the twin will be generated
            mln_config = MLNConfig.from_yaml(candidate_config)
            with open(f"{tmpdir}/eval_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(mln_config.to_yaml(), f, sort_keys=False, indent=4)

            for rep in range(nb_twins):
                # replace out paths according to sample number
                rep_ef = Path(tmpdir) / f"{rep}_{candidate_config['edges_filename']}"
                rep_cf = Path(tmpdir) / f"{rep}_{candidate_config['communities_filename']}"
                mln_config.edges_filename = str(rep_ef)
                mln_config.communities_filename = str(rep_cf)

                # generate the network
                MLNABCDGraphGenerator()(config=mln_config)
                twin = _prepare_network(load_edgelist(rep_ef))

                # compute correlation matrix and append it to the sample
                A_prime_n = get_comm_ami(net=twin, seed=rng_seed)
                A_primes.append(A_prime_n)

            # save computed A matrices in the sample
            A_primes = np.array(A_primes)
            np.save(f"{tmpdir}/A_primes.npy", A_primes)

        # average the sample, compute distance from the real network and return it as a loss
        std_A_primes = get_stacked_A_element_variance(A_primes)
        A_prime = np.mean(A_primes, axis=0)
        loss = criterium(A, A_prime)
        print("loss: %.5f" % loss, "std_A': %.5f" % std_A_primes, "dv: ", decision_vars)
        return loss

    return objective


def estimate_config_fancy(
    net: nd.MultilayerNetwork,
    log_dir: Path,
    save_logs: bool,
    criterium: str,
    decision_variables: list[str],
    cap_fixed_params: bool,
    nb_twins: int,
    nb_steps: int,
    d_max_iter: int,
    c_max_iter: int,
    t: int,
    eps: float,
    d: int,
    seed: int | None = None,
) -> tuple[dict[str, str], BaseMLNConfig]:
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
    l_map, fixed_mabcd_params = estimate_fixed_params(
        net=net,
        do_r=True if "r" in decision_variables else False,
        do_tau=True if "tau" in decision_variables else False,
        cap_fixed_params=cap_fixed_params,
        seed=seed,
    )
    A = get_comm_ami(net, seed)
    decision_space = get_decision_space(decision_variables, A.shape[0])
    criterium_func = get_criterium(criterium)
    objective = prepare_objective(
        fixed_mabcd_params=fixed_mabcd_params,
        A=A,
        decision_space=decision_space,
        criterium=criterium_func,
        nb_twins=nb_twins,
        d_max_iter=d_max_iter,
        c_max_iter=c_max_iter,
        t=t,
        eps=eps,
        d=d,
        rng_seed=seed,
        out_dir=log_dir if save_logs else None,
    )

    result = skopt.gp_minimize(
        func=objective,
        dimensions=decision_space,
        n_calls=nb_steps,
        noise="gaussian",
        random_state=seed,
        n_jobs=1,
    )
    print(
        f"[BEST SOLUTION in {np.where(result.func_vals == result.fun)[0][0].item() + 1}th step] ",
        "loss: %.5f" % result.fun,
        "dv: ",
        {dv.name: x for (dv, x) in zip(decision_space, result.x)},
    )

    if log_dir:
        plot_optimisation_process(result, log_dir / "trajectory.png")
        np.save(f"{log_dir}/A.npy", A)
        with open(f"{log_dir}/optim.txt", "w", encoding="utf-8") as f:
            f.write(result.__str__())
        SerialOptimizeResult(
            fun=result.fun,
            x=result.x,
            func_vals=result.func_vals,
            x_iters=result.x_iters,
        ).dump(f"{log_dir}/optim.pkl")
    
    dv_opt = convert_result(decision_space, result)
    if "r" in decision_variables:
        fixed_mabcd_params.layer_params["r"] = dv_opt["r"]
    if "tau" in decision_variables:
        fixed_mabcd_params.layer_params["tau"] = dv_opt["tau"]

    return l_map, fixed_mabcd_params
