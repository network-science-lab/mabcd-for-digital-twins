"""Main runner of the generator."""

from typing import Any

import juliacall
import numpy as np
import yaml
from tqdm import tqdm

from mfdt.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig
from mfdt.params_handler import create_out_dir


def read_mln_config_from_params(mln_config: dict[str, Any]) -> MLNConfig:
    with open(mln_config["params_path"], "r", encoding="utf-8") as f:
        dict_config = yaml.safe_load(f)
    for key, value in mln_config.items():
        if key != "params_path" and value is not None:
            dict_config[key] = value
    return MLNConfig.from_yaml(dict_config)


def run_experiments(config: dict[str, Any]) -> None:
    _mln_config = config["mln_config"]
    _mln_config["seed"] = config["run"]["rng_seed"]
    if _mln_config.get("params_path") is None:
        mln_config = MLNConfig.from_yaml(_mln_config)
    else:
        mln_config = read_mln_config_from_params(_mln_config)

    repetitions = config["generator"]["repetitions"]
    out_dir = create_out_dir(config["generator"]["out_dir"])
    e_name, e_stem = config["mln_config"]["edges_filename"].split(".")
    c_name, c_stem = config["mln_config"]["communities_filename"].split(".")

    p_bar = tqdm(np.arange(repetitions), desc="", leave=False, colour="green")
    for repetition in p_bar:
        p_bar.set_description_str("Repetition")
        mln_config.edges_filename = str(out_dir / f"{e_name}_{repetition}.{e_stem}")
        mln_config.communities_filename = str(out_dir / f"{c_name}_{repetition}.{c_stem}")
        MLNABCDGraphGenerator()(config=mln_config)
