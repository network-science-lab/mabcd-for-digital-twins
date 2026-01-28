"""Main entry point to run experiments."""

# TODO: consider adding hydra
# TODO: consider adding logging
# TODO: in the camera-ready version, run black or other code formatter on the codebase

import argparse
import yaml
import juliacall

from network_diffusion.utils import set_rng_seed

from mfdt.generator import run_experiments as re_generator
from mfdt.finder import run_experiments as re_finder
from mfdt.evaluator import run_experiments as re_evaluator


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file (default: config.yaml).",
        nargs="?",
        type=str,
        # default="scripts/configs/example_generate_1.yaml",
        # default="scripts/configs/example_generate_2/runner_config.yaml",
        # default="scripts/configs/example_find.yaml",
        default="scripts/configs/example_evaluate.yaml",
    )
    return parser.parse_args(*args)


def main():
    """Main entrypoint for the code."""

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")

    if random_seed := config["run"].get("random_seed"):
        print(f"Setting randomness seed as {random_seed}!")
        set_rng_seed(config["run"]["random_seed"])

    if (experiment_type := config["run"].get("experiment_type")) == "generate":
        entrypoint = re_generator
    elif experiment_type == "find":
        entrypoint = re_finder
    elif experiment_type == "evaluate":
        entrypoint = re_evaluator
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")

    print(f"Inferred experiment type as: {experiment_type}")
    entrypoint(config)


if __name__ == "__main__":
    print("Use only for VS Code debugging purposes!")
    main()
