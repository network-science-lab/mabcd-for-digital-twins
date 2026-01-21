"""A Python wrapper to the MLNABCDGraphGenerator Julia package."""

import tempfile
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from juliacall import JuliaError
from juliacall import Main as jl


@dataclass
class BaseMLNConfig:
    """A base class to keep mABCD estimables."""

    n: int
    edges_cor: pd.DataFrame
    layer_params: pd.DataFrame

    def _lp_yaml_helper(self) -> dict[str, Any]:
        lp_dict = {}
        for param in self.layer_params.columns:
            lp_dict[param] = list(self.layer_params.to_dict()[param].values())
        return lp_dict

    def to_yaml(self) -> dict[str, Any]:
        """Convert configuration into serialisable format."""
        return {
            "seed": None,
            "n": self.n,
            "edges_cor": self.edges_cor.to_numpy().tolist(),
            "layer_params": self._lp_yaml_helper(),
            "d_max_iter": None,
            "c_max_iter": None,
            "t": None,
            "eps": None,
            "d": None,
            "edges_filename": None,
            "communities_filename": None,
        }


@dataclass
class MLNConfig(BaseMLNConfig):
    """
    A wrapper for jl.MLNABCDGraphGenerator.MLNConfig.

    Note that this wrapper fixes a discrepancy in the original Julia class where some parameters
    are normalized while others are not. If an object of this class is used along with
    MLNABCDGraphGenerator, it implicitly converts its parameters to meet Julia's format and ranges.
    """
    seed: int
    n: int
    edges_cor: pd.DataFrame
    layer_params: pd.DataFrame
    d_max_iter: int
    c_max_iter: int
    t: int
    eps: float
    d: int
    edges_filename: str
    communities_filename: str

    def __post_init__(self) -> None:
        # TODO: consider adding validation with pydantic
        self._rng = np.random.default_rng(seed=self.seed)
        assert isinstance(self.seed, int)
        assert isinstance(self.n, int)
        assert isinstance(self.edges_cor, pd.DataFrame)
        assert isinstance(self.layer_params, pd.DataFrame)
        assert isinstance(self.d_max_iter, int)
        assert isinstance(self.c_max_iter, int)
        assert isinstance(self.t, int)
        assert isinstance(self.d, int)
        assert isinstance(self.eps, float)
        assert isinstance(self.d, int)
        assert isinstance(self.edges_filename, str)
        assert isinstance(self.communities_filename, str)

    @staticmethod
    def get_layer_params(lp: dict[str, Any] | str) -> pd.DataFrame:
        if isinstance(lp, str):
            return pd.read_csv(lp)
        elif isinstance(lp, dict):
            return pd.DataFrame(lp)
        else:
            raise ValueError(f"LP should be either dict or path to file.")

    @staticmethod
    def get_edges_cor(ec: list[list[float]] | str) -> pd.DataFrame:
        if isinstance(ec, str):
            return pd.read_csv(ec, index_col=0)
        elif isinstance(ec, list):
            return pd.DataFrame(
                ec,
                index=range(1, len(ec) + 1),
                columns=range(1, len(ec[0]) + 1),
            )
        raise ValueError(f"EC should be either list or path to file.")

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> "MLNConfig":
        """Read configuration from a YAML-like dictionary."""
        _config = config.copy()
        edges_cor = cls.get_edges_cor(config["edges_cor"])
        _config["edges_cor"] = edges_cor
        layer_params = cls.get_layer_params(config["layer_params"])
        _config["layer_params"] = layer_params
        return cls(**_config)

    def to_yaml(self) -> dict[str, Any]:
        """Convert configuration into serialisable format."""
        return {
            "seed": self.seed,
            "n": self.n,
            "edges_cor": self.edges_cor.to_numpy().tolist(),
            "layer_params": self._lp_yaml_helper(),
            "d_max_iter": self.d_max_iter,
            "c_max_iter": self.c_max_iter,
            "t": self.t,
            "eps": self.eps,
            "d": self.d,
            "edges_filename": self.edges_filename,
            "communities_filename": self.communities_filename,
        }

    def to_julia_csvs(self, edges_cor_path: str, layer_params_path: str) -> None:
        """
        Save configuration in Julia format where some parameters are not normalised as here.

        :param edges_cor_path: path of the edge correlation matrix file
        :param layer_params_path: path of the layer parameters file
        """
        self.edges_cor.to_csv(edges_cor_path)

        lp_df = self.layer_params.copy()
        assert all(lp_df["q"].between(0, 1))
        assert all(lp_df["delta"].between(0, 1))
        assert all(lp_df["Delta"].between(0, 1))
        assert all(lp_df["s"].between(0, 1))
        assert all(lp_df["S"].between(0, 1))
        lp_df["_q"] = lp_df["q"] * self.n
        lp_df["delta"] = (lp_df["delta"] * lp_df["_q"]).round(0).astype(int)
        lp_df["Delta"] = (lp_df["Delta"] * lp_df["_q"]).round(0).astype(int)
        lp_df["s"] = (lp_df["s"] * lp_df["_q"]).round(0).astype(int)
        lp_df["S"] = (lp_df["S"] * lp_df["_q"]).round(0).astype(int)
        lp_df[
            ["q", "tau", "r", "gamma", "delta", "Delta", "beta", "s", "S", "xi"]
        ].to_csv(layer_params_path, index=False)


class MLNABCDGraphGenerator:
    """A wrapper class for jl.MLNABCDGraphGenerator."""

    edges_filename = "edges.csv"
    layers_filename = "layers.csv"

    @staticmethod
    def install_julia_dependencies():
        jl.Pkg.add(url="https://github.com/bkamins/ABCDGraphGenerator.jl")
        jl.Pkg.add(url="https://github.com/KrainskiL/MLNABCDGraphGenerator.jl")

    def __call__(self, config: MLNConfig) -> None:
        try:
            jl.seval("using MLNABCDGraphGenerator")
        except JuliaError:
            self.install_julia_dependencies()
            jl.seval("using MLNABCDGraphGenerator")
        
        with tempfile.TemporaryDirectory() as tmpdir:

            # Save dataframes into temp dir
            edges_path = str(Path(tmpdir) / self.edges_filename)
            layers_path = str(Path(tmpdir) / self.layers_filename)
            config.to_julia_csvs(edges_cor_path=edges_path, layer_params_path=layers_path)

            # Load config. Since julia is called each time as a new process, we use a following
            # workaround to generate random, yet repetitive as a sequence, results
            config = jl.MLNABCDGraphGenerator.MLNConfig(
                int(config._rng.random() * 1000),
                config.n,
                edges_path,
                layers_path,
                config.d_max_iter,
                config.c_max_iter,
                config.t,
                config.eps,
                config.d,
                config.edges_filename,
                config.communities_filename,
            )

            # Active nodes
            active_nodes = jl.MLNABCDGraphGenerator.generate_active_nodes(config)

            # Degree Sequences
            degrees = jl.MLNABCDGraphGenerator.generate_degrees(config, active_nodes, False)

            # Sizes of communities
            com_sizes, coms = jl.MLNABCDGraphGenerator.generate_communities(config, active_nodes)

            # Generate ABCD graphs
            edges = jl.MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)

            # Map nodes and communities into agents
            edges = jl.MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
            coms = jl.MLNABCDGraphGenerator.map_communities_to_agents(config.n, coms, active_nodes)

            # Adjust edges correlation
            edges_rewired = jl.MLNABCDGraphGenerator.adjust_edges_correlation(
                config,
                edges,
                coms,
                active_nodes,
                False,
                False,
            )

            # Save edges to file
            jl.MLNABCDGraphGenerator.write_edges(config, edges_rewired)

            # Save communities to file
            jl.MLNABCDGraphGenerator.write_communities(config, coms)


if __name__ == "__main__":

    import yaml
    from pathlib import Path
    
    out_dir = Path("./examples/generate")
    out_dir.mkdir(exist_ok=True, parents=True)

    # generate from code
    n = 1000
    layer_params = MLNConfig.get_layer_params(
        lp={
            "q": [1, 0.75, 0.5, 0.25],
            "tau": [1, 0.75, 0.5, 0.25],
            "r": [1, 0.75, 0.5, 0.25],
            "gamma": [2.5, 2.5, 2.5, 2.5],
            "delta": [0.0020, 0.0027, 0.0040, 0.0080],
            "Delta": [0.0250, 0.0333, 0.0400, 0.0800],
            "beta": [1.5, 1.5, 1.7, 1.7],
            "s": [0.0080, 0.0107, 0.0160, 0.0320],
            "S": [0.0320, 0.0427, 0.0640, 0.1280],
            "xi": [0.2, 0.2, 0.2, 0.1],
        }
    )                                   
    edges_cor = MLNConfig.get_edges_cor(
        [
            [1.0, 0.15, 0.15, 0.12],
            [0.15, 1.0, 0.2, 0.1],
            [0.15, 0.2, 1.0, 0.2],
            [0.12, 0.1, 0.2, 1.0],
        ]
    )
    mln_config = MLNConfig(
        seed=43,
        n=n,
        edges_cor=edges_cor,
        layer_params=layer_params,
        d_max_iter=1000,
        c_max_iter=1000,
        t=100,
        eps=0.05,
        d=2,
        edges_filename=str(out_dir / "_edges.dat"),
        communities_filename=str(out_dir / "_communities.dat"),
    )
    MLNABCDGraphGenerator()(config=mln_config)

    # or from file
    with open("scripts/configs/example_generate_1.yaml", "r", encoding="utf-8") as file:
        _config = yaml.safe_load(file)
    config = _config["mln_config"]
    config["seed"] = _config["run"]["rng_seed"]
    config["edges_filename"] = str(out_dir / config["edges_filename"])
    config["communities_filename"] = str(out_dir / config["communities_filename"])
    mln_config = MLNConfig.from_yaml(config)
    MLNABCDGraphGenerator()(config=mln_config)
