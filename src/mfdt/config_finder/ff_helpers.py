"""Helpers for fancy methods for inferring mABCD configuration parameters."""

import tempfile
from pathlib import Path
from typing import Callable

import network_diffusion as nd
import numpy as np

from mfdt.config_finder import correlations, cr_helpers
from mfdt.params_handler import create_out_dir


def get_comm_ami(net: nd.MultilayerNetwork, seed: int | None = None) -> np.ndarray:
    """Get interlatyer 'correlations' (i.e. AMI) between partitions."""
    # net = net.to_multiplex()[0]  # TODO: decide if we need that?
    part_cor_raw = []
    for la_name, lb_name in cr_helpers.prepare_layer_pairs(list(net.layers.keys())):
        aligned_layers = cr_helpers.align_layers(net, la_name, lb_name, "destructive")
        part_ami = correlations.partitions_correlation(
            aligned_layers[la_name],
            aligned_layers[lb_name], 
            seed=seed,
        )
        part_cor_raw.append({(la_name, lb_name): part_ami})
    part_cor_df = cr_helpers.create_correlation_matrix(part_cor_raw)  # .round(3).fillna(0.0)
    return part_cor_df.to_numpy()


def get_stacked_A_element_variance(stacked_A: np.ndarray) -> float:
    n_samples, n_dims, _ = stacked_A.shape  # n_dims == _
    idx = np.tril_indices(n_dims, k=-1)
    vals = stacked_A[:, idx[0], idx[1]]
    return vals.std(axis=0).mean().item()


def frobenius_loss(A: np.ndarray, A_p: np.ndarray) -> float:
    """Frobenius loss."""
    fro_A = np.linalg.norm(A, ord="fro")
    fro_A_p = np.linalg.norm(A_p, ord="fro")
    return np.abs(fro_A_p - fro_A).item()


def dummy_loss(A: np.ndarray, A_p: np.ndarray) -> float:
    """Mean difference between A and A_p elements under the lower triangle."""
    d_A = np.abs(A_p - A)
    idcs = np.tril_indices(d_A.shape[0], k=-1)
    d_a = d_A[idcs[0], idcs[1]]  # d_A vals from the lower triangle without the diagonal
    return d_a.mean().item()


def get_criterium(name: str) -> Callable:
    if name == "frobenius":
        return frobenius_loss
    elif name == "dummy":
        return dummy_loss
    raise ValueError("Unknown criterium name")


def prepare_log_dir(out_dir: Path | None = None) -> Callable:
    """Prepare directory to store logs in if out_dir provided."""
    if out_dir:
        class OutDirServer:
            """Mock for TemporaryDirectory to store logs in a reachable dir."""
            def __init__(self, out_dir: Path) -> None:
                self.out_dir_base = out_dir
                self.call_nb = 0
            def __call__(self) -> "OutDirServer":
                return self
            def __enter__(self, *args, **kwargs) -> str:
                curr = self.out_dir_base / str(self.call_nb)
                self.call_nb += 1
                return str(create_out_dir(curr))
            def __exit__(self, *args, **kwargs) -> bool:
                return False
        return OutDirServer(out_dir)
        # @contextmanager
        # def mock_tmpdir():
        #     yield str(create_out_dir(out_dir))
    return tempfile.TemporaryDirectory
