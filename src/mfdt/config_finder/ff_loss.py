"""Definitions of the loss function used in the optimisation process."""

from typing import Callable

import numpy as np


def dummy_loss(A: np.ndarray, A_p: np.ndarray, **kwargs) -> float:
    """Mean difference between A and A_p elements under the lower triangle."""
    d_A = np.abs(A_p - A)
    idcs = np.tril_indices(d_A.shape[0], k=-1)
    d_a = d_A[idcs[0], idcs[1]]  # d_A vals from the lower triangle without the diagonal
    return 100 * d_a.mean().item()


def tau_loss(B: np.ndarray, B_prime: np.ndarray, **kwargs) -> np.float64:
    """Used for tau div score in paper."""
    l = B.shape[0]
    rss = ((B - B_prime) ** 2).sum()
    return np.sqrt(rss / (4 * l * (l - 1)))


def r_loss(A: np.ndarray, A_prime: np.ndarray, **kwargs) -> np.float64:
    """Used for r div score in paper."""
    l = A.shape[0]
    rss = ((A - A_prime) ** 2).sum()
    return np.sqrt(rss / (l * (l - 1)))


def r_tau_loss(
    A: np.ndarray,
    A_prime: np.ndarray,
    B: np.ndarray,
    B_prime: np.ndarray,
    **kwargs,
) -> np.float64:
    """Sum of two div scores for r and tau."""
    return r_loss(A, A_prime) + tau_loss(B, B_prime)


def get_criterium(name: str) -> Callable:
    if name == "dummy":  # I keep it only for debugging purposes
        return dummy_loss
    elif name == "r":
        return r_loss
    elif name == "tau":
        return tau_loss
    elif name == "r+tau":
        return r_tau_loss
    raise ValueError("Unknown criterium name")
