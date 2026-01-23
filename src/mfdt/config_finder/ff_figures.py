"""Functions to visualise the optimiastion process."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from skopt.plots import plot_convergence
from skopt.utils import OptimizeResult


def _plot_trajectory(ax: Axes, result: OptimizeResult) -> Axes:
    plot_convergence(("gp_min", result), ax=ax)
    ax.plot(
        np.arange(start=1, stop=len(result.func_vals)+1, step=1),
        result.func_vals,
        linestyle="--",
        linewidth=1.5,
        alpha=0.4,
        label="gp_curr",
    )
    ax.legend()
    ax.set_title("Covergence plot")
    return ax


def _plot_mesh(ax: Axes, result: "OptimizeResult") -> Axes:

    # Project solution space to 2D using PCA
    pca = PCA(n_components=2)
    r_2d = pca.fit_transform(result.x_iters)

    # Prepare a coarse mesh in PCA space
    n_grid = 100
    x_min, x_max = r_2d[:, 0].min(), r_2d[:, 0].max()
    y_min, y_max = r_2d[:, 1].min(), r_2d[:, 1].max()

    x_ax = np.linspace(x_min, x_max, n_grid)
    y_ax = np.linspace(y_min, y_max, n_grid)
    x_mesh, y_mesh = np.meshgrid(x_ax, y_ax)

    # Interpolate loss values over the mesh
    z_mesh = griddata(
        points=r_2d,
        values=result.func_vals,
        xi=(x_mesh, y_mesh),
        method="linear",
    )

    # Plot interpolated loss landscape
    mesh = ax.pcolormesh(
        x_mesh,
        y_mesh,
        z_mesh,
        norm=LogNorm(
            vmin=np.nanmin(z_mesh),
            vmax=np.nanmax(z_mesh),
        ),
        cmap="viridis_r",
        shading="auto",
    )
    ax.figure.colorbar(mesh, ax=ax, label="loss")

    # Plot sampled points
    ax.scatter(
        r_2d[:, 0],
        r_2d[:, 1],
        c=result.func_vals,
        cmap="viridis",
        edgecolor="k",
        s=40,
        zorder=3,
    )

    # Plot optimisation trajectory arrows
    dx = r_2d[1:, 0] - r_2d[:-1, 0]
    dy = r_2d[1:, 1] - r_2d[:-1, 1]
    ax.quiver(
        r_2d[:-1, 0],
        r_2d[:-1, 1],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.01,
        color="white",
        zorder=4,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Loss landscape (PCA, coarse interpolation)")
    return ax


def plot_optimisation_process(result: OptimizeResult, out_dir: Path) -> None:
    """Plot the optimisation process for a skopt result."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    fig.suptitle("Optimisation Results")
    ax[0] = _plot_trajectory(ax[0], result)
    ax[1] = _plot_mesh(ax[1], result)
    fig.tight_layout()
    fig.savefig(out_dir, dpi=300)
