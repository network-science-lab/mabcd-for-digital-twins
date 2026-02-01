"""Functions to visualise the optimiastion process."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from scipy.interpolate import RBFInterpolator
from skopt.plots import plot_convergence
from skopt.utils import OptimizeResult
from umap import UMAP


def _plot_trajectory(ax: Axes, result: OptimizeResult) -> Axes:
    plot_convergence(("gp_min", result), ax=ax)
    best_step = np.where(result.func_vals == result.fun)[0][0].item() + 1
    x = np.arange(start=1, stop=len(result.func_vals) + 1, step=1)
    ax.plot(
        x,
        result.func_vals,
        linestyle="--",
        linewidth=1.5,
        alpha=0.4,
        label="gp_curr",
    )
    ax.axvline(
        x=best_step,
        color="firebrick",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
    )
    ticks = list(ax.get_xticks())
    if best_step not in ticks:
        ticks.append(best_step)
        ticks = sorted(ticks)
        ax.set_xticks(ticks)
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick == best_step:
            label.set_color("firebrick")
    ax.legend()
    ax.set_title("Convergence plot")
    return ax


def _plot_mesh(ax: Axes, result: "OptimizeResult") -> Axes:
    reducer = UMAP(
        n_components=2,
        n_neighbors=len(result.x_iters) // 2,
        min_dist=0.05,
        metric="euclidean",
        init="pca",
        random_state=42,
    )
    r_2d = reducer.fit_transform(np.array(result.x_iters))

    # Prepare a coarse mesh in reduced space
    n_grid = 200
    x_min, x_max = r_2d[:, 0].min(), r_2d[:, 0].max()
    y_min, y_max = r_2d[:, 1].min(), r_2d[:, 1].max()

    x_ax = np.linspace(x_min, x_max, n_grid)
    y_ax = np.linspace(y_min, y_max, n_grid)
    x_mesh, y_mesh = np.meshgrid(x_ax, y_ax)

    # Interpolate loss values over the mesh
    rbf = RBFInterpolator(
        y=r_2d,
        d=result.func_vals,
        kernel="thin_plate_spline",
    )

    z_mesh = rbf(np.column_stack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)

    vals = z_mesh[np.isfinite(z_mesh)]
    vals_pos = vals[vals > 0]

    low = np.percentile(vals_pos, 2)
    high = np.percentile(vals_pos, 98)
    margin = 0.25

    vmin = low / (1 + margin)
    vmax = high * (1 + margin)
    z_plot = np.clip(z_mesh, vmin, vmax)

    # Plot interpolated loss landscape
    mesh = ax.pcolormesh(
        x_mesh,
        y_mesh,
        z_plot,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
        cmap="viridis_r",
        shading="auto",
        rasterized=True,
    )
    ax.figure.colorbar(mesh, ax=ax, label="loss")

    # Plot sampled points
    best_idx = np.nanargmin(result.func_vals)
    mask = np.ones(len(r_2d), dtype=bool)
    mask[[0, -1, best_idx]] = False
    ax.scatter(r_2d[0, 0], r_2d[0, 1], color="mediumpurple", s=50, zorder=4, label="Start")
    ax.scatter(r_2d[-1, 0], r_2d[-1, 1], color="orange", s=50, zorder=4, label="Finish")
    ax.scatter(r_2d[mask, 0], r_2d[mask, 1], c="slategray", s=50, zorder=3, label="Intermediate")
    ax.scatter(r_2d[best_idx, 0], r_2d[best_idx, 1], c="firebrick", s=50, zorder=4, label="Best")
    ax.legend(
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.3),
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
        width=0.007,
        color="white",
        zorder=4,
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Loss landscape & trajectory")
    return ax


def plot_optimisation_process(result: OptimizeResult, out_dir: Path) -> None:
    """Plot the optimisation process for a skopt result."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    fig.suptitle("Optimisation Results")
    ax[0] = _plot_trajectory(ax[0], result)
    ax[1] = _plot_mesh(ax[1], result)
    fig.tight_layout()
    fig.savefig(out_dir, dpi=300)
    fig.savefig(out_dir.parent / "trajectory.png", dpi=300)
    fig.savefig(
        out_dir.parent / "trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax = _plot_trajectory(ax, result)
    fig.tight_layout()
    fig.savefig(out_dir.parent / "only_trajectory.png", dpi=300)
    fig.savefig(
        out_dir.parent / "only_trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax = _plot_mesh(ax, result)
    fig.tight_layout()
    fig.savefig(out_dir.parent / "only_mesh.png", dpi=300)
    fig.savefig(
        out_dir.parent / "only_mesh.pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
