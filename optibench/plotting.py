"""Visualization helpers for optimization trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import PerformanceMetrics


def plot_convergence(metrics: PerformanceMetrics, *, save_path: str | Path | None = None) -> plt.Figure:
    """Plot objective values against iteration count."""
    fig, ax = plt.subplots(figsize=(6, 4))
    values = np.asarray(metrics.history, dtype=float)
    x = np.arange(len(values))
    ax.plot(x, values, marker="o", linewidth=1.5)
    ax.set_title(f"{metrics.optimizer_name} on {metrics.function_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_contour_with_path(
    objective,
    path: list[np.ndarray] | np.ndarray,
    *,
    bounds: tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
    save_path: str | Path | None = None,
):
    """Draw a contour plot of the objective surface with the optimization path."""
    trajectory = [np.asarray(point, dtype=float) for point in path]

    x_min, x_max, y_min, y_max = bounds
    x_values = np.linspace(x_min, x_max, 200)
    y_values = np.linspace(y_min, y_max, 200)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_mesh = np.empty_like(x_mesh, dtype=float)

    for idx in range(x_mesh.shape[0]):
        for jdx in range(x_mesh.shape[1]):
            z_mesh[idx, jdx] = objective(np.array([x_mesh[idx, jdx], y_mesh[idx, jdx]], dtype=float))

    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contourf(x_mesh, y_mesh, z_mesh, levels=30, cmap="viridis")
    ax.set_title("Optimization path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if trajectory:
        points = np.asarray(trajectory)
        if points.ndim == 2 and points.shape[1] == 2:
            ax.plot(points[:, 0], points[:, 1], "-o", color="red", linewidth=1.5)

    fig.colorbar(contour, ax=ax, label="Objective value")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


__all__ = ["plot_convergence", "plot_contour_with_path"]
