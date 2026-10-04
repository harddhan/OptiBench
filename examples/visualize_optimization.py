"""Create basic convergence and contour plots."""

from optibench.experiments import run_benchmark
from optibench.plotting import plot_convergence, plot_contour_with_path


if __name__ == "__main__":
    metrics = run_benchmark("rosenbrock", "gradient_descent")
    plot_convergence(metrics, save_path="results/rosenbrock_convergence.png")
    plot_contour_with_path(
        lambda xy: (1.0 - xy[0]) ** 2 + 100.0 * (xy[1] - xy[0] ** 2) ** 2,
        metrics.path,
        bounds=(-2.0, 2.0, -1.0, 3.0),
        save_path="results/rosenbrock_path.png",
    )
