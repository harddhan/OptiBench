"""Experiment orchestration for optimization benchmarks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .derivatives import BENCHMARK_GRADIENTS, BENCHMARK_HESSIANS
from .functions import BENCHMARK_FUNCTIONS
from .metrics import PerformanceMetrics, compute_metrics
from .optimizers import gradient_descent, newton_method

DEFAULT_STARTING_POINTS = {
    "sphere": np.array([2.0, -3.0, 1.5]),
    "rosenbrock": np.array([-1.5, 1.0]),
    "himmelblau": np.array([0.0, 0.0]),
}


def _make_gradient_descent(objective, gradient, hessian, x0):
    return gradient_descent(objective, gradient, x0, learning_rate=0.01, tol=1e-8, max_iter=5000)


def _make_newton(objective, gradient, hessian, x0):
    return newton_method(objective, gradient, hessian, x0, tol=1e-8, max_iter=50)


OPTIMIZER_REGISTRY = {
    "gradient_descent": _make_gradient_descent,
    "newton": _make_newton,
}


class BenchmarkRegistry(dict):
    """Convenience wrapper around benchmark metadata."""

    def register(self, name: str, objective: Callable[[np.ndarray], float]) -> None:
        self[name] = objective


def run_benchmark(function_name: str, optimizer_name: str, x0: np.ndarray | list[float] | None = None) -> PerformanceMetrics:
    """Run a single benchmark/optimizer configuration."""
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function: {function_name}")
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    objective = BENCHMARK_FUNCTIONS[function_name]
    gradient = BENCHMARK_GRADIENTS[function_name]
    hessian = BENCHMARK_HESSIANS[function_name]
    start = np.asarray(x0 if x0 is not None else DEFAULT_STARTING_POINTS[function_name], dtype=float)

    result = OPTIMIZER_REGISTRY[optimizer_name](objective, gradient, hessian, start)
    elapsed = 0.0
    return compute_metrics(function_name, optimizer_name, result, execution_time=elapsed)


def compare_optimizers(function_name: str, x0: np.ndarray | list[float] | None = None) -> list[PerformanceMetrics]:
    """Compare all implemented optimizers on a benchmark function."""
    return [run_benchmark(function_name, optimizer_name, x0) for optimizer_name in OPTIMIZER_REGISTRY]


__all__ = ["BenchmarkRegistry", "DEFAULT_STARTING_POINTS", "OPTIMIZER_REGISTRY", "compare_optimizers", "run_benchmark"]
