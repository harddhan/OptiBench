"""Analytical and numerical derivatives for the benchmark functions."""

from __future__ import annotations

import numpy as np

from .functions import _as_float_vector


def sphere_gradient(x: np.ndarray | list[float]) -> np.ndarray:
    """Return the analytical gradient of the Sphere function."""
    values = _as_float_vector(x, name="x")
    return 2.0 * values


def sphere_hessian(x: np.ndarray | list[float]) -> np.ndarray:
    """Return the Hessian of the Sphere function."""
    values = _as_float_vector(x, name="x")
    return 2.0 * np.eye(values.size, dtype=float)


def rosenbrock_gradient(x: np.ndarray | list[float]) -> np.ndarray:
    """Return the analytical gradient of the Rosenbrock objective."""
    values = _as_float_vector(x, name="x", expected_size=2)
    x1, x2 = values
    return np.array(
        [
            -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1**2),
            200.0 * (x2 - x1**2),
        ],
        dtype=float,
    )


def rosenbrock_hessian(x: np.ndarray | list[float]) -> np.ndarray:
    """Return the analytical Hessian of the Rosenbrock objective."""
    values = _as_float_vector(x, name="x", expected_size=2)
    x1, x2 = values
    h11 = 2.0 - 400.0 * x2 + 1200.0 * x1**2
    h12 = -400.0 * x1
    return np.array([[h11, h12], [h12, 200.0]], dtype=float)


def himmelblau_gradient(x: np.ndarray | list[float]) -> np.ndarray:
    """Return the analytical gradient of Himmelblau's function."""
    values = _as_float_vector(x, name="x", expected_size=2)
    x1, x2 = values
    a = x1**2 + x2 - 11.0
    b = x1 + x2**2 - 7.0
    return np.array(
        [
            4.0 * x1 * a + 2.0 * b,
            2.0 * a + 4.0 * x2 * b,
        ],
        dtype=float,
    )


def himmelblau_hessian(x: np.ndarray | list[float]) -> np.ndarray:
    """Return the analytical Hessian of Himmelblau's function."""
    values = _as_float_vector(x, name="x", expected_size=2)
    x1, x2 = values
    a = x1**2 + x2 - 11.0
    b = x1 + x2**2 - 7.0
    h11 = 4.0 * a + 8.0 * x1**2 + 2.0
    h12 = 4.0 * x1 + 4.0 * x2
    h22 = 2.0 + 4.0 * b + 8.0 * x2**2
    return np.array([[h11, h12], [h12, h22]], dtype=float)


def finite_difference_gradient(
    objective,
    x: np.ndarray | list[float],
    *,
    step_size: float = 1e-6,
) -> np.ndarray:
    """Approximate the gradient of an objective using central differences.

    This is mainly used as a validation tool for the analytical gradient. The
    chosen step size balances truncation error against floating-point noise.
    """
    values = _as_float_vector(x, name="x")
    gradient = np.zeros_like(values, dtype=float)
    for idx in range(values.size):
        delta = np.zeros_like(values, dtype=float)
        delta[idx] = step_size
        forward = objective(values + delta)
        backward = objective(values - delta)
        gradient[idx] = (forward - backward) / (2.0 * step_size)
    return gradient


BENCHMARK_GRADIENTS = {
    "sphere": sphere_gradient,
    "rosenbrock": rosenbrock_gradient,
    "himmelblau": himmelblau_gradient,
}

BENCHMARK_HESSIANS = {
    "sphere": sphere_hessian,
    "rosenbrock": rosenbrock_hessian,
    "himmelblau": himmelblau_hessian,
}

__all__ = [
    "BENCHMARK_GRADIENTS",
    "BENCHMARK_HESSIANS",
    "finite_difference_gradient",
    "himmelblau_gradient",
    "himmelblau_hessian",
    "rosenbrock_gradient",
    "rosenbrock_hessian",
    "sphere_gradient",
    "sphere_hessian",
]
