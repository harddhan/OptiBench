"""Objective function definitions for OptiBench."""

from __future__ import annotations

from typing import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _as_float_vector(x: ArrayLike, *, name: str = "x", expected_size: int | None = None) -> np.ndarray:
    """Convert a candidate vector to a 1D float NumPy array.

    Args:
        x: Input data to validate.
        name: Label used in exceptions.
        expected_size: Optional expected dimension for validation.

    Returns:
        A 1D floating-point NumPy array.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = np.ravel(arr)

    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} must have exactly {expected_size} entries; got {arr.size}.")
    return arr


def sphere(x: ArrayLike) -> float:
    """Evaluate the n-dimensional Sphere objective.

    The Sphere function is strictly convex and has a unique minimum at the origin.
    """
    values = _as_float_vector(x, name="x")
    return float(np.sum(values**2))


def rosenbrock(x: ArrayLike) -> float:
    """Evaluate the classic two-variable Rosenbrock objective."""
    values = _as_float_vector(x, name="x", expected_size=2)
    x1, x2 = values
    return float((1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2)


def himmelblau(x: ArrayLike) -> float:
    """Evaluate Himmelblau's function in two variables."""
    values = _as_float_vector(x, name="x", expected_size=2)
    x1, x2 = values
    a = x1**2 + x2 - 11.0
    b = x1 + x2**2 - 7.0
    return float(a**2 + b**2)


BENCHMARK_FUNCTIONS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "himmelblau": himmelblau,
}

__all__ = ["ArrayLike", "BENCHMARK_FUNCTIONS", "himmelblau", "rosenbrock", "sphere"]
