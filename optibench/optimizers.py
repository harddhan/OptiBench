"""Gradient-based optimizers implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class OptimizationResult:
    """Container for optimizer outputs."""

    x: np.ndarray
    fun: float
    nit: int
    grad_norm: float
    success: bool
    message: str
    path: list[np.ndarray] = field(default_factory=list)
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "x": np.asarray(self.x, dtype=float).tolist(),
            "fun": float(self.fun),
            "nit": int(self.nit),
            "grad_norm": float(self.grad_norm),
            "success": bool(self.success),
            "message": self.message,
            "path": [np.asarray(point, dtype=float).tolist() for point in self.path],
            "history": [float(value) for value in self.history],
        }


def gradient_descent(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | list[float],
    *,
    learning_rate: float = 1e-2,
    tol: float = 1e-8,
    max_iter: int = 5000,
) -> OptimizationResult:
    """Run gradient descent with a simple backtracking line search."""
    x = np.asarray(x0, dtype=float).copy()
    value = float(objective(x))
    path = [x.copy()]
    history = [value]

    for iteration in range(1, max_iter + 1):
        grad = np.asarray(gradient(x), dtype=float)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= tol:
            return OptimizationResult(
                x=x.copy(),
                fun=value,
                nit=iteration - 1,
                grad_norm=grad_norm,
                success=True,
                message="Gradient norm below tolerance.",
                path=path,
                history=history,
            )

        direction = -grad
        step = learning_rate
        candidate = x + step * direction
        candidate_value = float(objective(candidate))

        while candidate_value > value and step > 1e-12:
            step *= 0.5
            candidate = x + step * direction
            candidate_value = float(objective(candidate))

        x = candidate
        value = candidate_value
        path.append(x.copy())
        history.append(value)

        if np.linalg.norm(step * direction) <= tol:
            return OptimizationResult(
                x=x.copy(),
                fun=value,
                nit=iteration,
                grad_norm=float(np.linalg.norm(gradient(x))),
                success=True,
                message="Step size fell below tolerance.",
                path=path,
                history=history,
            )

    return OptimizationResult(
        x=x.copy(),
        fun=value,
        nit=max_iter,
        grad_norm=float(np.linalg.norm(gradient(x))),
        success=False,
        message="Maximum number of iterations reached.",
        path=path,
        history=history,
    )


def newton_method(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | list[float],
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> OptimizationResult:
    """Run a damped Newton method using a backtracking line search."""
    x = np.asarray(x0, dtype=float).copy()
    value = float(objective(x))
    path = [x.copy()]
    history = [value]

    for iteration in range(1, max_iter + 1):
        grad = np.asarray(gradient(x), dtype=float)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= tol:
            return OptimizationResult(
                x=x.copy(),
                fun=value,
                nit=iteration - 1,
                grad_norm=grad_norm,
                success=True,
                message="Gradient norm below tolerance.",
                path=path,
                history=history,
            )

        hess = np.asarray(hessian(x), dtype=float)
        try:
            direction = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            direction = -grad

        step = 1.0
        candidate = x + step * direction
        candidate_value = float(objective(candidate))

        while candidate_value > value and step > 1e-12:
            step *= 0.5
            candidate = x + step * direction
            candidate_value = float(objective(candidate))

        x = candidate
        value = candidate_value
        path.append(x.copy())
        history.append(value)

        if np.linalg.norm(step * direction) <= tol:
            return OptimizationResult(
                x=x.copy(),
                fun=value,
                nit=iteration,
                grad_norm=float(np.linalg.norm(gradient(x))),
                success=True,
                message="Newton step became too small.",
                path=path,
                history=history,
            )

    return OptimizationResult(
        x=x.copy(),
        fun=value,
        nit=max_iter,
        grad_norm=float(np.linalg.norm(gradient(x))),
        success=False,
        message="Maximum number of Newton iterations reached.",
        path=path,
        history=history,
    )


__all__ = ["OptimizationResult", "gradient_descent", "newton_method"]
