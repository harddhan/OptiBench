"""Utilities for summary statistics and performance comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .optimizers import OptimizationResult


@dataclass
class PerformanceMetrics:
    """Summary of one optimizer run on a benchmark objective."""

    function_name: str
    optimizer_name: str
    final_solution: np.ndarray
    final_objective: float
    iterations: int
    gradient_norm: float
    success: bool
    execution_time: float
    message: str
    history: list[float] = field(default_factory=list)
    path: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return {
            "function_name": self.function_name,
            "optimizer_name": self.optimizer_name,
            "final_solution": np.asarray(self.final_solution, dtype=float).tolist(),
            "final_objective": float(self.final_objective),
            "iterations": int(self.iterations),
            "gradient_norm": float(self.gradient_norm),
            "success": bool(self.success),
            "execution_time": float(self.execution_time),
            "message": self.message,
            "history": [float(v) for v in self.history],
            "path": [np.asarray(point, dtype=float).tolist() for point in self.path],
        }


def compute_metrics(
    function_name: str,
    optimizer_name: str,
    result: OptimizationResult,
    *,
    execution_time: float | None = None,
) -> PerformanceMetrics:
    """Turn an optimization result into a structured metrics record."""
    if execution_time is None:
        raise ValueError("execution_time must be provided.")
    return PerformanceMetrics(
        function_name=function_name,
        optimizer_name=optimizer_name,
        final_solution=np.asarray(result.x, dtype=float),
        final_objective=float(result.fun),
        iterations=int(result.nit),
        gradient_norm=float(result.grad_norm),
        success=bool(result.success),
        execution_time=float(execution_time),
        message=result.message,
        history=[float(v) for v in result.history],
        path=[np.asarray(point, dtype=float).copy() for point in result.path],
    )


def summarize_results(results: list[PerformanceMetrics]) -> dict[str, object]:
    """Create a compact summary and compare objective values across runs."""
    if not results:
        return {"count": 0, "best_function_value": None}

    best = min(results, key=lambda item: item.final_objective)
    return {
        "count": len(results),
        "best_function_value": float(best.final_objective),
        "best_optimizer": best.optimizer_name,
        "best_for_function": best.function_name,
        "runs": [item.to_dict() for item in results],
    }


__all__ = ["PerformanceMetrics", "compute_metrics", "summarize_results"]
