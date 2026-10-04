import numpy as np

from optibench.metrics import PerformanceMetrics, compute_metrics, summarize_results
from optibench.optimizers import OptimizationResult


def test_compute_metrics_and_summary():
    result = OptimizationResult(
        x=np.array([1.0, 2.0]),
        fun=0.25,
        nit=10,
        grad_norm=1e-7,
        success=True,
        message="Converged",
        path=[np.array([1.0, 2.0]), np.array([0.9, 1.9])],
        history=[0.5, 0.25],
    )
    metrics = compute_metrics("sphere", "gradient_descent", result, execution_time=0.123)
    assert metrics.function_name == "sphere"
    assert metrics.success is True
    summary = summarize_results([metrics])
    assert summary["best_function_value"] == 0.25
