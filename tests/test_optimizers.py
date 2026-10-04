import numpy as np

from optibench.derivatives import (
    himmelblau_gradient,
    himmelblau_hessian,
    rosenbrock_gradient,
    rosenbrock_hessian,
    sphere_gradient,
    sphere_hessian,
)
from optibench.functions import himmelblau, rosenbrock, sphere
from optibench.optimizers import gradient_descent, newton_method


def test_gradient_descent_sphere_converges():
    result = gradient_descent(sphere, sphere_gradient, np.array([2.0, -3.0]), learning_rate=0.05, tol=1e-10, max_iter=3000)
    assert result.success
    assert np.allclose(result.x, np.zeros(2), atol=1e-3)
    assert result.fun < 1e-6


def test_newton_method_sphere_converges():
    result = newton_method(sphere, sphere_gradient, sphere_hessian, np.array([1.5, -2.0]), tol=1e-10, max_iter=25)
    assert result.success
    assert np.allclose(result.x, np.zeros(2), atol=1e-6)
    assert result.fun < 1e-10


def test_newton_method_rosenbrock_converges():
    result = newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, np.array([-1.5, 1.0]), tol=1e-8, max_iter=80)
    assert result.success
    assert result.fun < 1e-4
