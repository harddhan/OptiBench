import numpy as np

from optibench.derivatives import (
    finite_difference_gradient,
    himmelblau_gradient,
    rosenbrock_gradient,
    sphere_gradient,
)
from optibench.functions import himmelblau, rosenbrock, sphere


def test_sphere_gradient_matches_finite_difference():
    x = np.array([1.25, -2.5, 0.75])
    analytical = sphere_gradient(x)
    numerical = finite_difference_gradient(sphere, x)
    assert np.allclose(analytical, numerical, atol=1e-5, rtol=1e-5)


def test_rosenbrock_gradient_matches_finite_difference():
    x = np.array([0.5, 2.0])
    analytical = rosenbrock_gradient(x)
    numerical = finite_difference_gradient(rosenbrock, x)
    assert np.allclose(analytical, numerical, atol=1e-5, rtol=1e-5)


def test_himmelblau_gradient_matches_finite_difference():
    x = np.array([1.0, 2.5])
    analytical = himmelblau_gradient(x)
    numerical = finite_difference_gradient(himmelblau, x)
    assert np.allclose(analytical, numerical, atol=1e-5, rtol=1e-5)
