import numpy as np

from optibench.functions import himmelblau, rosenbrock, sphere


def test_sphere_value():
    assert sphere(np.array([3.0, 4.0])) == 25.0
    assert sphere(np.array([-1.0, 2.0, 2.0])) == 9.0


def test_rosenbrock_value():
    assert rosenbrock(np.array([1.0, 1.0])) == 0.0
    assert np.isclose(rosenbrock(np.array([0.0, 0.0])), 1.0)


def test_himmelblau_value():
    assert np.isclose(himmelblau(np.array([3.0, 2.0])), 0.0)
    assert np.isclose(himmelblau(np.array([-2.805118, 3.131312])), 0.0, atol=1e-5)


def test_invalid_inputs_raise_errors():
    try:
        sphere([1.0, np.nan])
        raise AssertionError("Expected a ValueError for NaN input.")
    except ValueError:
        pass

    try:
        rosenbrock(np.array([1.0]))
        raise AssertionError("Expected a ValueError for wrong dimension.")
    except ValueError:
        pass
