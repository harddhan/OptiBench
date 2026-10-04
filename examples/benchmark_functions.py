"""Print values for each benchmark function."""

from optibench.functions import himmelblau, rosenbrock, sphere


if __name__ == "__main__":
    print("Sphere:", sphere([1.0, -2.0, 3.0]))
    print("Rosenbrock:", rosenbrock([1.0, 1.0]))
    print("Himmelblau:", himmelblau([3.0, 2.0]))
