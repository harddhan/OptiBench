"""Compare the implemented algorithms on one benchmark."""

from optibench.experiments import compare_optimizers


if __name__ == "__main__":
    results = compare_optimizers("rosenbrock")
    for item in results:
        print(item.to_dict())
