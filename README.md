# OptiBench

OptiBench is a modular Python benchmark suite for numerical optimization algorithms. It is designed for educational use in optimization coursework and emphasizes reproducibility, clear mathematical reasoning, and simple but rigorous implementation.

## Features

- Gradient descent and Newton's method implemented from scratch using NumPy.
- Benchmark functions: Sphere, Rosenbrock, and Himmelblau.
- Analytical gradients and Hessians with finite-difference validation.
- Statistics such as objective value, gradient norm, iteration count, runtime, and convergence status.
- Plotting utilities for convergence and contour-path visualization.
- A command-line interface and example scripts for quick experiments.

## Installation

`ash
python -m pip install -r requirements.txt
`

## Quick start

`ash
python -m optibench.cli --function sphere --algorithm all
`

## Benchmark functions

### Sphere

f(x) = sum_i x_i^2

### Rosenbrock

f(x, y) = (1 - x)^2 + 100(y - x^2)^2

### Himmelblau

f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

## Algorithms

- Gradient descent uses a backtracking line search to ensure descent.
- Newton's method solves the linear system H p = -g and uses a damped step if needed.

## Testing

`ash
pytest
`
