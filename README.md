# OptiBench

a python benchmark suite for numerical optimization algorithms, built from scratch using NumPy as part of my Optimization Techniques and Algorithms coursework.

Currently features:
- Gradient Descent and Newton's Method
- Sphere, Rosenbrock, and Himmelblau functions
- Analytical derivatives and finite-difference validation
- Performance metrics and convergence visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m optibench.cli --function sphere --optimizer all
```

## Testing

```bash
python -m pytest
```
