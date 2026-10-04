"""Command-line interface for OptiBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import compare_optimizers, run_benchmark


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run benchmark optimization experiments.")
    parser.add_argument("--function", choices=["sphere", "rosenbrock", "himmelblau"], default="sphere")
    parser.add_argument("--algorithm", choices=["gradient_descent", "newton", "all"], default="all")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for JSON results output.")
    return parser.parse_args()


def main() -> None:
    """Run the selected benchmark configuration."""
    args = parse_args()
    if args.algorithm == "all":
        results = compare_optimizers(args.function)
        payload = [item.to_dict() for item in results]
    else:
        result = run_benchmark(args.function, args.algorithm)
        payload = result.to_dict()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
