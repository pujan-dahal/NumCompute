"""
Reproducible benchmark script for NumCompute.

Run from the project root:
    python benchmark/run_benchmarks.py

The script compares simple NumPy/vectorised implementations against equivalent
Python-loop implementations and prints a compact timing table.
"""

import platform
import sys
import time
from pathlib import Path

import numpy as np

# Allow running the script directly from the benchmark/ folder or project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute.benchmarking import time_function
from numcompute.utils import sigmoid


def loop_mean(x):
    total = 0.0
    count = 0
    for value in x:
        total += float(value)
        count += 1
    return total / count


def vectorized_mean(x):
    return np.mean(x)


def loop_sigmoid(x):
    out = []
    for value in x:
        if value >= 0:
            out.append(1.0 / (1.0 + np.exp(-value)))
        else:
            exp_value = np.exp(value)
            out.append(exp_value / (1.0 + exp_value))
    return np.asarray(out)


def loop_squared_error(y_true, y_pred):
    total = 0.0
    count = 0
    for actual, predicted in zip(y_true, y_pred):
        error = actual - predicted
        total += error * error
        count += 1
    return total / count


def vectorized_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def run_case(name, vectorized_func, loop_func, args, repeats=5):
    vec = time_function(vectorized_func, *args, repeats=repeats, warmup=1)
    loop = time_function(loop_func, *args, repeats=repeats, warmup=1)
    speedup = loop["mean"] / vec["mean"] if vec["mean"] > 0 else np.inf
    return {
        "name": name,
        "vectorized": vec["mean"],
        "loop": loop["mean"],
        "speedup": speedup,
    }


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=100_000)
    y_true = rng.normal(size=100_000)
    y_pred = y_true + rng.normal(scale=0.1, size=100_000)

    cases = [
        run_case("mean", vectorized_mean, loop_mean, (x,)),
        run_case("sigmoid", sigmoid, loop_sigmoid, (x,)),
        run_case("mse", vectorized_squared_error, loop_squared_error, (y_true, y_pred)),
    ]

    print("NumCompute benchmark results")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy:  {np.__version__}")
    print(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"{'Task':<12} {'Vectorized (s)':<16} {'Loop (s)':<12} {'Speedup':<10}")
    print("-" * 54)
    for row in cases:
        print(
            f"{row['name']:<12} "
            f"{row['vectorized']:<16.6f} "
            f"{row['loop']:<12.6f} "
            f"{row['speedup']:<10.2f}x"
        )

    # quick correctness checks so benchmark results are meaningful
    assert np.isclose(vectorized_mean(x), loop_mean(x))
    assert np.allclose(sigmoid(x), loop_sigmoid(x))
    assert np.isclose(vectorized_squared_error(y_true, y_pred), loop_squared_error(y_true, y_pred))


if __name__ == "__main__":
    main()
