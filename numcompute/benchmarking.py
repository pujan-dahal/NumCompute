import time
import numpy as np


def time_function(func, *args, repeats=5, warmup=1, **kwargs):
    """Times a function over many runs"""
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    # Run first without timing
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    times = np.asarray(times, dtype=float)

    return {
        "times": times,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def compare_functions(func1, func2, args=(), kwargs1=None, kwargs2=None, repeats=5, warmup=1,
                      name1="vectorized", name2="loop"):
    """Compares how long two functions take"""
    if kwargs1 is None:
        kwargs1 = {}

    if kwargs2 is None:
        kwargs2 = {}

    result1 = time_function(func1, *args, repeats=repeats, warmup=warmup, **kwargs1)
    result2 = time_function(func2, *args, repeats=repeats, warmup=warmup, **kwargs2)

    speedup = result2["mean"] / result1["mean"]

    return {
        name1: result1,
        name2: result2,
        "speedup": speedup,
    }


def benchmark_suite(tasks, repeats=5, warmup=1):
    """Runs benchmarks for a list of tasks"""
    results = []

    for task in tasks:
        name = task["name"]
        func = task["func"]
        args = task.get("args", ())
        kwargs = task.get("kwargs", {})

        result = time_function(func, *args, repeats=repeats, warmup=warmup, **kwargs)

        results.append({
            "name": name,
            "mean": result["mean"],
            "std": result["std"],
            "min": result["min"],
            "max": result["max"],
        })

    return results


def print_benchmark_table(results):
    """Prints benchmark results in a table"""
    print(f"{'Task':<25} {'Mean (s)':<12} {'Std (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-" * 73)

    for result in results:
        print(
            f"{result['name']:<25} "
            f"{result['mean']:<12.6f} "
            f"{result['std']:<12.6f} "
            f"{result['min']:<12.6f} "
            f"{result['max']:<12.6f}"
        )


def benchmark_vectorized_vs_loop(vectorized_func, loop_func, args=(), repeats=5, warmup=1,
                                 vectorized_name="vectorized", loop_name="loop"):
    """Compares a vectorized function with a loop function"""
    results = compare_functions(
        vectorized_func,
        loop_func,
        args=args,
        repeats=repeats,
        warmup=warmup,
        name1=vectorized_name,
        name2=loop_name,
    )

    print(f"{vectorized_name} mean time: {results[vectorized_name]['mean']:.6f} s")
    print(f"{loop_name} mean time:       {results[loop_name]['mean']:.6f} s")
    print(f"Speedup: {results['speedup']:.2f}x")

    return results
