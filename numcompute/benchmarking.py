import time
import numpy as np


def time_function(func, *args, repeats=5, warmup=1, **kwargs):
    """
    Time a function over many runs

    Parameters
    ----------
    func : callable
        Function to time
    *args
        Positional arguments passed to func
    repeats : int
        Number of timed runs
    warmup : int
        Number of untimed runs before timing starts
    **kwargs
        Keyword arguments passed to func

    Returns
    -------
    dict
        Times array plus mean std min and max in seconds

    Raises
    ------
    ValueError
        If repeats is not positive or warmup is negative

    Complexity
    ----------
    Time O repeats plus warmup function calls
    Space O repeats
    """
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
    """
    Compare the runtime of two functions on the same input

    Parameters
    ----------
    func1 func2 : callable
        Functions to compare
    args : tuple
        Positional arguments passed to both functions
    kwargs1 kwargs2 : dict or None
        Keyword arguments for each function
    repeats : int
        Number of timed runs
    warmup : int
        Number of untimed runs
    name1 name2 : str
        Names used in the returned dictionary

    Returns
    -------
    dict
        Timing result for each function and speedup as func2 mean divided by func1 mean

    Raises
    ------
    ValueError
        If timing settings are invalid

    Complexity
    ----------
    Time O repeats plus warmup calls for both functions
    Space O repeats
    """
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
    """
    Run benchmarks for a list of named tasks

    Parameters
    ----------
    tasks : list
        Each task should contain name and func with optional args and kwargs
    repeats : int
        Number of timed runs for each task
    warmup : int
        Number of untimed runs for each task

    Returns
    -------
    list
        List of dictionaries with name mean std min and max

    Raises
    ------
    KeyError
        If a task is missing name or func
    ValueError
        If timing settings are invalid

    Complexity
    ----------
    Time O number of tasks times timing cost
    Space O number of tasks
    """
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
    """
    Print benchmark results in a simple table

    Parameters
    ----------
    results : list
        Benchmark summary dictionaries with name mean std min and max

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If a result is missing a required field

    Complexity
    ----------
    Time O r where r is number of results
    Space O 1
    """
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
    """
    Benchmark a vectorized function against a loop function

    Parameters
    ----------
    vectorized_func : callable
        Vectorized implementation to time
    loop_func : callable
        Loop based implementation to time
    args : tuple
        Positional arguments passed to both functions
    repeats : int
        Number of timed runs
    warmup : int
        Number of untimed runs
    vectorized_name loop_name : str
        Names used in printed output and returned data

    Returns
    -------
    dict
        Timing comparison and speedup

    Raises
    ------
    ValueError
        If timing settings are invalid

    Complexity
    ----------
    Time O timing cost for both functions
    Space O repeats
    """
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
