import numpy as np
import pytest

from numcompute.benchmarking import (
    time_function,
    compare_functions,
    benchmark_suite,
    print_benchmark_table,
    benchmark_vectorized_vs_loop,
)


def simple_sum(x):
    return np.sum(x)


def loop_sum(x):
    total = 0
    for value in x:
        total += value
    return total


def test_time_function_returns_expected_keys():
    x = np.array([1, 2, 3])

    result = time_function(simple_sum, x, repeats=3, warmup=1)

    assert "times" in result
    assert "mean" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result
    assert len(result["times"]) == 3


def test_time_function_invalid_repeats():
    x = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        time_function(simple_sum, x, repeats=0)


def test_time_function_invalid_warmup():
    x = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        time_function(simple_sum, x, warmup=-1)


def test_compare_functions():
    x = np.arange(100)

    result = compare_functions(
        simple_sum,
        loop_sum,
        args=(x,),
        repeats=2,
        warmup=1,
    )

    assert "vectorized" in result
    assert "loop" in result
    assert "speedup" in result
    assert result["speedup"] > 0


def test_compare_functions_custom_names():
    x = np.arange(100)

    result = compare_functions(
        simple_sum,
        loop_sum,
        args=(x,),
        repeats=2,
        warmup=1,
        name1="numpy_sum",
        name2="python_loop",
    )

    assert "numpy_sum" in result
    assert "python_loop" in result
    assert "speedup" in result


def test_benchmark_suite():
    x = np.arange(100)

    tasks = [
        {
            "name": "sum_test",
            "func": simple_sum,
            "args": (x,),
        }
    ]

    result = benchmark_suite(tasks, repeats=2, warmup=1)

    assert len(result) == 1
    assert result[0]["name"] == "sum_test"
    assert "mean" in result[0]
    assert "std" in result[0]
    assert "min" in result[0]
    assert "max" in result[0]


def test_print_benchmark_table(capsys):
    results = [
        {
            "name": "task1",
            "mean": 0.001,
            "std": 0.0001,
            "min": 0.0009,
            "max": 0.0012,
        }
    ]

    print_benchmark_table(results)

    captured = capsys.readouterr()

    assert "Task" in captured.out
    assert "task1" in captured.out
    assert "Mean" in captured.out


def test_benchmark_vectorized_vs_loop(capsys):
    x = np.arange(100)

    result = benchmark_vectorized_vs_loop(
        simple_sum,
        loop_sum,
        args=(x,),
        repeats=2,
        warmup=1,
    )

    captured = capsys.readouterr()

    assert "Speedup" in captured.out
    assert "speedup" in result
    assert result["speedup"] > 0
