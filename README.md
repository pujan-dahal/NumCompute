# NumCompute: A High-Performance NumPy Toolkit

NumCompute is a modular scientific-computing and machine-learning utility toolkit built using **plain Python and NumPy only**. It was developed for **Assignment 2.1: Programming Task 1** to demonstrate vectorised numerical programming, clean APIs, testing, benchmarking, and reusable ML-style workflow components.

The project simulates a small part of a machine-learning library such as scikit-learn, but every core component is implemented from scratch using NumPy rather than external ML/data libraries.

Authors:

- Akardhan Dewan (a1992862)
- Anshu Shrestha (a3201939)
- Pujan Dahal (a3190289)
- Shubham Kumar (a3173468)

---

## 1. Project checklist

| Assignment requirement                                         | Implemented location                        |
| -------------------------------------------------------------- | ------------------------------------------- |
| CSV input/output with missing values and dtype handling        | `numcompute/io.py`                          |
| Preprocessing: scaling, imputation, one-hot encoding           | `numcompute/preprocessing.py`               |
| Sorting, searching, top-k, quickselect, binary search          | `numcompute/sort_search.py`                 |
| Ranking and percentiles with tie handling                      | `numcompute/rank.py`, `numcompute/ranks.py` |
| Descriptive statistics, histograms, quantiles                  | `numcompute/stats.py`                       |
| Streaming statistics / Welford mean and variance               | `numcompute/stats.py`                       |
| Classification and regression metrics                          | `numcompute/metrics.py`                     |
| Finite-difference gradients and Jacobian                       | `numcompute/optim.py`                       |
| Lightweight pipeline abstraction                               | `numcompute/pipeline.py`                    |
| Utility functions: distances, activations, logsumexp, batching | `numcompute/utils.py`                       |
| Benchmark utilities                                            | `numcompute/benchmarking.py`                |
| Reproducible benchmark script                                  | `benchmark/run_benchmarks.py`               |
| Unit tests with edge cases                                     | `tests/`                                    |
| End-to-end demo notebook                                       | `demo/quickstart.ipynb`                     |

---

## 2. Folder structure

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── ranks.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
├── demo/
│   ├── numcompute_demo_data.csv
│   ├── quickstart.ipynb
│   └── quickstart_markdown.md
├── benchmark/
│   └── run_benchmarks.py
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 3. Installation

From the project root, install the package in editable mode:

```bash
python -m pip install -e .
```

Install test tools if needed:

```bash
python -m pip install numpy pytest
```

The runtime dependency is intentionally minimal:

```text
numpy
```

No external ML/DL/data libraries such as pandas, scikit-learn, TensorFlow, PyTorch, or SciPy are required for the core implementation.

---

## 4. Quick start example

```python
import numpy as np

from numcompute.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder
from numcompute.metrics import Classification, Regression
from numcompute.optim import grad
from numcompute.utils import softmax, logsumexp

X = np.array([
    [21.0, 170.0, 82.0],
    [22.0, np.nan, 76.0],
    [20.0, 165.0, np.nan],
    [24.0, 180.0, 91.0],
])

# Missing-value handling
imputer = SimpleImputer(strategy="mean")
X_filled = imputer.fit_transform(X)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Categorical encoding
categories = np.array([["A"], ["B"], ["A"], ["C"]], dtype=object)
encoded = OneHotEncoder().fit_transform(categories)

# Metrics
accuracy = Classification.accuracy([1, 0, 1, 1], [1, 0, 0, 1])
mse = Regression.mse(np.array([1.0, 2.0]), np.array([1.2, 1.8]))

# Gradient
objective = lambda x: x[0] ** 2 + 3 * x[1] ** 2
g = grad(objective, np.array([2.0, 3.0]))

# Numerically stable utilities
probs = softmax(np.array([1000.0, 1001.0, 1002.0]))
lse = logsumexp(np.array([1000.0, 1001.0, 1002.0]))
```

For a full demonstration, open:

```text
demo/quickstart.ipynb
```

A markdown version of the notebook is also provided at:

```text
demo/quickstart_markdown.md
```

---

## 5. Running the tests

Run all unit tests from the project root:

```bash
python -m pytest -q
```

The test suite covers more than the minimum requirement of 20 tests. It includes checks for:

- empty arrays and invalid shapes
- NaN handling
- all-equal values
- duplicate values and ties
- extreme and invalid `k` values
- string categories in one-hot encoding
- mixed integer/float CSV inference
- numerical-stability edge cases such as `logsumexp([-inf, -inf])`
- finite-difference gradients and Jacobian shapes
- pipeline validation and transformations

---

## 6. Running the benchmark script

Run:

```bash
python benchmark/run_benchmarks.py
```

The script compares vectorised NumPy implementations against equivalent Python-loop implementations and prints an environment-aware timing table.

Example output format:

| Task    | Vectorised implementation | Python-loop implementation | Reported value    |
| ------- | ------------------------- | -------------------------- | ----------------- |
| Mean    | `np.mean`                 | manual loop mean           | seconds + speedup |
| Sigmoid | stable vectorised sigmoid | manual loop sigmoid        | seconds + speedup |
| MSE     | vectorised squared error  | manual loop squared error  | seconds + speedup |

Exact timings depend on the computer, Python version, NumPy version, and system load. The benchmark script prints Python and NumPy version information to make the results reproducible.

---

## 7. Design choices

### Vectorisation

Core numerical operations use NumPy vectorisation where possible. This reduces slow Python-level loops and delegates array operations to optimized low-level routines. Python loops are only used where they are appropriate, such as finite-difference dimensions, Welford streaming updates, or small control-flow algorithms like quickselect.

### Modular API

The package is separated by responsibility:

- `io.py` handles data loading and dtype inference.
- `preprocessing.py` handles transformations before modelling.
- `sort_search.py` and `rank.py` provide reusable algorithmic utilities.
- `stats.py` provides descriptive and streaming statistics.
- `metrics.py` evaluates classification and regression outputs.
- `optim.py` estimates gradients and Jacobians.
- `pipeline.py` chains transformations using a simple estimator-style API.
- `utils.py` contains shared numerical helper functions.

This structure improves readability, testing, and maintainability.

### Numerical stability

The toolkit includes stable numerical forms such as:

- max-shifted `softmax`
- stable `logsumexp`
- overflow-safe `sigmoid`
- NaN-aware statistics and preprocessing
- safe handling of all-equal columns in scalers
- explicit validation for empty inputs and shape mismatches

---

## 8. Important API examples

### CSV loading

```python
from numcompute.io import IO

data = IO.load_csv("demo/numcompute_demo_data.csv")
print(data.data)
print(data.cols)
```

### Preprocessing pipeline

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

X_processed = pipe.fit_transform(X)
```

### Ranking and percentile

```python
from numcompute.rank import rank, percentile

rank([10, 20, 20, 30], method="average")
percentile([10, 20, 30, 40], 75)
```

### Streaming statistics

```python
from numcompute.stats import StreamingStats

stream = StreamingStats()
stream.update_many([1, 2, 3, 4, 5])
print(stream.mean)
print(stream.variance(ddof=0))
```

---

## 9. Team contribution notes

The final report should clearly identify each team member's contribution. Suggested format:

| Team member               | Main contribution                                               |
| ------------------------- | --------------------------------------------------------------- |
| Anshu Shrestha (a3201939) | IO, Preprocessing, test cases and documentation                 |
| Shubham Kumar (a3173468)  | Rank, Sort-search and test cases                                |
| Pujan Dahal (a3190289)    | Stats, Metrics, test cases and documentation                    |
| Akardhan Dewan (a1992862) | Optim, Pipeline, Utils, Benchmark, test cases and documentation |

---
