"""
Streaming benchmark for single tree and ensemble models.

Run from the project root with:
    python benchmark/stream_benchmark.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from numcompute.ensemble import EnsembleClassifier
from numcompute.tree import DecisionTreeClassifier


def make_stream(n_samples=600, random_state=7):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 4))
    y = ((X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.25) > 0).astype(int)
    return X, y


def run_model(model, X, y, chunk_size=60):
    start = time.perf_counter()
    scores = []
    for start_idx in range(0, X.shape[0], chunk_size):
        end_idx = start_idx + chunk_size
        X_chunk = X[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        model.partial_fit(X_chunk, y_chunk)
        pred = model.predict(X_chunk)
        scores.append(np.mean(pred == y_chunk))
    elapsed = time.perf_counter() - start
    return elapsed, float(np.mean(scores))


if __name__ == "__main__":
    X, y = make_stream()
    tree_time, tree_acc = run_model(DecisionTreeClassifier(max_depth=4), X, y)
    forest_time, forest_acc = run_model(EnsembleClassifier(n_estimators=7, max_depth=4, max_features=2, random_state=3), X, y)
    print("Streaming benchmark")
    print(f"DecisionTreeClassifier: {tree_time:.4f}s, mean chunk accuracy={tree_acc:.3f}")
    print(f"EnsembleClassifier:     {forest_time:.4f}s, mean chunk accuracy={forest_acc:.3f}")
