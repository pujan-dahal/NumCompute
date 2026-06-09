import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.tree import DecisionTreeClassifier


class NoTransformStep:
    def partial_fit(self, X):
        return self


class NoPartialFitFinal:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_pipeline_partial_fit_streams_transformer_and_model():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=2)),
    ])
    pipe.partial_fit(X[:2], y[:2])
    pipe.partial_fit(X[2:], y[2:])
    assert pipe.predict(X).shape == y.shape


def test_pipeline_partial_fit_rejects_middle_step_without_transform():
    pipe = Pipeline([
        ("bad", NoTransformStep()),
        ("tree", DecisionTreeClassifier()),
    ])
    with pytest.raises(TypeError, match="transform"):
        pipe.partial_fit([[1.0]], [0])


def test_pipeline_partial_fit_rejects_final_step_without_partial_fit():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", NoPartialFitFinal()),
    ])
    with pytest.raises(TypeError, match="partial_fit"):
        pipe.partial_fit([[1.0]], [0])
