import numpy as np
import pytest

from numcompute.pipeline import Pipeline, FeatureUnion, Compose


class AddOne:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) + 1


class MultiplyByTwo:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) * 2


class MeanModel:
    def fit(self, X, y):
        self.mean_value = np.mean(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_value)


def test_pipeline_fit_transform():
    X = np.array([[1, 2], [3, 4]])

    pipe = Pipeline([
        ("add", AddOne()),
        ("multiply", MultiplyByTwo())
    ])

    result = pipe.fit_transform(X)

    expected = np.array([[4, 6], [8, 10]])
    assert np.array_equal(result, expected)


def test_pipeline_predict():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([10, 20, 30])

    pipe = Pipeline([
        ("add", AddOne()),
        ("model", MeanModel())
    ])

    pipe.fit(X, y)
    result = pipe.predict(X)

    expected = np.array([20, 20, 20])
    assert np.array_equal(result, expected)


def test_pipeline_duplicate_step_names():
    with pytest.raises(ValueError):
        Pipeline([
            ("same", AddOne()),
            ("same", MultiplyByTwo())
        ])


def test_pipeline_invalid_step_format():
    with pytest.raises(TypeError):
        Pipeline([
            AddOne()
        ])


def test_feature_union():
    X = np.array([[1, 2], [3, 4]])

    union = FeatureUnion([
        ("add", AddOne()),
        ("multiply", MultiplyByTwo())
    ])

    result = union.fit_transform(X)

    expected = np.array([
        [2, 3, 2, 4],
        [4, 5, 6, 8]
    ])

    assert np.array_equal(result, expected)


def test_compose_works_like_pipeline():
    X = np.array([[1, 2]])

    compose = Compose([
        ("add", AddOne()),
        ("multiply", MultiplyByTwo())
    ])

    result = compose.fit_transform(X)

    expected = np.array([[4, 6]])
    assert np.array_equal(result, expected)
