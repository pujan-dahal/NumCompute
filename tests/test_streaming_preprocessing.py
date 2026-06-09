import numpy as np
import pytest

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_standard_scaler_partial_fit_matches_batch_fit_with_nan():
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]])
    stream = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:])
    batch = StandardScaler().fit(X)
    assert np.allclose(stream.mean_, batch.mean_, equal_nan=True)
    assert np.allclose(stream.std_, batch.std_, equal_nan=True)


def test_standard_scaler_partial_fit_rejects_feature_mismatch():
    scaler = StandardScaler().partial_fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Expected 2 features"):
        scaler.partial_fit([[1.0]])


def test_minmax_scaler_partial_fit_updates_range():
    X = np.array([[3.0, 10.0], [1.0, 20.0], [5.0, 15.0]])
    scaler = MinMaxScaler().partial_fit(X[:1]).partial_fit(X[1:])
    assert np.allclose(scaler.data_min_, [1.0, 10.0])
    assert np.allclose(scaler.data_max_, [5.0, 20.0])


def test_minmax_scaler_partial_fit_zero_variance_is_safe():
    scaler = MinMaxScaler().partial_fit([[2.0], [2.0]])
    assert np.allclose(scaler.transform([[2.0], [3.0]]), [[0.0], [1.0]])


def test_imputer_partial_fit_median_updates_across_chunks():
    X = np.array([[1.0, np.nan], [5.0, 6.0], [3.0, 8.0]])
    imputer = SimpleImputer(strategy="median").partial_fit(X[:1]).partial_fit(X[1:])
    assert np.allclose(imputer.statistics_, [3.0, 7.0])


def test_imputer_partial_fit_most_frequent_tie_uses_smallest_value():
    X = np.array([[2.0], [1.0], [2.0], [1.0]])
    imputer = SimpleImputer(strategy="most_frequent").partial_fit(X)
    assert imputer.statistics_[0] == 1.0


def test_onehot_partial_fit_expands_without_forgetting_old_categories():
    enc = OneHotEncoder().partial_fit(np.array([["red"], ["blue"]], dtype=object))
    enc.partial_fit(np.array([["green"], [None]], dtype=object))
    transformed = enc.transform(np.array([["red"], ["green"], ["unknown"]], dtype=object))
    assert transformed.shape == (3, 3)
    assert transformed[-1].sum() == 0.0


def test_onehot_partial_fit_rejects_feature_mismatch():
    enc = OneHotEncoder().partial_fit(np.array([["a", "x"]], dtype=object))
    with pytest.raises(ValueError, match="Expected 2 features"):
        enc.partial_fit(np.array([["b"]], dtype=object))
