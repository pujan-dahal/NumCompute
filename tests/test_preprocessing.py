import numpy as np
import pytest

from numcompute.preprocessing import (
    _BaseScaler,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer
)


# BaseScaler
def test_validate_converts_1d_to_2d():
    X = np.array([1, 2, 3])
    X_out = _BaseScaler._validate(X)
    assert X_out.shape == (3, 1)


def test_validate_raises_on_non_numeric():
    X = [["a", "b"], ["c", "d"]]
    with pytest.raises(ValueError):
        _BaseScaler._validate(X)


def test_validate_raises_on_high_dim():
    X = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        _BaseScaler._validate(X)


def test_validate_raises_if_not_fitted():
    scaler = StandardScaler()
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError):
        scaler.transform(X)


# Standard Scaler
def test_standard_scaler_basic():
    X = np.array([[1.0], [2.0], [3.0]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    assert np.allclose(np.mean(X_scaled), 0.0)
    assert np.allclose(np.std(X_scaled), 1.0)


def test_standard_scaler_zero_variance():
    X = np.array([[5.0], [5.0], [5.0]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    assert np.allclose(X_scaled, 0.0)


def test_standard_scaler_with_nan():
    X = np.array([[1.0], [np.nan], [3.0]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    assert np.isnan(X_scaled[1])


def test_standard_scaler_inverse():
    X = np.array([[1.0], [2.0], [3.0]])
    scaler = StandardScaler().fit(X)

    X_scaled = scaler.transform(X)
    X_inv = scaler.inverse_transform(X_scaled)

    assert np.allclose(X, X_inv)


def test_standard_scaler_feature_mismatch():
    X = np.array([[1.0, 2.0]])
    scaler = StandardScaler().fit(X)

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0]]))


# Min Max Scaler
def test_minmax_basic():
    X = np.array([[0.0], [5.0], [10.0]])
    scaler = MinMaxScaler().fit(X)

    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled, [[0.0], [0.5], [1.0]])


def test_minmax_zero_range():
    X = np.array([[5.0], [5.0]])
    scaler = MinMaxScaler().fit(X)

    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled, 0.0)


def test_minmax_custom_range():
    X = np.array([[0.0], [10.0]])
    scaler = MinMaxScaler(feature_range=(1, 2)).fit(X)

    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled, [[1.0], [2.0]])


def test_minmax_invalid_range():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 1))


def test_minmax_clip():
    X = np.array([[0.0], [10.0]])
    scaler = MinMaxScaler().fit(X)

    X_new = np.array([[-5.0], [15.0]])
    X_scaled = scaler.transform(X_new)

    assert np.all(X_scaled >= 0.0)
    assert np.all(X_scaled <= 1.0)


def test_minmax_inverse():
    X = np.array([[0.0], [10.0]])
    scaler = MinMaxScaler().fit(X)

    X_scaled = scaler.transform(X)
    X_inv = scaler.inverse_transform(X_scaled)

    assert np.allclose(X, X_inv)


# One Hot Encoder
def test_onehot_basic():
    X = np.array([[1], [2], [1]])
    enc = OneHotEncoder().fit(X)

    X_enc = enc.transform(X)
    assert X_enc.shape == (3, 2)


def test_onehot_drop_first():
    X = np.array([[1], [2], [3]])
    enc = OneHotEncoder(drop_first=True).fit(X)

    X_enc = enc.transform(X)
    assert X_enc.shape == (3, 2)


def test_onehot_with_nan():
    X = np.array([[1], [np.nan], [1]])
    enc = OneHotEncoder().fit(X)

    X_enc = enc.transform(X)
    assert np.all(X_enc[1] == 0)


def test_onehot_feature_mismatch():
    X = np.array([[1, 2]])
    enc = OneHotEncoder().fit(X)

    with pytest.raises(ValueError):
        enc.transform(np.array([[1]]))


# Simple Imputer

@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_imputer_numeric_strategies(strategy):
    X = np.array([[1.0], [np.nan], [3.0]])
    imp = SimpleImputer(strategy=strategy).fit(X)

    X_out = imp.transform(X)
    assert not np.isnan(X_out).any()


def test_imputer_most_frequent():
    X = np.array([[1.0], [2.0], [2.0], [np.nan]])
    imp = SimpleImputer(strategy="most_frequent").fit(X)

    X_out = imp.transform(X)
    assert X_out[-1] == 2.0


def test_imputer_constant():
    X = np.array([[1.0], [np.nan]])
    imp = SimpleImputer(strategy="constant", fill_value=99).fit(X)

    X_out = imp.transform(X)
    assert X_out[1] == 99


def test_imputer_all_nan_column():
    X = np.array([[np.nan], [np.nan]])
    imp = SimpleImputer(strategy="most_frequent").fit(X)

    X_out = imp.transform(X)
    assert np.isnan(X_out).all()


def test_imputer_invalid_strategy():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="invalid")


def test_imputer_feature_mismatch():
    X = np.array([[1.0, 2.0]])
    imp = SimpleImputer().fit(X)

    with pytest.raises(ValueError):
        imp.transform(np.array([[1.0]]))