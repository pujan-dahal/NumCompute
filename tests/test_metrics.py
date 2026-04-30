# test_metrics.py

import numpy as np
import pytest

from numcompute.metrics import Classification, Regression


def test_confusion_matrix_basic():
    """Test confusion matrix computation with a simple binary example."""
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
    assert tp == 1
    assert tn == 2
    assert fp == 0
    assert fn == 1


def test_all_correct_predictions():
    """Test metrics when all predictions are correct."""
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    assert Classification.accuracy(y_true, y_pred) == 1.0
    assert Classification.precision(y_true, y_pred) == 1.0
    assert Classification.recall(y_true, y_pred) == 1.0
    assert Classification.f1(y_true, y_pred) == 1.0


def test_all_incorrect_predictions():
    """Test metrics when all predictions are incorrect."""
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]

    assert Classification.accuracy(y_true, y_pred) == 0.0
    assert Classification.precision(y_true, y_pred) == 0.0
    assert Classification.recall(y_true, y_pred) == 0.0
    assert Classification.f1(y_true, y_pred) == 0.0


def test_precision_division_by_zero():
    """Test precision returns 0 when there are no positive predictions."""
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    assert Classification.precision(y_true, y_pred) == 0.0


def test_recall_division_by_zero():
    """Test recall returns 0 when there are no actual positives."""
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 1, 0]

    assert Classification.recall(y_true, y_pred) == 0.0


def test_f1_division_by_zero():
    """Test F1 returns 0 when both precision and recall are zero."""
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]

    assert Classification.f1(y_true, y_pred) == 0.0


def test_input_shape_mismatch_classification():
    """Test classification metrics raise error on mismatched shapes."""
    y_true = [1, 0, 1]
    y_pred = [1, 0]

    with pytest.raises(ValueError):
        Classification.accuracy(y_true, y_pred)


def test_numpy_array_input():
    """Test metrics accept NumPy array inputs."""
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 0])
    acc = Classification.accuracy(y_true, y_pred)
    assert isinstance(acc, float)


def test_non_binary_labels_behavior():
    """Test behavior with non-binary labels."""
    y_true = [2, 1, 2]
    y_pred = [2, 2, 1]
    tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
    assert tp == 0
    assert tn == 0


def test_mse_basic():
    """Test MSE returns zero for identical inputs."""
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert Regression.mse(y_true, y_pred) == 0.0


def test_mse_nonzero():
    """Test MSE computation with non-zero error."""
    y_true = [1, 2, 3]
    y_pred = [2, 2, 4]
    expected = 2 / 3
    assert np.isclose(Regression.mse(y_true, y_pred), expected)


def test_mse_with_negative_values():
    """Test MSE handles negative values correctly."""
    y_true = [-1, -2, -3]
    y_pred = [1, 2, 3]
    expected = np.mean(np.square(np.array(y_true) - np.array(y_pred)))
    assert np.isclose(Regression.mse(y_true, y_pred), expected)


def test_input_shape_mismatch_regression():
    """Test regression metrics raise error on mismatched shapes."""
    y_true = [1, 2, 3]
    y_pred = [1, 2]
    with pytest.raises(ValueError):
        Regression.mse(y_true, y_pred)


def test_mse_numpy_input():
    """Test MSE accepts NumPy arrays."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 3.0])
    result = Regression.mse(y_true, y_pred)
    assert isinstance(result, float)


def test_empty_input():
    """Test MSE behavior with empty inputs."""
    y_true = []
    y_pred = []
    result = Regression.mse(y_true, y_pred)
    assert np.isnan(result)