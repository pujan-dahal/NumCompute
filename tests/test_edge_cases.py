import numpy as np
import pytest

from numcompute.io import IO, CSVData, Column
from numcompute.metrics import Classification
from numcompute.preprocessing import OneHotEncoder
from numcompute.stats import Stats, StreamingStats
from numcompute.utils import logsumexp, sigmoid


def test_onehot_string_categories():
    X = np.array([["red"], ["blue"], ["red"]], dtype=object)
    enc = OneHotEncoder()
    out = enc.fit_transform(X)

    assert out.shape == (3, 2)
    assert np.all(out.sum(axis=1) == 1)


def test_csv_mixed_int_float_infers_float(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("a\n1\n2.5\n")

    data = IO.load_csv(str(p))

    assert data.cols[0].dtype == "float"
    assert np.isclose(data.data[1, 0], 2.5)


def test_logsumexp_all_negative_infinity():
    result = logsumexp(np.array([-np.inf, -np.inf]))
    assert result == -np.inf


def test_sigmoid_large_negative_has_no_overflow_warning():
    with np.errstate(over="raise"):
        result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

    assert np.allclose(result, np.array([0.0, 0.5, 1.0]))


def test_classification_accuracy_empty_raises():
    with pytest.raises(ValueError):
        Classification.accuracy([], [])


def test_streaming_stats_mean_and_variance():
    stream = StreamingStats()
    stream.update_many(np.array([1.0, 2.0, 3.0, 4.0]))

    assert np.isclose(stream.mean, 2.5)
    assert np.isclose(stream.variance(), np.var([1.0, 2.0, 3.0, 4.0]))


def test_stats_streaming_mean_matches_numpy():
    data = np.array([
        [1.0, 10.0, "A"],
        [2.0, np.nan, "B"],
        [3.0, 30.0, "C"],
    ], dtype=object)
    cols = [Column("x", "float"), Column("y", "float"), Column("label", "str")]
    csv_data = CSVData(data, cols)

    expected = np.nanmean(np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]]))
    assert np.isclose(Stats.streaming_mean(csv_data), expected)
