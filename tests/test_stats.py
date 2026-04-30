import numpy as np
import pytest

from numcompute.io import CSVData, Column
from numcompute.stats import Stats


def make_numeric_csv_data():
    """Create a CSVData instance with both numeric and non-numeric columns."""
    data = np.array([
        [20.0, 170.0, "A"],
        [22.0, np.nan, "B"],
        [24.0, 180.0, "C"],
    ], dtype=object)

    cols = [
        Column("age", "int"),
        Column("height", "float"),
        Column("group", "str"),
    ]

    return CSVData(data, cols)


def make_no_numeric_csv_data():
    """Create a CSVData instance with only non-numeric columns."""
    data = np.array([
        ["A", "Adelaide"],
        ["B", "Melbourne"],
    ], dtype=object)

    cols = [
        Column("group", "str"),
        Column("city", "str"),
    ]

    return CSVData(data, cols)


def test_get_numeric_cols():
    """Test numeric column detection."""
    data = make_numeric_csv_data()
    assert Stats._get_numeric_cols(data) == [0, 1]


def test_mean_all_numeric_values():
    """Test mean over all numeric values."""
    data = make_numeric_csv_data()

    result = Stats.mean(data)

    expected = np.nanmean(np.array([
        [20.0, 170.0],
        [22.0, np.nan],
        [24.0, 180.0],
    ]))

    assert np.isclose(result, expected)


def test_mean_axis_0_returns_dict():
    """Test column-wise mean."""
    data = make_numeric_csv_data()

    result = Stats.mean(data, axis=0)

    assert np.isclose(result["age"], 22.0)
    assert np.isclose(result["height"], 175.0)


def test_mean_axis_1_returns_array():
    """Test row-wise mean."""
    data = make_numeric_csv_data()

    result = Stats.mean(data, axis=1)

    expected = np.array([
        np.nanmean([20.0, 170.0]),
        np.nanmean([22.0, np.nan]),
        np.nanmean([24.0, 180.0]),
    ])

    assert np.allclose(result, expected)


def test_mean_invalid_axis():
    """Test invalid axis raises error."""
    data = make_numeric_csv_data()

    with pytest.raises(ValueError):
        Stats.mean(data, axis=3)


def test_median_all_numeric_values():
    """Test median over all numeric values."""
    data = make_numeric_csv_data()

    result = Stats.median(data)

    expected = np.nanmedian(np.array([
        [20.0, 170.0],
        [22.0, np.nan],
        [24.0, 180.0],
    ]))

    assert np.isclose(result, expected)


def test_median_axis_0_returns_dict():
    """Test column-wise median."""
    data = make_numeric_csv_data()

    result = Stats.median(data, axis=0)

    assert np.isclose(result["age"], 22.0)
    assert np.isclose(result["height"], 175.0)


def test_std_axis_0_returns_dict():
    """Test column-wise standard deviation."""
    data = make_numeric_csv_data()

    result = Stats.std(data, axis=0)

    assert np.isclose(result["age"], np.nanstd([20.0, 22.0, 24.0]))
    assert np.isclose(result["height"], np.nanstd([170.0, np.nan, 180.0]))


def test_minimum_all_values():
    """Test global minimum."""
    data = make_numeric_csv_data()

    assert np.isclose(Stats.minimum(data), 20.0)


def test_minimum_axis_0_returns_dict():
    """Test column-wise minimum."""
    data = make_numeric_csv_data()

    result = Stats.minimum(data, axis=0)

    assert np.isclose(result["age"], 20.0)
    assert np.isclose(result["height"], 170.0)


def test_maximum_all_values():
    """Test global maximum."""
    data = make_numeric_csv_data()

    assert np.isclose(Stats.maximum(data), 180.0)


def test_maximum_axis_0_returns_dict():
    """Test column-wise maximum."""
    data = make_numeric_csv_data()

    result = Stats.maximum(data, axis=0)

    assert np.isclose(result["age"], 24.0)
    assert np.isclose(result["height"], 180.0)


def test_quantile_all_values():
    """Test global quantile computation."""
    data = make_numeric_csv_data()

    result = Stats.quantile(data, 0.5)

    expected = np.nanquantile(np.array([
        [20.0, 170.0],
        [22.0, np.nan],
        [24.0, 180.0],
    ]), 0.5)

    assert np.isclose(result, expected)


def test_quantile_axis_0_returns_dict():
    """Test column-wise quantile."""
    data = make_numeric_csv_data()

    result = Stats.quantile(data, 0.5, axis=0)

    assert np.isclose(result["age"], 22.0)
    assert np.isclose(result["height"], 175.0)


def test_quantile_invalid_q():
    """Test invalid quantile value raises error."""
    data = make_numeric_csv_data()

    with pytest.raises(ValueError):
        Stats.quantile(data, 1.5)


def test_histogram_all_values():
    """Test global histogram."""
    data = make_numeric_csv_data()

    counts, bins = Stats.histogram(data, bins=3)

    assert len(counts) == 3
    assert len(bins) == 4


def test_histogram_axis_0_returns_dict():
    """Test column-wise histogram."""
    data = make_numeric_csv_data()

    result = Stats.histogram(data, bins=2, axis=0)

    assert "age" in result
    assert "height" in result

    age_counts, age_bins = result["age"]

    assert len(age_counts) == 2
    assert len(age_bins) == 3


def test_histogram_axis_1_returns_list():
    """Test row-wise histogram."""
    data = make_numeric_csv_data()

    result = Stats.histogram(data, bins=2, axis=1)

    assert isinstance(result, list)
    assert len(result) == 3


def test_histogram_invalid_axis():
    """Test invalid histogram axis raises error."""
    data = make_numeric_csv_data()

    with pytest.raises(ValueError):
        Stats.histogram(data, axis=5)


def test_no_numeric_columns_mean():
    """Test mean raises error when no numeric columns exist."""
    data = make_no_numeric_csv_data()

    with pytest.raises(ValueError):
        Stats.mean(data)


def test_no_numeric_columns_median():
    """Test median raises error when no numeric columns exist."""
    data = make_no_numeric_csv_data()

    with pytest.raises(ValueError):
        Stats.median(data)


def test_no_numeric_columns_std():
    """Test std raises error when no numeric columns exist."""
    data = make_no_numeric_csv_data()

    with pytest.raises(ValueError):
        Stats.std(data)