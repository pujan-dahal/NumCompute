"""
tests/test_io_preprocessing.py
================================
Unit tests for numcompute.io and numcompute.preprocessing.

Covers:
* Normal operation (happy paths)
* Edge cases: empty arrays, single row/column, all-NaN columns,
  constant columns, extreme values, mismatched feature counts
* Numerical correctness checks using np.allclose

Run with:  pytest tests/test_io_preprocessing.py -v

Author : (your name)
Day 4  : Initial test suite — 25 test cases total
"""

import os
import tempfile
import numpy as np
import pytest

# We import from the final merged files.  Adjust the import path
# to match your actual package layout once you consolidate the
# day-by-day files into io.py and preprocessing.py.
from numcompute.io import (
    load_csv,
    load_csv_safe,
    get_headers,
    select_columns,
    infer_dtypes,
    save_csv,
    load_csv_chunked,
)
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
)


# ===========================================================================
# Helpers — write temporary CSV files for io tests
# ===========================================================================

def _write_temp_csv(content: str) -> str:
    """Write *content* to a NamedTemporaryFile and return its path."""
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    fh.write(content)
    fh.close()
    return fh.name


def _cleanup(path: str) -> None:
    """Remove a temporary file, ignoring errors if it is already gone."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ===========================================================================
# io.py tests
# ===========================================================================

class TestLoadCsv:
    """Tests for load_csv — the primary CSV loader."""

    def test_basic_load(self):
        """Normal 3-row 2-column float CSV loads correctly."""
        path = _write_temp_csv("a,b\n1,2\n3,4\n5,6\n")
        try:
            data = load_csv(path)
            assert data.shape == (3, 2)
            assert np.allclose(data, [[1, 2], [3, 4], [5, 6]])
        finally:
            _cleanup(path)

    def test_no_header(self):
        """Files without a header row load correctly with skip_header=False."""
        path = _write_temp_csv("1,2\n3,4\n")
        try:
            data = load_csv(path, skip_header=False)
            assert data.shape == (2, 2)
        finally:
            _cleanup(path)

    def test_tab_delimiter(self):
        """Tab-separated files load with delimiter='\\t'."""
        path = _write_temp_csv("a\tb\n1\t2\n3\t4\n")
        try:
            data = load_csv(path, delimiter="\t")
            assert data.shape == (2, 2)
            assert np.allclose(data[0], [1, 2])
        finally:
            _cleanup(path)

    def test_missing_values_become_nan(self):
        """Empty cells are filled with np.nan by default."""
        path = _write_temp_csv("a,b\n1,\n3,4\n")
        try:
            data = load_csv(path)
            assert np.isnan(data[0, 1]), "Empty cell should be NaN"
        finally:
            _cleanup(path)

    def test_custom_fill_value(self):
        """Empty cells use *fill_value* when specified."""
        path = _write_temp_csv("a,b\n1,\n3,4\n")
        try:
            data = load_csv(path, fill_value=0.0)
            assert data[0, 1] == 0.0
        finally:
            _cleanup(path)

    def test_usecols(self):
        """usecols selects a subset of columns."""
        path = _write_temp_csv("a,b,c\n1,2,3\n4,5,6\n")
        try:
            data = load_csv(path, usecols=[0, 2])
            assert data.shape == (2, 2)
            assert np.allclose(data[:, 0], [1, 4])
            assert np.allclose(data[:, 1], [3, 6])
        finally:
            _cleanup(path)

    def test_output_always_2d(self):
        """Single-column files still return a 2-D array."""
        path = _write_temp_csv("a\n1\n2\n3\n")
        try:
            data = load_csv(path)
            assert data.ndim == 2
        finally:
            _cleanup(path)

    def test_file_not_found(self):
        """FileNotFoundError is raised for non-existent paths."""
        with pytest.raises(FileNotFoundError):
            load_csv("/tmp/__nonexistent_file_xyz__.csv")


class TestGetHeaders:
    """Tests for get_headers."""

    def test_basic_headers(self):
        path = _write_temp_csv("col1,col2,col3\n1,2,3\n")
        try:
            headers = get_headers(path)
            assert headers == ["col1", "col2", "col3"]
        finally:
            _cleanup(path)

    def test_headers_stripped(self):
        """Whitespace around header names is removed."""
        path = _write_temp_csv("  a  ,  b  \n1,2\n")
        try:
            headers = get_headers(path)
            assert headers == ["a", "b"]
        finally:
            _cleanup(path)


class TestSelectColumns:
    """Tests for select_columns."""

    def setup_method(self):
        self.data = np.arange(12).reshape(3, 4).astype(float)

    def test_select_two_columns(self):
        result = select_columns(self.data, [0, 2])
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], [0, 4, 8])
        assert np.allclose(result[:, 1], [2, 6, 10])

    def test_negative_index(self):
        """Negative column indices select from the right."""
        result = select_columns(self.data, [-1])
        assert np.allclose(result[:, 0], [3, 7, 11])

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            select_columns(self.data, [10])

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            select_columns(np.array([1, 2, 3]), [0])


class TestLoadCsvChunked:
    """Tests for the chunked streaming loader."""

    def test_total_rows_match(self):
        """All rows are yielded across chunks — no rows lost or duplicated."""
        rows_content = "\n".join(f"{i},{i*2}" for i in range(1, 21))
        path = _write_temp_csv(f"a,b\n{rows_content}\n")
        try:
            total = sum(chunk.shape[0] for chunk in load_csv_chunked(path, chunk_size=7))
            assert total == 20
        finally:
            _cleanup(path)

    def test_chunk_shapes(self):
        """Each chunk has at most chunk_size rows."""
        rows_content = "\n".join(f"{i}" for i in range(1, 11))
        path = _write_temp_csv(f"a\n{rows_content}\n")
        try:
            for chunk in load_csv_chunked(path, chunk_size=3):
                assert chunk.shape[0] <= 3
        finally:
            _cleanup(path)


# ===========================================================================
# preprocessing.py tests
# ===========================================================================

class TestStandardScaler:
    """Tests for StandardScaler."""

    def setup_method(self):
        self.X = np.array([[1., 2.], [3., 4.], [5., 6.]])

    def test_zero_mean_after_fit_transform(self):
        """Column means of the scaled output are essentially zero."""
        X_s = StandardScaler().fit_transform(self.X)
        assert np.allclose(X_s.mean(axis=0), 0.0, atol=1e-10)

    def test_unit_std_after_fit_transform(self):
        """Column stds of the scaled output are essentially one."""
        X_s = StandardScaler().fit_transform(self.X)
        assert np.allclose(X_s.std(axis=0), 1.0, atol=1e-10)

    def test_inverse_transform_recovers_original(self):
        """Inverse transform restores the original values."""
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X)
        X_rec = scaler.inverse_transform(X_s)
        assert np.allclose(X_rec, self.X)

    def test_constant_column_not_nan(self):
        """A column with zero variance doesn't produce NaN or inf."""
        X = np.array([[1., 5.], [1., 3.], [1., 7.]])
        X_s = StandardScaler().fit_transform(X)
        assert np.all(np.isfinite(X_s)), "Constant column must remain finite"

    def test_with_nan_in_data(self):
        """NaNs in the input don't corrupt the column statistics."""
        X = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        scaler = StandardScaler()
        scaler.fit(X)
        # Mean for col 0 should be 3 (avg of 1, 3, 5); col 1 should be 5
        assert np.isclose(scaler.mean_[0], 3.0)
        assert np.isclose(scaler.mean_[1], 5.0)

    def test_feature_mismatch_raises(self):
        """transform() raises ValueError when feature counts differ."""
        scaler = StandardScaler().fit(self.X)
        X_bad = np.array([[1., 2., 3.]])
        with pytest.raises(ValueError, match="features"):
            scaler.transform(X_bad)

    def test_transform_before_fit_raises(self):
        """Calling transform() before fit() raises RuntimeError."""
        with pytest.raises(RuntimeError):
            StandardScaler().transform(self.X)

    def test_1d_input_accepted(self):
        """1-D input arrays are automatically reshaped to (n, 1)."""
        x = np.array([1., 2., 3., 4., 5.])
        X_s = StandardScaler().fit_transform(x)
        assert X_s.ndim == 2
        assert X_s.shape[1] == 1


class TestMinMaxScaler:
    """Tests for MinMaxScaler."""

    def setup_method(self):
        self.X = np.array([[1., 2.], [3., 4.], [5., 6.]])

    def test_min_is_zero_max_is_one(self):
        """Default range [0, 1]: column mins → 0, maxes → 1."""
        X_s = MinMaxScaler().fit_transform(self.X)
        assert np.allclose(X_s.min(axis=0), 0.0)
        assert np.allclose(X_s.max(axis=0), 1.0)

    def test_custom_range(self):
        """Custom feature_range is respected."""
        X_s = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.X)
        assert np.allclose(X_s.min(axis=0), -1.0)
        assert np.allclose(X_s.max(axis=0), 1.0)

    def test_inverse_transform(self):
        """Inverse transform recovers original data."""
        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(self.X)
        assert np.allclose(scaler.inverse_transform(X_s), self.X)

    def test_constant_column(self):
        """Constant columns don't produce NaN or inf."""
        X = np.array([[3., 1.], [3., 5.], [3., 9.]])
        X_s = MinMaxScaler().fit_transform(X)
        assert np.all(np.isfinite(X_s))

    def test_invalid_range_raises(self):
        """feature_range with min >= max raises ValueError at init."""
        with pytest.raises(ValueError):
            MinMaxScaler(feature_range=(5, 2))

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MinMaxScaler().transform(self.X)


class TestOneHotEncoder:
    """Tests for OneHotEncoder."""

    def test_basic_encode(self):
        """Three categories produce a (n, 3) output matrix."""
        X = np.array([[0], [1], [2], [1], [0]])
        out = OneHotEncoder().fit_transform(X)
        assert out.shape == (5, 3)
        # Row 0 → category 0 → first column should be 1
        assert np.allclose(out[0], [1, 0, 0])
        assert np.allclose(out[1], [0, 1, 0])
        assert np.allclose(out[2], [0, 0, 1])

    def test_drop_first(self):
        """drop_first=True removes the first dummy column per feature."""
        X = np.array([[0], [1], [2]])
        out = OneHotEncoder(drop_first=True).fit_transform(X)
        assert out.shape == (3, 2)

    def test_two_columns(self):
        """Two categorical columns produce concatenated indicator columns."""
        X = np.array([[0, 0], [1, 1], [0, 1]])
        enc = OneHotEncoder()
        out = enc.fit_transform(X)
        # Col 0 has 2 categories → 2 indicator cols;
        # Col 1 has 2 categories → 2 indicator cols; total = 4
        assert out.shape == (3, 4)

    def test_unseen_category_all_zeros(self):
        """A category not seen during fit produces an all-zero indicator row."""
        X_train = np.array([[0], [1], [2]])
        X_test = np.array([[9]])   # category 9 was not in training data
        enc = OneHotEncoder().fit(X_train)
        out = enc.transform(X_test)
        assert np.allclose(out, 0)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            OneHotEncoder().transform(np.array([[0], [1]]))


class TestSimpleImputer:
    """Tests for SimpleImputer."""

    def test_mean_strategy(self):
        """NaNs are replaced by the column mean."""
        X = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        X_out = SimpleImputer(strategy="mean").fit_transform(X)
        # Col 0 mean = 3.0; col 1 mean = 5.0
        assert X_out[0, 0] == 1.0    # not NaN, unchanged
        assert X_out[0, 1] == 5.0    # was NaN, now 5.0

    def test_median_strategy(self):
        """NaNs are replaced by the column median."""
        X = np.array([[1., np.nan], [3., 2.], [5., 8.]])
        X_out = SimpleImputer(strategy="median").fit_transform(X)
        # Col 1 non-NaN values: [2, 8] → median = 5.0
        assert X_out[0, 1] == 5.0

    def test_constant_strategy(self):
        """NaNs are replaced by a fixed constant."""
        X = np.array([[1., np.nan], [3., 4.]])
        X_out = SimpleImputer(strategy="constant", fill_value=-99.).fit_transform(X)
        assert X_out[0, 1] == -99.0

    def test_most_frequent_strategy(self):
        """NaNs are replaced by the most common value in the column."""
        X = np.array([[1., np.nan], [2., 3.], [2., 3.], [3., 5.]])
        X_out = SimpleImputer(strategy="most_frequent").fit_transform(X)
        # Col 0 mode = 2 (appears twice); so X[0, 0] is untouched (it's 1)
        # NaN is in col 1, mode of col 1 (non-NaN: 3, 3, 5) = 3
        assert X_out[0, 1] == 3.0

    def test_no_nans_unchanged(self):
        """Arrays with no NaNs are returned unchanged."""
        X = np.array([[1., 2.], [3., 4.]])
        X_out = SimpleImputer().fit_transform(X)
        assert np.allclose(X_out, X)

    def test_invalid_strategy_raises(self):
        """Unknown strategy raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            SimpleImputer(strategy="wizard")

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            SimpleImputer().transform(np.array([[1., 2.]]))

    def test_all_nan_column(self):
        """A column that is entirely NaN gets a NaN statistic (not a crash)."""
        X = np.array([[np.nan, 2.], [np.nan, 4.]])
        imp = SimpleImputer(strategy="mean").fit(X)
        # The statistic for col 0 should itself be NaN (no data to average)
        assert np.isnan(imp.statistics_[0])