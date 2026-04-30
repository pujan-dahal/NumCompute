"""
Unit tests for rank.py module.

This test suite validates ranking and percentile computations
with proper handling of edge cases.

Coverage includes:
- Empty arrays
- All-equal values
- Duplicate values (tie handling)
- NaN handling
- Multiple percentile queries
- Non-contiguous arrays

Framework: pytest
"""

import numpy as np
import pytest

from numcompute.rank import rank, rankdata, rank_with_ties, percentile, percentiles


# ---------------- EMPTY ----------------

def test_empty_rank():
    """Test rank returns empty array for empty input."""
    x = np.array([])
    assert len(rank(x)) == 0


def test_empty_percentile():
    """Test percentile raises error for empty input."""
    with pytest.raises(ValueError):
        percentile(np.array([]), 50)


# ---------------- ALL EQUAL ----------------

def test_all_equal_average():
    """Test average ranking for identical values."""
    x = np.array([5, 5, 5])
    r = rank(x, method="average")
    assert np.all(r == r[0])


def test_all_equal_dense():
    """Test dense ranking for identical values."""
    x = np.array([3, 3, 3])
    r = rank(x, method="dense")
    assert np.all(r == 0)


# ---------------- TIES ----------------

def test_rank_average_ties():
    """Test average ranking assigns same rank to tied values."""
    x = np.array([1, 2, 2, 3])
    r = rank(x, method="average")
    assert r[1] == r[2]


def test_rank_dense_ties():
    """Test dense ranking produces consecutive ranks."""
    x = np.array([1, 2, 2, 3])
    r = rank(x, method="dense")
    assert set(r) == {0, 1, 2}


# ---------------- ORDINAL ----------------

def test_rank_ordinal():
    """Test ordinal ranking assigns unique ranks."""
    x = np.array([30, 10, 20])
    r = rank(x, method="ordinal")
    assert sorted(r) == [0, 1, 2]


def test_rankdata_alias():
    """Test rankdata behaves as ordinal ranking."""
    x = np.array([10, 20, 30])
    r = rankdata(x)
    assert np.all(r == [0, 1, 2])


def test_rank_with_ties_alias():
    """Test rank_with_ties behaves as average ranking."""
    x = np.array([1, 2, 2])
    r = rank_with_ties(x)
    assert r[1] == r[2]


# ---------------- NaN HANDLING ----------------

def test_percentile_with_nan():
    """Test percentile ignores NaN values."""
    x = np.array([1, 2, np.nan, 4])
    assert percentile(x, 50) == 2


def test_percentile_all_nan():
    """Test percentile raises error when all values are NaN."""
    with pytest.raises(ValueError):
        percentile(np.array([np.nan, np.nan]), 50)


# ---------------- PERCENTILE ----------------

def test_percentile_basic():
    """Test percentile computation with linear interpolation."""
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50) == 2.5


def test_percentile_lower():
    """Test percentile with 'lower' interpolation."""
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50, interpolation="lower") == 2


def test_percentile_higher():
    """Test percentile with 'higher' interpolation."""
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50, interpolation="higher") == 3


def test_percentile_midpoint():
    """Test percentile with 'midpoint' interpolation."""
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50, interpolation="midpoint") == 2.5


def test_percentiles_multiple():
    """Test multiple percentile computation."""
    x = np.array([1, 2, 3, 4])
    ps = percentiles(x, [25, 50, 75])
    assert len(ps) == 3


# ---------------- NON-CONTIGUOUS rank ----------------

def test_non_contiguous_rank():
    """Test rank works with non-contiguous arrays."""
    x = np.arange(10)[::2]
    r = rank(x)
    assert len(r) == len(x)