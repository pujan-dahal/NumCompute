"""
Unit tests for sort_search.py module.

This test suite validates sorting, top-k selection, searching,
and selection algorithms under normal and edge-case conditions.

Coverage includes:
- Empty arrays
- All-equal values
- Duplicate values
- Extreme k values
- Non-contiguous arrays
- Correctness of indices and shapes

Framework: pytest
"""

import numpy as np
import pytest

from numcompute.sort_search import (
    stable_sort, multi_key_sort, topk, top_k, top_k_sorted,
    quickselect, binary_search
)


# ---------------- STABLE SORT ----------------

def test_stable_sort_basic():
    """Test stable_sort returns correctly sorted array."""
    x = np.array([3, 1, 2])
    assert np.all(stable_sort(x) == np.array([1, 2, 3]))


def test_stable_sort_indices():
    """Test stable_sort returns valid indices when requested."""
    x = np.array([3, 1, 2])
    vals, idx = stable_sort(x, return_indices=True)
    assert len(idx) == len(x)


# ---------------- MULTI-KEY SORT ----------------

def test_multi_key_sort_basic():
    """Test multi-key sorting on 2D array."""
    X = np.array([[1, 2], [1, 1], [0, 3]])
    sorted_X = multi_key_sort(X, keys=[0, 1])
    assert sorted_X[0][0] == 0


def test_multi_key_sort_descending():
    """Test multi-key sorting in descending order."""
    X = np.array([[1, 2], [1, 1], [0, 3]])
    sorted_X = multi_key_sort(X, keys=[0], ascending=False)
    assert sorted_X[0][0] == 1


def test_multi_key_invalid_key():
    """Test error handling for invalid column index."""
    X = np.array([[1, 2]])
    with pytest.raises(ValueError):
        multi_key_sort(X, keys=[5])


# ---------------- TOP-K (1D) ----------------

def test_topk_basic():
    """Test topk returns largest k elements correctly."""
    x = np.array([1, 5, 3, 2])
    vals, _ = topk(x, 2)
    assert np.all(vals == np.array([5, 3]))


def test_topk_smallest():
    """Test topk returns smallest k elements when largest=False."""
    x = np.array([1, 5, 3, 2])
    vals, _ = topk(x, 2, largest=False)
    assert np.all(vals == np.array([1, 2]))


def test_topk_all_equal():
    """Test topk on array with all equal values."""
    x = np.array([7, 7, 7])
    vals, _ = topk(x, 2)
    assert np.all(vals == 7)


def test_topk_invalid_k():
    """Test topk raises error for invalid k."""
    with pytest.raises(ValueError):
        topk(np.array([1, 2]), 5)


# ---------------- TOP-K (ND) ----------------

def test_top_k_axis():
    """Test top_k along specified axis."""
    X = np.array([[1, 5, 3], [2, 4, 6]])
    vals, _ = top_k(X, 2, axis=1)
    assert vals.shape == (2, 2)


def test_top_k_sorted():
    """Test top_k_sorted returns sorted results."""
    X = np.array([[1, 5, 3]])
    vals, _ = top_k_sorted(X, 2, axis=1)
    assert np.all(vals[0] == np.array([5, 3]))


# ---------------- NON-CONTIGUOUS topk----------------

def test_non_contiguous_topk():
    """Test topk works with non-contiguous (strided) arrays."""
    x = np.arange(10)[::2]
    vals, _ = topk(x, 2)
    assert len(vals) == 2


# ---------------- QUICKSELECT ----------------

def test_quickselect_basic():
    """Test quickselect returns correct k-th smallest element."""
    x = np.array([4, 2, 1, 3])
    assert quickselect(x, 2) == 3


def test_quickselect_negative():
    """Test quickselect with negative values."""
    x = np.array([-5, -1, -3])
    assert quickselect(x, 1) == -3


def test_quickselect_single():
    """Test quickselect on single-element array."""
    x = np.array([10])
    assert quickselect(x, 0) == 10


def test_quickselect_invalid_k():
    """Test quickselect raises error for invalid k."""
    with pytest.raises(ValueError):
        quickselect(np.array([1, 2]), 5)


# ---------------- BINARY SEARCH ----------------

def test_binary_search_found():
    """Test binary_search finds existing element."""
    x = np.array([1, 2, 3, 4])
    _, found = binary_search(x, 3)
    assert found is True


def test_binary_search_not_found():
    """Test binary_search correctly reports missing element."""
    x = np.array([1, 2, 4])
    _, found = binary_search(x, 3)
    assert found is False


def test_binary_search_insertion_index():
    """Test binary_search returns correct insertion index."""
    x = np.array([1, 2, 4])
    idx, _ = binary_search(x, 3)
    assert idx == 2


def test_binary_search_empty():
    """Test binary_search on empty array."""
    x = np.array([])
    _, found = binary_search(x, 1)
    assert found is False