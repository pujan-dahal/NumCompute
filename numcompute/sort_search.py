import numpy as np

#Top-K

def top_k(X, k, axis=-1):
    """
    Return the top-k values and their indices along a given axis.

    Parameters
    ----------
    X : np.ndarray
        Input array.
    k : int
        Number of top elements to select.
    axis : int, optional
        Axis along which to select top-k.

    Returns
    -------
    values : np.ndarray
        Top-k values (unsorted).
    indices : np.ndarray
        Indices of top-k values.

    Raises
    ------
    ValueError
        If k is invalid.
    """
    X = np.asarray(X)

    if k <= 0:
        raise ValueError("k must be positive")
    if k > X.shape[axis]:
        raise ValueError("k cannot be greater than axis length")

    idx = np.argpartition(X, -k, axis=axis)[..., -k:]
    values = np.take_along_axis(X, idx, axis=axis)

    return values, idx


def top_k_sorted(X, k, axis=-1):
    """
    Return the top-k values sorted in descending order.

    Returns
    -------
    values_sorted : np.ndarray
    indices_sorted : np.ndarray
    """
    values, idx = top_k(X, k, axis)

    order = np.argsort(-values, axis=axis)
    values_sorted = np.take_along_axis(values, order, axis=axis)
    idx_sorted = np.take_along_axis(idx, order, axis=axis)

    return values_sorted, idx_sorted


#Quickselect

def quickselect(arr, k):
    """
    Select the k-th smallest element (0-based index).

    Parameters
    ----------
    arr : np.ndarray
    k : int

    Returns
    -------
    float or int

    Notes
    -----
    Average complexity: O(n)
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ValueError("quickselect only supports 1D arrays")
    if k < 0 or k >= len(arr):
        raise ValueError("k out of bounds")

    while True:
        if len(arr) == 1:
            return arr[0]

        pivot = arr[len(arr) // 2]

        lows = arr[arr < pivot]
        highs = arr[arr > pivot]
        pivots = arr[arr == pivot]

        if k < len(lows):
            arr = lows
        elif k < len(lows) + len(pivots):
            return pivot
        else:
            k = k - len(lows) - len(pivots)
            arr = highs



# Binary Search

def binary_search(arr, target):
    """
    Perform binary search on a sorted array.

    Parameters
    ----------
    arr : np.ndarray
        Sorted 1D array.
    target : scalar

    Returns
    -------
    int
        Index of target if found, else -1.
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ValueError("binary_search only supports 1D arrays")

    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1