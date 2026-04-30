import numpy as np

#sort

def stable_sort(a, axis=-1, return_indices=False):

    """
    Perform a stable sort along a specified axis.

    Parameters
    ----------
    a : np.ndarray
        Input array of arbitrary shape.
    axis : int, optional (default=-1)
        Axis along which to sort.
    return_indices : bool, optional (default=False)
        If True, also return the indices that would sort the array.

    Returns
    -------
    sorted_array : np.ndarray
        Array sorted along the specified axis.
    indices : np.ndarray, optional
        Indices of the sorted elements (only if return_indices=True).

    Raises
    ------
    ValueError
        If axis is invalid.

    Notes
    -----
    Uses NumPy's stable sorting algorithm (`kind='stable'`).

    Time Complexity
    ---------------
    O(n log n)

    Space Complexity
    ----------------
    O(n)
    """
    a = np.asarray(a)

    if return_indices:
        indices = np.argsort(a, axis=axis, kind="stable")
        values = np.take_along_axis(a, indices, axis=axis)
        return values, indices

    return np.sort(a, axis=axis, kind="stable")


def multi_key_sort(X, keys, ascending=True, return_indices=False):
    """
    Sort rows of a 2D array based on multiple column keys.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input 2D array.
    keys : int or list of int
        Column indices to sort by (priority order).
    ascending : bool or list of bool, optional
        Sort order for each key. If a single bool is provided, it is applied to all keys.
    return_indices : bool, optional
        If True, return the row indices used for sorting.

    Returns
    -------
    sorted_X : np.ndarray
        Sorted array.
    indices : np.ndarray, optional
        Indices of rows after sorting.

    Raises
    ------
    ValueError
        If X is not 2D, keys are invalid, or ascending length mismatches keys.

    Notes
    -----
    Sorting is stable across keys (last key has lowest priority).

    Time Complexity
    ---------------
    O(k * n log n), where k = number of keys

    Space Complexity
    ----------------
    O(n)
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if isinstance(keys, int):
        keys = [keys]

    if len(keys) == 0:
        raise ValueError("keys must not be empty")

    for key in keys:
        if key < 0 or key >= X.shape[1]:
            raise ValueError("key index out of range")

    if isinstance(ascending, bool):
        ascending = [ascending] * len(keys)

    if len(ascending) != len(keys):
        raise ValueError("ascending must match keys")

    order = np.arange(X.shape[0])

    for key, asc in reversed(list(zip(keys, ascending))):
        values = X[order, key]
        sorted_idx = np.argsort(values, kind="stable")

        if not asc:
            sorted_idx = sorted_idx[::-1]

        order = order[sorted_idx]

    sorted_X = X[order]

    if return_indices:
        return sorted_X, order

    return sorted_X


def topk(values, k, largest=True, return_indices=True):
    """
    Select the top-k elements from a 1D array using partial sorting.

    Parameters
    ----------
    values : np.ndarray of shape (n,)
        Input array.
    k : int
        Number of elements to select.
    largest : bool, optional (default=True)
        If True, select largest k elements; otherwise smallest k.
    return_indices : bool, optional
        If True, return indices along with values.

    Returns
    -------
    selected_values : np.ndarray of shape (k,)
        Top-k values (sorted).
    indices : np.ndarray of shape (k,), optional
        Indices of selected values.

    Raises
    ------
    ValueError
        If input is not 1D or k is invalid.

    Notes
    -----
    Uses `np.argpartition` for efficient partial sorting.

    Time Complexity
    ---------------
    O(n)

    Space Complexity
    ----------------
    O(k)
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    if k <= 0 or k > len(values):
        raise ValueError("k must be between 1 and len(values)")

    if largest:
        indices = np.argpartition(values, -k)[-k:]
        indices = indices[np.argsort(values[indices])[::-1]]
    else:
        indices = np.argpartition(values, k - 1)[:k]
        indices = indices[np.argsort(values[indices])]

    selected_values = values[indices]

    if return_indices:
        return selected_values, indices

    return selected_values


#Top-K

def top_k(X, k, axis=-1):
    """
    Compute top-k values along a specified axis.

    Parameters
    ----------
    X : np.ndarray
        Input array of arbitrary shape.
    k : int
        Number of elements to select.
    axis : int, optional
        Axis along which to compute top-k.

    Returns
    -------
    values : np.ndarray
        Top-k values (unsorted).
    indices : np.ndarray
        Indices of the selected values.

    Raises
    ------
    ValueError
        If k is invalid.

    Time Complexity
    ---------------
    O(n)

    Space Complexity
    ----------------
    O(k)
    """
    
    X = np.asarray(X)

    if k <= 0:
        raise ValueError("k must be positive")
    if k > X.shape[axis]:
        raise ValueError("k cannot be greater than axis length")

    indices  = np.argpartition(X, -k, axis=axis)
    slicer = [slice(None)] * X.ndim
    slicer[axis] = slice(-k, None)

    indices = indices[tuple(slicer)]

    values = np.take_along_axis(X, indices, axis=axis)

    return values, indices


def top_k_sorted(X, k, axis=-1):
    """
    Compute sorted top-k values along a specified axis.

    Parameters
    ----------
    X : np.ndarray
    k : int
    axis : int

    Returns
    -------
    values_sorted : np.ndarray
        Top-k values sorted in descending order.
    indices_sorted : np.ndarray
        Corresponding indices.

    Time Complexity
    ---------------
    O(n + k log k)

    Space Complexity
    ----------------
    O(k)
    """
    
    values, indices  = top_k(X, k, axis=axis)

    order = np.argsort(-values, axis=axis)
    values_sorted = np.take_along_axis(values, order, axis=axis)
    indices_sorted  = np.take_along_axis(indices, order, axis=axis)

    return values_sorted, indices_sorted 


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
        The k-th smallest element.
    Parameters
    ----------
    arr : np.ndarray
    k : int

    Notes
    -----
    Average complexity: O(n)
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ValueError("quickselect only supports 1D arrays")
    if k < 0 or k >= len(arr):
        raise ValueError("k out of bounds")
    arr = arr.copy()

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
    arr : np.ndarray of shape (n,)
        Sorted 1D array.
    target : scalar
        Value to search for.

    Returns
    -------
    insertion_index : int
        Index where the target is found or should be inserted.
    found : bool
        True if target exists in the array, else False.

    Raises
    ------
    ValueError
        If input is not 1D.

    Notes
    -----
    Uses lower-bound search.

    Time Complexity
    ---------------
    O(log n)

    Space Complexity
    ----------------
    O(1)
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ValueError("binary_search only supports 1D arrays")

    left = 0
    right = len(arr)

    while left < right:
        mid = (left + right) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    insertion_index = left
    found = (insertion_index < len(arr) and bool (arr[insertion_index] == target))


    return insertion_index, found