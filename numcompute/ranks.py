import numpy as np

#ranking

def rank(data, method="average"):
    """
    Compute ranks of elements with tie handling.

    Parameters
    ----------
    data : np.ndarray of shape (n,)
        Input 1D array.
    method : {'average', 'dense', 'ordinal'}, optional
        Ranking method:
        - 'average': average rank for ties
        - 'dense': consecutive ranks without gaps
        - 'ordinal': unique ranks based on order

    Returns
    -------
    ranks : np.ndarray of shape (n,)
        Rank values (float for 'average', int otherwise).

    Raises
    ------
    ValueError
        If input is not 1D or method is invalid.

    Notes
    -----
    Stable sorting ensures consistent tie handling.

    Time Complexity
    ---------------
    O(n log n)

    Space Complexity
    ----------------
    O(n)
    """
    
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("rankdata only supports 1D arrays")
    if method not in ["average", "dense", "ordinal"]:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'")

    n = len(data)

    if n == 0:
        return np.array([])


    sorted_idx = np.argsort(data, kind="stable")
    sorted_data = data[sorted_idx]

    if method == "ordinal":
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)
        return ranks

    ranks = np.zeros(n, dtype=float)

    i = 0
    dense_rank = 0

    while i < n:
        j = i

        while j < n and sorted_data[j] == sorted_data[i]:
            j += 1

        if method == "average":
            rank_value = (i + j - 1) / 2.0
        else:
            rank_value = dense_rank
            dense_rank += 1

        ranks[sorted_idx[i:j]] = rank_value
        i = j


    return ranks

def rankdata(x):
    return rank(x, method="ordinal")



def rank_with_ties(x):
    return rank(x, method="average")
    


#percentiles

def percentile(x, q, interpolation="linear"):
    """
    Compute the q-th percentile of the data.

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        Input data.
    q : float
        Percentile value in range [0, 100].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}
        Interpolation method.

    Returns
    -------
    float
        Computed percentile value.

    Raises
    ------
    ValueError
        If input is invalid, empty, or contains only NaNs.

    Notes
    -----
    NaN values are ignored before computation.

    Time Complexity
    ---------------
    O(n log n)

    Space Complexity
    ----------------
    O(n)
    """
  
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("percentile only supports 1D arrays")
    if x.size == 0:
        raise ValueError("empty array")
    if not (0 <= q <= 100):
        raise ValueError("q must be between 0 and 100")
    if interpolation not in ["linear", "lower", "higher", "midpoint"]:
        raise ValueError("interpolation must be 'linear', 'lower', 'higher', or 'midpoint'")

    x = x[~np.isnan(x)]

    if x.size == 0:
        raise ValueError("array contains only NaN values")


    x_sorted = np.sort(x)
    n = len(x_sorted)

    pos = (q / 100) * (n - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if interpolation  == "lower":
        return x_sorted[lower]
    if interpolation == "higher":
        return x_sorted[upper]

    if interpolation == "midpoint":
        return (x_sorted[lower] + x_sorted[upper]) / 2.0

    if lower == upper:
        return x_sorted[lower]

    weight = pos - lower
    return (1 - weight) * x_sorted[lower] + weight * x_sorted[upper]


def percentiles(x, qs, interpolation="linear"):
    """
    Compute multiple percentiles.

    Parameters
    ----------
    x : np.ndarray of shape (n,)
    qs : array-like
        Sequence of percentile values.
    interpolation : str

    Returns
    -------
    np.ndarray
        Array of percentile values.

    Time Complexity
    ---------------
    O(m * n log n), where m = len(qs)

    Space Complexity
    ----------------
    O(m)
    """
    
    return np.array([percentile(x, q, interpolation=interpolation) for q in qs])