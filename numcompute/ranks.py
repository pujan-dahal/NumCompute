import numpy as np

#ranking

def rankdata(x):
    """
    Assign ranks to data without tie handling (ordinal ranking).

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
        Ranks (0-based).
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("rankdata only supports 1D arrays")

    temp = np.argsort(x)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))

    return ranks


def rank_with_ties(x):
    """
    Assign ranks with average tie handling.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
        Ranks (float).

    Example
    -------
    [10, 20, 20, 40] -> [0, 1.5, 1.5, 3]
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("rank_with_ties only supports 1D arrays")

    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]

    ranks = np.zeros(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1

        avg_rank = (i + j - 1) / 2.0
        ranks[sorted_idx[i:j]] = avg_rank

        i = j

    return ranks

#percentiles

def percentile(x, q):
    """
    Compute the q-th percentile.

    Parameters
    ----------
    x : np.ndarray
    q : float (0-100)

    Returns
    -------
    float
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("percentile only supports 1D arrays")
    if not (0 <= q <= 100):
        raise ValueError("q must be between 0 and 100")

    x_sorted = np.sort(x)
    n = len(x_sorted)

    if n == 0:
        raise ValueError("empty array")

    pos = (q / 100) * (n - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if lower == upper:
        return x_sorted[lower]

    weight = pos - lower
    return (1 - weight) * x_sorted[lower] + weight * x_sorted[upper]


def percentiles(x, qs):
    """
    Compute multiple percentiles.

    Parameters
    ----------
    x : np.ndarray
    qs : array-like

    Returns
    -------
    np.ndarray
    """
    return np.array([percentile(x, q) for q in qs])