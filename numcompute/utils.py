import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors

    Parameters
    ----------
    a b : array like
        Input vectors with the same shape

    Returns
    -------
    float
        Square root of the sum of squared differences

    Raises
    ------
    ValueError
        If a and b do not have the same shape

    Complexity
    ----------
    Time O n
    Space O n
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    """
    Compute Manhattan distance between two vectors

    Parameters
    ----------
    a b : array like
        Input vectors with the same shape

    Returns
    -------
    float
        Sum of absolute differences

    Raises
    ------
    ValueError
        If a and b do not have the same shape

    Complexity
    ----------
    Time O n
    Space O n
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    return np.sum(np.abs(a - b))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors

    Parameters
    ----------
    a b : array like
        Input vectors with the same shape

    Returns
    -------
    float
        Cosine similarity between a and b

    Raises
    ------
    ValueError
        If shapes do not match or either vector has zero length

    Complexity
    ----------
    Time O n
    Space O n
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for zero vectors")

    return np.dot(a, b) / (norm_a * norm_b)


def sigmoid(x):
    """
    Compute sigmoid activation

    Parameters
    ----------
    x : array like
        Input values with any shape

    Returns
    -------
    numpy.ndarray
        Sigmoid values with the same shape as x

    Complexity
    ----------
    Time O n
    Space O n
    """
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Compute ReLU activation

    Parameters
    ----------
    x : array like
        Input values with any shape

    Returns
    -------
    numpy.ndarray
        ReLU values with the same shape as x

    Complexity
    ----------
    Time O n
    Space O n
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)


def softmax(x, axis=-1):
    """
    Compute softmax in a numerically stable way

    Parameters
    ----------
    x : array like
        Input values with any shape
    axis : int
        Axis used for the softmax calculation

    Returns
    -------
    numpy.ndarray
        Softmax values with the same shape as x

    Notes
    -----
    The maximum value is subtracted before exponentiation to reduce overflow

    Complexity
    ----------
    Time O n
    Space O n
    """
    x = np.asarray(x, dtype=float)

    # Shift by max value for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def logsumexp(x, axis=None, keepdims=False):
    """
    Compute log sum exp in a numerically stable way

    Parameters
    ----------
    x : array like
        Input values with any shape
    axis : int or None
        Axis used for the calculation
    keepdims : bool
        If true the reduced axis is kept in the output

    Returns
    -------
    numpy.ndarray or float
        Log sum exp result

    Notes
    -----
    The maximum value is subtracted before exponentiation to reduce overflow

    Complexity
    ----------
    Time O n
    Space O n
    """
    x = np.asarray(x, dtype=float)

    max_x = np.max(x, axis=axis, keepdims=True)
    out = max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))

    if not keepdims:
        out = np.squeeze(out, axis=axis)

    return out

def top_k_indices(x, k, largest=True):
    """
    Return indices of the top k values

    Parameters
    ----------
    x : array like
        One dimensional input array
    k : int
        Number of values to return
    largest : bool
        If true return largest values otherwise return smallest values

    Returns
    -------
    numpy.ndarray
        Indices of the selected values with shape k

    Raises
    ------
    ValueError
        If x is not one dimensional or k is out of range

    Complexity
    ----------
    Time O n average because argpartition is used
    Space O k plus partition workspace
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array")

    if k <= 0 or k > len(x):
        raise ValueError("k must be between 1 and len(x)")

    if largest:
        idx = np.argpartition(x, -k)[-k:]
        idx = idx[np.argsort(x[idx])[::-1]]
    else:
        idx = np.argpartition(x, k - 1)[:k]
        idx = idx[np.argsort(x[idx])]

    return idx


def top_k_values(x, k, largest=True):
    """
    Return the top k values

    Parameters
    ----------
    x : array like
        One dimensional input array
    k : int
        Number of values to return
    largest : bool
        If true return largest values otherwise return smallest values

    Returns
    -------
    numpy.ndarray
        Selected values with shape k

    Raises
    ------
    ValueError
        If x is not one dimensional or k is out of range

    Complexity
    ----------
    Time O n average because argpartition is used
    Space O k plus partition workspace
    """
    x = np.asarray(x)

    idx = top_k_indices(x, k, largest=largest)
    return x[idx]


def make_batches(X, batch_size, shuffle=False, random_state=None):
    """
    Yield mini batches from input data

    Parameters
    ----------
    X : array like
        Input data where the first axis is samples
    batch_size : int
        Number of rows in each batch
    shuffle : bool
        If true rows are shuffled before batching
    random_state : int or None
        Seed for reproducible shuffling

    Yields
    ------
    numpy.ndarray
        Batch of X with at most batch_size rows

    Raises
    ------
    ValueError
        If batch_size is not positive

    Complexity
    ----------
    Time O n
    Space O n for indices
    """
    X = np.asarray(X)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(X))

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx]


def make_batches_xy(X, y, batch_size, shuffle=False, random_state=None):
    """
    Yield mini batches from X and y together

    Parameters
    ----------
    X : array like
        Input data where the first axis is samples
    y : array like
        Target values with the same number of rows as X
    batch_size : int
        Number of rows in each batch
    shuffle : bool
        If true rows are shuffled before batching
    random_state : int or None
        Seed for reproducible shuffling

    Yields
    ------
    tuple
        Batch of X and matching batch of y

    Raises
    ------
    ValueError
        If X and y row counts differ or batch_size is not positive

    Complexity
    ----------
    Time O n
    Space O n for indices
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(X))

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
