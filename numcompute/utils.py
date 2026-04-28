import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    """
    Compute Manhattan distance between two vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    return np.sum(np.abs(a - b))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
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
    Compute sigmoid activation.
    """
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Compute ReLU activation.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)


def softmax(x, axis=-1):
    """
    Compute softmax in a numerically stable way.
    """
    x = np.asarray(x, dtype=float)

    # Shift by max value for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def logsumexp(x, axis=None, keepdims=False):
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    """
    x = np.asarray(x, dtype=float)

    max_x = np.max(x, axis=axis, keepdims=True)
    out = max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))

    if not keepdims:
        out = np.squeeze(out, axis=axis)

    return out