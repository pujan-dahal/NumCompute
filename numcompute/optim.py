import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """
    Compute the gradient of a scalar function using finite differences.
    """
    x = np.asarray(x, dtype=float)

    # Convert scalar input into a 1D array
    if x.ndim == 0:
        x = x.reshape(1)

    if h <= 0:
        raise ValueError("h must be positive")

    if method not in ["central", "forward"]:
        raise ValueError("method must be 'central' or 'forward'")

    # Array to store gradient values
    g = np.zeros_like(x, dtype=float)

    # For forward difference, f(x) is reused for every dimension
    if method == "forward":
        fx = f(x)

    # Change one variable at a time and approximate partial derivative
    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += h

        if method == "central":
            x2 = x.copy()
            x2[i] -= h
            g[i] = (f(x1) - f(x2)) / (2 * h)
        else:
            g[i] = (f(x1) - fx) / h

    return g