import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar function using finite differences

    Parameters
    ----------
    f : callable
        Function that takes x and returns one number
    x : array like
        Point where the gradient is calculated
    h : float
        Step size used for the finite difference
    method : str
        Either central or forward

    Returns
    -------
    numpy.ndarray
        Gradient values with the same length as x

    Raises
    ------
    ValueError
        If h is not positive or method is invalid

    Complexity
    ----------
    Time O n function evaluations
    Space O n
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


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian matrix of a vector function using finite differences

    Parameters
    ----------
    F : callable
        Function that takes x and returns one or more values
    x : array like
        Point where the Jacobian is calculated
    h : float
        Step size used for the finite difference
    method : str
        Either central or forward

    Returns
    -------
    numpy.ndarray
        Matrix with shape n_outputs by n_inputs

    Raises
    ------
    ValueError
        If h is not positive or method is invalid

    Complexity
    ----------
    Time O n function evaluations where n is the number of inputs
    Space O m n where m is outputs and n is inputs
    """
    x = np.asarray(x, dtype=float)

    # Convert scalar input into 1D array
    if x.ndim == 0:
        x = x.reshape(1)

    if h <= 0:
        raise ValueError("h must be positive")

    if method not in ["central", "forward"]:
        raise ValueError("method must be 'central' or 'forward'")

    # Evaluate function once to know output size
    y = np.asarray(F(x), dtype=float)

    if y.ndim == 0:
        y = y.reshape(1)

    # Jacobian has rows = outputs and columns = inputs
    J = np.zeros((len(y), len(x)), dtype=float)

    # For forward difference, F(x) is reused
    if method == "forward":
        Fx = y

    # Change one input variable at a time
    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += h

        if method == "central":
            x2 = x.copy()
            x2[i] -= h
            J[:, i] = (np.asarray(F(x1), dtype=float) - np.asarray(F(x2), dtype=float)) / (2 * h)
        else:
            J[:, i] = (np.asarray(F(x1), dtype=float) - Fx) / h

    return J


def line_search(f, x, direction, alpha=1.0, rho=0.5, c=1e-4, max_iter=50):
    """
    Find a step size using backtracking line search

    Parameters
    ----------
    f : callable
        Objective function that takes x and returns one number
    x : array like
        Starting point
    direction : array like
        Direction to move from x
    alpha : float
        Starting step size
    rho : float
        Amount used to shrink alpha each time
    c : float
        Armijo condition constant
    max_iter : int
        Maximum number of shrink steps

    Returns
    -------
    float
        Step size accepted by the search

    Raises
    ------
    ValueError
        If shapes do not match or search settings are invalid

    Complexity
    ----------
    Time O max_iter times gradient cost
    Space O n
    """
    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)

    # Convert scalar inputs into 1D arrays
    if x.ndim == 0:
        x = x.reshape(1)
    if direction.ndim == 0:
        direction = direction.reshape(1)

    if x.shape != direction.shape:
        raise ValueError("x and direction must have the same shape")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if not (0 < rho < 1):
        raise ValueError("rho must be between 0 and 1")

    if not (0 < c < 1):
        raise ValueError("c must be between 0 and 1")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    # Compute gradient at current point
    g = grad(f, x)
    fx = f(x)

    # Directional derivative
    slope = np.dot(g, direction)

    # Reduce step size until Armijo condition is satisfied
    for _ in range(max_iter):
        x_new = x + alpha * direction
        if f(x_new) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho

    return alpha
