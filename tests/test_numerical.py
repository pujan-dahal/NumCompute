"""
Test Module for Numerical Optimization Functions

This module contains unit tests for the numerical optimization functions
including gradient computation, Jacobian estimation, and line search.

Tests cover:
    - Gradient computation with central and forward differences
    - Jacobian matrix computation for vector-valued functions
    - Line search with Armijo condition
    - Error handling for invalid inputs
"""

import numpy as np
import pytest

from numerical import grad, jacobian, line_search


# Tests for gradient computation
def test_grad_central_quadratic():
    """Test gradient with central difference on quadratic function"""
    def f(x):
        return x[0] ** 2 + 3 * x[1] ** 2

    x = np.array([2.0, 3.0])
    result = grad(f, x, method="central")

    expected = np.array([4.0, 18.0])
    assert np.allclose(result, expected, atol=1e-4)


def test_grad_forward_quadratic():
    """Test gradient with forward difference on quadratic function"""
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([1.0, 2.0])
    result = grad(f, x, method="forward")

    expected = np.array([2.0, 4.0])
    assert np.allclose(result, expected, atol=1e-3)


def test_grad_scalar_input():
    """Test gradient with scalar input"""
    def f(x):
        return x[0] ** 2

    result = grad(f, 3.0)

    assert np.allclose(result, np.array([6.0]), atol=1e-4)


def test_grad_invalid_h():
    """Test gradient raises error for invalid step size"""
    def f(x):
        return np.sum(x)

    with pytest.raises(ValueError):
        grad(f, [1, 2], h=0)


def test_grad_invalid_method():
    """Test gradient raises error for invalid method"""
    def f(x):
        return np.sum(x)

    with pytest.raises(ValueError):
        grad(f, [1, 2], method="invalid")


# Tests for Jacobian computation
def test_jacobian_vector_function():
    """Test Jacobian for vector-valued function"""
    def F(x):
        return np.array([
            x[0] + x[1],
            x[0] * x[1]
        ])

    x = np.array([2.0, 3.0])
    result = jacobian(F, x)

    expected = np.array([
        [1.0, 1.0],
        [3.0, 2.0]
    ])

    assert np.allclose(result, expected, atol=1e-4)


def test_jacobian_scalar_output():
    """Test Jacobian for scalar output function"""
    def F(x):
        return x[0] ** 2 + x[1]

    x = np.array([3.0, 4.0])
    result = jacobian(F, x)

    expected = np.array([[6.0, 1.0]])
    assert np.allclose(result, expected, atol=1e-4)


def test_jacobian_invalid_method():
    """Test Jacobian raises error for invalid method"""
    def F(x):
        return x

    with pytest.raises(ValueError):
        jacobian(F, [1, 2], method="bad")


# Tests for line search
def test_line_search_returns_positive_alpha():
    """Test line search returns positive step size"""
    def f(x):
        return np.sum((x - 1) ** 2)

    x = np.array([0.0, 0.0])
    direction = -grad(f, x)

    alpha = line_search(f, x, direction)

    assert alpha > 0
    assert alpha <= 1.0


def test_line_search_decreases_function_value():
    """Test line search produces sufficient decrease"""
    def f(x):
        return np.sum((x - 1) ** 2)

    x = np.array([0.0, 0.0])
    direction = -grad(f, x)

    alpha = line_search(f, x, direction)

    assert f(x + alpha * direction) <= f(x)


def test_line_search_shape_mismatch():
    """Test line search raises error for shape mismatch"""
    def f(x):
        return np.sum(x ** 2)

    with pytest.raises(ValueError):
        line_search(f, np.array([1, 2]), np.array([1, 2, 3]))


def test_line_search_invalid_alpha():
    """Test line search raises error for invalid alpha"""
    def f(x):
        return np.sum(x ** 2)

    with pytest.raises(ValueError):
        line_search(f, [1, 2], [-1, -1], alpha=0)


def test_line_search_invalid_rho():
    """Test line search raises error for invalid rho"""
    def f(x):
        return np.sum(x ** 2)

    with pytest.raises(ValueError):
        line_search(f, [1, 2], [-1, -1], rho=1.5)

// 

def test_line_search_invalid_c():
    """Test line search raises error for invalid c parameter"""
    def f(x):
        return np.sum(x ** 2)

    with pytest.raises(ValueError):
        line_search(f, [1, 2], [-1, -1], c=2)

        //