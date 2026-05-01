import numpy as np
import pytest

from numcompute.utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    sigmoid,
    relu,
    softmax,
    logsumexp,
    top_k_indices,
    top_k_values,
    make_batches,
    make_batches_xy,
)


def test_euclidean_distance():
    assert np.isclose(euclidean_distance([0, 0], [3, 4]), 5.0)


def test_manhattan_distance():
    assert np.isclose(manhattan_distance([1, 2, 3], [4, 0, 6]), 8.0)


def test_distance_shape_mismatch():
    with pytest.raises(ValueError):
        euclidean_distance([1, 2], [1, 2, 3])


def test_cosine_similarity_same_direction():
    assert np.isclose(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)


def test_cosine_similarity_opposite_direction():
    assert np.isclose(cosine_similarity([1, 0], [-1, 0]), -1.0)


def test_cosine_similarity_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])


def test_sigmoid():
    result = sigmoid(np.array([0.0]))
    assert np.allclose(result, np.array([0.5]))


def test_relu():
    result = relu(np.array([-2, 0, 3]))
    expected = np.array([0, 0, 3])
    assert np.array_equal(result, expected)


def test_softmax_sum_is_one():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert np.isclose(np.sum(result), 1.0)


def test_softmax_2d_axis():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = softmax(x, axis=1)
    assert np.allclose(np.sum(result, axis=1), np.array([1.0, 1.0]))


def test_softmax_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)
    assert np.isclose(np.sum(result), 1.0)
    assert not np.any(np.isnan(result))


def test_logsumexp():
    x = np.array([1.0, 2.0, 3.0])
    result = logsumexp(x)
    expected = np.log(np.sum(np.exp(x)))
    assert np.allclose(result, expected)


def test_logsumexp_axis():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = logsumexp(x, axis=1)
    expected = np.log(np.sum(np.exp(x), axis=1))
    assert np.allclose(result, expected)


def test_top_k_indices_largest():
    x = np.array([10, 5, 30, 20])
    result = top_k_indices(x, 2, largest=True)
    expected = np.array([2, 3])
    assert np.array_equal(result, expected)


def test_top_k_indices_smallest():
    x = np.array([10, 5, 30, 20])
    result = top_k_indices(x, 2, largest=False)
    expected = np.array([1, 0])
    assert np.array_equal(result, expected)


def test_top_k_values_largest():
    x = np.array([10, 5, 30, 20])
    result = top_k_values(x, 2)
    expected = np.array([30, 20])
    assert np.array_equal(result, expected)


def test_top_k_invalid_k():
    with pytest.raises(ValueError):
        top_k_indices([1, 2, 3], 0)


def test_top_k_non_1d_array():
    with pytest.raises(ValueError):
        top_k_indices(np.array([[1, 2], [3, 4]]), 2)


def test_make_batches():
    X = np.array([1, 2, 3, 4, 5])
    batches = list(make_batches(X, batch_size=2))

    assert len(batches) == 3
    assert np.array_equal(batches[0], np.array([1, 2]))
    assert np.array_equal(batches[1], np.array([3, 4]))
    assert np.array_equal(batches[2], np.array([5]))


def test_make_batches_invalid_batch_size():
    with pytest.raises(ValueError):
        list(make_batches([1, 2, 3], batch_size=0))


def test_make_batches_xy():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([10, 20, 30, 40])

    batches = list(make_batches_xy(X, y, batch_size=2))

    assert len(batches) == 2
    assert np.array_equal(batches[0][0], np.array([[1], [2]]))
    assert np.array_equal(batches[0][1], np.array([10, 20]))


def test_make_batches_xy_length_mismatch():
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20])

    with pytest.raises(ValueError):
        list(make_batches_xy(X, y, batch_size=2))


def test_make_batches_shuffle_reproducible():
    X = np.array([1, 2, 3, 4, 5])

    batches1 = list(make_batches(X, batch_size=2, shuffle=True, random_state=42))
    batches2 = list(make_batches(X, batch_size=2, shuffle=True, random_state=42))

    joined1 = np.concatenate(batches1)
    joined2 = np.concatenate(batches2)

    assert np.array_equal(joined1, joined2)
