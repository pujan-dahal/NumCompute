import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier


def sample_tree_data():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 1.2],
        [0.2, 0.1],
        [1.2, 1.1],
    ])
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y


def test_tree_fit_predicts_training_shape():
    X, y = sample_tree_data()
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape
    assert np.mean(pred == y) >= 0.8


def test_tree_partial_fit_accumulates_chunks():
    X, y = sample_tree_data()
    clf = DecisionTreeClassifier(max_depth=2)
    clf.partial_fit(X[:3], y[:3], classes=[0, 1])
    clf.partial_fit(X[3:], y[3:], classes=[0, 1])
    assert clf._X_seen.shape == X.shape
    assert set(clf.classes_.tolist()) == {0, 1}


def test_tree_entropy_criterion_runs():
    X, y = sample_tree_data()
    clf = DecisionTreeClassifier(max_depth=3, criterion="entropy")
    clf.fit(X, y)
    assert clf.predict([[0.05, 0.05]])[0] == 0


def test_tree_max_depth_zero_uses_majority_leaf():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0, 1])
    clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert np.all(clf.predict([[0.0], [9.0]]) == 1)


def test_tree_handles_nan_features_without_crashing():
    X = np.array([[0.0], [np.nan], [1.0], [1.2], [0.2]])
    y = np.array([0, 1, 1, 1, 0])
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    pred = clf.predict([[np.nan], [0.1]])
    assert pred.shape == (2,)


def test_tree_tie_resolution_is_deterministic():
    X = np.array([[0.0], [0.0], [0.0], [0.0]])
    y = np.array([1, 0, 1, 0])
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.predict([[0.0]])[0] == 0


def test_tree_rejects_bad_criterion():
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion="misclassification")


def test_tree_rejects_feature_mismatch_in_partial_fit():
    clf = DecisionTreeClassifier().partial_fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="Expected 2 features"):
        clf.partial_fit([[1.0]], [0])


def test_tree_rejects_empty_training_data():
    with pytest.raises(ValueError, match="empty"):
        DecisionTreeClassifier().fit(np.empty((0, 2)), np.array([]))


def test_tree_predict_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict([[1.0, 2.0]])
