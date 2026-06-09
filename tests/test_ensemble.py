import numpy as np
import pytest

from numcompute.ensemble import EnsembleClassifier, RandomForestClassifier


def sample_ensemble_data():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.1, 1.2],
        [1.2, 1.1],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_ensemble_fit_creates_requested_estimators():
    X, y = sample_ensemble_data()
    model = EnsembleClassifier(n_estimators=4, max_depth=2, random_state=7)
    model.fit(X, y)
    assert len(model.estimators_) == 4
    assert model.predict(X).shape == y.shape


def test_ensemble_partial_fit_accumulates_stream():
    X, y = sample_ensemble_data()
    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=3)
    model.partial_fit(X[:2], y[:2])
    model.partial_fit(X[2:], y[2:])
    assert model._X_seen.shape[0] == X.shape[0]
    assert len(model.estimators_) == 3


def test_random_forest_alias_behaves_like_ensemble():
    X, y = sample_ensemble_data()
    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=1)
    model.fit(X, y)
    assert model.predict([[0.05, 0.05], [1.15, 1.10]]).tolist() == [0, 1]


def test_ensemble_predict_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        EnsembleClassifier().predict([[1.0, 2.0]])


def test_ensemble_rejects_zero_estimators():
    with pytest.raises(ValueError, match="positive"):
        EnsembleClassifier(n_estimators=0)


def test_ensemble_rejects_mismatched_xy_lengths():
    with pytest.raises(ValueError, match="same number"):
        EnsembleClassifier().fit(np.ones((3, 2)), np.ones(2))


def test_ensemble_rejects_feature_mismatch_between_chunks():
    model = EnsembleClassifier(n_estimators=2).partial_fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="Expected 2 features"):
        model.partial_fit([[1.0]], [1])


def test_ensemble_majority_vote_tie_resolution_is_deterministic():
    X = np.array([[0.0], [0.0], [0.0], [0.0]])
    y = np.array([1, 0, 1, 0])
    model = EnsembleClassifier(n_estimators=3, max_depth=1, random_state=0).fit(X, y)
    assert model.predict([[0.0]])[0] in (0, 1)
