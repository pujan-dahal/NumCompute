import numpy as np

from numcompute.ensemble import EnsembleClassifier
from numcompute.metrics import StreamingClassificationMetric
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder
from numcompute.stats import StreamingStats
from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier


def make_data():
    X = np.array([
        [0.0, 0.1], [0.2, 0.0], [1.0, 1.2], [1.1, 0.9],
        [0.1, 0.2], [1.2, 1.1], [0.3, 0.1], [1.3, 1.4],
    ])
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    return X, y


def test_decision_tree_fit_predict_training_data():
    X, y = make_data()
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.mean(pred == y) >= 0.75


def test_decision_tree_partial_fit_two_chunks():
    X, y = make_data()
    model = DecisionTreeClassifier(max_depth=2)
    model.partial_fit(X[:4], y[:4])
    model.partial_fit(X[4:], y[4:])
    assert model._X_seen.shape[0] == X.shape[0]
    assert model.predict(X).shape == y.shape


def test_decision_tree_rejects_bad_shapes():
    model = DecisionTreeClassifier()
    try:
        model.fit(np.ones((3, 2)), np.ones(2))
    except ValueError as exc:
        assert "same number" in str(exc)
    else:
        raise AssertionError("ValueError not raised")


def test_decision_tree_predict_before_fit_raises():
    model = DecisionTreeClassifier()
    try:
        model.predict([[1, 2]])
    except RuntimeError:
        assert True
    else:
        raise AssertionError("RuntimeError not raised")


def test_ensemble_partial_fit_predict():
    X, y = make_data()
    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=0)
    model.partial_fit(X[:4], y[:4])
    model.partial_fit(X[4:], y[4:])
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert len(model.estimators_) == 3


def test_ensemble_rejects_zero_estimators():
    try:
        EnsembleClassifier(n_estimators=0)
    except ValueError:
        assert True
    else:
        raise AssertionError("ValueError not raised")


def test_streaming_metric_accuracy_and_reset():
    metric = StreamingClassificationMetric(metric="accuracy")
    metric.update([0, 1], [0, 0])
    metric.update([1, 1], [1, 1])
    assert metric.result() == 0.75
    metric.reset()
    assert metric.result() == 0.0


def test_streaming_metric_window():
    metric = StreamingClassificationMetric(metric="accuracy", window_size=2)
    metric.update([0, 1, 1], [1, 1, 1])
    assert metric.result() == 1.0


def test_streaming_confusion_matrix_multiclass():
    metric = StreamingClassificationMetric()
    metric.update([0, 1, 2], [0, 2, 2])
    matrix = metric.confusion_matrix(labels=[0, 1, 2])
    assert matrix.shape == (3, 3)
    assert matrix[1, 2] == 1


def test_standard_scaler_partial_fit_matches_full_mean():
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    scaler = StandardScaler()
    scaler.partial_fit(X[:1])
    scaler.partial_fit(X[1:])
    assert np.allclose(scaler.mean_, np.nanmean(X, axis=0))


def test_simple_imputer_partial_fit_mean():
    X = np.array([[1.0, np.nan], [3.0, 5.0]])
    imputer = SimpleImputer(strategy="mean")
    imputer.partial_fit(X[:1])
    imputer.partial_fit(X[1:])
    out = imputer.transform(X)
    assert not np.isnan(out).any()
    assert out[0, 1] == 5.0


def test_onehot_partial_fit_expands_categories():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([['a'], ['b']], dtype=object))
    enc.partial_fit(np.array([['c']], dtype=object))
    out = enc.transform(np.array([['a'], ['c']], dtype=object))
    assert out.shape == (2, 3)


def test_streaming_stats_mean_variance_quantile():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]])
    stats = StreamingStats()
    stats.update_stats(X[:2])
    stats.update_stats(X[2:])
    assert np.allclose(stats.mean, np.nanmean(X, axis=0))
    assert stats.variance().shape == (2,)
    assert stats.quantile(0.5).shape == (2,)


def test_pipeline_partial_fit_with_tree():
    X, y = make_data()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])
    pipe.partial_fit(X[:4], y[:4])
    pipe.partial_fit(X[4:], y[4:])
    pred = pipe.predict(X)
    assert pred.shape == y.shape


def test_stream_trainer_logs_chunks():
    X, y = make_data()
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=2), classes=[0, 1])
    trainer.partial_fit_score(X[:4], y[:4])
    trainer.partial_fit_score(X[4:], y[4:])
    assert len(trainer.logs["chunk_accuracy"]) == 2
    assert trainer.logs["memory_bytes"][-1] > 0
