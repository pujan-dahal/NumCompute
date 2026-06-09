import numpy as np
import pytest

from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier


class NoPartialFitModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class BadPredictModel:
    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([0])


def stream_data():
    X = np.array([[0.0], [0.2], [1.0], [1.2], [0.1], [1.1]])
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y


def test_stream_trainer_logs_fit_and_score_per_chunk():
    X, y = stream_data()
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=2), classes=[0, 1])
    trainer.partial_fit_score(X[:3], y[:3])
    trainer.partial_fit_score(X[3:], y[3:])
    assert trainer.logs["chunk"] == [1, 2]
    assert len(trainer.logs["cumulative_accuracy"]) == 2
    assert trainer.logs["memory_bytes"][-1] > trainer.logs["memory_bytes"][0]


def test_stream_trainer_score_chunk_returns_float_accuracy():
    X, y = stream_data()
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=2))
    trainer.fit_chunk(X, y)
    score = trainer.score_chunk(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_stream_trainer_rejects_model_without_partial_fit():
    trainer = StreamTrainer(NoPartialFitModel())
    with pytest.raises(TypeError, match="partial_fit"):
        trainer.fit_chunk([[1.0]], [0])


def test_stream_trainer_rejects_prediction_shape_mismatch():
    trainer = StreamTrainer(BadPredictModel())
    trainer.fit_chunk([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="same shape"):
        trainer.score_chunk([[0.0], [1.0]], [0, 1])


def test_stream_trainer_uses_custom_metric_window():
    from numcompute.metrics import StreamingClassificationMetric

    X, y = stream_data()
    metric = StreamingClassificationMetric(metric="accuracy", window_size=3)
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=2), metric=metric)
    trainer.partial_fit_score(X[:3], y[:3])
    trainer.partial_fit_score(X[3:], y[3:])
    assert trainer.metric.y_true_.size == 3
