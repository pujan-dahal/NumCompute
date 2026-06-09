import numpy as np
import pytest

from numcompute.metrics import StreamingClassificationMetric


def test_streaming_accuracy_accumulates_multiple_chunks():
    metric = StreamingClassificationMetric(metric="accuracy")
    metric.update([0, 1], [0, 0]).update([1, 1], [1, 1])
    assert metric.result() == 0.75


def test_streaming_precision_recall_and_f1_handle_zero_division():
    precision = StreamingClassificationMetric(metric="precision", positive_label=1).update([0, 0], [1, 1])
    recall = StreamingClassificationMetric(metric="recall", positive_label=1).update([0, 0], [0, 0])
    f1 = StreamingClassificationMetric(metric="f1", positive_label=1).update([0, 0], [0, 0])
    assert precision.result() == 0.0
    assert recall.result() == 0.0
    assert f1.result() == 0.0


def test_streaming_metric_rolling_window_keeps_recent_samples_only():
    metric = StreamingClassificationMetric(metric="accuracy", window_size=3)
    metric.update([0, 0, 0], [0, 0, 0])
    metric.update([1, 1], [0, 1])
    assert metric.y_true_.tolist() == [0, 1, 1]
    assert metric.result() == 2 / 3


def test_streaming_metric_confusion_matrix_multiclass():
    metric = StreamingClassificationMetric().update(["a", "b", "c"], ["a", "c", "c"])
    cm = metric.confusion_matrix(labels=["a", "b", "c"])
    assert cm.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_streaming_metric_reset_returns_zero_result():
    metric = StreamingClassificationMetric().update([1], [1])
    metric.reset()
    assert metric.result() == 0.0


def test_streaming_metric_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric"):
        StreamingClassificationMetric(metric="auc")


def test_streaming_metric_rejects_mismatched_lengths():
    metric = StreamingClassificationMetric()
    with pytest.raises(ValueError):
        metric.update([0, 1], [0])
