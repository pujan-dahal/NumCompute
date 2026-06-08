"""
Streaming training helpers for NumCompute.

StreamTrainer coordinates a model, optional pipeline steps and metric logging
for chunk-wise learning experiments.
"""

import sys
import numpy as np

from numcompute.metrics import StreamingClassificationMetric


class StreamTrainer:
    """
    Manage chunk-wise model training and logging.

    Parameters
    ----------
    model : object
        Estimator implementing partial_fit and predict.
    metric : object or None, default None
        Streaming metric object with update and result methods.
    classes : array like or None, default None
        Optional fixed class labels passed to compatible estimators.
    """

    def __init__(self, model, metric=None, classes=None):
        self.model = model
        self.metric = metric if metric is not None else StreamingClassificationMetric(metric="accuracy")
        self.classes = None if classes is None else np.asarray(classes)
        self.logs = {
            "chunk": [],
            "chunk_accuracy": [],
            "cumulative_accuracy": [],
            "memory_bytes": [],
        }
        self._chunk_index = 0

    def fit_chunk(self, X, y):
        """Fit the model on one stream chunk."""
        if not hasattr(self.model, "partial_fit"):
            raise TypeError("model must implement partial_fit(X, y)")
        try:
            self.model.partial_fit(X, y, classes=self.classes)
        except TypeError:
            self.model.partial_fit(X, y)
        return self

    def score_chunk(self, X, y):
        """Predict and log performance for one stream chunk."""
        y_pred = self.model.predict(X)
        chunk_accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y)))
        self.metric.update(y, y_pred)
        cumulative_accuracy = float(self.metric.result())
        self._chunk_index += 1
        self.logs["chunk"].append(self._chunk_index)
        self.logs["chunk_accuracy"].append(chunk_accuracy)
        self.logs["cumulative_accuracy"].append(cumulative_accuracy)
        self.logs["memory_bytes"].append(self._estimate_memory())
        return chunk_accuracy

    def partial_fit_score(self, X, y):
        """Fit one chunk and score it immediately after the update."""
        self.fit_chunk(X, y)
        return self.score_chunk(X, y)

    def _estimate_memory(self):
        total = sys.getsizeof(self.model)
        for attr in ("_X_seen", "_y_seen"):
            value = getattr(self.model, attr, None)
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return int(total)
