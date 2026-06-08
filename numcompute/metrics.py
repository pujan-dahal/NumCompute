"""
Metrics calculation utilities for NumCompute.

Provides common evaluation metrics for machine learning tasks:
- Classification: confusion matrix, accuracy, precision, recall, F1 score
- Regression: mean squared error (MSE)

All functions operate on array-like inputs and internally convert them to NumPy arrays.
Inputs must have matching shapes.
"""
import numpy as np


class Classification:
    @staticmethod
    def _validate_inputs(y_true, y_pred):
        """
        Validate and standardize classification inputs.

        Converts inputs to NumPy arrays and ensures matching shapes.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Validated (y_true, y_pred) arrays.

        Raises
        ------
        ValueError
            If shapes of y_true and y_pred do not match.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        if y_true.size == 0:
            raise ValueError("classification inputs must not be empty")

        return y_true, y_pred

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Compute confusion matrix components for binary classification.

        Assumes labels are binary (0 or 1).

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        tuple[int, int, int, int]
            (tp, tn, fp, fn)
        """
        y_true, y_pred = Classification._validate_inputs(y_true, y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp, tn, fp, fn

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Compute classification accuracy.

        Returns
        -------
        float
        """
        y_true, y_pred = Classification._validate_inputs(y_true, y_pred)

        tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
        return (tp + tn) / (tp + tn + fp + fn)

    @staticmethod
    def precision(y_true, y_pred):
        """
        Compute precision (TP / (TP + FP)).

        Returns
        -------
        float
        """
        y_true, y_pred = Classification._validate_inputs(y_true, y_pred)
        tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
        return tp / (tp + fp) if tp + fp > 0 else 0

    @staticmethod
    def recall(y_true, y_pred):
        """
        Compute recall (TP / (TP + FN)).

        Returns
        -------
        float
        """
        y_true, y_pred = Classification._validate_inputs(y_true, y_pred)
        tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
        return tp / (tp + fn) if tp + fn > 0 else 0

    @staticmethod
    def f1(y_true, y_pred):
        """
        Compute F1 score.

        Returns
        -------
        float
        """
        y_true, y_pred = Classification._validate_inputs(y_true, y_pred)
        precision = Classification.precision(y_true, y_pred)
        recall = Classification.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


class Regression:
    @staticmethod
    def _validate_inputs(y_true, y_pred):
        """
        Validate and standardize regression inputs.

        Converts inputs to NumPy arrays and ensures matching shapes.

        Raises
        ------
        ValueError
            If shapes do not match.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        return y_true, y_pred

    @staticmethod
    def mse(y_true, y_pred):
        """
        Compute mean squared error (MSE).

        Returns
        -------
        float
        """
        y_true, y_pred = Regression._validate_inputs(y_true, y_pred)
        return np.mean(np.square(y_true - y_pred))


class StreamingClassificationMetric:
    """
    Accumulate classification metrics over stream chunks.

    Parameters
    ----------
    metric : {'accuracy', 'precision', 'recall', 'f1'}, default 'accuracy'
        Metric returned by result().
    positive_label : int or str, default 1
        Positive class used for binary precision, recall and F1.
    window_size : int or None, default None
        Optional rolling window measured in samples.
    """

    def __init__(self, metric="accuracy", positive_label=1, window_size=None):
        if metric not in {"accuracy", "precision", "recall", "f1"}:
            raise ValueError("metric must be one of accuracy, precision, recall, f1")
        self.metric = metric
        self.positive_label = positive_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset accumulated metric state."""
        self.y_true_ = np.asarray([], dtype=object)
        self.y_pred_ = np.asarray([], dtype=object)
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Update metric state using one stream chunk."""
        y_true, y_pred = Classification._validate_inputs(y_true_chunk, y_pred_chunk)
        self.y_true_ = np.concatenate([self.y_true_, y_true.astype(object)])
        self.y_pred_ = np.concatenate([self.y_pred_, y_pred.astype(object)])
        if self.window_size is not None and self.y_true_.size > self.window_size:
            self.y_true_ = self.y_true_[-self.window_size:]
            self.y_pred_ = self.y_pred_[-self.window_size:]
        return self

    def result(self):
        """Return the selected accumulated metric."""
        if self.y_true_.size == 0:
            return 0.0
        y_true = self.y_true_
        y_pred = self.y_pred_
        if self.metric == "accuracy":
            return float(np.mean(y_true == y_pred))
        tp = np.sum((y_true == self.positive_label) & (y_pred == self.positive_label))
        fp = np.sum((y_true != self.positive_label) & (y_pred == self.positive_label))
        fn = np.sum((y_true == self.positive_label) & (y_pred != self.positive_label))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if self.metric == "precision":
            return float(precision)
        if self.metric == "recall":
            return float(recall)
        return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

    def confusion_matrix(self, labels=None):
        """Return an accumulated multi-class confusion matrix."""
        if labels is None:
            labels = np.unique(np.concatenate([self.y_true_, self.y_pred_]))
        labels = np.asarray(labels)
        matrix = np.zeros((labels.size, labels.size), dtype=int)
        for row_idx, true_label in enumerate(labels):
            for col_idx, pred_label in enumerate(labels):
                matrix[row_idx, col_idx] = np.sum((self.y_true_ == true_label) & (self.y_pred_ == pred_label))
        return matrix
