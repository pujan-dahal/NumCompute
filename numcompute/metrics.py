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