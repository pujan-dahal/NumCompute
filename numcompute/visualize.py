"""
Visualisation utilities for NumCompute.

The functions intentionally depend only on matplotlib so they can be reused in
scripts, notebooks and benchmark outputs without extra plotting libraries.
"""

import numpy as np
import matplotlib.pyplot as plt


def _finalise_plot(save_path=None, show=True):
    """Save or show the active matplotlib figure."""
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return plt.gcf()


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None, show=True):
    """
    Plot one metric across stream chunks.

    Parameters
    ----------
    metric_values : array like
        Values ordered by chunk.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str or None
        Optional path used to save the figure.
    show : bool
        Whether to display the figure immediately.
    """
    values = np.asarray(metric_values, dtype=float)
    plt.figure()
    plt.plot(np.arange(1, values.size + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    return _finalise_plot(save_path=save_path, show=show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison", ylabel="Metric", save_path=None, show=True):
    """Compare two metric sequences over stream chunks."""
    values1 = np.asarray(metric1, dtype=float)
    values2 = np.asarray(metric2, dtype=float)
    plt.figure()
    plt.plot(np.arange(1, values1.size + 1), values1, marker="o", label=labels[0])
    plt.plot(np.arange(1, values2.size + 1), values2, marker="s", label=labels[1])
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finalise_plot(save_path=save_path, show=show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth", save_path=None, show=True):
    """Visualise predictions and actual labels for the latest stream chunk."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    x_axis = np.arange(y_true.size)
    plt.figure()
    plt.scatter(x_axis, y_true, label="Ground truth", marker="o")
    plt.scatter(x_axis, y_pred, label="Prediction", marker="x")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Class label")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finalise_plot(save_path=save_path, show=show)
