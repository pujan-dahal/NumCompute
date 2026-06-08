"""
Tree ensemble models for NumCompute.

The module implements a compact random-forest style classifier using multiple
DecisionTreeClassifier instances. Each tree receives a bootstrap sample from the
cumulative stream cache on every update.
"""

import numpy as np

from numcompute.tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    Random-forest style ensemble classifier.

    Parameters
    ----------
    n_estimators : int, default 5
        Number of trees in the ensemble.
    max_depth : int, default 5
        Maximum depth for each tree.
    min_samples_split : int, default 2
        Minimum samples required to split in each tree.
    max_features : int or None, default None
        Number of features considered at each split.
    random_state : int or None, default None
        Seed for bootstrap sampling.
    """

    def __init__(self, n_estimators=5, max_depth=5, min_samples_split=2, max_features=None, random_state=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self._X_seen = None
        self._y_seen = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """Fit the ensemble on a complete batch."""
        X, y = self._validate_xy(X, y)
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._fit_estimators()
        return self

    def partial_fit(self, X, y, classes=None):
        """Update the ensemble with a new stream chunk."""
        X, y = self._validate_xy(X, y)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        if self._X_seen is None:
            self._X_seen = X.copy()
            self._y_seen = y.copy()
        else:
            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])
        self.classes_ = np.unique(self._y_seen if classes is None else np.concatenate([self._y_seen, np.asarray(classes)]))
        self.n_features_in_ = self._X_seen.shape[1]
        self._fit_estimators()
        return self

    def predict(self, X):
        """Predict labels using majority vote across trees."""
        if not self.estimators_:
            raise RuntimeError("EnsembleClassifier has not been fitted")
        X = self._validate_X(X)
        tree_predictions = np.asarray([tree.predict(X) for tree in self.estimators_])
        output = []
        for col in tree_predictions.T:
            labels, counts = np.unique(col, return_counts=True)
            output.append(labels[np.argmax(counts)])
        return np.asarray(output)

    def _fit_estimators(self):
        self.estimators_ = []
        n_samples = self._X_seen.shape[0]
        for idx in range(self.n_estimators):
            sample_idx = self._rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + idx,
            )
            tree.fit(self._X_seen[sample_idx], self._y_seen[sample_idx])
            self.estimators_.append(tree)

    def _validate_X(self, X):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"X must be numeric. Original error: {exc}")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        return X

    def _validate_xy(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("training data must not be empty")
        return X, y


class RandomForestClassifier(EnsembleClassifier):
    """Alias for the random-forest style EnsembleClassifier."""

    pass
