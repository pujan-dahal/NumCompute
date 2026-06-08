"""
Decision tree models for NumCompute.

Provides a small decision tree classifier with a stream-compatible API. The
classifier keeps cumulative chunks and rebuilds a depth-limited tree on each
partial update. This keeps the implementation simple, deterministic and easy to
inspect for learning purposes while still supporting online-style usage.
"""

import numpy as np


class _TreeNode:
    """Internal node used by DecisionTreeClassifier."""

    def __init__(self, prediction, feature_index=None, threshold=None, left=None, right=None):
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        """Return True when this node has no children."""
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum tree depth.
    min_samples_split : int, default 2
        Minimum samples required to attempt a split.
    max_features : int or None, default None
        Number of features considered per split. None uses all features.
    criterion : {'gini', 'entropy'}, default 'gini'
        Impurity function used to score splits.
    random_state : int or None, default None
        Seed used when max_features is smaller than the feature count.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted class labels learned from training data.
    root_ : _TreeNode
        Root node of the fitted tree.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, criterion="gini", random_state=None):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._X_seen = None
        self._y_seen = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit the tree using a complete batch of data.

        Parameters
        ----------
        X : array like
            Feature matrix with shape (n_samples, n_features).
        y : array like
            Target labels with shape (n_samples,).

        Returns
        -------
        DecisionTreeClassifier
            The fitted classifier.
        """
        X, y = self._validate_xy(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def partial_fit(self, X, y, classes=None):
        """
        Update the classifier with a new stream chunk.

        New data is appended to the internal training cache and the tree is
        rebuilt. Rebuilding is acceptable here because the assignment focuses on
        transparent algorithm design rather than production-scale throughput.
        """
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
        self.root_ = self._build_tree(self._X_seen, self._y_seen, depth=0)
        return self

    def predict(self, X):
        """
        Predict labels for input samples.

        Parameters
        ----------
        X : array like
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier has not been fitted")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return np.asarray([self._predict_row(row, self.root_) for row in X])

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

    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _impurity(self, y):
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(probs * probs)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _candidate_features(self, n_features):
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        if self.max_features <= 0:
            raise ValueError("max_features must be positive or None")
        return np.sort(self._rng.choice(n_features, size=self.max_features, replace=False))

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in self._candidate_features(n_features):
            feature_values = X[:, feature_index]
            valid_mask = ~np.isnan(feature_values)
            values = np.unique(feature_values[valid_mask])
            if values.size <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = np.nan_to_num(feature_values, nan=np.inf) <= threshold
                right_mask = ~left_mask
                left_count = np.sum(left_mask)
                right_count = n_samples - left_count
                if left_count == 0 or right_count == 0:
                    continue
                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])
                child_impurity = (left_count / n_samples) * left_impurity + (right_count / n_samples) * right_impurity
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        prediction = self._majority_class(y)
        if depth >= self.max_depth or y.size < self.min_samples_split or np.unique(y).size == 1:
            return _TreeNode(prediction=prediction)

        feature_index, threshold, gain = self._best_split(X, y)
        if feature_index is None or gain <= 0.0:
            return _TreeNode(prediction=prediction)

        feature_values = X[:, feature_index]
        left_mask = np.nan_to_num(feature_values, nan=np.inf) <= threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _TreeNode(prediction=prediction, feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _predict_row(self, row, node):
        while not node.is_leaf:
            value = row[node.feature_index]
            if np.isnan(value) or value > node.threshold:
                node = node.right
            else:
                node = node.left
        return node.prediction
