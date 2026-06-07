"""Small chunk based decision tree classifier built with only NumPy."""
from __future__ import annotations
import numpy as np


class _Node:
    """Internal tree node used by DecisionTreeClassifier."""

    __slots__ = ("prediction", "proba", "feature", "threshold", "left", "right", "depth")

    def __init__(self, prediction=None, proba=None, feature=None, threshold=None, left=None, right=None, depth=0):
        self.prediction = prediction
        self.proba = proba
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth

    @property
    def is_leaf(self):
        """bool: True when this node does not split any further."""
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    """Depth-limited decision tree classifier with chunk-wise partial_fit."""

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, criterion="gini", random_state=None):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None

        self._X = None
        self._y = None

    def fit(self, X, y):
        """Fit the tree from scratch on one complete dataset."""
        self._X = self._y = None
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """Update the tree using a new incoming chunk."""
        X, y = self._validate_xy(X_chunk, y_chunk)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        if classes is not None and self.classes_ is None:
            self.classes_ = np.asarray(classes)

        # keep the old chunks, then rebuild
        self._X = X.copy() if self._X is None else np.vstack([self._X, X])
        self._y = y.copy() if self._y is None else np.concatenate([self._y, y])
        self.classes_ = np.unique(self._y) if self.classes_ is None else np.unique(np.concatenate([self.classes_, np.unique(y)]))
        self.root_ = self._build(self._X, self._y, depth=0)
        return self

    def predict(self, X):
        """Predict class labels for X."""
        if self.root_ is None:
            raise ValueError("This DecisionTreeClassifier has not been fitted yet")
        X = self._validate_x(X)
        return np.array([self._predict_row(row, self.root_) for row in X])

    def predict_proba(self, X):
        """Predict class probabilities for each row in X."""
        if self.root_ is None:
            raise ValueError("This DecisionTreeClassifier has not been fitted yet")
        X = self._validate_x(X)
        out = []
        for row in X:
            node = self.root_
            while not node.is_leaf:
                val = row[node.feature]
                node = node.left if (np.isnan(val) or val <= node.threshold) else node.right
            out.append(node.proba)
        return np.vstack(out)

    def _validate_x(self, X):
        """Convert feature input to a clean 2-D float array."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X

    def _validate_xy(self, X, y):
        """Validate matching feature and target chunks."""
        X = self._validate_x(X)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} samples vs {y.shape[0]} labels")
        return X, y

    def _class_counts(self, y):
        """Count labels using the fitted class order."""
        return np.array([np.sum(y == c) for c in self.classes_], dtype=float)

    def _leaf(self, y, depth):
        """Create a leaf node with majority-class prediction."""
        counts = self._class_counts(y)
        prediction = self.classes_[int(np.argmax(counts))]
        total = counts.sum()
        proba = counts / total if total else np.ones(len(self.classes_)) / len(self.classes_)
        return _Node(prediction=prediction, proba=proba, depth=depth)

    def _impurity(self, y):
        """Compute Gini or entropy impurity for labels in a node."""
        if y.size == 0:
            return 0.0
        p = self._class_counts(y) / y.size
        p = p[p > 0]
        if self.criterion == "entropy":
            return float(-np.sum(p * np.log2(p)))
        return float(1.0 - np.sum(p * p))

    def _feature_indices(self, n_features):
        """Choose which feature columns are allowed for the next split."""
        if self.max_features is None:
            k = n_features
        elif isinstance(self.max_features, str):
            k = max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            k = max(1, int(np.ceil(self.max_features * n_features)))
        else:
            k = int(self.max_features)
        k = min(max(k, 1), n_features)
        return self.rng_.choice(n_features, size=k, replace=False) if k < n_features else np.arange(n_features)

    def _best_split(self, X, y):
        """Find the feature and threshold that gives the best impurity gain."""
        n, d = X.shape
        base = self._impurity(y)
        best_gain, best_feat, best_thr = 0.0, None, None

        for j in self._feature_indices(d):
            col = X[:, j]
            clean = col[~np.isnan(col)]
            # if a column is all missing or constant, splitting on it will not help.
            if clean.size == 0 or np.nanmin(clean) == np.nanmax(clean):
                continue

            thresholds = np.unique(clean)
            # limit candidates so large chunks do not make the demo painfully slow.
            if thresholds.size > 16:
                thresholds = np.quantile(clean, np.linspace(0.05, 0.95, 16))

            less_equal = col[:, None] <= thresholds[None, :]
            nan_mask = np.isnan(col)[:, None]
            left_masks = less_equal | nan_mask
            left_counts = left_masks.sum(axis=0)
            right_counts = n - left_counts
            valid = (left_counts > 0) & (right_counts > 0)

            for idx in np.where(valid)[0]:
                mask = left_masks[:, idx]
                weighted = (mask.sum() * self._impurity(y[mask]) + (~mask).sum() * self._impurity(y[~mask])) / n
                gain = base - weighted
                if gain > best_gain + 1e-12:
                    best_gain, best_feat, best_thr = gain, int(j), float(thresholds[idx])
        return best_feat, best_thr

    def _build(self, X, y, depth):
        """Recursively grow a tree from the current retained stream."""
        if y.size == 0 or np.unique(y).size == 1 or depth >= self.max_depth or y.size < self.min_samples_split:
            return self._leaf(y, depth)

        feat, thr = self._best_split(X, y)
        if feat is None:
            return self._leaf(y, depth)

        mask = (X[:, feat] <= thr) | np.isnan(X[:, feat])
        if mask.all() or (~mask).all():
            return self._leaf(y, depth)

        leaf = self._leaf(y, depth)
        return _Node(
            prediction=leaf.prediction,
            proba=leaf.proba,
            feature=feat,
            threshold=thr,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
            depth=depth,
        )

    def _predict_row(self, row, node):
        """Walk one row down the tree until a leaf is reached."""
        while not node.is_leaf:
            val = row[node.feature]
            node = node.left if (np.isnan(val) or val <= node.threshold) else node.right
        return node.prediction
