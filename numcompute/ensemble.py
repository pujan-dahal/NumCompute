"""Small tree ensemble used in the streaming demo."""
import numpy as np
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Bagging ensemble of decision trees that can learn chunk by chunk."""

    def __init__(self, n_estimators=5, max_depth=5, min_samples_split=2, max_features="sqrt", criterion="gini", bootstrap=True, random_state=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                criterion=criterion,
                random_state=None if random_state is None else random_state + i,
            )
            for i in range(self.n_estimators)
        ]
        self.classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        """Fit the ensemble from scratch on a full dataset."""
        self.__init__(self.n_estimators, self.max_depth, self.min_samples_split, self.max_features, self.criterion, self.bootstrap, self.random_state)
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """Update every tree using one incoming data chunk."""
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y_chunk).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of samples")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        self.classes_ = np.unique(y) if self.classes_ is None else np.unique(np.concatenate([self.classes_, np.unique(y)]))
        if classes is not None:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.asarray(classes)]))

        n = X.shape[0]
        for est in self.estimators_:
            if self.bootstrap:
                # each tree sees a slightly different version of the same chunk
                idx = self.rng_.integers(0, n, size=n)
                est.partial_fit(X[idx], y[idx], classes=self.classes_)
            else:
                est.partial_fit(X, y, classes=self.classes_)
        return self

    def predict(self, X):
        """Predict labels by majority vote across all trees."""
        if self.classes_ is None:
            raise ValueError("This EnsembleClassifier has not been fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        votes = np.vstack([est.predict(X) for est in self.estimators_]).T
        preds = []
        for row in votes:
            counts = np.array([np.sum(row == c) for c in self.classes_])
            preds.append(self.classes_[int(np.argmax(counts))])
        return np.asarray(preds)

    def predict_proba(self, X):
        """Average class probabilities from all trees."""
        if self.classes_ is None:
            raise ValueError("This EnsembleClassifier has not been fitted yet")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        proba = np.zeros((X_arr.shape[0], len(self.classes_)))
        for est in self.estimators_:
            est_proba = est.predict_proba(X_arr)
            for k, c in enumerate(est.classes_):
                target = np.where(self.classes_ == c)[0][0]
                proba[:, target] += est_proba[:, k]
        return proba / self.n_estimators

