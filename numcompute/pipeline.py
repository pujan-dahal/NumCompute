import numpy as np


class Transformer:
    """
    Base class for transformers.
    """

    def fit(self, X, y=None):
        """
        Fit the transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input data.
        """
        raise NotImplementedError("transform must be implemented")

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the data.
        """
        self.fit(X, y)
        return self.transform(X)


class Estimator:
    """
    Base class for estimators.
    """

    def fit(self, X, y):
        """
        Fit the estimator.
        """
        raise NotImplementedError("fit must be implemented")

    def predict(self, X):
        """
        Predict using the estimator.
        """
        raise NotImplementedError("predict must be implemented")


