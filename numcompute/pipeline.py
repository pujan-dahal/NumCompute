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

class Compose:
    """
    Chain multiple transformers and an optional final estimator.
    """

    def __init__(self, steps):
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list")

        self.steps = steps

    def fit(self, X, y=None):
        """
        Fit all steps in order.
        """
        Xt = X

        # Fit and transform all steps except the last one
        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' must implement fit and transform")
            Xt = step.fit_transform(Xt, y)

        # Fit the last step
        name, last_step = self.steps[-1]

        if hasattr(last_step, "predict"):
            last_step.fit(Xt, y)
        elif hasattr(last_step, "transform"):
            last_step.fit(Xt, y)
        else:
            raise TypeError(f"Step '{name}' must implement transform or predict")

        return self

    def transform(self, X):
        """
        Apply all transform steps in order.
        """
        Xt = X

        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' does not implement transform")
            Xt = step.transform(Xt)

        return Xt

    def fit_transform(self, X, y=None):
        """
        Fit all steps and return transformed output.
        """
        Xt = X

        for name, step in self.steps:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' must implement fit and transform")
            Xt = step.fit_transform(Xt, y)

        return Xt

    def predict(self, X):
        """
        Apply all transformers, then predict using the final estimator.
        """
        Xt = X

        # Transform through all steps except the last one
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' does not implement transform")
            Xt = step.transform(Xt)

        # Final step must be an estimator
        name, last_step = self.steps[-1]

        if not hasattr(last_step, "predict"):
            raise TypeError(f"Final step '{name}' does not implement predict")

        return last_step.predict(Xt)


class FeatureUnion:
    """
    Apply multiple transformers in parallel and combine their outputs.
    """

    def __init__(self, transformers):
        if not isinstance(transformers, list) or len(transformers) == 0:
            raise ValueError("transformers must be a non-empty list")

        self.transformers = transformers

    def fit(self, X, y=None):
        """
        Fit all transformers.
        """
        for name, transformer in self.transformers:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError(f"Transformer '{name}' must implement fit and transform")
            transformer.fit(X, y)

        return self

    def transform(self, X):
        """
        Transform data with all transformers and concatenate outputs.
        """
        outputs = []

        for name, transformer in self.transformers:
            Xt = transformer.transform(X)
            Xt = np.asarray(Xt)

            # Convert 1D output into column form
            if Xt.ndim == 1:
                Xt = Xt.reshape(-1, 1)

            outputs.append(Xt)

        return np.hstack(outputs)

    def fit_transform(self, X, y=None):
        """
        Fit all transformers, transform data, and concatenate outputs.
        """
        outputs = []

        for name, transformer in self.transformers:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError(f"Transformer '{name}' must implement fit and transform")

            Xt = transformer.fit_transform(X, y)
            Xt = np.asarray(Xt)

            # Convert 1D output into column form
            if Xt.ndim == 1:
                Xt = Xt.reshape(-1, 1)

            outputs.append(Xt)

        return np.hstack(outputs)


