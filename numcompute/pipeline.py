import numpy as np


class Pipeline:
    def __init__(self, steps):
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list")

        self.steps = steps
        self.named_steps = {}

        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError("each step must be a tuple like ('name', object)")

            name, obj = step

            if not isinstance(name, str) or name == "":
                raise ValueError("step name must be a non-empty string")

            if name in self.named_steps:
                raise ValueError("step names must be unique")

            if obj is None:
                raise ValueError("step object cannot be None")

            self.named_steps[name] = obj

    def fit(self, X, y=None):
        X_current = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"step '{name}' must have fit() and transform()")

            step.fit(X_current)
            X_current = step.transform(X_current)

        last_name, last_step = self.steps[-1]

        if hasattr(last_step, "fit"):
            if y is None:
                last_step.fit(X_current)
            else:
                last_step.fit(X_current, y)

        return self

    def transform(self, X):
        X_current = X

        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"step '{name}' does not have transform()")

            X_current = step.transform(X_current)

        return X_current

    def fit_transform(self, X, y=None):
        X_current = X

        for name, step in self.steps:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"step '{name}' must have fit() and transform()")

            step.fit(X_current)
            X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        X_current = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"step '{name}' does not have transform()")

            X_current = step.transform(X_current)

        last_name, last_step = self.steps[-1]

        if not hasattr(last_step, "predict"):
            raise TypeError(f"last step '{last_name}' does not have predict()")

        return last_step.predict(X_current)


class FeatureUnion:
    def __init__(self, transformers):
        if not isinstance(transformers, list) or len(transformers) == 0:
            raise ValueError("transformers must be a non-empty list")

        self.transformers = transformers
        self.named_transformers = {}

        for item in transformers:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each transformer must be a tuple like ('name', object)")

            name, transformer = item

            if name in self.named_transformers:
                raise ValueError("transformer names must be unique")

            self.named_transformers[name] = transformer

    def fit(self, X, y=None):
        for name, transformer in self.transformers:
            if not hasattr(transformer, "fit"):
                raise TypeError(f"transformer '{name}' does not have fit()")

            transformer.fit(X)

        return self

    def transform(self, X):
        outputs = []

        for name, transformer in self.transformers:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"transformer '{name}' does not have transform()")

            output = np.asarray(transformer.transform(X))

            if output.ndim == 1:
                output = output.reshape(-1, 1)

            outputs.append(output)

        return np.hstack(outputs)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class Compose(Pipeline):
    pass