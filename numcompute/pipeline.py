import numpy as np


class Pipeline:
    """
    Run named processing steps one after another

    Parameters
    ----------
    steps : list
        List of name and object pairs
        Middle steps should have fit and transform
        The last step can have fit transform or predict depending on use

    Attributes
    ----------
    steps : list
        Original ordered list of steps
    named_steps : dict
        Dictionary used to access steps by name

    Raises
    ------
    ValueError
        If steps is empty names are invalid names repeat or objects are None
    TypeError
        If a step is not a name and object pair

    Complexity
    ----------
    Time O s to validate where s is number of steps
    Space O s
    """

    def __init__(self, steps):
        """
        Set up the pipeline with named steps

        Parameters
        ----------
        steps : list
            Non empty list of tuples like name and object

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If steps is empty or contains invalid duplicate names
        TypeError
            If any step is not a tuple with two values

        Complexity
        ----------
        Time O s
        Space O s
        """
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
        """
        Fit the pipeline on input data

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features
        y : array like or None
            Optional target values for the final estimator

        Returns
        -------
        Pipeline
            The fitted pipeline

        Raises
        ------
        TypeError
            If a middle step does not have fit and transform

        Complexity
        ----------
        Time O sum of step fit and transform costs
        Space depends on transformed data size
        """
        X_current = X

        # Fit and transform every step before the last one
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
        """
        Transform data through every step

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features

        Returns
        -------
        array like
            Transformed data from the final step

        Raises
        ------
        TypeError
            If any step does not have transform

        Complexity
        ----------
        Time O sum of step transform costs
        Space depends on transformed data size
        """
        X_current = X

        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"step '{name}' does not have transform()")

            X_current = step.transform(X_current)

        return X_current

    def fit_transform(self, X, y=None):
        """
        Fit each step then transform the data

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features
        y : array like or None
            Optional target values

        Returns
        -------
        array like
            Final transformed data

        Raises
        ------
        TypeError
            If any step does not have fit and transform

        Complexity
        ----------
        Time O sum of step fit and transform costs
        Space depends on transformed data size
        """
        X_current = X

        for name, step in self.steps:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"step '{name}' must have fit() and transform()")

            step.fit(X_current)
            X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        """
        Transform data through preprocessing steps then predict

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features

        Returns
        -------
        array like
            Predictions returned by the last step

        Raises
        ------
        TypeError
            If a preprocessing step cannot transform or the last step cannot predict

        Complexity
        ----------
        Time O transform costs plus final predict cost
        Space depends on transformed data and prediction size
        """
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
    """
    Run many transformers on the same data and join their outputs

    Parameters
    ----------
    transformers : list
        List of name and transformer pairs

    Attributes
    ----------
    transformers : list
        Original ordered list of transformers
    named_transformers : dict
        Dictionary used to access transformers by name

    Raises
    ------
    ValueError
        If transformers is empty or names repeat
    TypeError
        If a transformer entry is not a name and object pair

    Complexity
    ----------
    Time O t to validate where t is number of transformers
    Space O t
    """

    def __init__(self, transformers):
        """
        Set up the feature union

        Parameters
        ----------
        transformers : list
            Non empty list of tuples like name and transformer

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If transformers is empty or names repeat
        TypeError
            If any transformer is not a tuple with two values

        Complexity
        ----------
        Time O t
        Space O t
        """
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
        """
        Fit every transformer on the same input data

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features
        y : array like or None
            Optional target values currently not used

        Returns
        -------
        FeatureUnion
            The fitted feature union

        Raises
        ------
        TypeError
            If a transformer does not have fit

        Complexity
        ----------
        Time O sum of transformer fit costs
        Space O one transformer at a time
        """
        for name, transformer in self.transformers:
            if not hasattr(transformer, "fit"):
                raise TypeError(f"transformer '{name}' does not have fit()")

            transformer.fit(X)

        return self

    def transform(self, X):
        """
        Transform data with each transformer and join the columns

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features

        Returns
        -------
        numpy.ndarray
            Joined feature matrix with shape n_samples by total_output_features

        Raises
        ------
        TypeError
            If a transformer does not have transform

        Complexity
        ----------
        Time O sum of transformer transform costs plus output join cost
        Space O size of all transformer outputs
        """
        outputs = []

        for name, transformer in self.transformers:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"transformer '{name}' does not have transform()")

            output = np.asarray(transformer.transform(X))

            # Make one dimensional output into one column
            if output.ndim == 1:
                output = output.reshape(-1, 1)

            outputs.append(output)

        return np.hstack(outputs)

    def fit_transform(self, X, y=None):
        """
        Fit all transformers then return the joined output

        Parameters
        ----------
        X : array like
            Input data with shape n_samples by n_features
        y : array like or None
            Optional target values currently not used

        Returns
        -------
        numpy.ndarray
            Joined feature matrix

        Raises
        ------
        TypeError
            If a transformer does not have fit or transform

        Complexity
        ----------
        Time O fit costs plus transform costs
        Space O size of all transformer outputs
        """
        self.fit(X, y)
        return self.transform(X)


class Compose(Pipeline):
    """
    Alias for Pipeline

    This gives another name for the same pipeline behaviour
    """

    pass



def _pipeline_partial_fit(self, X, y=None):
    """
    Incrementally fit pipeline steps and the final estimator.

    Middle steps use partial_fit when available, otherwise fit. Each middle step
    must still implement transform so the next step receives updated features.
    """
    X_current = X
    for name, step in self.steps[:-1]:
        if hasattr(step, "partial_fit"):
            step.partial_fit(X_current)
        elif hasattr(step, "fit"):
            step.fit(X_current)
        else:
            raise TypeError(f"step '{name}' must have partial_fit() or fit()")
        if not hasattr(step, "transform"):
            raise TypeError(f"step '{name}' must have transform()")
        X_current = step.transform(X_current)
    last_name, last_step = self.steps[-1]
    if not hasattr(last_step, "partial_fit"):
        raise TypeError(f"last step '{last_name}' must have partial_fit()")
    if y is None:
        last_step.partial_fit(X_current)
    else:
        last_step.partial_fit(X_current, y)
    return self


Pipeline.partial_fit = _pipeline_partial_fit
