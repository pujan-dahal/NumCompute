"""
numcompute/preprocessing.py
============================
Data preprocessing transformers for NumCompute.
 
Every class follows the same three-method contract:
 
    transformer.fit(X)            # learn parameters from training data
    transformer.transform(X)      # apply the learned transformation
    transformer.fit_transform(X)  # both at once (convenience shortcut)
 
All hot paths use NumPy vectorised operations.  No Python loops over
rows or columns anywhere in the production code.
 
 
Author : (Anshu Shrestha)
"""
 
import numpy as np
from typing import Optional, Tuple, List
 

# Internal base class — shared boilerplate

 
class _BaseScaler:
    """Shared boilerplate for all transformers.
 
    Provides ``fit_transform`` so subclasses only need to implement
    ``fit`` and ``transform``.  Not part of the public API.
    """
 
    def fit(self, X: np.ndarray) -> "_BaseScaler":
        raise NotImplementedError
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
 
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to *X*, transform *X*, and return the result.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
 
        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)
 
    @staticmethod
    def _validate(
        X: np.ndarray,
        fitted_attr: Optional[str] = None,
        obj=None,
    ) -> np.ndarray:
        """Coerce *X* to a 2-D float64 array; optionally check fit state.
 
        Parameters
        ----------
        X : array-like
            Will be coerced to ``np.float64``.
        fitted_attr : str, optional
            Name of an attribute that exists only after fitting.
        obj : object, optional
            The transformer instance to inspect.
 
        Returns
        -------
        np.ndarray, shape (n_samples, n_features), dtype float64
 
        Raises
        ------
        ValueError
            If *X* cannot be represented as a 2-D numeric array.
        RuntimeError
            If the transformer has not been fitted yet.
        """
        try:
            X = np.array(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"X must be convertible to a numeric NumPy array. "
                f"Original error: {exc}"
            )
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if fitted_attr is not None and obj is not None:
            if not hasattr(obj, fitted_attr):
                raise RuntimeError(
                    "This transformer has not been fitted yet. "
                    "Call fit(X) before transform(X)."
                )
        return X
# ===========================================================================
# StandardScaler


class StandardScaler(_BaseScaler):
    """
    Standardize features by removing the mean and scaling to unit variance.

    In simple terms, this makes each column have:
    - mean ≈ 0
    - standard deviation ≈ 1

    This is useful for many machine learning models where features
    need to be on a similar scale.

    Formula:
        X_scaled = (X - mean) / std

    Notes
    -----
    - Columns with zero variance (constant values) are left unchanged.
    - NaN values are ignored when calculating mean and std.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        ddof: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        with_mean : bool
            If True, subtract the mean from each feature.
        with_std : bool
            If True, scale features to unit variance.
        ddof : int
            Degrees of freedom for standard deviation.
            0 = population std (default), 1 = sample std.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Learn the mean and standard deviation from the data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        self.n_features_in_ = X.shape[1]

        self.mean_ = np.nanmean(X, axis=0) if self.with_mean else None

        if self.with_std:
            std = np.nanstd(X, axis=0, ddof=self.ddof)
            self.std_ = np.where(std == 0.0, 1.0, std)
        else:
            self.std_ = None

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization using the learned parameters.

        Parameters
        ----------
        X : np.ndarray
            Data to transform

        Returns
        -------
        np.ndarray
            Scaled data
        """
        X = self._validate(X, fitted_attr="mean_", obj=self)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        X_out = X.copy()

        if self.with_mean:
            X_out -= self.mean_

        if self.with_std:
            X_out /= self.std_

        return X_out

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled data back to the original values.

        Parameters
        ----------
        X_scaled : np.ndarray
            Standardized data

        Returns
        -------
        np.ndarray
            Original scale data
        """
        X_scaled = self._validate(X_scaled, fitted_attr="mean_", obj=self)

        X_out = X_scaled.copy()

        if self.with_std:
            X_out *= self.std_

        if self.with_mean:
            X_out += self.mean_

        return X_out


# ===========================================================================
# MinMaxScaler

class MinMaxScaler(_BaseScaler):
    """
    Scale features to a fixed range (default: 0 to 1).

    This rescales each column so that the minimum value becomes `min`
    and the maximum becomes `max`.

    Formula:
        X_scaled = (X - X_min) / (X_max - X_min)

    Then adjusted to the target range:
        X_scaled = X_scaled * (max - min) + min

    Notes
    -----
    - Constant columns are mapped to the lower bound of the range.
    - Works well when you want bounded values (e.g. for neural networks).
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        """
        Parameters
        ----------
        feature_range : tuple (min, max)
            Desired output range. Must satisfy min < max.
        """
        rmin, rmax = feature_range

        if rmin >= rmax:
            raise ValueError(
                f"feature_range must have min < max, got {feature_range}"
            )

        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """
        Learn the min and max values from the data.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        self
        """
        X = self._validate(X)
        self.n_features_in_ = X.shape[1]

        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)

        data_range = self.data_max_ - self.data_min_

        rmin, rmax = self.feature_range
        range_width = rmax - rmin

        safe_range = np.where(data_range == 0.0, 1.0, data_range)

        self.scale_ = range_width / safe_range
        self.min_ = rmin - self.data_min_ * self.scale_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data to the specified range.

        Parameters
        ----------
        X : np.ndarray
            Data to transform

        Returns
        -------
        np.ndarray
            Scaled data
        """
        X = self._validate(X, fitted_attr="data_min_", obj=self)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        X_out = X * self.scale_ + self.min_

        rmin, rmax = self.feature_range
        np.clip(X_out, rmin, rmax, out=X_out)

        return X_out

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled data back to original values.

        Parameters
        ----------
        X_scaled : np.ndarray
            Scaled data

        Returns
        -------
        np.ndarray
            Original data
        """
        X_scaled = self._validate(X_scaled, fitted_attr="data_min_", obj=self)

        return (X_scaled - self.min_) / self.scale_