"""
numcompute.preprocessing
=======================

Preprocessing utilities for NumCompute.

Provides:
- Feature scaling (StandardScaler, MinMaxScaler)
- Encoding (OneHotEncoder)
- Missing value handling (SimpleImputer)

Author: Anshu Shrestha
"""

import numpy as np
from typing import Optional, Tuple, List


class _BaseScaler:
    """
    Base class for all preprocessing transformers.

    Defines a consistent API similar to scikit-learn:
    - fit(X)
    - transform(X)
    - fit_transform(X)
    """

    def fit(self, X: np.ndarray) -> "_BaseScaler":
        """
        Fit the transformer to the data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            _BaseScaler: Fitted transformer.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the transformer and apply transformation.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.fit(X).transform(X)

    @staticmethod
    def _validate(
        X: np.ndarray,
        fitted_attr: Optional[str] = None,
        obj=None,
    ) -> np.ndarray:
        """
        Validate and standardize input data.

        Ensures:
        - Input is numeric
        - Input is 2D
        - Transformer is fitted before transform

        Args:
            X (np.ndarray): Input data.
            fitted_attr (Optional[str]): Attribute name to check if fitted.
            obj (object): Transformer instance.

        Returns:
            np.ndarray: Validated 2D array.

        Raises:
            ValueError: If input cannot be converted or shape is invalid.
            RuntimeError: If transformer is not fitted.
        """
        try:
            X = np.array(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "X must be convertible to a numeric NumPy array. "
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


class StandardScaler(_BaseScaler):
    """
    Standardize features by removing mean and scaling to unit variance.

    Formula:
        X_scaled = (X - mean) / std

    Handles missing values using NaN-aware statistics.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        ddof: int = 0,
    ) -> None:
        """
        Initialize StandardScaler.

        Args:
            with_mean (bool): Whether to center data.
            with_std (bool): Whether to scale to unit variance.
            ddof (int): Delta degrees of freedom for std calculation.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute mean and standard deviation.

        Args:
            X (np.ndarray): Input data.

        Returns:
            StandardScaler: Fitted scaler.
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
        Apply standardization.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
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
        Revert scaling to original values.

        Args:
            X_scaled (np.ndarray): Scaled data.

        Returns:
            np.ndarray: Original data.
        """
        X_scaled = self._validate(X_scaled, fitted_attr="mean_", obj=self)

        X_out = X_scaled.copy()

        if self.with_std:
            X_out *= self.std_

        if self.with_mean:
            X_out += self.mean_

        return X_out


class MinMaxScaler(_BaseScaler):
    """
    Scale features to a specified range.

    Formula:
        X_scaled = (X - min) / (max - min) * (rmax - rmin) + rmin
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        """
        Initialize MinMaxScaler.

        Args:
            feature_range (Tuple[float, float]): Desired output range.
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
        Compute min and max values for scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            MinMaxScaler: Fitted scaler.
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
        Apply min-max scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
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
        Revert scaled data to original range.

        Args:
            X_scaled (np.ndarray): Scaled data.

        Returns:
            np.ndarray: Original data.
        """
        X_scaled = self._validate(X_scaled, fitted_attr="data_min_", obj=self)
        return (X_scaled - self.min_) / self.scale_


class OneHotEncoder(_BaseScaler):
    """
    Convert categorical features into one-hot encoded vectors.

    Each category is represented as a binary vector.
    """

    def __init__(
        self,
        drop_first: bool = False,
        dtype=np.float64,
    ) -> None:
        """
        Initialize OneHotEncoder.

        Args:
            drop_first (bool): Whether to drop first category (avoid dummy trap).
            dtype (type): Output data type.
        """
        self.drop_first = drop_first
        self.dtype = dtype
        self.categories_: Optional[List[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """
        Learn unique categories per feature.

        Args:
            X (np.ndarray): Input categorical data.

        Returns:
            OneHotEncoder: Fitted encoder.
        """
        X = self._validate(X)
        self.n_features_in_ = X.shape[1]

        self.categories_ = []

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            unique_vals = np.unique(col[~np.isnan(col)])
            self.categories_.append(unique_vals)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical data into one-hot encoding.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Encoded data.
        """
        X = self._validate(X, fitted_attr="categories_", obj=self)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        output_blocks = []

        for col_idx, cats in enumerate(self.categories_):
            col = X[:, col_idx]
            indicators = (col[:, None] == cats[None, :]).astype(self.dtype)

            if self.drop_first:
                indicators = indicators[:, 1:]

            output_blocks.append(indicators)

        return np.hstack(output_blocks)


class SimpleImputer(_BaseScaler):
    """
    Replace missing values using a specified strategy.

    Supported strategies:
    - mean
    - median
    - most_frequent
    - constant
    """

    _VALID_STRATEGIES = {
        "mean",
        "median",
        "most_frequent",
        "constant"
    }

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: float = 0.0,
    ) -> None:
        """
        Initialize SimpleImputer.

        Args:
            strategy (str): Imputation strategy.
            fill_value (float): Used when strategy="constant".
        """
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose one of {sorted(self._VALID_STRATEGIES)}"
            )

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        """
        Compute imputation statistics.

        Args:
            X (np.ndarray): Input data.

        Returns:
            SimpleImputer: Fitted imputer.
        """
        X = self._validate(X)
        self.n_features_in_ = X.shape[1]

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)

        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)

        elif self.strategy == "most_frequent":
            stats = np.empty(X.shape[1])

            for col_idx in range(X.shape[1]):
                col = X[:, col_idx]
                col = col[~np.isnan(col)]

                if col.size == 0:
                    stats[col_idx] = np.nan
                else:
                    unique_vals, counts = np.unique(
                        col,
                        return_counts=True
                    )
                    stats[col_idx] = unique_vals[np.argmax(counts)]

            self.statistics_ = stats

        else:
            self.statistics_ = np.full(
                X.shape[1],
                self.fill_value
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace missing values.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Imputed data.
        """
        X = self._validate(
            X,
            fitted_attr="statistics_",
            obj=self
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        X_out = X.copy()
        nan_mask = np.isnan(X_out)

        for col_idx in range(X_out.shape[1]):
            col_mask = nan_mask[:, col_idx]

            if col_mask.any():
                X_out[col_mask, col_idx] = self.statistics_[col_idx]

        return X_out