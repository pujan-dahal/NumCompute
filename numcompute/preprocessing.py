"""
Author : (Anshu Shrestha)
"""
 
import numpy as np
from typing import Optional, Tuple, List

class _BaseScaler:
    """Shared base class for transformers."""

    def fit(self, X: np.ndarray) -> "_BaseScaler":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def _validate(
        X: np.ndarray,
        fitted_attr: Optional[str] = None,
        obj=None,
    ) -> np.ndarray:
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


class StandardScaler(_BaseScaler):
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        ddof: int = 0,
    ) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
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
        X_scaled = self._validate(X_scaled, fitted_attr="mean_", obj=self)

        X_out = X_scaled.copy()

        if self.with_std:
            X_out *= self.std_

        if self.with_mean:
            X_out += self.mean_

        return X_out


class MinMaxScaler(_BaseScaler):
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
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
        X_scaled = self._validate(X_scaled, fitted_attr="data_min_", obj=self)
        return (X_scaled - self.min_) / self.scale_


class OneHotEncoder(_BaseScaler):
    def __init__(
        self,
        drop_first: bool = False,
        dtype=np.float64,
    ) -> None:
        self.drop_first = drop_first
        self.dtype = dtype
        self.categories_: Optional[List[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        X = self._validate(X)
        self.n_features_in_ = X.shape[1]

        self.categories_ = []

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            unique_vals = np.unique(col[~np.isnan(col)])
            self.categories_.append(unique_vals)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
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