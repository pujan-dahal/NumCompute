"""
Statistical utilities for NumCompute.

Provides basic descriptive statistics and histogram computation for CSV data.
Only numeric columns (dtype 'int' or 'float') are considered; all others are ignored.
"""

import numpy as np
from numcompute.io import CSVData


class Stats:
    @staticmethod
    def _get_numeric_cols(data: CSVData):
        """
        Identify numeric columns in the dataset.

        Parameters
        ----------
        data : CSVData
            Input CSV data object.

        Returns
        -------
        list[int]
            Indices of columns with dtype 'int' or 'float'.
        """
        
        return [idx for idx, col in enumerate(data.cols) if col.dtype in ['int', 'float']]

    @staticmethod
    def mean(data: CSVData, axis: int | None = None, ignore_nan: bool = True):
        """
        Compute the mean of numeric values.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        axis : {None, 0, 1}, optional
            Axis along which to compute the mean:
            - None : mean of all numeric values
            - 0    : column-wise mean (returns dict)
            - 1    : row-wise mean (returns array)
        ignore_nan : bool, default True
            Whether to ignore NaN values.

        Returns
        -------
        float or dict[str, float] or np.ndarray
            Mean value(s) depending on axis.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        mean_func = np.nanmean if ignore_nan else np.mean

        if axis is None:
            return mean_func(numeric_data)
        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = mean_func(numeric_data, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1:
            return mean_func(numeric_data, axis=1)
        else:
            raise ValueError("axis must be None, 0, or 1")

    @staticmethod
    def median(data: CSVData, axis: int | None = None, ignore_nan: bool = True):
        """
        Compute the median of numeric values.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        axis : {None, 0, 1}, optional
            Axis along which to compute the median.
        ignore_nan : bool, default True
            Whether to ignore NaN values.

        Returns
        -------
        float or dict[str, float] or np.ndarray
            Median value(s) depending on axis.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        median_func = np.nanmedian if ignore_nan else np.median

        if axis is None:
            return median_func(numeric_data)
        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            medians = median_func(numeric_data, axis=0)
            return dict(zip(selected_cols, medians))
        elif axis == 1:
            return median_func(numeric_data, axis=1)
        else:
            raise ValueError("axis must be None, 0, or 1")

    @staticmethod
    def std(data: CSVData, axis: int | None = None, ignore_nan: bool = True):
        """
        Compute the standard deviation of numeric values.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        axis : {None, 0, 1}, optional
            Axis along which to compute the standard deviation.
        ignore_nan : bool, default True
            Whether to ignore NaN values.

        Returns
        -------
        float or dict[str, float] or np.ndarray
            Standard deviation value(s) depending on axis.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        std_func = np.nanstd if ignore_nan else np.std

        if axis is None:
            return std_func(numeric_data)
        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            stds = std_func(numeric_data, axis=0)
            return dict(zip(selected_cols, stds))
        elif axis == 1:
            return std_func(numeric_data, axis=1)
        else:
            raise ValueError("axis must be None, 0, or 1")

    @staticmethod
    def minimum(data: CSVData, axis: int | None = None):
        """
        Compute the minimum of numeric values (NaNs ignored).

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        axis : {None, 0, 1}, optional
            Axis along which to compute the minimum.

        Returns
        -------
        float or dict[str, float] or np.ndarray
            Minimum value(s) depending on axis.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        if axis is None:
            return np.nanmin(numeric_data)
        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            mins = np.nanmin(numeric_data, axis=0)
            return dict(zip(selected_cols, mins))
        elif axis == 1:
            return np.nanmin(numeric_data, axis=1)
        else:
            raise ValueError("axis must be None, 0, or 1")

    @staticmethod
    def maximum(data: CSVData, axis: int | None = None):
        """
        Compute the maximum of numeric values (NaNs ignored).

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        axis : {None, 0, 1}, optional
            Axis along which to compute the maximum.

        Returns
        -------
        float or dict[str, float] or np.ndarray
            Maximum value(s) depending on axis.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        if axis is None:
            return np.nanmax(numeric_data)
        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            maxs = np.nanmax(numeric_data, axis=0)
            return dict(zip(selected_cols, maxs))
        elif axis == 1:
            return np.nanmax(numeric_data, axis=1)
        else:
            raise ValueError("axis must be None, 0, or 1")

    @staticmethod
    def quantile(data: CSVData, q, axis: int | None = None, ignore_nan=True):
        """
        Compute the q-th quantile of numeric values.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        q : float
            Quantile to compute, must be between 0 and 1.
        axis : {None, 0, 1}, optional
            Axis along which to compute the quantile.
        ignore_nan : bool, default True
            Whether to ignore NaN values.

        Returns
        -------
        float or dict[str, float] or np.ndarray
            Quantile value(s) depending on axis.
        """
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")

        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        quantile_func = np.nanquantile if ignore_nan else np.quantile

        if axis is None:
            return quantile_func(numeric_data, q)
        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            quantiles = quantile_func(numeric_data, q, axis=0)
            return dict(zip(selected_cols, quantiles))
        elif axis == 1:
            return quantile_func(numeric_data, q, axis=1)
        else:
            raise ValueError("axis must be None, 0, or 1")

    @staticmethod
    def histogram(data: CSVData, bins: int = 10, axis: int | None = None):
        """
        Compute histograms of numeric values.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        bins : int, default 10
            Number of histogram bins.
        axis : {None, 0, 1}, optional
            Axis along which to compute histograms:
            - None : single histogram for all numeric values
            - 0    : per-column histograms (dict)
            - 1    : per-row histograms (list)

        Returns
        -------
        tuple or dict[str, tuple] or list[tuple]
            Histogram(s) as (counts, bin_edges).
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)

        if axis is None:
            clean = numeric_data[~np.isnan(numeric_data)].flatten()
            return np.histogram(clean, bins=bins)

        elif axis == 0:
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            histograms = {}

            for idx, col_name in enumerate(selected_cols):
                col_data = numeric_data[:, idx]
                clean = col_data[~np.isnan(col_data)]

                if clean.size == 0:
                    histograms[col_name] = (np.array([]), np.array([]))
                else:
                    histograms[col_name] = np.histogram(clean, bins=bins)

            return histograms

        elif axis == 1:
            histograms = []

            for row in numeric_data:
                clean = row[~np.isnan(row)]

                if clean.size == 0:
                    histograms.append((np.array([]), np.array([])))
                else:
                    histograms.append(np.histogram(clean, bins=bins))

            return histograms

        else:
            raise ValueError("axis must be None, 0, or 1")
    @staticmethod
    def streaming_mean(data: CSVData, ignore_nan: bool = True):
        """
        Compute global mean over numeric columns using Welford's streaming update.

        This processes values one at a time, so it is useful for large datasets
        or streamed chunks where keeping all values in memory is not ideal.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        ignore_nan : bool, default True
            Whether to skip NaN values.

        Returns
        -------
        float
            Streaming mean of all numeric values.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")

        values = data.data[:, numeric_col_idx].astype(float).ravel()
        stream = StreamingStats(ignore_nan=ignore_nan)
        stream.update_many(values)
        return stream.mean

    @staticmethod
    def streaming_variance(data: CSVData, ddof: int = 0, ignore_nan: bool = True):
        """
        Compute global variance over numeric columns using Welford's algorithm.

        Parameters
        ----------
        data : CSVData
            Input CSV data.
        ddof : int, default 0
            Delta degrees of freedom. Use 0 for population variance and 1 for
            sample variance.
        ignore_nan : bool, default True
            Whether to skip NaN values.

        Returns
        -------
        float
            Streaming variance of all numeric values.
        """
        numeric_col_idx = Stats._get_numeric_cols(data)
        if len(numeric_col_idx) == 0:
            raise ValueError("No numeric columns found")

        values = data.data[:, numeric_col_idx].astype(float).ravel()
        stream = StreamingStats(ignore_nan=ignore_nan)
        stream.update_many(values)
        return stream.variance(ddof=ddof)


class StreamingStats:
    """
    Maintain running statistics for numeric stream chunks.

    The class tracks count, mean, variance and raw values for exact quantiles.
    """

    def __init__(self):
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.m2_ = None
        self.values_ = None

    def update_stats(self, X_chunk):
        """Update running statistics from a new chunk."""
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X_chunk must be 2-D, got shape {X.shape}")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=float)
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.m2_ = np.zeros(X.shape[1], dtype=float)
            self.values_ = [np.asarray([], dtype=float) for _ in range(X.shape[1])]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        valid = ~np.isnan(X)
        chunk_count = valid.sum(axis=0).astype(float)
        safe_values = np.where(valid, X, 0.0)
        chunk_mean = np.divide(safe_values.sum(axis=0), chunk_count, out=np.zeros(X.shape[1]), where=chunk_count > 0)
        centered = np.where(valid, X - chunk_mean, 0.0)
        chunk_m2 = np.sum(centered * centered, axis=0)
        old_count = self.count_
        new_count = old_count + chunk_count
        delta = chunk_mean - self.mean_
        self.mean_ = np.where(new_count > 0, self.mean_ + delta * np.divide(chunk_count, new_count, out=np.zeros_like(new_count), where=new_count > 0), self.mean_)
        self.m2_ = self.m2_ + chunk_m2 + delta * delta * old_count * np.divide(chunk_count, new_count, out=np.zeros_like(new_count), where=new_count > 0)
        self.count_ = new_count
        for col_idx in range(X.shape[1]):
            vals = X[:, col_idx][valid[:, col_idx]]
            if vals.size:
                self.values_[col_idx] = np.concatenate([self.values_[col_idx], vals])
        return self

    def mean(self):
        """Return running column means."""
        return self.mean_.copy()

    def variance(self, ddof=0):
        """Return running column variances."""
        denom = np.maximum(self.count_ - ddof, 1.0)
        return self.m2_ / denom

    def quantile(self, q):
        """Return exact running quantiles for each column."""
        return np.asarray([np.quantile(vals, q) if vals.size else np.nan for vals in self.values_])

    def histogram(self, bins=10):
        """Return histograms for each running numeric column."""
        return [np.histogram(vals, bins=bins) for vals in self.values_]


class StreamingStats:
    """
    Maintain streaming statistics for scalars or numeric chunks.

    Supports the original scalar API (`update`, `update_many`, `mean`,
    `variance`) and the Assignment 2.2 chunk API (`update_stats`, `quantile`,
    `histogram`).
    """

    def __init__(self, ignore_nan: bool = True):
        self.ignore_nan = ignore_nan
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.m2_ = None
        self.values_ = None

    def update(self, value):
        """Add one scalar value to the stream."""
        return self.update_stats(np.asarray([value], dtype=float).reshape(-1, 1))

    def update_many(self, values):
        """Add multiple scalar values to the stream."""
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.update_stats(values)

    def update_stats(self, X_chunk):
        """Update running statistics from a new chunk."""
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X_chunk must be 2-D, got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X_chunk must contain at least one row")
        if not self.ignore_nan and np.isnan(X).any():
            X = np.full_like(X, np.nan, dtype=float)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=float)
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.m2_ = np.zeros(X.shape[1], dtype=float)
            self.values_ = [np.asarray([], dtype=float) for _ in range(X.shape[1])]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        valid = ~np.isnan(X)
        chunk_count = valid.sum(axis=0).astype(float)
        safe_values = np.where(valid, X, 0.0)
        chunk_mean = np.divide(safe_values.sum(axis=0), chunk_count, out=np.zeros(X.shape[1]), where=chunk_count > 0)
        centered = np.where(valid, X - chunk_mean, 0.0)
        chunk_m2 = np.sum(centered * centered, axis=0)
        old_count = self.count_
        new_count = old_count + chunk_count
        delta = chunk_mean - self.mean_
        self.mean_ = np.where(new_count > 0, self.mean_ + delta * np.divide(chunk_count, new_count, out=np.zeros_like(new_count), where=new_count > 0), self.mean_)
        self.m2_ = self.m2_ + chunk_m2 + delta * delta * old_count * np.divide(chunk_count, new_count, out=np.zeros_like(new_count), where=new_count > 0)
        self.count_ = new_count
        for col_idx in range(X.shape[1]):
            vals = X[:, col_idx][valid[:, col_idx]]
            if vals.size:
                self.values_[col_idx] = np.concatenate([self.values_[col_idx], vals])
        return self

    def _require_values(self):
        """Raise a clear error when the stream has no valid values."""
        if self.count_ is None or np.sum(self.count_) == 0:
            raise ValueError("stream contains no valid values")

    @property
    def mean(self):
        """Return current mean as a scalar or column vector."""
        self._require_values()
        return float(self.mean_[0]) if self.mean_.size == 1 else self.mean_.copy()

    def variance(self, ddof: int = 0):
        """Return current variance as a scalar or column vector."""
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        self._require_values()
        denom = self.count_ - ddof
        var = np.divide(self.m2_, denom, out=np.full_like(self.m2_, np.nan), where=denom > 0)
        return float(var[0]) if var.size == 1 else var

    def std(self, ddof: int = 0):
        """Return current standard deviation."""
        return np.sqrt(self.variance(ddof=ddof))

    def quantile(self, q):
        """Return exact running quantiles for each column."""
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        self._require_values()
        values = np.asarray([np.quantile(vals, q) if vals.size else np.nan for vals in self.values_])
        return float(values[0]) if values.size == 1 else values

    def histogram(self, bins=10):
        """Return histograms for each running numeric column."""
        if bins <= 0:
            raise ValueError("bins must be positive")
        self._require_values()
        return [np.histogram(vals, bins=bins) for vals in self.values_]
