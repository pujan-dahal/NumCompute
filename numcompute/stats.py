"""
Statistical utilities for NumCompute.
Works on CSV data loaded as string arrays by extracting numeric columns.
"""

import numpy as np
from io import CSVData

class Stats:
    @staticmethod
    def _get_numeric_cols(data: CSVData):
        return [idx for idx, col in enumerate(data.cols) if col.dtype in ['int', 'float']]

    @staticmethod
    def mean(data: CSVData, axis: int | None = None, ignore_nan: bool = True):
        # select numeric columns
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        mean_func = np.mean
        if ignore_nan:
            mean_func = np.nanmean
        if axis is None: # all data
            return mean_func(numeric_data)
        elif axis == 0: # column wise
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = mean_func(numeric_data, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1: # rowwise
            return mean_func(numeric_data, axis = 1)
        else:
            raise ValueError("axis must be None, 0, or 1")
        
    @staticmethod
    def median(data: CSVData, axis: int | None = None, ignore_nan: bool = True):
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        median_func = np.mean
        if ignore_nan:
            median_func = np.nanmean
        if axis is None: # all data
            return median_func(numeric_data)
        elif axis == 0: # column wise
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = median_func(numeric_data, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1: # rowwise
            return median_func(numeric_data, axis = 1)
        else:
            raise ValueError("axis must be None, 0, or 1")
        
    @staticmethod
    def std(data: CSVData, axis: int | None = None, ignore_nan: bool = True):
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        std_func = np.std
        if ignore_nan:
            std_func = np.nanstd
        if axis is None: # all data
            return std_func(numeric_data)
        elif axis == 0: # column wise
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = std_func(numeric_data, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1: # rowwise
            return std_func(numeric_data, axis = 1)
        else:
            raise ValueError("axis must be None, 0, or 1")
        
    @staticmethod
    def minimum(data: CSVData, axis: int | None = None):
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        if axis is None: # all data
            return np.nanmin(numeric_data)
        elif axis == 0: # column wise
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = np.nanmin(numeric_data, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1: # rowwise
            return np.nanmin(numeric_data, axis = 1)
        else:
            raise ValueError("axis must be None, 0, or 1")
        
    @staticmethod
    def maximum(data: CSVData, axis: int | None = None):
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        if axis is None: # all data
            return np.nanmax(numeric_data)
        elif axis == 0: # column wise
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = np.nanmax(numeric_data, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1: # rowwise
            return np.nanmax(numeric_data, axis = 1)
        else:
            raise ValueError("axis must be None, 0, or 1")
        
    @staticmethod
    def quantile(data: CSVData, q, axis: int | None = None, ignore_nan=True):
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        quantile_func = np.quantile
        if ignore_nan:
            quantile_func = np.nanquantile
        if axis is None: # all data
            return quantile_func(numeric_data, q)
        elif axis == 0: # column wise
            selected_cols = [data.cols[i].name for i in numeric_col_idx]
            means = quantile_func(numeric_data, q, axis=0)
            return dict(zip(selected_cols, means))
        elif axis == 1: # rowwise
            return quantile_func(numeric_data, q, axis = 1)
        else:
            raise ValueError("axis must be None, 0, or 1")
        
    @staticmethod
    def histogram(data: CSVData, bins: int = 10, axis: int | None = None):
        numeric_col_idx = Stats._get_numeric_cols(data)
        numeric_data = data.data[:, numeric_col_idx].astype(float)
        if axis is None:
            clean = numeric_data[~np.isnan(numeric_data)].flatten()
            return np.histogram(clean, bins=bins)
        elif axis == 0: # column-wise
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
            # row-wise
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