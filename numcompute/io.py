"""
numcompute.io
=============

CSV loading utilities for NumCompute.

Provides functionality to:
- Load CSV files into structured NumPy arrays
- Automatically infer column data types
- Handle missing values
- Support both numeric and categorical data

Author: Anshu Shrestha
"""

import numpy as np
from typing import List


class Data:
    """
    Base class representing a generic dataset.

    This class serves as a parent for specific data formats
    such as CSVData. It can be extended in the future to support
    additional data sources (e.g., JSON, databases).
    """
    pass


class Column:
    """
    Represents metadata for a single column in a dataset.

    Attributes:
        name (str): Name of the column.
        dtype (str): Inferred data type of the column.
    """

    def __init__(self, name: str, dtype: str):
        """
        Initialize a Column instance.

        Args:
            name (str): Column name.
            dtype (str): Data type of the column.
        """
        self.__name = name
        self.__dtype = dtype

    @property
    def name(self) -> str:
        """
        Get the column name.

        Returns:
            str: Column name.
        """
        return self.__name

    @property
    def dtype(self) -> str:
        """
        Get the column data type.

        Returns:
            str: Column data type.
        """
        return self.__dtype

    def __repr__(self) -> str:
        """
        String representation of the Column.

        Returns:
            str: Readable column description.
        """
        return f"Name: {self.__name}, Data Type: {self.__dtype}"


class CSVData(Data):
    """
    Represents CSV dataset with data and column metadata.

    Attributes:
        data (np.ndarray): Processed dataset.
        cols (List[Column]): List of column metadata.
    """

    def __init__(self, data: np.ndarray, cols: List[Column]):
        """
        Initialize a CSVData object.

        Args:
            data (np.ndarray): Processed data array.
            cols (List[Column]): Column metadata.
        """
        self.__data = data
        self.__cols = cols

    @property
    def data(self) -> np.ndarray:
        """
        Get dataset values.

        Returns:
            np.ndarray: Dataset array.
        """
        return self.__data

    @property
    def cols(self) -> List[Column]:
        """
        Get column metadata.

        Returns:
            List[Column]: List of columns.
        """
        return self.__cols

    def __repr__(self) -> str:
        """
        String representation of the dataset.

        Returns:
            str: Summary of dataset shape and headers.
        """
        header_list = [f"{col.name}: {col.dtype}" for col in self.__cols]
        return f"Shape: {self.__data.shape}\nHeaders: {', '.join(header_list)}"


class IO:
    """
    Utility class for handling CSV input/output operations.

    Provides static methods for:
    - Inferring column metadata
    - Converting raw CSV data to appropriate types
    - Loading CSV files into structured objects
    """

    @staticmethod
    def _get_cols(data: np.ndarray, has_headers: bool) -> List[Column]:
        """
        Infer column names and data types from raw CSV data.

        The dtype decision uses all non-missing values in each column, not only
        the first value. This avoids treating ["1", "2.5"] as an integer
        column and accidentally truncating 2.5 to 2.

        Args:
            data (np.ndarray): Raw CSV data as string array.
            has_headers (bool): Whether first row contains headers.

        Returns:
            List[Column]: List of inferred column metadata.
        """
        def is_null(val):
            return str(val).strip().lower() in ("", "none", "null", "nan")

        def is_bool(val):
            return str(val).strip().lower() in ("true", "false")

        def is_int(val):
            try:
                value = float(str(val).strip())
                return value.is_integer()
            except (ValueError, TypeError):
                return False

        def is_float(val):
            try:
                float(str(val).strip())
                return True
            except (ValueError, TypeError):
                return False

        num_cols = data.shape[1]

        if has_headers:
            col_names = data[0].astype(str).tolist()
        else:
            col_names = [f"col_{i}" for i in range(num_cols)]

        start_row = int(has_headers)
        cols = []

        for col_idx in range(num_cols):
            raw_col = data[start_row:, col_idx]
            valid_values = [val for val in raw_col if not is_null(val)]

            if len(valid_values) == 0:
                dtype = "NoneType"
            elif all(is_bool(val) for val in valid_values):
                dtype = "bool"
            elif all(is_int(val) for val in valid_values):
                dtype = "int"
            elif all(is_float(val) for val in valid_values):
                dtype = "float"
            else:
                dtype = "str"

            cols.append(Column(
                name=col_names[col_idx],
                dtype=dtype
            ))

        return cols

    @staticmethod
    def _convert_dtype(
        data: np.ndarray,
        has_headers: bool,
        cols: List[Column],
        fill_value: str
    ) -> np.ndarray:
        """
        Convert raw string data into appropriate NumPy dtypes.

        Handles integer, float, boolean, string, and all-missing columns.
        Mixed-type columns are kept as string/object columns.

        Args:
            data (np.ndarray): Raw CSV data.
            has_headers (bool): Whether headers are present.
            cols (List[Column]): Column metadata.
            fill_value (str): Value used for missing entries.

        Returns:
            np.ndarray: Converted dataset.
        """
        data = data[int(has_headers):]

        def is_null(val):
            return str(val).strip().lower() in ("", "none", "null", "nan")

        v_is_null = np.vectorize(is_null)
        converted_cols = []

        fill_lookup = {
            "nan": np.nan,
            "none": None,
        }
        fill = fill_lookup.get(str(fill_value).lower(), fill_value)

        for idx, col in enumerate(cols):
            raw_col = data[:, idx]
            dtype = col.dtype
            null_mask = v_is_null(raw_col)

            if dtype == "int":
                converted = np.full(raw_col.shape, np.nan, dtype=float)
                converted[~null_mask] = raw_col[~null_mask].astype(float).astype(int)

            elif dtype == "float":
                converted = np.full(raw_col.shape, np.nan, dtype=float)
                converted[~null_mask] = raw_col[~null_mask].astype(float)

            elif dtype == "bool":
                lowered = np.char.lower(raw_col.astype(str))
                converted = np.empty(raw_col.shape, dtype=object)
                converted[null_mask] = fill
                converted[~null_mask] = (lowered[~null_mask] == "true")

            elif dtype == "str":
                converted = np.empty(raw_col.shape, dtype=object)
                converted[null_mask] = fill
                converted[~null_mask] = raw_col[~null_mask].astype(str)

            else:
                converted = np.full(raw_col.shape, fill, dtype=object)

            converted_cols.append(converted)

        return np.column_stack(converted_cols)

    @staticmethod
    def load_csv(
        filepath: str,
        delimiter: str = ",",
        has_headers: bool = True,
        missing_value: str = "",
        fill_value: str = "nan"
    ) -> CSVData:
        """
        Load a CSV file and return a structured CSVData object.

        Features:
        - Automatic type inference
        - Missing value handling
        - Support for mixed data types

        Args:
            filepath (str): Path to CSV file.
            delimiter (str, optional): Field separator. Defaults to ",".
            has_headers (bool, optional): Whether first row contains headers.
            missing_value (str, optional): Representation of missing values in file.
            fill_value (str, optional): Replacement value ("nan" or "none").

        Returns:
            CSVData: Processed dataset with metadata.
        """
        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=False,
            dtype=str,
            encoding="utf-8",
            missing_values=missing_value,
            filling_values=fill_value,
            ndmin=2 # minimum 2 dimensions as we need it when only one column is there in the csv file
        )

        cols = IO._get_cols(data=data, has_headers=has_headers)

        converted_data = IO._convert_dtype(
            data=data,
            has_headers=has_headers,
            cols=cols,
            fill_value=fill_value
        )

        return CSVData(data=converted_data, cols=cols)
