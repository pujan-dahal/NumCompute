"""
Basic CSV loading utilities for NumCompute.
Supports numeric + categorical data and missing values.
"""

import numpy as np
from typing import List

class Data:
    """Base class for Data"""
    pass

class Column:
    """Column class containing column name and data type"""
    def __init__(self, name, dtype):
        self.__name = name
        self.__dtype = dtype

    @property
    def name(self):
        return self.__name
    
    @property
    def dtype(self):
        return self.__dtype
    
    def __repr__(self):
        return f"Name: {self.__name}, Data Type: {self.__dtype}"


class CSVData(Data):
    """Class for CSV Data"""
    def __init__(self, data: np.ndarray, cols: List[Column]):
        self.__data = data
        self.__cols = cols

    @property
    def data(self):
        return self.__data
    
    @property
    def cols(self):
        return self.__cols
    
    def __repr__(self):
        header_list = [f"{col.name}: {col.dtype}" for col in self.__cols]
        return f"Shape: {self.__data.shape}\nHeaders: {', '.join(header_list)}"
    

class IO:
    @staticmethod
    def _get_cols(
        data: np.ndarray,
        has_headers: bool
    ) -> List[Column]:
        """
        Read column names and data types from the CSV data.
        """
        def is_int(val):
            try:
                int(val)
                return True
            except (ValueError, TypeError):
                return False

        def is_float(val):
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                return False

        def is_bool(val):
            return str(val).strip().lower() in ("true", "false")

        def is_null(val):
            return str(val).strip().lower() in ("", "none", "null", "nan")

        def identify_type(val):
            val = str(val).strip()
            if is_null(val):   return "NoneType"
            if is_bool(val):   return "bool"
            if is_int(val):    return "int"
            if is_float(val):  return "float"
            return "str"

        # vectorized functions to identify data types
        v_is_null = np.vectorize(is_null)
        v_identify_type = np.vectorize(identify_type)

        num_cols = data.shape[1]
        
        # mask the null values
        null_mask = v_is_null(data)
        valid_mask = ~null_mask

        cols = []
        
        if has_headers:
            col_names = data[0].astype(str).tolist()
        else:
            col_names = [f"col_{i}" for i in range(num_cols)] # if no headers header names are col_0, col_1, ...
        
        for col_idx in range(num_cols):
            # if headers start from 1th row
            col_valid = valid_mask[int(has_headers):, col_idx]
            if not col_valid.any():
                dtype = "NoneType"
                continue
            # get valid non-null values from this column
            valid_values = data[int(has_headers):, col_idx][col_valid]
            types = v_identify_type(valid_values)

            # take data type of first valid value
            dtype = types[0]

            cols.append(Column(
                name=col_names[col_idx],
                dtype=dtype
            ))

        return cols
    
    @staticmethod
    def _convert_dtype(data: np.ndarray, has_headers: bool, cols: List[Column], fill_value: str):
        # select data rows
        data = data[int(has_headers):]
        
        # create vectorized null mask
        def is_null(val):
            return str(val).strip().lower() in ("", "none", "null", "nan")

        v_is_null = np.vectorize(is_null)

        converted_cols = []

        for idx, col in enumerate(cols):
            raw_col = data[:, idx]
            dtype = col.dtype

            fill_dict = {
                'nan': np.nan,
                'none': None
            }
            fill = fill_dict[fill_value.lower()]

            # null detection
            null_mask = v_is_null(raw_col)

            # fully null column
            if np.all(null_mask):
                if dtype == 'str':
                    converted = np.full(raw_col.shape, None, dtype=object)
                else:
                    converted = np.full(raw_col.shape, np.nan)
                converted_cols.append(converted)
                continue

            elif dtype == "int":
                # int cannot store NaN, use float array
                converted = np.full(raw_col.shape, np.nan, dtype=float)

                valid_values = raw_col[~null_mask].astype(float)
                converted[~null_mask] = valid_values.astype(int)

            elif dtype == "float":
                converted = np.full(raw_col.shape, np.nan, dtype=float)

                valid_values = raw_col[~null_mask].astype(float)
                converted[~null_mask] = valid_values

            elif dtype == "bool":
                lowered = np.char.lower(raw_col.astype(str))

                # build as object array first
                converted = np.empty(raw_col.shape, dtype=object)
                converted[null_mask] = fill
                converted[~null_mask] = (lowered[~null_mask] == "true")

            elif dtype == "str":
                # keep strings as object so None can coexist
                converted = np.empty(raw_col.shape, dtype=object)
                converted[null_mask] = fill
                converted[~null_mask] = raw_col[~null_mask].astype(str)

            else:
                # fallback for NoneType / unknown dtype
                converted = np.where(
                    null_mask,
                    np.nan,
                    raw_col
                ).astype(object)

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
        Load a CSV file into a NumPy array.

        Supports:
        - numeric columns
        - categorical columns
        - missing values

        Returns:
            CSVData object
        """

        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=False,
            dtype=str,
            encoding="utf-8",
            missing_values=missing_value,
            filling_values=fill_value
        )

        cols = IO._get_cols(
            data=data,
            has_headers=has_headers
        )
        # convert data types to appropriate types
        converted_data = IO._convert_dtype(data=data, has_headers=has_headers, cols=cols, fill_value=fill_value)
        csv_data_obj = CSVData(data=converted_data, cols=cols)

        return csv_data_obj