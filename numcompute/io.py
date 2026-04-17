"""
numcompute/io.py
================

Basic CSV loading utilities for NumCompute.

This is the starting point for handling tabular data using NumPy.
Focus is on simplicity and core functionality.

Author: ( Anshu Shrestha)
"""

import numpy as np
from typing import Iterator, List


# ===========================================================================
# Core Functions 
# ===========================================================================

def load_csv(
    filepath: str,
    delimiter: str = ",",
    skip_header: bool = True,
) -> np.ndarray:
    """
    Load a CSV file into a NumPy array.

    Parameters
    ----------
    filepath : str
        Path to the CSV file
    delimiter : str
        Column separator (default: comma)
    skip_header : bool
        Skip first row if it contains column names

    Returns
    -------
    np.ndarray
        2D array of data
    """
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=int(skip_header),
        dtype=float
    )

    # Ensure output is always 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data


def get_headers(
    filepath: str,
    delimiter: str = ","
) -> List[str]:
    """
    Read column names from the first row of a CSV file.

    Parameters
    ----------
    filepath : str
        Path to file
    delimiter : str
        Column separator

    Returns
    -------
    list of str
        Column names
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    return [col.strip() for col in first_line.split(delimiter)]


def load_csv_chunked(
    filepath: str,
    chunk_size: int = 1000,
    delimiter: str = ",",
    skip_header: bool = True,
) -> Iterator[np.ndarray]:
    """
    Load a CSV file in chunks (useful for large files).

    Parameters
    ----------
    filepath : str
        Path to CSV file
    chunk_size : int
        Number of rows per chunk
    delimiter : str
        Column separator
    skip_header : bool
        Skip first row

    Yields
    ------
    np.ndarray
        Chunk of data
    """
    with open(filepath, "r") as f:
        if skip_header:
            next(f)

        buffer = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            buffer.append(line)

            if len(buffer) == chunk_size:
                yield _parse_chunk(buffer, delimiter)
                buffer = []

        if buffer:
            yield _parse_chunk(buffer, delimiter)


# ===========================================================================
# Helper Function
# ===========================================================================

def _parse_chunk(lines: List[str], delimiter: str) -> np.ndarray:
    """
    Convert raw lines into a NumPy array.
    """
    rows = []
    for line in lines:
        values = [float(x) for x in line.split(delimiter)]
        rows.append(values)

    return np.array(rows)
    