import numpy as np
import pytest

from numcompute.io import IO, CSVData, Column


def test_column_properties():
    col = Column("age", "int")

    assert col.name == "age"
    assert col.dtype == "int"
    assert "age" in repr(col)
    assert "int" in repr(col)


def test_csvdata_properties():
    data = np.array([[1, 2], [3, 4]])
    cols = [Column("a", "int"), Column("b", "int")]

    csv_data = CSVData(data, cols)

    assert np.array_equal(csv_data.data, data)
    assert csv_data.cols == cols
    assert "Shape" in repr(csv_data)
    assert "Headers" in repr(csv_data)


def test_load_csv_with_headers_and_missing_values(tmp_path):
    csv_file = tmp_path / "sample.csv"

    csv_file.write_text(
        "age,height,name,passed\n"
        "20,170.5,Alice,true\n"
        "21,,Bob,false\n"
        ",180.0,,true\n"
    )

    result = IO.load_csv(
        filepath=str(csv_file),
        delimiter=",",
        has_headers=True
    )

    assert isinstance(result, CSVData)
    assert result.data.shape == (3, 4)

    col_names = [col.name for col in result.cols]
    col_types = [col.dtype for col in result.cols]

    assert col_names == ["age", "height", "name", "passed"]
    assert col_types == ["int", "float", "str", "bool"]

    age_col = result.data[:, 0].astype(float)
    height_col = result.data[:, 1].astype(float)

    assert np.isclose(age_col[0], 20.0)
    assert np.isclose(age_col[1], 21.0)
    assert np.isnan(age_col[2])

    assert np.isclose(height_col[0], 170.5)
    assert np.isnan(height_col[1])
    assert np.isclose(height_col[2], 180.0)

    assert result.data[0, 2] == "Alice"
    assert result.data[1, 2] == "Bob"
    assert result.data[0, 3] is True
    assert result.data[1, 3] is False


def test_load_csv_without_headers(tmp_path):
    csv_file = tmp_path / "no_headers.csv"

    csv_file.write_text(
        "1,2.5\n"
        "3,4.5\n"
    )

    result = IO.load_csv(
        filepath=str(csv_file),
        delimiter=",",
        has_headers=False
    )

    assert result.data.shape == (2, 2)

    col_names = [col.name for col in result.cols]
    col_types = [col.dtype for col in result.cols]

    assert col_names == ["col_0", "col_1"]
    assert col_types == ["int", "float"]

    assert np.allclose(result.data.astype(float), np.array([[1, 2.5], [3, 4.5]]))


def test_load_csv_with_tab_delimiter(tmp_path):
    csv_file = tmp_path / "tab_file.tsv"

    csv_file.write_text(
        "age\theight\n"
        "20\t170\n"
        "21\t175\n"
    )

    result = IO.load_csv(
        filepath=str(csv_file),
        delimiter="\t",
        has_headers=True
    )

    assert result.data.shape == (2, 2)

    col_names = [col.name for col in result.cols]
    assert col_names == ["age", "height"]

    assert np.allclose(result.data.astype(float), np.array([[20, 170], [21, 175]]))


def test_load_csv_string_column(tmp_path):
    csv_file = tmp_path / "strings.csv"

    csv_file.write_text(
        "name,city\n"
        "Alice,Adelaide\n"
        "Bob,Melbourne\n"
    )

    result = IO.load_csv(
        filepath=str(csv_file),
        delimiter=",",
        has_headers=True
    )

    col_types = [col.dtype for col in result.cols]

    assert col_types == ["str", "str"]
    assert result.data[0, 0] == "Alice"
    assert result.data[1, 1] == "Melbourne"


def test_load_csv_boolean_column(tmp_path):
    csv_file = tmp_path / "bools.csv"

    csv_file.write_text(
        "passed\n"
        "true\n"
        "false\n"
        "true\n"
    )

    result = IO.load_csv(
        filepath=str(csv_file),
        delimiter=",",
        has_headers=True
    )

    assert result.cols[0].dtype == "bool"
    assert result.data[0, 0] is True
    assert result.data[1, 0] is False
    assert result.data[2, 0] is True


def test_load_csv_invalid_file_raises_error():
    with pytest.raises(Exception):
        IO.load_csv("file_that_does_not_exist.csv")
