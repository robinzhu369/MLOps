"""Tests for tools/file_tools.py and tools/metadata_tools.py."""

import io
import os

import pandas as pd
import pytest

from tools.file_tools import save_uploaded_file, inspect_file_format, read_dataframe
from tools.metadata_tools import extract_file_metadata


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 35, None, 45],
        "income": [50000.0, 60000.0, 75000.0, 80000.0, 120000.0],
        "apply_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "bad_flag": [0, 0, 1, 0, 1],
    })
    path = str(tmp_path / "loan_application.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_excel(tmp_path):
    """Create a sample Excel file for testing."""
    df = pd.DataFrame({
        "account_id": ["A001", "A002", "A003"],
        "balance": [1000.0, 2000.0, 3000.0],
    })
    path = str(tmp_path / "accounts.xlsx")
    df.to_excel(path, index=False)
    return path


class TestSaveUploadedFile:
    def test_save_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = b"col1,col2\n1,a\n2,b\n"
        file_obj = io.BytesIO(content)

        result = save_uploaded_file(file_obj, "test.csv", "my_project")

        assert result["file_name"] == "test.csv"
        assert result["file_format"] == "csv"
        assert result["file_id"] != ""
        assert os.path.exists(result["file_path"])

        with open(result["file_path"], "rb") as f:
            assert f.read() == content

    def test_unsupported_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_obj = io.BytesIO(b"data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            save_uploaded_file(file_obj, "test.txt", "my_project")


class TestInspectFileFormat:
    def test_inspect_csv(self, sample_csv):
        result = inspect_file_format(sample_csv)

        assert result["file_format"] == "csv"
        assert result["readable"] is True
        assert result["error"] == ""
        assert result["file_size_bytes"] > 0

    def test_inspect_excel(self, sample_excel):
        result = inspect_file_format(sample_excel)

        assert result["file_format"] == "xlsx"
        assert result["readable"] is True

    def test_inspect_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            inspect_file_format("/nonexistent/file.csv")


class TestExtractFileMetadata:
    def test_basic_metadata(self, sample_csv):
        result = extract_file_metadata(sample_csv)

        assert result["n_rows"] == 5
        assert result["n_cols"] == 5
        assert "customer_id" in result["columns"]
        assert "bad_flag" in result["columns"]

    def test_column_profiles(self, sample_csv):
        result = extract_file_metadata(sample_csv)
        profiles = result["column_profiles"]

        # Numeric column
        assert profiles["income"]["dtype"] == "float64"
        assert profiles["income"]["missing_count"] == 0
        assert profiles["income"]["mean"] is not None

        # Column with missing values
        assert profiles["age"]["missing_count"] == 1
        assert profiles["age"]["missing_rate"] == 0.2

    def test_sample_values(self, sample_csv):
        result = extract_file_metadata(sample_csv, sample_size=3)
        samples = result["sample_values_masked"]

        assert len(samples["customer_id"]) == 3
        assert len(samples["income"]) == 3

    def test_masking_long_values(self):
        """Long string values should be partially masked."""
        from tools.metadata_tools import _get_masked_samples
        series = pd.Series(["a" * 30, "b" * 25])
        masked = _get_masked_samples(series, n=2)

        assert "***" in masked[0]
        assert len(masked[0]) < 30


class TestReadDataframe:
    def test_read_csv(self, sample_csv):
        df = read_dataframe(sample_csv)
        assert len(df) == 5
        assert "customer_id" in df.columns

    def test_read_excel(self, sample_excel):
        df = read_dataframe(sample_excel)
        assert len(df) == 3
        assert "account_id" in df.columns
