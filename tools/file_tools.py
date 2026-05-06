"""File upload and inspection tools."""

import os
import shutil
import uuid
from typing import IO

import pandas as pd

from core.constants import SUPPORTED_FILE_FORMATS


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_uploaded_file(
    file_obj: IO[bytes],
    file_name: str,
    project_name: str,
) -> dict:
    """Save an uploaded file to the project's raw_files directory.

    Returns metadata about the saved file.
    """
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext not in SUPPORTED_FILE_FORMATS:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {SUPPORTED_FILE_FORMATS}"
        )

    raw_dir = os.path.join("artifacts", project_name, "raw_files")
    _ensure_dir(raw_dir)

    dest_path = os.path.join(raw_dir, file_name)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file_obj, f)

    file_id = uuid.uuid4().hex[:12]

    return {
        "file_id": file_id,
        "file_name": file_name,
        "file_path": dest_path,
        "file_format": ext,
    }


def save_uploaded_file_from_path(
    source_path: str,
    project_name: str,
) -> dict:
    """Save a file from a local path to the project's raw_files directory."""
    file_name = os.path.basename(source_path)
    with open(source_path, "rb") as f:
        return save_uploaded_file(f, file_name, project_name)


def inspect_file_format(file_path: str) -> dict:
    """Inspect a file and return basic format information.

    Returns file format, size, and whether it can be read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = os.path.basename(file_path)
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    file_size = os.path.getsize(file_path)

    readable = False
    error_msg = ""

    try:
        df = _read_file(file_path, ext, nrows=5)
        readable = True
    except Exception as e:
        error_msg = str(e)

    return {
        "file_name": file_name,
        "file_format": ext,
        "file_size_bytes": file_size,
        "readable": readable,
        "error": error_msg,
    }


def _read_file(file_path: str, ext: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a file into a DataFrame based on its extension."""
    if ext == "csv":
        return pd.read_csv(file_path, nrows=nrows)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(file_path, nrows=nrows)
    elif ext == "parquet":
        df = pd.read_parquet(file_path)
        if nrows is not None:
            df = df.head(nrows)
        return df
    else:
        raise ValueError(f"Unsupported format: {ext}")


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read a file into a full DataFrame."""
    ext = file_path.rsplit(".", 1)[-1].lower()
    return _read_file(file_path, ext)
