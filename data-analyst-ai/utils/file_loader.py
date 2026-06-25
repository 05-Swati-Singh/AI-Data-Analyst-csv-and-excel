from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


def load_uploaded_file(uploaded_file) -> tuple[pd.DataFrame, dict[str, Any]]:
    file_name = uploaded_file.name
    suffix = Path(file_name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()
    file_bytes = BytesIO(raw_bytes)
    file_signature = hashlib.sha256(raw_bytes).hexdigest()

    if suffix == ".csv":
        dataframe = pd.read_csv(file_bytes)
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    dataframe.columns = dataframe.columns.astype(str).str.strip()
    metadata = {
        "file_name": file_name,
        "file_type": suffix.lstrip("."),
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "file_signature": file_signature,
    }
    return dataframe, metadata
