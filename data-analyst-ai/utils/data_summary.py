from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    column_types: pd.DataFrame
    missing_summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    correlation: pd.DataFrame
    sample_preview: pd.DataFrame


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    detected: list[str] = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            detected.append(column)
            continue
        if pd.api.types.is_object_dtype(df[column]):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.75:
                detected.append(column)
    return detected


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    date_columns = _detect_date_columns(df)
    categorical_columns = [column for column in df.columns if column not in numeric_columns and column not in date_columns]

    missing_counts = df.isna().sum()
    missing_summary = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_percent": np.round((missing_counts.values / max(len(df), 1)) * 100, 2),
        }
    )

    if numeric_columns:
        numeric_summary = df[numeric_columns].describe().transpose().reset_index().rename(columns={"index": "column"})
    else:
        numeric_summary = pd.DataFrame()

    cat_rows = []
    for column in categorical_columns[:25]:
        series = df[column].astype(str)
        value_counts = series.value_counts(dropna=True)
        cat_rows.append(
            {
                "column": column,
                "unique_values": series.nunique(dropna=True),
                "top_value": value_counts.index[0] if not value_counts.empty else None,
                "top_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
            }
        )
    categorical_summary = pd.DataFrame(cat_rows)

    correlation = df[numeric_columns].corr(numeric_only=True).round(3) if len(numeric_columns) > 1 else pd.DataFrame()

    column_types = pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": df.notna().sum().values,
            "nulls": df.isna().sum().values,
        }
    )

    profile = DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        date_columns=date_columns,
        column_types=column_types,
        missing_summary=missing_summary,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        correlation=correlation,
        sample_preview=df.head(10),
    )

    return profile.__dict__


def format_profile_for_llm(profile: dict[str, Any]) -> str:
    column_types = profile["column_types"]
    missing_summary = profile["missing_summary"]
    numeric_summary = profile["numeric_summary"]
    categorical_summary = profile["categorical_summary"]

    lines = [
        f"Rows: {profile['rows']}",
        f"Columns: {profile['columns']}",
        f"Missing cells: {profile['missing_cells']}",
        f"Duplicate rows: {profile['duplicate_rows']}",
        f"Numeric columns: {', '.join(profile['numeric_columns']) if profile['numeric_columns'] else 'None'}",
        f"Categorical columns: {', '.join(profile['categorical_columns']) if profile['categorical_columns'] else 'None'}",
        f"Date columns: {', '.join(profile['date_columns']) if profile['date_columns'] else 'None'}",
        "",
        "Column types:",
        column_types.to_string(index=False),
        "",
        "Missing summary:",
        missing_summary.head(20).to_string(index=False),
    ]

    if not numeric_summary.empty:
        lines.extend(["", "Numeric summary:", numeric_summary.head(20).to_string(index=False)])
    if not categorical_summary.empty:
        lines.extend(["", "Categorical summary:", categorical_summary.head(20).to_string(index=False)])

    return "\n".join(lines)
