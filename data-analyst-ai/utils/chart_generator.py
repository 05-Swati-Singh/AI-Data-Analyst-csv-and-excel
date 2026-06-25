from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class ChartSpec:
    chart_type: str = "bar"
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    title: str = "Analysis Chart"
    aggregation: str = "sum"
    top_n: int = 10


def _resolve_aggregation(series: pd.Series, aggregation: str) -> Any:
    aggregation = aggregation.lower()
    if aggregation == "mean":
        return series.mean()
    if aggregation == "median":
        return series.median()
    if aggregation == "max":
        return series.max()
    if aggregation == "min":
        return series.min()
    if aggregation == "count":
        return series.count()
    return series.sum()


def _aggregate_dataframe(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if not spec.x_column:
        raise ValueError("No x-axis column was identified for the chart.")

    working_df = df.copy()
    x_column = spec.x_column
    y_column = spec.y_column

    if y_column and y_column in working_df.columns and pd.api.types.is_numeric_dtype(working_df[y_column]):
        working_df = working_df[[x_column, y_column]].copy()
        working_df = working_df.dropna(subset=[x_column, y_column])
        grouped = working_df.groupby(x_column, dropna=False)[y_column].apply(lambda values: _resolve_aggregation(values, spec.aggregation)).reset_index()
        grouped = grouped.sort_values(y_column, ascending=False)
    else:
        grouped = working_df.groupby(x_column, dropna=False).size().reset_index(name="count")
        grouped = grouped.sort_values("count", ascending=False)

    if spec.chart_type in {"bar", "pie"} and len(grouped) > spec.top_n:
        grouped = grouped.head(spec.top_n).copy()
    return grouped


def build_chart(df: pd.DataFrame, spec: ChartSpec) -> dict[str, Any]:
    try:
        aggregated = _aggregate_dataframe(df, spec)
        chart_type = spec.chart_type.lower()

        if chart_type == "bar":
            y_axis = spec.y_column if spec.y_column in aggregated.columns else aggregated.columns[-1]
            figure = px.bar(aggregated, x=aggregated.columns[0], y=y_axis, title=spec.title)
        elif chart_type == "line":
            x_axis = aggregated.columns[0]
            y_axis = spec.y_column if spec.y_column in aggregated.columns else aggregated.columns[-1]
            figure = px.line(aggregated.sort_values(x_axis), x=x_axis, y=y_axis, markers=True, title=spec.title)
        elif chart_type == "pie":
            y_axis = spec.y_column if spec.y_column in aggregated.columns else aggregated.columns[-1]
            figure = px.pie(aggregated, names=aggregated.columns[0], values=y_axis, title=spec.title)
        elif chart_type == "scatter":
            if not spec.x_column or not spec.y_column:
                raise ValueError("Scatter plots require both x and y columns.")
            subset = df[[spec.x_column, spec.y_column]].dropna()
            figure = px.scatter(subset, x=spec.x_column, y=spec.y_column, title=spec.title)
            aggregated = subset
        elif chart_type == "histogram":
            histogram_column = spec.y_column or spec.x_column
            if histogram_column is None:
                raise ValueError("Histogram plots require a numeric column.")
            subset = df[[histogram_column]].dropna()
            figure = px.histogram(subset, x=histogram_column, title=spec.title)
            aggregated = subset
        else:
            raise ValueError(f"Unsupported chart type: {spec.chart_type}")

        figure.update_layout(
            template="plotly_white",
            title=dict(x=0.02, xanchor="left"),
            margin=dict(l=20, r=20, t=60, b=20),
            height=520,
            legend_title_text="",
        )
        return {
            "figure": figure,
            "dataframe": aggregated,
            "summary": f"Generated a {chart_type} chart using {spec.x_column} and {spec.y_column or 'count'}.",
        }
    except Exception as exc:
        return {"error": str(exc)}


def build_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    figure = px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale=["#e0f2fe", "#0f766e"],
        aspect="auto",
        title="Numeric Correlation Matrix",
    )
    figure.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=60, b=20))
    return figure
