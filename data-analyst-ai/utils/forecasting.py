from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    forecast_frame: pd.DataFrame
    figure: Any
    model_name: str
    summary: str


def detect_forecast_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    date_candidates = []
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()

    for column in df.columns:
        lower = column.lower()
        if any(keyword in lower for keyword in ["date", "month", "time", "quarter", "year"]):
            date_candidates.append(column)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_candidates.append(column)
            continue
        if pd.api.types.is_object_dtype(df[column]):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.75:
                date_candidates.append(column)

    return date_candidates, numeric_candidates


def _infer_frequency(date_series: pd.Series) -> str:
    ordered = pd.to_datetime(date_series, errors="coerce").dropna().sort_values()
    if len(ordered) < 3:
        return "M"
    deltas = ordered.diff().dropna().dt.days
    median_delta = deltas.median()
    if median_delta <= 3:
        return "D"
    if median_delta <= 10:
        return "W"
    return "M"


def forecast_time_series(df: pd.DataFrame, date_column: str | None, target_column: str | None, periods: int = 1) -> dict[str, Any]:
    try:
        if not date_column or not target_column:
            raise ValueError("Forecasting requires both a date column and a numeric target column.")

        frame = df[[date_column, target_column]].copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, target_column])

        if frame.empty:
            raise ValueError("Selected columns do not contain enough valid date and numeric values.")

        frequency = _infer_frequency(frame[date_column])
        resampled = frame.set_index(date_column)[target_column].resample(frequency).sum().dropna().reset_index()

        if len(resampled) < 3:
            raise ValueError("Not enough time-series history to build a stable forecast.")

        resampled["step"] = np.arange(len(resampled))
        model = LinearRegression()
        model.fit(resampled[["step"]], resampled[target_column])

        residuals = resampled[target_column] - model.predict(resampled[["step"]])
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        future_steps = np.arange(len(resampled), len(resampled) + periods)
        future_dates = pd.date_range(start=resampled[date_column].max(), periods=periods + 1, freq=frequency)[1:]
        predictions = model.predict(future_steps.reshape(-1, 1))
        margin = 1.96 * residual_std if residual_std > 0 else max(abs(predictions).mean() * 0.08, 1.0)

        forecast_frame = pd.DataFrame(
            {
                "date": future_dates,
                "forecast": predictions,
                "lower_bound": predictions - margin,
                "upper_bound": predictions + margin,
            }
        )

        historical = resampled[[date_column, target_column]].rename(columns={date_column: "date", target_column: "actual"})
        forecast_series = forecast_frame.rename(columns={"forecast": "predicted"})
        combined = pd.concat(
            [
                historical.assign(predicted=np.nan),
                forecast_series.assign(actual=np.nan),
            ],
            ignore_index=True,
        )

        figure = go.Figure()
        figure.add_trace(go.Scatter(x=historical["date"], y=historical["actual"], mode="lines+markers", name="Actual"))
        figure.add_trace(go.Scatter(x=forecast_frame["date"], y=forecast_frame["forecast"], mode="lines+markers", name="Forecast"))
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame["lower_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame["upper_bound"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(15,118,110,0.18)",
                line=dict(width=0),
                name="Confidence interval",
            )
        )
        figure.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=60, b=20), title="Forecast vs. Historical Data")

        summary = (
            f"Forecasted {periods} future period(s) using a linear trend model over {len(resampled)} aggregated observations. "
            f"Average uncertainty band: ±{margin:,.2f}."
        )
        return {
            "forecast_frame": forecast_frame,
            "figure": figure,
            "model_name": "LinearRegression",
            "summary": summary,
        }
    except Exception as exc:
        return {"error": str(exc)}
