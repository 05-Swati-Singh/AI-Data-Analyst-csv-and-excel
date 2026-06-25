from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    lowered = {column.lower(): column for column in df.columns}
    for keyword in keywords:
        for lower_name, original in lowered.items():
            if keyword in lower_name:
                return original
    return None


def _find_numeric_column(df: pd.DataFrame, preferred_keywords: list[str]) -> str | None:
    candidate = _find_column(df, preferred_keywords)
    if candidate is not None and pd.api.types.is_numeric_dtype(df[candidate]):
        return candidate
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns[0] if numeric_columns else None


def _find_date_column(df: pd.DataFrame) -> str | None:
    candidate = _find_column(df, ["date", "month", "quarter", "year", "time"])
    if candidate is not None:
        return candidate
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
    return None


def _rank_entities(df: pd.DataFrame, entity_column: str, metric_column: str, top_n: int = 5) -> pd.DataFrame:
    working = df[[entity_column, metric_column]].copy()
    working = working.dropna(subset=[entity_column, metric_column])
    ranked = working.groupby(entity_column)[metric_column].sum().reset_index().sort_values(metric_column, ascending=False)
    return ranked.head(top_n)


def generate_business_insights(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    metric_column = _find_numeric_column(df, ["revenue", "sales", "profit", "amount", "value", "total", "price"])
    entity_column = _find_column(df, ["product", "customer", "region", "category", "segment", "brand", "store", "channel"])
    customer_column = _find_column(df, ["customer", "client", "buyer"])
    region_column = _find_column(df, ["region", "market", "territory"])
    product_column = _find_column(df, ["product", "item", "sku", "service"])
    date_column = _find_date_column(df)

    insights: dict[str, Any] = {
        "kpis": [],
        "revenue_trends": [],
        "top_products": pd.DataFrame(),
        "underperforming_products": pd.DataFrame(),
        "customer_insights": pd.DataFrame(),
        "regional_insights": pd.DataFrame(),
        "data_quality": [],
    }

    insights["kpis"].append(f"Rows analyzed: {profile['rows']:,}")
    insights["kpis"].append(f"Columns analyzed: {profile['columns']:,}")
    insights["kpis"].append(f"Missing values: {profile['missing_cells']:,}")

    if metric_column is not None:
        total_metric = float(pd.to_numeric(df[metric_column], errors="coerce").fillna(0).sum())
        insights["kpis"].append(f"Total {metric_column}: {total_metric:,.2f}")

    if date_column is not None and metric_column is not None:
        monthly = df[[date_column, metric_column]].copy()
        monthly[date_column] = pd.to_datetime(monthly[date_column], errors="coerce")
        monthly[metric_column] = pd.to_numeric(monthly[metric_column], errors="coerce")
        monthly = monthly.dropna(subset=[date_column, metric_column])
        if not monthly.empty:
            monthly = monthly.set_index(date_column)[metric_column].resample("M").sum().reset_index()
            if len(monthly) >= 2:
                latest = float(monthly[metric_column].iloc[-1])
                previous = float(monthly[metric_column].iloc[-2])
                growth = ((latest - previous) / previous * 100) if previous != 0 else np.nan
                if not np.isnan(growth):
                    insights["revenue_trends"].append(
                        f"Latest monthly {metric_column} was {latest:,.2f}; prior month was {previous:,.2f}; growth was {growth:,.1f}%"
                    )
                else:
                    insights["revenue_trends"].append(f"Latest monthly {metric_column} was {latest:,.2f}.")

    if product_column is not None and metric_column is not None:
        top_products = _rank_entities(df, product_column, metric_column)
        bottom_products = df[[product_column, metric_column]].dropna().groupby(product_column)[metric_column].sum().reset_index().sort_values(metric_column, ascending=True).head(5)
        insights["top_products"] = top_products
        insights["underperforming_products"] = bottom_products
        if not top_products.empty:
            leader = top_products.iloc[0]
            insights["revenue_trends"].append(f"Top product: {leader[product_column]} with {leader[metric_column]:,.2f} in combined value.")

    if customer_column is not None and metric_column is not None:
        insights["customer_insights"] = _rank_entities(df, customer_column, metric_column)

    if region_column is not None and metric_column is not None:
        insights["regional_insights"] = _rank_entities(df, region_column, metric_column)

    if profile["missing_cells"] > 0:
        missing_sorted = profile["missing_summary"].sort_values("missing_count", ascending=False).head(5)
        for _, row in missing_sorted.iterrows():
            if row["missing_count"] > 0:
                insights["data_quality"].append(f"{row['column']} contains {int(row['missing_count'])} missing values ({row['missing_percent']:.2f}%).")

    if not insights["data_quality"]:
        insights["data_quality"].append("No significant data quality issues were detected in the visible sample.")

    if entity_column and metric_column:
        insights["kpis"].append(f"Primary entity column: {entity_column}")
        insights["kpis"].append(f"Primary metric column: {metric_column}")

    return insights


def build_insights_report(insights: dict[str, Any]) -> str:
    sections = ["<h4>Key KPIs</h4>"]
    sections.extend([f"<li>{item}</li>" for item in insights.get("kpis", [])])

    if insights.get("revenue_trends"):
        sections.append("<h4>Revenue Trends</h4>")
        sections.extend([f"<li>{item}</li>" for item in insights["revenue_trends"]])

    for title, frame_key in [
        ("Top Performing Products", "top_products"),
        ("Underperforming Products", "underperforming_products"),
        ("Customer Insights", "customer_insights"),
        ("Regional Insights", "regional_insights"),
    ]:
        frame = insights.get(frame_key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            sections.append(f"<h4>{title}</h4>")
            sections.append(frame.to_html(index=False, classes="table table-striped", border=0))

    sections.append("<h4>Data Quality</h4>")
    sections.extend([f"<li>{item}</li>" for item in insights.get("data_quality", [])])
    return "<ul>" + "".join(sections) + "</ul>"
