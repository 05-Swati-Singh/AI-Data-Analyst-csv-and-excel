from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from llm.prompts import ANSWER_SYSTEM_PROMPT, QUERY_ROUTER_SYSTEM_PROMPT


@dataclass
class QueryPlan:
    intent: str = "analysis"
    chart_type: str | None = None
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    aggregation: str = "sum"
    forecast_periods: int = 0
    top_n: int = 10
    needs_chart: bool = False
    explanation: str = ""


def get_chat_model(api_key: str | None, model_name: str) -> ChatGoogleGenerativeAI | None:
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name, temperature=0.1)


def _pick_column(question: str, columns: list[str], keywords: list[str]) -> str | None:
    lowered_columns = {column.lower(): column for column in columns}
    question_lower = question.lower()

    for keyword in keywords:
        if keyword in question_lower:
            for lower_name, original in lowered_columns.items():
                if keyword in lower_name:
                    return original

    for lower_name, original in lowered_columns.items():
        if any(keyword in lower_name for keyword in keywords):
            return original
    return None


def _heuristic_plan(question: str, columns: list[str]) -> QueryPlan:
    question_lower = question.lower()
    chart_type = None
    intent = "analysis"
    needs_chart = False
    aggregation = "sum"
    forecast_periods = 0

    if any(keyword in question_lower for keyword in ["forecast", "predict", "next month", "next quarter", "future", "projection"]):
        intent = "forecast"
        forecast_periods = 3 if "quarter" in question_lower else 1
        chart_type = "line"
        needs_chart = True
    elif any(keyword in question_lower for keyword in ["show", "plot", "chart", "graph", "visualize", "display", "compare"]):
        intent = "chart"
        needs_chart = True

    x_column = _pick_column(question, columns, ["region", "category", "product", "customer", "segment", "brand", "store", "channel", "month", "date", "year", "quarter"])
    y_column = _pick_column(question, columns, ["revenue", "sales", "profit", "amount", "value", "total", "price", "quantity", "units"])

    if x_column is None:
        date_candidates = [column for column in columns if any(token in column.lower() for token in ["date", "month", "year", "quarter", "time"])]
        x_column = date_candidates[0] if date_candidates else None

    if y_column is None:
        metric_candidates = [column for column in columns if any(token in column.lower() for token in ["revenue", "sales", "profit", "amount", "value", "total", "price", "quantity", "units"])]
        y_column = metric_candidates[0] if metric_candidates else None

    if any(keyword in question_lower for keyword in ["top", "highest", "largest", "most", "best"]):
        aggregation = "sum"
        needs_chart = True
        if "top 5" in question_lower:
            top_n = 5
        else:
            top_n = 10
    elif any(keyword in question_lower for keyword in ["average", "avg", "mean"]):
        aggregation = "mean"
        top_n = 10
    else:
        top_n = 10

    if x_column and y_column:
        if any(keyword in question_lower for keyword in ["trend", "monthly", "time", "over time"]):
            chart_type = "line"
        elif any(keyword in question_lower for keyword in ["distribution", "spread", "histogram"]):
            chart_type = "histogram"
        elif any(keyword in question_lower for keyword in ["share", "composition", "breakdown"]):
            chart_type = "pie"
        elif chart_type is None:
            chart_type = "bar"
        needs_chart = True

    explanation = f"Using {x_column or 'available grouping columns'} and {y_column or 'available metric columns'} with {aggregation} aggregation."
    return QueryPlan(
        intent=intent,
        chart_type=chart_type,
        x_column=x_column,
        y_column=y_column,
        aggregation=aggregation,
        forecast_periods=forecast_periods,
        top_n=top_n,
        needs_chart=needs_chart,
        explanation=explanation,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "null", "none", "nan"}:
            return None
        return cleaned
    return str(value)


def build_query_plan(question: str, profile_text: str, columns: list[str], llm: ChatOpenAI | None = None) -> QueryPlan:
    heuristic = _heuristic_plan(question, columns)
    if llm is None:
        return heuristic

    messages = [
        SystemMessage(content=QUERY_ROUTER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Dataset columns: {', '.join(columns)}\n\nDataset profile:\n{profile_text}\n\nUser question: {question}\n\nHeuristic suggestion: {heuristic.__dict__}\n\nReturn the final JSON only."
        ),
    ]

    try:
        response = llm.invoke(messages)
        parsed = _extract_json(response.content)
        if not parsed:
            return heuristic
        return QueryPlan(
            intent=str(parsed.get("intent", heuristic.intent)),
            chart_type=_normalize_optional_string(parsed.get("chart_type")) or heuristic.chart_type,
            x_column=_normalize_optional_string(parsed.get("x_column")) or heuristic.x_column,
            y_column=_normalize_optional_string(parsed.get("y_column")) or heuristic.y_column,
            color_column=_normalize_optional_string(parsed.get("color_column")) or heuristic.color_column,
            aggregation=str(parsed.get("aggregation", heuristic.aggregation)),
            forecast_periods=int(parsed.get("forecast_periods", heuristic.forecast_periods) or 0),
            top_n=int(parsed.get("top_n", heuristic.top_n) or heuristic.top_n),
            needs_chart=bool(parsed.get("needs_chart", heuristic.needs_chart)),
            explanation=str(parsed.get("explanation", heuristic.explanation)),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini routing failed: {exc}") from exc


def generate_answer(question: str, plan: QueryPlan, payload: dict[str, Any], llm: ChatOpenAI | None = None) -> str:
    if llm is None:
        return _fallback_answer(question, plan, payload)

    compact_payload = _summarize_payload(payload)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nPlan: {plan.__dict__}\n\nAnalysis payload:\n{compact_payload}\n\nWrite a concise business answer with reasoning and any caveats."),
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini answer generation failed: {exc}") from exc


def _summarize_payload(payload: dict[str, Any]) -> str:
    if "summary" in payload:
        return str(payload["summary"])
    if "forecast_frame" in payload:
        return payload["forecast_frame"].head(10).to_string(index=False)
    if "dataframe" in payload and payload["dataframe"] is not None:
        data = payload["dataframe"]
        if hasattr(data, "head"):
            return data.head(10).to_string(index=False)
    if isinstance(payload, dict):
        lines = []
        for key, value in payload.items():
            if hasattr(value, "head"):
                lines.append(f"{key}:\n{value.head(5).to_string(index=False)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n\n".join(lines)
    return str(payload)


def _fallback_answer(question: str, plan: QueryPlan, payload: dict[str, Any]) -> str:
    if "forecast_frame" in payload:
        frame = payload["forecast_frame"]
        forecast_value = float(frame["forecast"].iloc[0]) if not frame.empty else 0.0
        return (
            f"I forecasted {plan.forecast_periods} future period(s) using the detected time series. "
            f"The next predicted value is {forecast_value:,.2f}. "
            f"Reasoning: {plan.explanation}"
        )

    if "dataframe" in payload and payload["dataframe"] is not None:
        data = payload["dataframe"]
        if hasattr(data, "iloc") and not data.empty:
            top_row = data.iloc[0]
            x_value = top_row.iloc[0]
            y_value = top_row.iloc[-1]
            if isinstance(y_value, (int, float)):
                metric_text = f"{y_value:,.2f}"
            else:
                metric_text = str(y_value)
            return f"The strongest result in the aggregated analysis is {x_value} with {metric_text} on the selected metric. Reasoning: {plan.explanation}"

    if payload.get("revenue_trends"):
        trend = payload["revenue_trends"][0]
        return f"{trend} Reasoning: I summarized the dataset using the strongest available metric and entity columns."

    return f"I reviewed the dataset for your question: {question}. Reasoning: {plan.explanation} The answer is based on the available columns and data quality profile."
