from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from llm.agent import build_query_plan, generate_answer, get_chat_model
from utils.chart_generator import ChartSpec, build_chart, build_correlation_heatmap
from utils.data_summary import build_dataset_profile, format_profile_for_llm
from utils.file_loader import load_uploaded_file
from utils.forecasting import detect_forecast_columns, forecast_time_series
from utils.insights import build_insights_report, generate_business_insights


APP_TITLE = "AI Data Analyst"
APP_ICON = "📊"
DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", os.getenv("OPENAI_MODEL", "gemini-1.5-flash"))


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.14), transparent 20%),
                linear-gradient(180deg, #f8fbff 0%, #eef4fa 100%);
            color: #0f172a;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 24px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(135deg, rgba(8, 18, 31, 0.96), rgba(15, 118, 110, 0.88));
            color: white;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
        }

        .hero h1, .hero p {
            margin: 0;
        }

        .hero p {
            opacity: 0.86;
            margin-top: 0.4rem;
            font-size: 1rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1rem 1rem 0.85rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }

        .metric-card .label {
            color: #64748b;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-card .value {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .metric-card .sub {
            color: #64748b;
            margin-top: 0.25rem;
            font-size: 0.92rem;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 0.8rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }

        .small-note {
            color: #64748b;
            font-size: 0.92rem;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.72);
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.05);
            padding: 0.25rem 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "chat_history": [],
        "df": None,
        "uploaded_name": "",
        "uploaded_signature": "",
        "profile": None,
        "file_loaded": False,
        "last_chart": None,
        "last_forecast": None,
        "last_insights": None,
        "last_response": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <p>Upload CSV or Excel data, ask questions in plain English, generate charts, uncover business insights, and forecast future sales.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(profile: dict[str, object]) -> None:
    rows = profile.get("rows", 0)
    columns = profile.get("columns", 0)
    missing_cells = profile.get("missing_cells", 0)
    duplicate_rows = profile.get("duplicate_rows", 0)
    numeric_columns = profile.get("numeric_columns", [])

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "Rows", rows, "records loaded"),
        (col2, "Columns", columns, "fields detected"),
        (col3, "Missing Cells", missing_cells, "values to review"),
        (col4, "Duplicates", duplicate_rows, "rows repeated"),
    ]
    for column, label, value, subtext in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="sub">{subtext}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"Numeric columns detected: {len(numeric_columns)}")


def build_analysis_report(
    df: pd.DataFrame,
    profile: dict[str, object],
    insights: dict[str, object] | None = None,
    forecast_result: dict[str, object] | None = None,
) -> str:
    lines = [
        f"AI Data Analyst Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Dataset: {st.session_state.uploaded_name or 'Uploaded file'}",
        f"Rows: {profile['rows']:,}",
        f"Columns: {profile['columns']:,}",
        f"Missing cells: {profile['missing_cells']:,}",
        f"Duplicate rows: {profile['duplicate_rows']:,}",
        "",
        "Numeric columns:",
        ", ".join(profile['numeric_columns']) if profile['numeric_columns'] else "None detected",
        "",
        "Categorical columns:",
        ", ".join(profile['categorical_columns']) if profile['categorical_columns'] else "None detected",
        "",
        "Date columns:",
        ", ".join(profile['date_columns']) if profile['date_columns'] else "None detected",
    ]

    if insights:
        lines.extend(["", "Key insights:"])
        for item in insights.get("kpis", [])[:5]:
            lines.append(f"- {item}")
        for item in insights.get("revenue_trends", [])[:5]:
            lines.append(f"- {item}")
        for item in insights.get("data_quality", [])[:5]:
            lines.append(f"- {item}")

    if forecast_result and not forecast_result.get("error"):
        forecast_frame = forecast_result.get("forecast_frame")
        lines.extend(["", "Forecast summary:", str(forecast_result.get("summary", "No summary available."))])
        if isinstance(forecast_frame, pd.DataFrame) and not forecast_frame.empty:
            lines.extend(["", forecast_frame.to_string(index=False)])

    lines.extend(["", "Suggested follow-up questions:"])
    lines.extend(
        [
            "- Which category contributes the most revenue?",
            "- Show the monthly trend for sales or revenue.",
            "- What are the top 5 products or regions?",
            "- Forecast the next month of sales.",
        ]
    )
    return "\n".join(lines)


def render_dataset_overview(df: pd.DataFrame, profile: dict[str, object]) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    st.write("A structured snapshot of the uploaded file. Use this panel to validate the data before chatting with the analyst.")

    left, right = st.columns([1.2, 0.8])
    with left:
        st.dataframe(df.head(15), use_container_width=True, height=340)
    with right:
        st.markdown("**Column types**")
        st.dataframe(profile["column_types"], use_container_width=True, height=340)

    st.markdown("**Missing value statistics**")
    missing_df = profile["missing_summary"]
    if missing_df.empty or missing_df["missing_count"].sum() == 0:
        st.success("No missing values were detected in this dataset.")
    else:
        missing_chart = px.bar(
            missing_df.sort_values("missing_count", ascending=False),
            x="column",
            y="missing_count",
            color="missing_percent",
            color_continuous_scale=["#d1fae5", "#0f766e"],
            labels={"missing_count": "Missing values", "missing_percent": "Missing %"},
            title="Missing Values by Column",
        )
        st.plotly_chart(missing_chart, use_container_width=True)
        st.dataframe(missing_df, use_container_width=True)

    st.markdown("**Descriptive statistics**")
    numeric_summary = profile["numeric_summary"]
    if not numeric_summary.empty:
        st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.info("No numeric columns were available for descriptive statistics.")

    st.markdown("**Correlation analysis**")
    correlation = profile["correlation"]
    if correlation is not None and not correlation.empty and correlation.shape[0] > 1:
        heatmap = build_correlation_heatmap(correlation)
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Correlation analysis requires at least two numeric columns.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_insights_section(df: pd.DataFrame, profile: dict[str, object]) -> dict[str, object]:
    insights = generate_business_insights(df, profile)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Business Insights")
    st.caption("Automatically generated findings from the uploaded dataset.")
    st.markdown(build_insights_report(insights), unsafe_allow_html=True)
    st.download_button(
        "Download analysis report",
        data=build_analysis_report(df, profile, insights=insights),
        file_name=f"{st.session_state.uploaded_name or 'analysis_report'}.txt".replace(" ", "_"),
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return insights


def render_forecast_section(df: pd.DataFrame) -> dict[str, object] | None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Sales Forecasting")
    st.caption("Forecast future revenue or sales from a date column and a numeric target column.")

    date_candidates, value_candidates = detect_forecast_columns(df)
    if not date_candidates or not value_candidates:
        st.warning("No clear date and numeric columns were detected automatically. Select suitable columns in the controls below.")

    forecast_controls = st.columns([1, 1, 0.7, 0.7])
    with forecast_controls[0]:
        date_column = st.selectbox(
            "Date column",
            options=list(df.columns),
            index=list(df.columns).index(date_candidates[0]) if date_candidates else 0,
        )
    with forecast_controls[1]:
        numeric_defaults = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        target_column = st.selectbox(
            "Target column",
            options=list(df.columns),
            index=list(df.columns).index(value_candidates[0]) if value_candidates else (list(df.columns).index(numeric_defaults[0]) if numeric_defaults else 0),
        )
    with forecast_controls[2]:
        horizon_label = st.selectbox("Forecast horizon", ["Next month", "Next quarter"], index=0)
    with forecast_controls[3]:
        run_forecast = st.button("Run forecast", use_container_width=True)

    forecast_output: dict[str, object] | None = None
    periods = 1 if horizon_label == "Next month" else 3
    if run_forecast:
        with st.spinner("Generating forecast..."):
            forecast_output = forecast_time_series(df=df, date_column=date_column, target_column=target_column, periods=periods)
        if forecast_output.get("error"):
            st.error(forecast_output["error"])
        else:
            st.metric("Forecast model", forecast_output["model_name"])
            st.metric("Future periods", periods)
            st.dataframe(forecast_output["forecast_frame"], use_container_width=True)
            st.plotly_chart(forecast_output["figure"], use_container_width=True)
            st.info(forecast_output["summary"])

    st.markdown("</div>", unsafe_allow_html=True)
    return forecast_output


def infer_default_chart_spec(question: str, df: pd.DataFrame) -> ChartSpec:
    plan = build_query_plan(question, format_profile_for_llm(build_dataset_profile(df)), list(df.columns), llm=None)
    return ChartSpec(
        chart_type=plan.chart_type or "bar",
        x_column=plan.x_column,
        y_column=plan.y_column,
        color_column=plan.color_column,
        title=plan.explanation or question,
        aggregation=plan.aggregation,
        top_n=plan.top_n,
    )


def handle_chat_question(question: str, df: pd.DataFrame, profile: dict[str, object], llm) -> dict[str, object]:
    profile_text = format_profile_for_llm(profile)
    plan_error: str | None = None
    try:
        plan = build_query_plan(question, profile_text, list(df.columns), llm=llm)
    except Exception as exc:
        plan = build_query_plan(question, profile_text, list(df.columns), llm=None)
        plan_error = str(exc)

    response_payload: dict[str, object] = {
        "plan": plan,
        "answer": "",
        "chart": None,
        "forecast": None,
        "analysis_frame": None,
        "llm_error": plan_error,
    }

    if plan.intent == "forecast":
        forecast_result = forecast_time_series(
            df=df,
            date_column=plan.x_column or profile.get("date_columns", [None])[0],
            target_column=plan.y_column or profile.get("numeric_columns", [None])[0],
            periods=plan.forecast_periods,
        )
        if forecast_result.get("error"):
            response_payload["answer"] = forecast_result["error"]
        else:
            response_payload["forecast"] = forecast_result
            try:
                response_payload["answer"] = generate_answer(question, plan, forecast_result, llm=llm)
            except Exception as exc:
                response_payload["llm_error"] = response_payload["llm_error"] or str(exc)
                response_payload["answer"] = generate_answer(question, plan, forecast_result, llm=None)
        return response_payload

    if plan.needs_chart or plan.chart_type:
        chart_spec = ChartSpec(
            chart_type=plan.chart_type or "bar",
            x_column=plan.x_column,
            y_column=plan.y_column,
            color_column=plan.color_column,
            title=plan.explanation or question,
            aggregation=plan.aggregation,
            top_n=plan.top_n,
        )
        chart_result = build_chart(df, chart_spec)
        if chart_result.get("error"):
            response_payload["answer"] = chart_result["error"]
        else:
            response_payload["chart"] = chart_result
            response_payload["analysis_frame"] = chart_result.get("dataframe")
            try:
                response_payload["answer"] = generate_answer(question, plan, chart_result, llm=llm)
            except Exception as exc:
                response_payload["llm_error"] = response_payload["llm_error"] or str(exc)
                response_payload["answer"] = generate_answer(question, plan, chart_result, llm=None)
        return response_payload

    insights = generate_business_insights(df, profile)
    try:
        response_payload["answer"] = generate_answer(question, plan, insights, llm=llm)
    except Exception as exc:
        response_payload["llm_error"] = response_payload["llm_error"] or str(exc)
        response_payload["answer"] = generate_answer(question, plan, insights, llm=None)
    return response_payload


def render_chat_section(df: pd.DataFrame, profile: dict[str, object], llm) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Chat Analyst")
    st.caption("Ask questions in natural language. The assistant will analyze the data and explain the result.")

    suggested_questions = [
        "Which region generated the highest revenue?",
        "What are the top 5 products by sales?",
        "Show the monthly sales trend.",
        "Forecast next month's sales.",
    ]
    st.markdown("**Suggested questions**")
    suggestion_columns = st.columns(2)
    selected_suggestion = None
    for index, suggestion in enumerate(suggested_questions):
        with suggestion_columns[index % 2]:
            if st.button(suggestion, key=f"suggested_question_{index}", use_container_width=True):
                selected_suggestion = suggestion

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chart") is not None:
                st.plotly_chart(message["chart"], use_container_width=True)
            if message.get("forecast") is not None:
                st.dataframe(message["forecast"]["forecast_frame"], use_container_width=True)
                st.plotly_chart(message["forecast"]["figure"], use_container_width=True)

    question = st.chat_input("Ask about revenue, trends, products, customers, or forecasts") or selected_suggestion
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Analyzing your dataset..."):
            result = handle_chat_question(question, df, profile, llm)

        if result.get("llm_error"):
            st.warning(f"Gemini API call failed, so the app used the local fallback: {result['llm_error']}")

        assistant_message = {"role": "assistant", "content": result["answer"]}
        if result.get("chart") is not None:
            assistant_message["chart"] = result["chart"]["figure"]
        if result.get("forecast") is not None:
            assistant_message["forecast"] = result["forecast"]

        st.session_state.chat_history.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result.get("chart") is not None:
                st.plotly_chart(result["chart"]["figure"], use_container_width=True)
                if result["chart"].get("dataframe") is not None:
                    st.dataframe(result["chart"]["dataframe"], use_container_width=True)
            if result.get("forecast") is not None:
                st.dataframe(result["forecast"]["forecast_frame"], use_container_width=True)
                st.plotly_chart(result["forecast"]["figure"], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(df: pd.DataFrame | None) -> tuple[str, str | None, object | None]:
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("Production-style analytics workspace")

    env_key = os.getenv("GOOGLE_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    sidebar_key = st.sidebar.text_input("Google AI Studio API Key", type="password", help="Optional. Overrides the .env value during this session.")
    effective_key = sidebar_key.strip() or env_key

    if effective_key:
        st.sidebar.success("Google API key configured")
    else:
        st.sidebar.warning("Google API key missing")

    model_name = st.sidebar.text_input("Gemini model", value=DEFAULT_MODEL)
    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    if df is not None:
        st.sidebar.markdown("**Dataset details**")
        st.sidebar.write(f"Rows: {len(df):,}")
        st.sidebar.write(f"Columns: {len(df.columns):,}")
        st.sidebar.write(f"File: {st.session_state.uploaded_name or 'Uploaded dataset'}")
        st.sidebar.write("Columns:")
        st.sidebar.caption(", ".join(df.columns.astype(str).tolist()))

    return model_name, effective_key, uploaded_file


def build_sidebar_info_box() -> None:
    st.sidebar.markdown(
        """
        <div class="section-card">
            <strong>Workflow</strong><br/>
            1. Upload a CSV or Excel file.<br/>
            2. Ask a question in plain English.<br/>
            3. Review charts, insights, and forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_empty_state() -> None:
    st.info("Upload a CSV or Excel file to start the analysis workflow.")
    st.write("The app will automatically profile the dataset, build charts, and answer questions using the uploaded file.")


def main() -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    apply_custom_styles()
    initialize_state()

    model_name, api_key, uploaded_file = render_sidebar(st.session_state.df)
    build_sidebar_info_box()

    if uploaded_file is not None:
        try:
            df, metadata = load_uploaded_file(uploaded_file)
            new_signature = metadata.get("file_signature", "")
            previous_signature = st.session_state.uploaded_signature
            if new_signature and new_signature != previous_signature:
                st.session_state.chat_history = []
                st.session_state.last_response = ""
                st.session_state.last_chart = None
                st.session_state.last_forecast = None
                st.session_state.last_insights = None
            st.session_state.df = df
            st.session_state.uploaded_name = metadata["file_name"]
            st.session_state.uploaded_signature = new_signature
            st.session_state.profile = build_dataset_profile(df)
            st.session_state.file_loaded = True
        except Exception as exc:
            st.sidebar.error(f"Unable to read file: {exc}")

    render_hero()

    if not st.session_state.file_loaded or st.session_state.df is None:
        create_empty_state()
        return

    df = st.session_state.df
    profile = st.session_state.profile or build_dataset_profile(df)
    render_metric_cards(profile)

    tabs = st.tabs(["Overview", "Chat", "Forecasting", "Insights", "Charts"])
    with tabs[0]:
        render_dataset_overview(df, profile)
    with tabs[1]:
        llm = get_chat_model(api_key, model_name)
        render_chat_section(df, profile, llm)
    with tabs[2]:
        forecast_result = render_forecast_section(df)
        st.session_state.last_forecast = forecast_result
    with tabs[3]:
        insights = render_insights_section(df, profile)
        st.session_state.last_insights = insights
    with tabs[4]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Quick Chart Builder")
        st.caption("Create a visualization from a natural-language style chart specification.")

        chart_question = st.text_input(
            "Describe the chart you want",
            value="Show sales by region",
            help="Examples: Show monthly revenue, compare product performance, plot profit by category.",
        )
        chart_spec = infer_default_chart_spec(chart_question, df)
        chart_result = build_chart(df, chart_spec)
        if chart_result.get("error"):
            st.warning(chart_result["error"])
        else:
            st.plotly_chart(chart_result["figure"], use_container_width=True)
            st.dataframe(chart_result["dataframe"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
