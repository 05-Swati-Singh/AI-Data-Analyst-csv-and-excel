# AI Data Analyst

An AI-powered data analysis app for CSV and Excel files built with Streamlit, Pandas, Plotly, LangChain, and Google Gemini.

## Features

- Upload CSV and Excel files
- Automatic dataset profiling and quality checks
- Natural-language chat with AI-generated analysis
- Suggested questions for faster analyst prompts
- Dynamic Plotly chart generation
- Business insights and KPI summaries
- Time-series forecasting for sales and revenue
- Downloadable analysis report
- Session-state chat history

## Project Structure

```text
data-analyst-ai/
├── app.py
├── requirements.txt
├── .env
├── data/
├── utils/
│   ├── file_loader.py
│   ├── data_summary.py
│   ├── chart_generator.py
│   ├── forecasting.py
│   └── insights.py
├── llm/
│   ├── agent.py
│   └── prompts.py
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Google AI Studio API key to `.env`.

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-1.5-flash
```

4. Start the app.

```bash
streamlit run app.py
```

## Example Questions

- Which region generated the highest revenue?
- What are the top 5 selling products?
- Show monthly sales trends.
- Predict next month's sales.
- Which customers contribute most to revenue?
- Generate a profit analysis report.

## Notes

- Forecasting is implemented with scikit-learn for portability.
- The app uses deterministic pandas analysis for the core calculations and LangChain/Gemini for query routing and narrative responses.
- If no API key is configured, the app still loads the dataset and generates local analytics, but AI responses will be limited.
- Loading a new file clears the existing chat history so the conversation stays aligned with the active dataset.
