# 📊 AI Data Analyst for CSV & Excel Files

An AI-powered data analysis assistant that enables users to upload CSV and Excel files, ask questions in natural language, generate insights, create visualizations, and perform exploratory data analysis without writing SQL or Python code.

## 🚀 Features

* 📂 Upload CSV and Excel datasets
* 🤖 Natural Language Data Analysis using LLMs
* 📈 Automatic Data Visualization
* 🔍 Smart Data Exploration & Insights
* 📊 Statistical Summaries
* 📉 Trend & Forecast Analysis
* ⚡ Interactive Streamlit Interface
* 🧠 AI-generated Business Insights
* 📋 Dataset Overview & Health Checks

---

## 📸 Application Screenshots

### Dashboard

<img width="1277" height="607" alt="Screenshot 2026-06-25 225636" src="https://github.com/user-attachments/assets/3e3d2b5a-55b9-47be-934f-a3508d851aca" />


### AI Analysis Results

<img width="1278" height="641" alt="Screenshot 2026-06-26 001043" src="https://github.com/user-attachments/assets/66a11b8f-d8dd-4cbf-aa34-5d1fe0afaf84" />


---

## 🏗️ Project Architecture

```text
User Uploads Dataset
        │
        ▼
Data Processing Layer
        │
        ▼
AI Agent (LLM)
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Insights Charts Forecasts
        │
        ▼
Streamlit UI
```

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI & Data Processing

* LangChain
* Google Gemini API
* Pandas
* NumPy

### Visualization

* Matplotlib
* Plotly

### Forecasting & Analytics

* Statistical Analysis
* Time Series Forecasting

---

## 📂 Project Structure

```text
data-analyst-ai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── llm/
│   ├── agent.py
│   └── prompts.py
│
├── utils/
│   ├── file_loader.py
│   ├── data_summary.py
│   ├── chart_generator.py
│   ├── forecasting.py
│   └── insights.py
│
└── data/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd AI-Data-Analyst-csv-and-excel
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

* Business Intelligence
* Sales Analysis
* Financial Reporting
* Marketing Analytics
* Customer Segmentation
* Forecasting & Trend Detection
* Operational Performance Monitoring

---

## 📈 Future Enhancements

* Multi-file Analysis
* Database Connectivity
* Advanced Forecasting Models
* PDF Report Generation
* Dashboard Export
* Real-time Data Sources
* Conversational Analytics

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is intended for educational and research purposes.

---

### Author

**Swati Singh**

Built with ❤️ using Python, Streamlit, LangChain, and Gemini AI.
