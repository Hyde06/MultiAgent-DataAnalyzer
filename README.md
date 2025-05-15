# Multi-Agent Data Analysis Automation

## Overview
This application is a modern, multi-agent data analysis platform for sales and business intelligence. It combines deterministic analytics, advanced forecasting, interactive visualizations, and natural language AI summaries—each powered by a dedicated agent. Users can also chat with their data for ad-hoc insights.

## Key Features
- **Multi-Agent Architecture:** Each analysis domain (Sales, BI, Forecast, Visualization) is handled by a specialized agent that provides both deterministic outputs and an AI-generated summary.
- **Agent-Based Tabs:** Each main tab (Sales, BI, Forecast, Visualization) shows raw analysis/visualizations first, followed by a natural language summary from the agent.
- **Comprehensive AI Analysis:** A dedicated AI Analysis agent synthesizes all agent summaries into a detailed, actionable business report.
- **Conversational Data Chat:** Chat with your data using an LLM-powered assistant, with full context from all analyses.
- **Advanced Visualizations:** Both sales-specific and advanced (PCA, clustering, correlation) visualizations, each with AI explanations.
- **Modern UI/UX:** Clean, responsive interface with chat bubbles, auto-scroll, and intuitive navigation.

## Project Structure
- `main.py` — Streamlit app entry point, orchestrates agents and UI
- `core/agents.py` — Agent classes for each analysis domain and the AI Analysis agent
- `core/sales_analyzer.py`, `core/business_intelligence.py`, `core/advanced_forecasting.py`, `core/advanced_analysis.py` — Deterministic analysis logic
- `core/llm_utils.py` — LLM (e.g., Gemini, OpenAI) integration for summaries and chat
- `requirements.txt` — Python dependencies
- `.gitignore` — Excludes data, notebooks, secrets, and build artifacts

## Setup
1. **Clone the repository**
2. **Create a virtual environment**
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your `.env` file** with your LLM API key(s)
5. **Run the app:**
   ```bash
   streamlit run main.py
   ```

## Usage
- **Upload your CSV data** in the sidebar.
- **Navigate tabs:**
  - **Sales Analysis, BI, Forecasting, Visualizations:**
    - See deterministic outputs/visuals first
    - Read the agent's AI summary below
  - **AI Analysis:**
    - Read a comprehensive, cross-agent business report
  - **Data Chat:**
    - Ask questions and get conversational, context-aware answers

## The Agent System
- **SalesAgent, BIAgent, ForecastAgent, VisualizationAgent, SalesVisualizationAgent:**
  - Each runs deterministic analysis and generates an LLM summary for its tab
- **AIAnalysisAgent:**
  - Synthesizes all agent summaries into a final business report
- **Data Chat:**
  - Uses all agent outputs as context for LLM-powered Q&A

## Development & Contribution
- **Testing:**
  - Run tests with `pytest`
- **Formatting:**
  - Format code with `black`
- **Extending:**
  - Add new agents for other domains as needed

## License
MIT 