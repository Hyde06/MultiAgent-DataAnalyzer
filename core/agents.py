import pandas as pd
from typing import Any, Dict
import plotly.express as px
from core.sales_analyzer import SalesDataAnalyzer
from core.business_intelligence import BusinessIntelligence
from core.advanced_forecasting import AdvancedForecasting
from core.advanced_analysis import AdvancedAnalyzer
from core.llm_utils import generate_text

class BaseAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_results = None
        self.summary = None
    def analyze(self):
        raise NotImplementedError
    def summarize(self):
        raise NotImplementedError

class SalesAgent(BaseAgent):
    def analyze(self):
        self.analyzer = SalesDataAnalyzer(self.df)
        self.analysis_results = {
            'key_metrics': self.analyzer._calculate_key_metrics(),
            'sales_trends': self.analyzer.analyze_sales_trends(),
            'product_performance': self.analyzer.analyze_product_performance(),
            'customer_segments': self.analyzer.analyze_customer_segments(),
        }
        return self.analysis_results
    def summarize(self):
        prompt = f"""
You are a sales analytics expert. Given the following sales analysis results, write a clear, concise summary for a business audience. Use bullet points and highlight key findings and trends.

{self.analysis_results}

Summary:
"""
        self.summary = generate_text(prompt)
        return self.summary

class BIAgent(BaseAgent):
    def analyze(self):
        self.analyzer = BusinessIntelligence(self.df)
        self.analysis_results = {
            'kpis': self.analyzer.track_kpis(),
            'territory': self.analyzer.analyze_territories(),
            'competitors': self.analyzer.analyze_competitors(),
            'inventory': self.analyzer.optimize_inventory(),
            'clv': self.analyzer.calculate_customer_lifetime_value(),
        }
        return self.analysis_results
    def summarize(self):
        prompt = f"""
You are a business intelligence expert. Given the following BI analysis results, write a clear, concise summary for a business audience. Use bullet points and highlight actionable insights.

{self.analysis_results}

Summary:
"""
        self.summary = generate_text(prompt)
        return self.summary

class ForecastAgent(BaseAgent):
    def analyze(self):
        self.analyzer = AdvancedForecasting(self.df)
        self.analysis_results = self.analyzer.forecast_with_prophet(14)
        return self.analysis_results
    def summarize(self):
        prompt = f"""
You are a forecasting expert. Given the following forecast results, write a clear, concise summary for a business audience. Use bullet points and highlight key projections and uncertainties.

{self.analysis_results}

Summary:
"""
        self.summary = generate_text(prompt)
        return self.summary

class VisualizationAgent(BaseAgent):
    def analyze(self):
        self.analyzer = AdvancedAnalyzer(self.df)
        self.analysis_results = self.analyzer.generate_advanced_visualizations()
        return self.analysis_results
    def summarize(self):
        prompt = f"""
You are a data visualization expert. Given the following visual analysis results, write a clear, concise summary for a business audience. Use bullet points and highlight what the visualizations reveal about the data.

{self.analysis_results}

Summary:
"""
        self.summary = generate_text(prompt)
        return self.summary

class SalesVisualizationAgent(BaseAgent):
    def analyze(self):
        self.analyzer = SalesDataAnalyzer(self.df)
        visualizations = []
        # Sales trends
        sales_trends = self.analyzer.analyze_sales_trends()
        if sales_trends and 'daily_trend' in sales_trends:
            fig = px.line(sales_trends['daily_trend'], 
                         x=sales_trends['daily_trend'].columns[0],
                         y=sales_trends['daily_trend'].columns[1],
                         title="Daily Sales Trend")
            visualizations.append(("Daily Sales Trend", fig))
        # Product performance
        product_perf = self.analyzer.analyze_product_performance()
        if product_perf and 'top_products' in product_perf:
            fig = px.bar(product_perf['top_products'].reset_index(),
                        x=product_perf['top_products'].index.name,
                        y='sum',
                        title="Top 10 Products by Sales")
            visualizations.append(("Top Products", fig))
        # Customer segments
        customer_segments = self.analyzer.analyze_customer_segments()
        if customer_segments and 'customer_segments' in customer_segments:
            fig = px.box(customer_segments['customer_segments'],
                        x='segment',
                        y='sum',
                        title="Customer Segment Distribution")
            visualizations.append(("Customer Segments", fig))
        self.analysis_results = visualizations
        return visualizations
    def summarize(self):
        prompt = f"""
You are a sales data visualization expert. Given the following sales visualizations, write a clear, concise summary for a business audience. Use bullet points and highlight what the visualizations reveal about sales trends, products, and customers.

{[title for title, _ in self.analysis_results]}

Summary:
"""
        self.summary = generate_text(prompt)
        return self.summary

class AIAnalysisAgent:
    def __init__(self, agent_summaries: Dict[str, str]):
        self.agent_summaries = agent_summaries
        self.final_report = None
    def synthesize_report(self):
        prompt = f"""
You are a senior business analyst. Given the following summaries from specialized agents (Sales, BI, Forecast, Visualization), synthesize a comprehensive, actionable business report. Use clear headings, subheadings, and bullet points. Reference each agent's findings and provide overall recommendations.

{self.agent_summaries}

Final Report:
"""
        self.final_report = generate_text(prompt)
        return self.final_report 