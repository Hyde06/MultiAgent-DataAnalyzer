import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from core.advanced_analysis import AdvancedAnalyzer
from core.column_analyzer import ColumnAnalyzer

class SalesDataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize the sales data analyzer with a DataFrame"""
        self.df = df
        self.advanced_analyzer = AdvancedAnalyzer(df)
        self.column_analyzer = ColumnAnalyzer(df)
        
    def analyze_sales_trends(self) -> Dict[str, Any]:
        """Analyze sales trends over time"""
        # Assuming there's a date column and sales amount column
        date_col = self._find_date_column()
        sales_col = self._find_sales_column()
        
        if date_col and sales_col:
            # Convert to datetime if not already
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Daily sales trend
            daily_sales = self.df.groupby(date_col)[sales_col].sum().reset_index()
            
            # Monthly sales trend
            monthly_sales = self.df.groupby(self.df[date_col].dt.to_period('M'))[sales_col].sum()
            
            # Year-over-year growth
            yoy_growth = self._calculate_yoy_growth(monthly_sales)
            
            return {
                'daily_trend': daily_sales,
                'monthly_trend': monthly_sales,
                'yoy_growth': yoy_growth
            }
        return {}
    
    def analyze_product_performance(self) -> Dict[str, Any]:
        """Analyze product performance metrics"""
        product_col = self._find_product_column()
        sales_col = self._find_sales_column()
        
        if product_col and sales_col:
            # Product sales analysis
            product_sales = self.df.groupby(product_col)[sales_col].agg([
                'sum', 'mean', 'count'
            ]).sort_values('sum', ascending=False)
            
            # Top performing products
            top_products = product_sales.head(10)
            
            # Product category analysis if category column exists
            category_col = self._find_category_column()
            if category_col:
                category_sales = self.df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
            else:
                category_sales = None
                
            return {
                'product_sales': product_sales,
                'top_products': top_products,
                'category_sales': category_sales
            }
        return {}
    
    def analyze_customer_segments(self) -> Dict[str, Any]:
        """Analyze customer segments and behavior"""
        customer_col = self._find_customer_column()
        sales_col = self._find_sales_column()
        
        if customer_col and sales_col:
            # Customer value analysis
            customer_value = self.df.groupby(customer_col)[sales_col].agg([
                'sum', 'mean', 'count'
            ]).sort_values('sum', ascending=False)
            
            # RFM Analysis
            rfm = self._calculate_rfm()
            
            # Customer segments based on value
            segments = pd.qcut(customer_value['sum'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
            customer_value['segment'] = segments
            
            return {
                'customer_value': customer_value,
                'rfm_analysis': rfm,
                'customer_segments': customer_value
            }
        return {}
    
    def generate_sales_insights(self) -> Dict[str, Any]:
        """Generate comprehensive sales insights"""
        insights = {
            'sales_trends': self.analyze_sales_trends(),
            'product_performance': self.analyze_product_performance(),
            'customer_segments': self.analyze_customer_segments(),
            'correlation_analysis': self.advanced_analyzer.analyze_correlations()
        }
        
        # Add key metrics
        insights['key_metrics'] = self._calculate_key_metrics()
        
        return insights
    
    def _find_date_column(self) -> str:
        """Find the date column in the dataset"""
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        return date_columns[0] if date_columns else None
    
    def _find_sales_column(self) -> str:
        """Find the sales amount column in the dataset"""
        sales_columns = [col for col in self.df.columns if 'sales' in col.lower() or 'amount' in col.lower() or 'revenue' in col.lower()]
        return sales_columns[0] if sales_columns else None
    
    def _find_product_column(self) -> str:
        """Find the product column in the dataset"""
        product_columns = [col for col in self.df.columns if 'product' in col.lower() or 'item' in col.lower()]
        return product_columns[0] if product_columns else None
    
    def _find_customer_column(self) -> str:
        """Find the customer column in the dataset"""
        customer_columns = [col for col in self.df.columns if 'customer' in col.lower() or 'client' in col.lower()]
        return customer_columns[0] if customer_columns else None
    
    def _find_category_column(self) -> str:
        """Find the category column in the dataset"""
        category_columns = [col for col in self.df.columns if 'category' in col.lower() or 'type' in col.lower()]
        return category_columns[0] if category_columns else None
    
    def _calculate_yoy_growth(self, monthly_sales: pd.Series) -> float:
        """Calculate year-over-year growth rate"""
        if len(monthly_sales) >= 12:
            current_year = monthly_sales[-12:].sum()
            previous_year = monthly_sales[-24:-12].sum()
            return ((current_year - previous_year) / previous_year) * 100
        return 0.0
    
    def _calculate_rfm(self) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        date_col = self._find_date_column()
        customer_col = self._find_customer_column()
        sales_col = self._find_sales_column()
        
        if all([date_col, customer_col, sales_col]):
            # Convert to datetime
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Calculate RFM metrics
            rfm = self.df.groupby(customer_col).agg({
                date_col: lambda x: (self.df[date_col].max() - x.max()).days,  # Recency
                sales_col: ['count', 'sum']  # Frequency and Monetary
            })
            
            rfm.columns = ['recency', 'frequency', 'monetary']
            return rfm
        return pd.DataFrame()
    
    def _calculate_key_metrics(self) -> Dict[str, float]:
        """Calculate key sales metrics"""
        sales_col = self._find_sales_column()
        if sales_col:
            return {
                'total_sales': self.df[sales_col].sum(),
                'average_order_value': self.df[sales_col].mean(),
                'total_orders': len(self.df),
                'unique_customers': self.df[self._find_customer_column()].nunique() if self._find_customer_column() else 0
            }
        return {} 