import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from core.sales_analyzer import SalesDataAnalyzer

class BusinessIntelligence:
    def __init__(self, df: pd.DataFrame):
        """Initialize the business intelligence analyzer with a DataFrame"""
        self.df = df
        self.sales_analyzer = SalesDataAnalyzer(df)
        self.date_col = self.sales_analyzer._find_date_column()
        self.sales_col = self.sales_analyzer._find_sales_column()
        self.product_col = self.sales_analyzer._find_product_column()
        self.territory_col = self._find_territory_column()
        self.competitor_col = self._find_competitor_column()
        
    def track_kpis(self, goals: Dict[str, float] = None) -> Dict[str, Any]:
        """Track key performance indicators and compare against goals"""
        if not self.sales_col:
            return {}
            
        # Calculate KPIs
        kpis = {
            'total_sales': self.df[self.sales_col].sum(),
            'average_order_value': self.df[self.sales_col].mean(),
            'sales_growth': self._calculate_growth_rate(),
            'customer_retention': self._calculate_retention_rate(),
            'profit_margin': self._calculate_profit_margin(),
            'inventory_turnover': self._calculate_inventory_turnover()
        }
        
        # Compare with goals if provided
        if goals:
            kpis['goal_achievement'] = {
                kpi: (value / goals.get(kpi, value)) * 100 
                for kpi, value in kpis.items() 
                if kpi in goals
            }
        
        return kpis
    
    def analyze_territories(self) -> Dict[str, Any]:
        """Analyze sales performance by territory"""
        if not (self.territory_col and self.sales_col):
            return {}
            
        # Territory performance metrics
        territory_metrics = self.df.groupby(self.territory_col).agg({
            self.sales_col: ['sum', 'mean', 'count'],
            self.product_col: 'nunique'
        }).round(2)
        
        # Territory growth rates
        territory_growth = self._calculate_territory_growth()
        
        # Territory clustering
        territory_clusters = self._cluster_territories()
        
        return {
            'metrics': territory_metrics,
            'growth': territory_growth,
            'clusters': territory_clusters
        }
    
    def analyze_competitors(self) -> Dict[str, Any]:
        """Analyze competitor performance and market share"""
        if not (self.competitor_col and self.sales_col):
            return {}
            
        # Market share analysis
        market_shares = self.df.groupby(self.competitor_col)[self.sales_col].sum()
        total_sales = market_shares.sum()
        market_shares = (market_shares / total_sales * 100).round(2)
        
        # Competitor growth rates
        competitor_growth = self._calculate_competitor_growth()
        
        # Price comparison
        price_comparison = self._analyze_price_comparison()
        
        return {
            'market_shares': market_shares,
            'growth_rates': competitor_growth,
            'price_comparison': price_comparison
        }
    
    def optimize_inventory(self) -> Dict[str, Any]:
        """Generate inventory optimization recommendations"""
        if not (self.product_col and self.sales_col):
            return {}
            
        # Calculate inventory metrics
        inventory_metrics = self._calculate_inventory_metrics()
        
        # Generate recommendations
        recommendations = self._generate_inventory_recommendations(inventory_metrics)
        
        return {
            'metrics': inventory_metrics,
            'recommendations': recommendations
        }
    
    def calculate_customer_lifetime_value(self) -> Dict[str, Any]:
        """Calculate customer lifetime value and segment customers"""
        if not (self.sales_col and self.date_col):
            return {}
            
        # Calculate CLV metrics
        clv_metrics = self._calculate_clv_metrics()
        
        # Segment customers based on CLV
        customer_segments = self._segment_customers_by_clv(clv_metrics)
        
        return {
            'metrics': clv_metrics,
            'segments': customer_segments
        }
    
    def _find_territory_column(self) -> str:
        """Find the territory column in the dataset"""
        territory_columns = [col for col in self.df.columns if 'territory' in col.lower() or 'region' in col.lower()]
        return territory_columns[0] if territory_columns else None
    
    def _find_competitor_column(self) -> str:
        """Find the competitor column in the dataset"""
        competitor_columns = [col for col in self.df.columns if 'competitor' in col.lower() or 'brand' in col.lower()]
        return competitor_columns[0] if competitor_columns else None
    
    def _calculate_growth_rate(self) -> float:
        """Calculate overall sales growth rate"""
        if not (self.date_col and self.sales_col):
            return 0.0
            
        monthly_sales = self.df.groupby(pd.Grouper(key=self.date_col, freq='ME'))[self.sales_col].sum()
        if len(monthly_sales) < 2:
            return 0.0
            
        return ((monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2] * 100).round(2)
    
    def _calculate_retention_rate(self) -> float:
        """Calculate customer retention rate"""
        # Implementation depends on available customer data
        return 0.0
    
    def _calculate_profit_margin(self) -> float:
        """Calculate profit margin"""
        # Implementation depends on available cost data
        return 0.0
    
    def _calculate_inventory_turnover(self) -> float:
        """Calculate inventory turnover ratio"""
        # Implementation depends on available inventory data
        return 0.0
    
    def _calculate_territory_growth(self) -> pd.Series:
        """Calculate growth rates for each territory"""
        if not (self.territory_col and self.date_col and self.sales_col):
            return pd.Series()
            
        monthly_territory_sales = self.df.groupby([self.territory_col, pd.Grouper(key=self.date_col, freq='ME')])[self.sales_col].sum()
        territory_growth = monthly_territory_sales.groupby(self.territory_col).apply(
            lambda x: ((x.iloc[-1] - x.iloc[-2]) / x.iloc[-2] * 100).round(2) if len(x) >= 2 else 0.0
        )
        return territory_growth
    
    def _cluster_territories(self) -> Dict[str, List[str]]:
        """Cluster territories based on performance metrics"""
        if not (self.territory_col and self.sales_col):
            return {}
            
        # Prepare features for clustering
        features = self.df.groupby(self.territory_col).agg({
            self.sales_col: ['sum', 'mean', 'std']
        }).fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Group territories by cluster
        territory_clusters = {}
        for i, territory in enumerate(features.index):
            cluster = f"Cluster {clusters[i] + 1}"
            if cluster not in territory_clusters:
                territory_clusters[cluster] = []
            territory_clusters[cluster].append(territory)
        
        return territory_clusters
    
    def _calculate_competitor_growth(self) -> pd.Series:
        """Calculate growth rates for each competitor"""
        if not (self.competitor_col and self.date_col and self.sales_col):
            return pd.Series()
            
        monthly_competitor_sales = self.df.groupby([self.competitor_col, pd.Grouper(key=self.date_col, freq='M')])[self.sales_col].sum()
        competitor_growth = monthly_competitor_sales.groupby(self.competitor_col).apply(
            lambda x: ((x.iloc[-1] - x.iloc[-2]) / x.iloc[-2] * 100).round(2) if len(x) >= 2 else 0.0
        )
        return competitor_growth
    
    def _analyze_price_comparison(self) -> pd.DataFrame:
        """Analyze price comparison with competitors"""
        # Implementation depends on available price data
        return pd.DataFrame()
    
    def _calculate_inventory_metrics(self) -> Dict[str, Any]:
        """Calculate inventory-related metrics"""
        # Implementation depends on available inventory data
        return {}
    
    def _generate_inventory_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate inventory optimization recommendations"""
        # Implementation depends on inventory metrics
        return []
    
    def _calculate_clv_metrics(self) -> Dict[str, Any]:
        """Calculate customer lifetime value metrics"""
        if not (self.sales_col and self.date_col):
            return {}
            
        # Calculate average purchase value
        avg_purchase = self.df[self.sales_col].mean()
        
        # Calculate purchase frequency
        purchase_freq = self.df.groupby(self.date_col)[self.sales_col].count().mean()
        
        # Calculate customer lifespan (in months)
        customer_lifespan = 12  # Default value, should be calculated based on actual data
        
        # Calculate CLV
        clv = avg_purchase * purchase_freq * customer_lifespan
        
        return {
            'average_purchase_value': avg_purchase,
            'purchase_frequency': purchase_freq,
            'customer_lifespan': customer_lifespan,
            'customer_lifetime_value': clv
        }
    
    def _segment_customers_by_clv(self, metrics: Dict[str, Any]) -> Dict[str, List[str]]:
        """Segment customers based on their lifetime value"""
        # Implementation depends on available customer data
        return {} 