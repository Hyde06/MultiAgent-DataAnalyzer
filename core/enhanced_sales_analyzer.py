import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from mlxtend.frequent_patterns import apriori, association_rules
from core.sales_analyzer import SalesDataAnalyzer

class EnhancedSalesAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize the enhanced sales analyzer with a DataFrame"""
        self.df = df
        self.sales_analyzer = SalesDataAnalyzer(df)
        self.date_col = self.sales_analyzer._find_date_column()
        self.sales_col = self.sales_analyzer._find_sales_column()
        self.product_col = self.sales_analyzer._find_product_column()
        self.price_col = self._find_price_column()
        
    def analyze_seasonality(self) -> Dict[str, Any]:
        """Analyze seasonal patterns in sales data"""
        if not (self.date_col and self.sales_col):
            return {}
            
        # Convert to datetime if not already
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        
        # Prepare time series data
        daily_sales = self.df.groupby(self.date_col)[self.sales_col].sum()
        
        # Perform seasonal decomposition
        try:
            decomposition = seasonal_decompose(daily_sales, period=30)  # Assuming monthly seasonality
            
            # Test for stationarity
            adf_result = adfuller(daily_sales.dropna())
            
            # Calculate seasonal strength
            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal.var() + decomposition.resid.var()))
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'is_stationary': adf_result[1] < 0.05,
                'seasonal_strength': seasonal_strength,
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1]
            }
        except Exception as e:
            print(f"Error in seasonality analysis: {str(e)}")
            return {}
    
    def forecast_sales(self, forecast_periods: int = 30) -> Dict[str, Any]:
        """Forecast future sales using multiple models"""
        if not (self.date_col and self.sales_col):
            return {}
            
        # Prepare time series data
        daily_sales = self.df.groupby(self.date_col)[self.sales_col].sum()
        
        # Create features for prediction
        df_features = pd.DataFrame(index=daily_sales.index)
        df_features['sales'] = daily_sales
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month
        df_features['year'] = df_features.index.year
        df_features['day_of_year'] = df_features.index.dayofyear
        
        # Prepare training data
        X = df_features.drop('sales', axis=1)
        y = df_features['sales']
        
        # Train multiple models
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        forecasts = {}
        for name, model in models.items():
            model.fit(X, y)
            forecasts[name] = model.predict(X)
        
        # Generate future dates
        last_date = daily_sales.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)
        
        # Prepare future features
        future_features = pd.DataFrame(index=future_dates)
        future_features['day_of_week'] = future_features.index.dayofweek
        future_features['month'] = future_features.index.month
        future_features['year'] = future_features.index.year
        future_features['day_of_year'] = future_features.index.dayofyear
        
        # Generate forecasts
        future_forecasts = {}
        for name, model in models.items():
            future_forecasts[name] = model.predict(future_features)
        
        return {
            'historical': daily_sales,
            'forecasts': future_forecasts,
            'future_dates': future_dates,
            'model_performance': {
                name: np.mean(np.abs(forecasts[name] - y)) for name in models.keys()
            }
        }
    
    def analyze_market_basket(self, min_support: float = 0.01) -> Dict[str, Any]:
        """Perform market basket analysis"""
        if not (self.date_col and self.product_col):
            return {}
            
        try:
            # Create transaction data
            transactions = self.df.groupby([self.date_col, self.product_col]).size().unstack().fillna(0)
            
            # Convert to boolean (purchased/not purchased)
            transactions_binary = (transactions > 0).astype(bool)
            
            # Generate frequent itemsets
            frequent_itemsets = apriori(transactions_binary, min_support=min_support, use_colnames=True)
            
            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            
            # Convert frozensets to lists for better compatibility
            if not rules.empty:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
                rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
            
            # Get top product combinations
            top_combinations = rules.sort_values('lift', ascending=False).head(10)
            
            return {
                'frequent_itemsets': frequent_itemsets,
                'association_rules': rules,
                'top_combinations': top_combinations
            }
        except Exception as e:
            print(f"Error in market basket analysis: {str(e)}")
            return {
                'frequent_itemsets': pd.DataFrame(),
                'association_rules': pd.DataFrame(),
                'top_combinations': pd.DataFrame()
            }
    
    def analyze_price_elasticity(self) -> Dict[str, Any]:
        """Analyze price elasticity of demand"""
        if not (self.price_col and self.sales_col and self.product_col):
            return {}
            
        try:
            # Calculate price elasticity for each product
            elasticity_results = {}
            
            for product in self.df[self.product_col].unique():
                product_data = self.df[self.df[self.product_col] == product]
                
                if len(product_data) > 1:
                    # Calculate price and quantity changes
                    price_changes = product_data[self.price_col].pct_change()
                    quantity_changes = product_data[self.sales_col].pct_change()
                    
                    # Handle division by zero and NaN values
                    valid_changes = ~(np.isnan(price_changes) | np.isnan(quantity_changes) | (price_changes == 0))
                    if valid_changes.any():
                        elasticity = (quantity_changes[valid_changes] / price_changes[valid_changes]).mean()
                    else:
                        elasticity = np.nan
                    
                    elasticity_results[product] = {
                        'elasticity': elasticity,
                        'is_elastic': abs(elasticity) > 1 if not np.isnan(elasticity) else False,
                        'price_mean': product_data[self.price_col].mean(),
                        'quantity_mean': product_data[self.sales_col].mean()
                    }
            
            # Calculate overall elasticity excluding NaN values
            valid_elasticities = [r['elasticity'] for r in elasticity_results.values() if not np.isnan(r['elasticity'])]
            overall_elasticity = np.mean(valid_elasticities) if valid_elasticities else np.nan
            
            return {
                'product_elasticity': elasticity_results,
                'overall_elasticity': overall_elasticity
            }
        except Exception as e:
            print(f"Error in price elasticity analysis: {str(e)}")
            return {
                'product_elasticity': {},
                'overall_elasticity': np.nan
            }
    
    def generate_enhanced_insights(self) -> Dict[str, Any]:
        """Generate comprehensive enhanced sales insights"""
        insights = {
            'seasonality': self.analyze_seasonality(),
            'forecast': self.forecast_sales(),
            'market_basket': self.analyze_market_basket(),
            'price_elasticity': self.analyze_price_elasticity()
        }
        
        # Add basic sales insights
        insights.update(self.sales_analyzer.generate_sales_insights())
        
        return insights
    
    def _find_price_column(self) -> str:
        """Find the price column in the dataset"""
        price_columns = [col for col in self.df.columns if 'price' in col.lower() or 'cost' in col.lower()]
        return price_columns[0] if price_columns else None 