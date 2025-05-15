import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.sales_analyzer import SalesDataAnalyzer

class AdvancedForecasting:
    def __init__(self, df: pd.DataFrame):
        """Initialize the advanced forecasting analyzer with a DataFrame"""
        self.df = df
        self.sales_analyzer = SalesDataAnalyzer(df)
        self.date_col = self.sales_analyzer._find_date_column()
        self.sales_col = self.sales_analyzer._find_sales_column()
        
    def forecast_with_arima(self, forecast_periods: int = 30) -> Dict[str, Any]:
        """Forecast sales using ARIMA model"""
        if not (self.date_col and self.sales_col):
            return {}
            
        try:
            # Prepare time series data
            daily_sales = self.df.groupby(self.date_col)[self.sales_col].sum()
            
            # Fit ARIMA model
            model = ARIMA(daily_sales, order=(5,1,0))  # Example order, should be optimized
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            
            # Calculate confidence intervals
            forecast_ci = model_fit.get_forecast(steps=forecast_periods).conf_int()
            
            # Generate future dates
            last_date = daily_sales.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)
            
            # Calculate model performance
            predictions = model_fit.predict(start=0, end=len(daily_sales)-1)
            mae = mean_absolute_error(daily_sales, predictions)
            rmse = np.sqrt(mean_squared_error(daily_sales, predictions))
            
            return {
                'historical': daily_sales,
                'forecast': forecast,
                'confidence_intervals': forecast_ci,
                'future_dates': future_dates,
                'model_performance': {
                    'mae': mae,
                    'rmse': rmse
                }
            }
        except Exception as e:
            print(f"Error in ARIMA forecasting: {str(e)}")
            return {}
    
    def forecast_with_prophet(self, forecast_periods: int = 30) -> Dict[str, Any]:
        """Forecast sales using Prophet model"""
        if not (self.date_col and self.sales_col):
            return {}
            
        try:
            # Prepare data for Prophet
            daily_sales = self.df.groupby(self.date_col)[self.sales_col].sum().reset_index()
            daily_sales.columns = ['ds', 'y']
            
            # Fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            model.fit(daily_sales)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=forecast_periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Calculate model performance
            predictions = forecast['yhat'][:len(daily_sales)]
            mae = mean_absolute_error(daily_sales['y'], predictions)
            rmse = np.sqrt(mean_squared_error(daily_sales['y'], predictions))
            
            return {
                'historical': daily_sales,
                'forecast': forecast,
                'model_performance': {
                    'mae': mae,
                    'rmse': rmse
                },
                'components': {
                    'trend': forecast['trend'],
                    'yearly': forecast['yearly'],
                    'weekly': forecast['weekly']
                }
            }
        except Exception as e:
            print(f"Error in Prophet forecasting: {str(e)}")
            return {}
    
    def compare_forecasting_models(self, forecast_periods: int = 30) -> Dict[str, Any]:
        """Compare different forecasting models"""
        arima_results = self.forecast_with_arima(forecast_periods)
        prophet_results = self.forecast_with_prophet(forecast_periods)
        
        comparison = {
            'arima': arima_results.get('model_performance', {}),
            'prophet': prophet_results.get('model_performance', {}),
            'best_model': None
        }
        
        # Determine best model based on MAE
        if arima_results and prophet_results:
            arima_mae = arima_results['model_performance']['mae']
            prophet_mae = prophet_results['model_performance']['mae']
            comparison['best_model'] = 'arima' if arima_mae < prophet_mae else 'prophet'
        
        return comparison
    
    def generate_forecast_visualization(self, model_type: str = 'prophet') -> go.Figure:
        """Generate visualization for the forecast"""
        if model_type == 'arima':
            results = self.forecast_with_arima()
        else:
            results = self.forecast_with_prophet()
            
        if not results:
            return go.Figure()
        
        if model_type == 'prophet':
            forecast = results['forecast']
            historical = results['historical']
            fig = go.Figure()
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical['ds'],
                y=historical['y'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            # Upper CI
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper CI',
                line=dict(color='rgba(255,0,0,0.2)'),
                showlegend=True
            ))
            # Lower CI (fill to upper)
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower CI',
                line=dict(color='rgba(255,0,0,0.2)'),
                fill='tonexty',
                showlegend=True
            ))
            fig.update_layout(
                title='Sales Forecast (PROPHET)',
                xaxis_title='Date',
                yaxis_title='Sales',
                showlegend=True
            )
            return fig
        else:
            # ... existing ARIMA plotting code ...
            return go.Figure() 