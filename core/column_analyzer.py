import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class ColumnAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize the column analyzer with a DataFrame"""
        self.df = df
        self.column_types = self._analyze_column_types()
        
    def _analyze_column_types(self) -> Dict[str, str]:
        """Analyze and categorize columns based on their characteristics"""
        column_types = {}
        
        for col in self.df.columns:
            # Get basic column info
            unique_ratio = self.df[col].nunique() / len(self.df)
            dtype = str(self.df[col].dtype)
            
            # Check for reference/ID columns
            if self._is_reference_column(col, unique_ratio):
                column_types[col] = 'reference'
            # Check for date/time columns
            elif self._is_datetime_column(col, dtype):
                column_types[col] = 'datetime'
            # Check for categorical columns
            elif self._is_categorical_column(col, unique_ratio, dtype):
                column_types[col] = 'categorical'
            # Check for numeric columns
            elif self._is_numeric_column(dtype):
                column_types[col] = 'numeric'
            else:
                column_types[col] = 'other'
                
        return column_types
    
    def _is_reference_column(self, col: str, unique_ratio: float) -> bool:
        """Check if a column is likely a reference/ID column"""
        # Check column name patterns
        ref_patterns = [
            r'id$', r'key$', r'code$', r'number$', r'num$', r'no$',
            r'order', r'reference', r'transaction', r'ticket'
        ]
        
        name_match = any(re.search(pattern, col.lower()) for pattern in ref_patterns)
        
        # Check if column has high uniqueness
        is_unique = unique_ratio > 0.9
        
        # Check if column contains only alphanumeric values
        is_alphanumeric = self.df[col].astype(str).str.match(r'^[A-Za-z0-9]+$').all()
        
        return name_match or (is_unique and is_alphanumeric)
    
    def _is_datetime_column(self, col: str, dtype: str) -> bool:
        """Check if a column is likely a datetime column"""
        # Check column name patterns
        date_patterns = [
            r'date$', r'time$', r'datetime$', r'day$', r'month$', r'year$',
            r'created', r'updated', r'start', r'end'
        ]
        
        name_match = any(re.search(pattern, col.lower()) for pattern in date_patterns)
        
        # Check dtype
        is_datetime = 'datetime' in dtype.lower()
        
        return name_match or is_datetime
    
    def _is_categorical_column(self, col: str, unique_ratio: float, dtype: str) -> bool:
        """Check if a column is likely a categorical column"""
        # Check if column has low cardinality
        is_low_cardinality = unique_ratio < 0.1
        
        # Check if column is object type
        is_object = dtype == 'object'
        
        return is_low_cardinality and is_object
    
    def _is_numeric_column(self, dtype: str) -> bool:
        """Check if a column is numeric"""
        return 'int' in dtype.lower() or 'float' in dtype.lower()
    
    def get_important_columns(self) -> Dict[str, List[str]]:
        """Get lists of important columns by type"""
        important_cols = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'reference': []
        }
        
        for col, col_type in self.column_types.items():
            if col_type in important_cols:
                important_cols[col_type].append(col)
                
        return important_cols
    
    def get_column_importance(self) -> Dict[str, float]:
        """Calculate importance score for each column"""
        importance_scores = {}
        
        for col, col_type in self.column_types.items():
            score = 0.0
            
            if col_type == 'reference':
                score = 0.1  # Low importance for reference columns
            elif col_type == 'numeric':
                # Higher importance for numeric columns with more unique values
                unique_ratio = self.df[col].nunique() / len(self.df)
                score = 0.5 + (0.5 * unique_ratio)
            elif col_type == 'categorical':
                # Medium importance for categorical columns
                score = 0.4
            elif col_type == 'datetime':
                # Medium-high importance for datetime columns
                score = 0.6
            else:
                score = 0.3  # Default importance for other columns
                
            importance_scores[col] = score
            
        return importance_scores
    
    def get_analysis_columns(self) -> Dict[str, List[str]]:
        """Get columns recommended for different types of analysis"""
        important_cols = self.get_important_columns()
        
        return {
            'visualization': important_cols['numeric'] + important_cols['categorical'],
            'correlation': important_cols['numeric'],
            'clustering': important_cols['numeric'],
            'time_series': important_cols['datetime'] + important_cols['numeric'],
            'categorical_analysis': important_cols['categorical']
        } 