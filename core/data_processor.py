import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor with necessary directories"""
        self.raw_data_dir = Path(RAW_DATA_DIR)
        self.processed_data_dir = Path(PROCESSED_DATA_DIR)
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from a CSV file with automatic encoding detection"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded data from {file_path} using {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error reading file with {encoding} encoding: {str(e)}")
                    continue
            
            # If all encodings fail, try with error handling
            try:
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
                logger.warning(f"Loaded data with encoding errors replaced: {file_path}")
                return df
            except Exception as e:
                logger.error(f"Failed to load data with all encoding attempts: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values and outliers"""
        if df is None:
            return None
            
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                # For numeric columns, fill with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                # For categorical/text columns, fill with mode
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        logger.info("Data cleaning completed")
        return df_clean
        
    def prepare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for analysis by generating basic statistics"""
        if df is None:
            return None
            
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'summary': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        logger.info("Data preparation completed")
        return stats
        
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> bool:
        """Save processed data to the processed data directory"""
        try:
            output_path = self.processed_data_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
            
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a CSV file and return statistics and processed data"""
        try:
            # Load data with automatic encoding detection
            df = self.load_data(file_path)
            if df is None:
                logger.error(f"Failed to load data from {file_path}")
                return None
            
            # Convert data types to ensure Arrow compatibility
            df = self._convert_dtypes(df)
            
            # Generate statistics
            stats = self._generate_statistics(df)
            
            # Save processed file
            output_file = f"processed_{os.path.basename(file_path)}"
            output_path = os.path.join(self.processed_dir, output_file)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            return {
                'data': df,
                'statistics': stats,
                'output_file': output_file
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
            
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame dtypes to ensure Arrow compatibility"""
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Convert object columns to string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        # Convert numeric columns to float64
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].astype('float64')
            
        # Convert datetime columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
                    
        return df
        
    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistics about the DataFrame"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        } 