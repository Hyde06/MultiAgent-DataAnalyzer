import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.data_processor import DataProcessor
from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'numeric_col': [1, 2, np.nan, 4, 5],
        'categorical_col': ['A', 'B', 'C', np.nan, 'A'],
        'text_col': ['text1', 'text2', 'text3', 'text4', np.nan]
    })

@pytest.fixture
def data_processor():
    """Create a DataProcessor instance for testing"""
    return DataProcessor()

def test_load_data(data_processor, tmp_path):
    """Test data loading functionality"""
    # Create a temporary CSV file
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    
    # Test loading
    loaded_df = data_processor.load_data(str(file_path))
    assert loaded_df is not None
    assert loaded_df.shape == (3, 2)
    assert list(loaded_df.columns) == ['A', 'B']

def test_clean_data(data_processor, sample_data):
    """Test data cleaning functionality"""
    cleaned_df = data_processor.clean_data(sample_data)
    
    assert cleaned_df is not None
    assert cleaned_df.shape == sample_data.shape
    assert cleaned_df.isnull().sum().sum() == 0  # No missing values
    
    # Check if numeric column was filled with median
    assert cleaned_df['numeric_col'].median() == 3.0
    
    # Check if categorical column was filled with mode
    assert cleaned_df['categorical_col'].mode()[0] == 'A'

def test_prepare_data(data_processor, sample_data):
    """Test data preparation functionality"""
    stats = data_processor.prepare_data(sample_data)
    
    assert stats is not None
    assert 'shape' in stats
    assert 'columns' in stats
    assert 'dtypes' in stats
    assert 'summary' in stats
    assert 'missing_values' in stats
    assert 'unique_values' in stats
    
    # Check specific statistics
    assert stats['shape'] == (5, 3)
    assert len(stats['columns']) == 3
    assert len(stats['unique_values']) == 3

def test_save_processed_data(data_processor, sample_data, tmp_path):
    """Test saving processed data"""
    # Set processed data directory to temporary path
    data_processor.processed_data_dir = tmp_path
    
    # Test saving
    result = data_processor.save_processed_data(sample_data, "test_output.csv")
    assert result is True
    
    # Verify file exists
    output_file = tmp_path / "test_output.csv"
    assert output_file.exists()
    
    # Verify content
    loaded_df = pd.read_csv(output_file)
    assert loaded_df.shape == sample_data.shape
    assert list(loaded_df.columns) == list(sample_data.columns)

def test_process_file(data_processor, tmp_path):
    """Test complete data processing pipeline"""
    # Create test input file
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['a', 'b', 'c', np.nan]
    })
    input_file = tmp_path / "test_input.csv"
    df.to_csv(input_file, index=False)
    
    # Set directories to temporary path
    data_processor.raw_data_dir = tmp_path
    data_processor.processed_data_dir = tmp_path
    
    # Process file
    result = data_processor.process_file(str(input_file))
    
    assert result is not None
    assert 'data' in result
    assert 'statistics' in result
    assert 'output_file' in result
    
    # Verify output file exists
    output_file = tmp_path / result['output_file']
    assert output_file.exists()
    
    # Verify processed data
    processed_df = result['data']
    assert processed_df.shape == (4, 2)
    assert processed_df.isnull().sum().sum() == 0  # No missing values 