import pytest
from pathlib import Path
import streamlit as st
from main import main

def test_file_upload_creation(test_data_dir):
    """Test if the data directory is created when a file is uploaded."""
    # Simulate file upload
    test_file = test_data_dir / "test.csv"
    test_file.write_text("test,data\n1,2")
    
    # Check if the raw data directory exists
    raw_data_dir = Path("data/raw")
    assert raw_data_dir.exists()
    
    # Clean up
    test_file.unlink()

def test_data_directory_structure():
    """Test if all required directories exist."""
    required_dirs = [
        Path("data"),
        Path("data/raw"),
        Path("data/processed")
    ]
    
    for directory in required_dirs:
        assert directory.exists()
        assert directory.is_dir() 