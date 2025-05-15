import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load test environment variables
load_dotenv()

@pytest.fixture
def test_data_dir():
    """Create and return a temporary test data directory."""
    test_dir = Path("tests/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

@pytest.fixture
def sample_csv_data(test_data_dir):
    """Create a sample CSV file for testing."""
    csv_path = test_data_dir / "sample.csv"
    csv_content = """date,value,category
2024-01-01,100,A
2024-01-02,150,B
2024-01-03,200,A
2024-01-04,250,B"""
    
    with open(csv_path, "w") as f:
        f.write(csv_content)
    
    return csv_path

@pytest.fixture
def mock_llm_response():
    """Return a mock LLM response for testing."""
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "This is a mock response for testing purposes."
                }]
            }
        }]
    } 