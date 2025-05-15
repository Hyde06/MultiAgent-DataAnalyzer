import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create necessary directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Agent Configuration
AGENT_CONFIG = {
    "data_cleaner": {
        "name": "Data Cleaner",
        "description": "Cleans and preprocesses the input data",
        "tools": ["pandas", "numpy"]
    },
    "analyzer": {
        "name": "Data Analyzer",
        "description": "Analyzes the data and generates insights",
        "tools": ["pandas", "numpy", "scikit-learn"]
    },
    "visualizer": {
        "name": "Data Visualizer",
        "description": "Creates visualizations from the analyzed data",
        "tools": ["plotly", "streamlit"]
    }
}

# Model Configuration
MODEL_CONFIG = {
    "default_model": "gemini-1.5-flash",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "top_p": 0.8,
    "top_k": 40,
    "rate_limit": {
        "requests_per_minute": 15,
        "requests_per_day": 1500
    }
} 