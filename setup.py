from setuptools import setup, find_packages

setup(
    name="multi_agent_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "crewai>=0.1.0",
        "langchain>=0.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "chromadb>=0.4.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.0",
        "google-cloud-aiplatform>=1.36.0",
        "google-generativeai>=0.3.0",
        "requests>=2.31.0"
    ],
    python_requires=">=3.9",
) 