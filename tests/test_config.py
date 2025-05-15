import pytest
import os
from core.config import (
    GOOGLE_API_KEY,
    GOOGLE_PROJECT_ID,
    GOOGLE_LOCATION,
    MODEL_CONFIG,
    AGENT_CONFIG
)

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_environment_variables():
    """Test if required environment variables are set."""
    assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY is not set"
    assert isinstance(GOOGLE_API_KEY, str), "GOOGLE_API_KEY should be a string"
    assert len(GOOGLE_API_KEY) > 0, "GOOGLE_API_KEY should not be empty"

def test_model_config():
    """Test if model configuration is properly set."""
    required_keys = ["default_model", "temperature", "max_output_tokens", "top_p", "top_k"]
    for key in required_keys:
        assert key in MODEL_CONFIG, f"Missing {key} in MODEL_CONFIG"
    
    assert isinstance(MODEL_CONFIG["temperature"], float)
    assert isinstance(MODEL_CONFIG["max_output_tokens"], int)
    assert isinstance(MODEL_CONFIG["top_p"], float)
    assert isinstance(MODEL_CONFIG["top_k"], int)

def test_agent_config():
    """Test if agent configuration is properly set."""
    required_agents = ["data_cleaner", "analyzer", "visualizer"]
    for agent in required_agents:
        assert agent in AGENT_CONFIG, f"Missing {agent} in AGENT_CONFIG"
        
        agent_config = AGENT_CONFIG[agent]
        assert "name" in agent_config
        assert "description" in agent_config
        assert "tools" in agent_config
        assert isinstance(agent_config["tools"], list) 