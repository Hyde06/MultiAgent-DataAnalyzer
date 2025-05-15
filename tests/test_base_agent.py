import pytest
from agents.base_agent import BaseAgent

class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""
    def process(self, data):
        return {"result": data}

def test_agent_initialization():
    """Test agent initialization."""
    agent = MockAgent("Test Agent", "Test Description", ["tool1", "tool2"])
    assert agent.name == "Test Agent"
    assert agent.description == "Test Description"
    assert agent.tools == ["tool1", "tool2"]
    assert agent.memory == {}

def test_memory_operations():
    """Test memory operations."""
    agent = MockAgent("Test Agent", "Test Description", [])
    
    # Test storing data
    agent.store_in_memory("test_key", "test_value")
    assert agent.memory["test_key"] == "test_value"
    
    # Test retrieving data
    assert agent.get_from_memory("test_key") == "test_value"
    assert agent.get_from_memory("nonexistent_key") is None
    
    # Test clearing memory
    agent.clear_memory()
    assert agent.memory == {} 