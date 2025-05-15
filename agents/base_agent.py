from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, description: str, tools: list):
        self.name = name
        self.description = description
        self.tools = tools
        self.memory = {}
    
    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process the input data and return results.
        
        Args:
            data: Input data to be processed
            
        Returns:
            Dict containing the processing results
        """
        pass
    
    def store_in_memory(self, key: str, value: Any) -> None:
        """Store data in agent's memory."""
        self.memory[key] = value
    
    def get_from_memory(self, key: str) -> Any:
        """Retrieve data from agent's memory."""
        return self.memory.get(key)
    
    def clear_memory(self) -> None:
        """Clear agent's memory."""
        self.memory.clear() 