import pytest
import os
from core.llm_utils import generate_text, list_available_models, rate_limiter
from core.config import GOOGLE_API_KEY

def test_api_key_configuration():
    """Test if API key is properly configured"""
    assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY is not set in environment variables"
    assert len(GOOGLE_API_KEY) > 0, "GOOGLE_API_KEY is empty"

def test_rate_limiter():
    """Test rate limiter functionality"""
    # Test initial state
    assert rate_limiter.can_make_request() is True, "Rate limiter should allow initial request"
    
    # Test minute limit
    for _ in range(rate_limiter.requests_per_minute):
        rate_limiter.add_request()
    
    assert rate_limiter.can_make_request() is False, "Rate limiter should block after reaching minute limit"
    
    # Test wait time calculation
    wait_time = rate_limiter.get_wait_time()
    assert wait_time > 0, "Wait time should be positive when rate limit is reached"

def test_text_generation():
    """Test basic text generation"""
    prompt = "Write a one-sentence summary of artificial intelligence."
    response = generate_text(prompt)
    
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"

def test_model_listing():
    """Test model listing functionality"""
    models = list_available_models()
    assert isinstance(models, list), "Should return a list of models"
    assert "models/gemini-1.5-flash" in models, "models/gemini-1.5-flash should be in available models"

if __name__ == "__main__":
    # Run tests
    test_api_key_configuration()
    print("✓ API key configuration test passed")
    
    test_rate_limiter()
    print("✓ Rate limiter test passed")
    
    test_text_generation()
    print("✓ Text generation test passed")
    
    test_model_listing()
    print("✓ Model listing test passed")
    
    print("\nAll tests completed successfully!") 