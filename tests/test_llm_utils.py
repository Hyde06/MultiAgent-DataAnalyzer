import pytest
import os
import time
import traceback
from core.llm_utils import generate_text, generate_structured_response, list_available_models

def test_list_models():
    """Test listing available models."""
    models = list_available_models()
    assert isinstance(models, list)
    if models:  # If we have models available
        assert all(isinstance(model, str) for model in models)

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_generate_text():
    """Test basic text generation."""
    # First check if we have any models available
    models = list_available_models()
    if not models:
        pytest.skip("No models available")
    
    # Try up to 3 times with increasing delays
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            prompt = "What is 2+2?"
            print(f"Sending prompt: {prompt}")
            response = generate_text(prompt)
            print(f"Received response: {response}")
            assert isinstance(response, str)
            assert len(response) > 0
            return  # Test passed
        except Exception as e:
            print(f"\nError on attempt {attempt + 1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            
            if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            raise  # Re-raise if not a rate limit error or we're out of retries

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_generate_structured_response():
    """Test structured response generation."""
    # First check if we have any models available
    models = list_available_models()
    if not models:
        pytest.skip("No models available")
    
    # Try up to 3 times with increasing delays
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            prompt = "What are the three primary colors?"
            output_format = {
                "colors": ["color1", "color2", "color3"],
                "explanation": "string"
            }
            print(f"Sending prompt: {prompt}")
            print(f"Output format: {output_format}")
            response = generate_structured_response(prompt, output_format)
            print(f"Received response: {response}")
            assert isinstance(response, dict)
            assert "colors" in response or "raw_response" in response
            return  # Test passed
        except Exception as e:
            print(f"\nError on attempt {attempt + 1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            
            if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            raise  # Re-raise if not a rate limit error or we're out of retries 