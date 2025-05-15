import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from core.config import GOOGLE_API_KEY, MODEL_CONFIG

# Update to use the correct API endpoint for Gemini 1.5 Flash
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models"
LIST_MODELS_URL = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"

class RateLimiter:
    def __init__(self):
        self.requests_per_minute = MODEL_CONFIG["rate_limit"]["requests_per_minute"]
        self.requests_per_day = MODEL_CONFIG["rate_limit"]["requests_per_day"]
        self.minute_requests = []
        self.daily_requests = []
        self.last_request_time = None

    def can_make_request(self) -> bool:
        current_time = datetime.now()
        
        # Clean up old requests
        self.minute_requests = [t for t in self.minute_requests if current_time - t < timedelta(minutes=1)]
        self.daily_requests = [t for t in self.daily_requests if current_time - t < timedelta(days=1)]
        
        # Check rate limits
        if len(self.minute_requests) >= self.requests_per_minute:
            return False
        if len(self.daily_requests) >= self.requests_per_day:
            return False
            
        return True

    def add_request(self):
        current_time = datetime.now()
        self.minute_requests.append(current_time)
        self.daily_requests.append(current_time)
        self.last_request_time = current_time

    def get_wait_time(self) -> float:
        if not self.last_request_time:
            return 0
            
        current_time = datetime.now()
        minute_elapsed = (current_time - self.last_request_time).total_seconds()
        
        if len(self.minute_requests) >= self.requests_per_minute:
            return max(60 - minute_elapsed, 0)
            
        return 0

# Initialize rate limiter
rate_limiter = RateLimiter()

def handle_rate_limit(response: requests.Response) -> None:
    """
    Handle rate limit response by extracting retry delay and waiting.
    
    Args:
        response: The response object from the API call
    """
    try:
        error_data = response.json().get("error", {})
        if error_data.get("code") == 429:  # Rate limit error
            wait_time = rate_limiter.get_wait_time()
            if wait_time > 0:
                print(f"Rate limit hit. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                return
            time.sleep(5)  # Default fallback
    except Exception as e:
        print(f"Error handling rate limit: {str(e)}")
        time.sleep(5)  # Default fallback

def make_api_request(url: str, method: str = "GET", headers: Dict = None, data: Dict = None, max_retries: int = 5) -> requests.Response:
    """
    Make API request with retry logic for rate limits.
    
    Args:
        url: The API endpoint URL
        method: HTTP method (GET or POST)
        headers: Request headers
        data: Request data for POST requests
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response object from the API
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Check rate limits before making request
            if not rate_limiter.can_make_request():
                wait_time = rate_limiter.get_wait_time()
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue

            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 429:  # Rate limit error
                handle_rate_limit(response)
                retry_count += 1
                continue
                
            response.raise_for_status()
            rate_limiter.add_request()  # Record successful request
            return response
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = min(300, 2 ** (retry_count + 1) * 10)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retry_count += 1
            else:
                raise Exception(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

def list_available_models() -> List[str]:
    """
    List all available models from the Gemini API.
    
    Returns:
        List of available model names
    """
    try:
        print(f"Attempting to list models from: {LIST_MODELS_URL}")
        response = make_api_request(LIST_MODELS_URL)
        print(f"List models response status: {response.status_code}")
        print(f"List models response: {response.text}")
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            print(f"Available models: {model_names}")
            return model_names
        else:
            print(f"Failed to list models: {response.text}")
            return []
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def generate_text(prompt: str) -> str:
    """
    Generate text using Gemini 1.5 Flash model via REST API.
    
    Args:
        prompt: The input prompt for text generation
        
    Returns:
        Generated text response
    """
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": MODEL_CONFIG["temperature"],
            "maxOutputTokens": MODEL_CONFIG["max_output_tokens"],
            "topP": MODEL_CONFIG["top_p"],
            "topK": MODEL_CONFIG["top_k"]
        }
    }
    
    try:
        model_name = MODEL_CONFIG["default_model"]
        generate_url = f"{GEMINI_API_URL}/{model_name}:generateContent?key={GOOGLE_API_KEY}"
        
        print(f"Using model: {model_name}")
        print(f"Generation URL: {generate_url}")
        
        response = make_api_request(generate_url, method="POST", headers=headers, data=data)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        else:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            print(f"Error: {error_msg}")
            raise Exception(error_msg)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        raise

def generate_structured_response(prompt: str, output_format: dict) -> Dict[str, Any]:
    """
    Generate structured response using Gemini model.
    
    Args:
        prompt: The input prompt
        output_format: Dictionary describing the expected output format
        
    Returns:
        Structured response as dictionary
    """
    formatted_prompt = f"{prompt}\n\nPlease provide the response in the following JSON format: {json.dumps(output_format, indent=2)}"
    
    try:
        response_text = generate_text(formatted_prompt)
        
        # Try to parse the response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, return the raw text
            return {"raw_response": response_text}
    except Exception as e:
        print(f"Error generating structured response: {str(e)}")
        raise

def generate_batch_responses(prompts: List[str]) -> List[str]:
    """
    Generate multiple responses in batch.
    
    Args:
        prompts: List of prompts to process
        
    Returns:
        List of generated responses
    """
    responses = []
    for prompt in prompts:
        try:
            response = generate_text(prompt)
            responses.append(response)
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            responses.append(f"Error: {str(e)}")
    return responses 