import os
import time
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def verify_api_key():
    """
    Verify that the OpenAI API key is valid by making a simple API call.
    
    Raises:
        ValueError: If API key is missing or invalid
        Exception: If API call fails for any other reason
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    logger.info("Verifying OpenAI API key...")
    try:
        # Make a minimal API call to verify the key works
        response = client.models.list()
        logger.info("OpenAI API key verified successfully")
        return True
    except Exception as e:
        error_msg = f"Failed to verify OpenAI API key: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def generate_draft(customer_message, context=None):
    """
    Generate a draft reply based on the customer message.
    
    Args:
        customer_message: Customer message string
        context: Optional dict (not used in simplified version, kept for compatibility)
    
    Returns:
        Tuple of (draft_text: str, metadata: dict) where metadata contains:
        - model: str
        - latency_ms: int
        - token_usage: dict with prompt_tokens, completion_tokens, total_tokens
    """
    # Simple prompt that will be concatenated with customer message
    system_prompt = "You are a helpful customer support agent with a fun and engaging attitude. Draft a short, professional, and friendly reply to the customer's message."
    
    start_time = time.time()
    
    try:
        # Prepare request parameters
        request_params = {
            "model": "gpt-4o-mini",  # Using cost-effective model, can be made configurable
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": customer_message}
            ],
            "temperature": 0.9,
            "max_tokens": 500
        }
        
        # Log request parameters
        logger.info(f"OpenAI API Request: {json.dumps(request_params, indent=2)}")
        
        # Use the same parameters for the API call
        response = client.chat.completions.create(**request_params)
        
        # Log API response
        response_data = {
            "id": response.id,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        logger.info(f"OpenAI API Response: {json.dumps(response_data, indent=2)}")
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        draft = response.choices[0].message.content
        usage = response.usage

        # Convert usage object to dict - try model_dump() (Pydantic v2), fallback to dict() (Pydantic v1)
        if hasattr(usage, 'model_dump'):
            usage_dict = usage.model_dump()
        elif hasattr(usage, 'dict'):
            usage_dict = usage.dict()
        else:
            # Fallback: extract known attributes
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        
        logger.info(f"OpenAI API Usage: {json.dumps(usage_dict, indent=2)}")
        
        metadata = {
            "model": response.model,
            "latency_ms": latency_ms,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        }
        
        return draft, metadata
        
    except Exception as e:
        # Fallback in case of API error
        logger.error(f"OpenAI API Error: {str(e)}", exc_info=True)
        latency_ms = int((time.time() - start_time) * 1000)
        metadata = {
            "model": "error",
            "latency_ms": latency_ms,
            "token_usage": {},
            "error": str(e)
        }
        return "I apologize, but I encountered an error while generating the draft. Please try again.", metadata
        