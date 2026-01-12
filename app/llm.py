import os
import time
import logging
import json
from typing import Optional
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


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """
    Calculate cost in USD based on model and token usage.
    
    Args:
        model: OpenAI model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
    
    Returns:
        Cost in USD or None if cost cannot be calculated
    """
    
    # These can be overridden via environment variables or updated as needed
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},  # 'leagcy' model according to openai pricing page
    }
    
    # Check if pricing is available for this model
    if model not in pricing:
        return None
    
    model_pricing = pricing[model]
    input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]
    
    return input_cost + output_cost


def generate_draft(customer_message, context=None, snippets=None, trace=None, prompt_version: Optional[str] = None):
    """
    Generate a draft reply based on the customer message.
    
    Args:
        customer_message: Customer message string
        context: Optional dict (not used in simplified version, kept for compatibility)
        snippets: Optional list of snippet dicts with source_id and excerpt
        trace: Optional Langfuse trace object for observability
        prompt_version: Optional prompt version identifier
    
    Returns:
        Tuple of (draft_text: str, metadata: dict) where metadata contains:
        - model: str
        - latency_ms: int
        - token_usage: dict with prompt_tokens, completion_tokens, total_tokens
    """
    # Get prompt version from env if not provided
    if prompt_version is None:
        prompt_version = os.getenv("PROMPT_VERSION", "v1")
    
    # Build system prompt with knowledge base snippets if available
    system_prompt = "You are a helpful customer support agent with a fun and engaging attitude. Draft a short, professional, and friendly reply to the customer's message."
    
    user_content = customer_message
    
    # Inject snippets if any found
    if snippets and len(snippets) > 0:
        snippets_text = "\n\nRelevant knowledge base information:\n"
        for i, snippet in enumerate(snippets, 1):
            snippets_text += f"\n[{i}] From {snippet['source_id']}:\n{snippet['excerpt']}\n"
        
        user_content += snippets_text
    
    start_time = time.time()
    
    # Prepare request parameters
    model = "gpt-4o-mini"  # Using cost-effective model, can be made configurable
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    try:
        request_params = {
            "model": model,
            "messages": messages,
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

        cost = _calculate_cost(response.model, usage.prompt_tokens, usage.completion_tokens)
        
        metadata = {
            "model": response.model,
            "latency_ms": latency_ms,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "cost": cost,
            "prompt_version": prompt_version
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
            "error": str(e),
            "prompt_version": prompt_version
        }
        return "I apologize, but I encountered an error while generating the draft. Please try again.", metadata
        