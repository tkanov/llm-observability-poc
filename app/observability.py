import os
from contextlib import contextmanager
from typing import Optional, Dict, Any
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Initialize Langfuse client
_langfuse_client: Optional[Langfuse] = None


def get_langfuse_client() -> Optional[Langfuse]:
    """
    Get or initialize Langfuse client.
    
    Returns:
        Langfuse client instance or None if credentials are not configured
    """
    global _langfuse_client
    
    if _langfuse_client is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if public_key and secret_key:
            _langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
        else:
            # Return None if credentials not configured - tracing will be disabled
            return None
    
    return _langfuse_client


@contextmanager
def start_trace(ticket_id: str, env: Optional[str] = None, prompt_version: Optional[str] = None):
    """
    Start a trace for observability using Langfuse.
    
    Args:
        ticket_id: Ticket identifier for tagging the trace
        env: Environment name (e.g., 'prod', 'dev') - defaults to ENV env var
        prompt_version: Version identifier for the prompt - defaults to PROMPT_VERSION env var
    
    Yields:
        Trace object that can be used to create spans, or None if tracing is disabled
    """
    langfuse = get_langfuse_client()
    
    if langfuse is None:
        # Tracing disabled - yield None for backward compatibility
        try:
            yield None
        finally:
            pass
        return
    
    # Get environment and prompt version from kwargs or env vars
    trace_env = env or os.getenv("ENV", "unknown")
    trace_prompt_version = prompt_version or os.getenv("PROMPT_VERSION", "v1")
    
    # Create trace as a top-level span (in Langfuse v2, spans act as traces)
    trace = langfuse.start_span(
        name="draft_reply",
        metadata={
            "ticket_id": ticket_id,
            "env": trace_env,
            "prompt_version": trace_prompt_version,
        },
    )
    
    # Update trace with tags using update_trace method
    trace.update_trace(
        tags=[trace_env, trace_prompt_version, f"ticket:{ticket_id}"],
    )
    
    try:
        yield trace
    finally:
        # Flush to ensure all data is sent
        langfuse.flush()
