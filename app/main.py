from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.llm import generate_draft, verify_api_key
from app.observability import start_trace

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """Verify OpenAI API key on app startup."""
    verify_api_key()


class DraftRequest(BaseModel):
    ticket_id: str
    subject: str
    customer_message: str
    language: Optional[str] = None


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/draft-reply')
def draft_reply(request: DraftRequest):
    with start_trace(ticket_id=request.ticket_id):
        # Generate CS response draft using OpenAI
        # Simple prompt + customer message concatenation happens in llm.py
        draft, metadata = generate_draft(request.customer_message)
        
        # Build response
        # Citations are empty for simplified version (as per spec v0)
        citations = []
        
        return {
            'draft': draft,
            'citations': citations,
            'metadata': metadata
        }