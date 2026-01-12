import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.llm import generate_draft, verify_api_key
from app.retrieval import retrieve

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
    prompt_version = os.getenv("PROMPT_VERSION", "v1")
    
    # Retrieve relevant snippets from knowledge base
    snippets = retrieve(request.customer_message)
    
    # Generate CS response draft using OpenAI with snippets injected
    draft = generate_draft(
        request.customer_message, 
        snippets=snippets,
        prompt_version=prompt_version
    )
    
    # Build citations from retrieved snippets
    citations = []
    if snippets:
        for snippet in snippets:
            citations.append({
                "source_id": snippet["source_id"],
                "excerpt": snippet["excerpt"]
            })
    
    return {
        'draft': draft,
        'citations': citations,
    }
