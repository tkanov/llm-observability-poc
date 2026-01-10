
### Endpoints

- `GET /health` → `{ "status": "ok" }`
    
- `POST /draft-reply` → returns `{ draft, citations, metadata }`
    

### Request/response shape

**Input**

- `ticket_id` (string)
    
- `subject` (string)
    
- `customer_message` (string)
    

**Output**

- `draft` (string)
    
- `citations` (list of `{source_id, excerpt}`)
    
- `metadata` (model, latency_ms, token_usage/cost if available)
    

## Minimal internals

### Modules

- `app/main.py` – FastAPI wiring
    
- `app/llm.py` – one function: `generate_draft(prompt, context) -> text`
    
- `app/retrieval.py` – one function: `retrieve(customer_message) -> [snippets]` 
    
- `app/observability.py` – one function: `start_trace(...) / span(...)` wrapper
    

### Data

- `data/kb/` with short Markdown files (refund policy, SLA, troubleshooting, pricing, etc.)