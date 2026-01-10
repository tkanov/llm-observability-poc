A single API endpoint that:
- takes a support ticket, 
- optionally retrieves a few KB snippets, 
- calls an LLM, 
- and returns a draft

with end-to-end tracing and a repeatable evaluation run.

### Repo layout

- app/ (FastAPI + logic)

- data/kb/ (markdown knowledge base)


### How to run this

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI app:
   ```bash
   uvicorn app.main:app --reload
   ```

4. The API will be available at `http://localhost:8000`
   - Health check: `GET http://localhost:8000/health`
   - API docs: `http://localhost:8000/docs`