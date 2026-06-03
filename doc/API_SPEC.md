# API Reference

The RAG ingestion HTTP API. The contract is **document-centric**: clients submit documents,
get a `document_id`, and poll its status. Jobs are an internal detail and never appear in the
contract (except a debug-only `jobs` field under `?verbose=true`).

Interactive OpenAPI docs are served by the app at **`/docs`** (and the raw schema at
`/openapi.json`).

## Architecture in one line

Two processes share two databases: the **API** (`app/main.py`) enqueues work and serves
status; one or more **workers** (`run_worker.py`) consume the queue, run the pipeline, and
persist results. `QUEUE_DB_URL` is the pgQueuer job queue; `DOCUMENT_DB_URL` is the
document/chunk/embedding store.

## Running locally

Dev runs in the conda env `tarn.rag` (Python 3.12).

1. **Install the runtime extras** (the base env already has fastapi/httpx; the queue and
   storage backends are optional extras):

   ```bash
   conda run -n tarn.rag pip install -e ".[api,queue,postgres,embed,parsers]"
   ```

2. **Configure the environment** — copy `.env.example` to `.env` and set the two database
   URLs. `QUEUE_DB_URL` must be PostgreSQL (pgQueuer); `DOCUMENT_DB_URL` may be Postgres
   (pgvector) or SQLite for local use:

   ```bash
   QUEUE_DB_URL=postgresql://user:pass@localhost:5432/rag_queue
   DOCUMENT_DB_URL=sqlite:///./rag_docs.db        # or postgresql://…/rag_docs
   EMBEDDING_MODEL=sentence-transformers/all-minilm-l6-v2
   EMBEDDING_DIMENSION=384                          # must match the model
   ```

   Install pgQueuer's tables in the queue DB once (see pgQueuer's CLI/docs).

3. **Start the API:**

   ```bash
   conda run -n tarn.rag uvicorn app.main:app --reload
   ```

4. **Start a worker** (run as many as you want for parallelism):

   ```bash
   conda run -n tarn.rag python run_worker.py
   ```

> No Postgres handy? The full test suite exercises the same flow on an in-memory queue + SQLite
> with no external services: `conda run -n tarn.rag python -m pytest -q`.

## Endpoints

Base path: `/v1/ingest`.

Both ingest endpoints accept an optional **`parser`** field that selects the PDF
text-extraction backend for the whole request — `"pypdf"` (default) or `"pdfplumber"` (better
tables/layout). It applies to PDFs only; omit it for the default. An unknown value is rejected
with **422**.

### `POST /v1/ingest/` — ingest from file paths

The worker's load stage reads the files.

```jsonc
// request
{ "file_paths": ["/data/doc1.pdf", "/data/doc2.txt"], "parser": "pdfplumber" }
```

### `POST /v1/ingest/content` — ingest pre-loaded content

A client-supplied `source_id` becomes the `document_id`; otherwise one is assigned. Extra keys
flow into the document metadata.

```jsonc
// request
{ "documents": [ { "content": "full text…", "source_id": "doc-1" } ] }
```

Both endpoints return the same shape (HTTP 200):

```jsonc
{
  "documents": [ { "document_id": "doc-1", "status": "queued" } ],
  "documents_queued": 1
}
```

### `GET /v1/ingest/documents/{document_id}/status` — poll status

Query param `verbose=true` adds the debug-only per-job `jobs` breakdown. Returns **404** if the
document is unknown.

```jsonc
// 200
{
  "document_id": "doc-1",
  "status": "complete",
  "chunk_count": 12,
  "embedding_count": 12,
  "jobs": null            // a list of per-job records when ?verbose=true
}
```

**`status`** is derived from persisted data:

| value | meaning |
|-------|---------|
| `pending` | queued, nothing persisted yet |
| `in_progress` | some chunks/embeddings written, not all |
| `complete` | every chunk has an embedding |
| `failed` | at least one of the document's jobs failed |

## Example

```bash
curl -s localhost:8000/v1/ingest/content \
  -H 'content-type: application/json' \
  -d '{"documents":[{"content":"hello world","source_id":"doc-1"}]}'
# -> {"documents":[{"document_id":"doc-1","status":"queued"}],"documents_queued":1}

curl -s localhost:8000/v1/ingest/documents/doc-1/status
# -> {"document_id":"doc-1","status":"complete","chunk_count":1,"embedding_count":1,"jobs":null}
```
