# tarn.rag

**tarnrag** — composable, DAG-based RAG **ingestion + retrieval + generation** engines: documents in, a
queryable vector index out, optionally a grounded answer with a proof tree. Pure library — there is no HTTP
layer here (a consuming REST API lives in the separate **tiqtasq.backend** repo).

- **Ingestion** — a config-driven pipeline (extract → enrich → clean → chunk → embed) writing one store
  (a §8 index: documents · chunks · vectors · FTS · metadata), embedded (SQLite + sqlite-vec/FTS5) or
  distributed (Postgres + pgvector, pgQueuer workers).
- **Retrieval** — config-driven, swappable Components: dense / sparse retrievers → RRF fusion → license &
  scope pre-filter → auto-merge → cross-encoder rerank, with layout-grade provenance on every hit.
- **Generation** — multi-hop reasoning (single-hop / iterative / decomposition) → grounding check →
  proof tree + evidence, over the retrieval substrate. LLM-pluggable; retrieval stays LLM-free by default.

## Install

```bash
pip install tarn-rag
```

The base install is the lightweight SQLite/embedded path. Backends and features are **extras**:

| Extra | Adds |
|---|---|
| `onnx` | local ONNX embedding (`onnxruntime` + `tokenizers`) |
| `embeddings-api` | hosted embedding APIs (OpenAI / Voyage / Gemini) |
| `parsers` | PDF/HTML document loaders (`pypdf`, `pdfplumber`, `beautifulsoup4`) |
| `docling` | high-fidelity layout-aware PDF extraction (heavy) |
| `generation` | the LLM backend for the generation layer (`anthropic`) |
| `console` | the interactive `tarnrag` console UI (`rich`) |
| `postgres` / `queue` | the distributed backends (`asyncpg` + `pgvector` / `pgqueuer`) |
| `all` | the common embedded runtime (`onnx`, `parsers`, `embeddings-api`, `generation`, `console`) |

```bash
pip install "tarn-rag[onnx,parsers]"     # local embedder + PDF/HTML extraction
pip install "tarn-rag[all]"              # the full single-process runtime
pip install "tarn-rag[postgres,queue]"  # the distributed backends
```

## Quickstart

The high-level facade (`TarnRag`) wires ingestion + retrieval + generation over one store; each call returns
an `Outcome` (the value) plus a `Report` of any non-fatal issues:

```python
import asyncio
from tarnrag import TarnRag

async def main():
    async with TarnRag("config.json") as tarn:   # a JSON Settings file
        await tarn.ingest(["docs/"])             # ingest files / directories
        hits = await tarn.retrieve("how do I inspect a tank?", top_k=8)
        answer = await tarn.ask("…")             # grounded answer + proof tree (needs an LLM key)
        print(answer.value.answer)

asyncio.run(main())
```

Or drive the engines directly (each built from `Settings` via `create()`):

```python
from tarnrag import IngestionEngine, RetrievalEngine

engine = await IngestionEngine.create()
ids = await engine.ingest_paths(["/data/spec.pdf"])

async with await RetrievalEngine.create() as r:   # validates schema + embedding fingerprint
    results = await r.search_text("how do I inspect a tank?", top_k=8)
```

An interactive console (needs the `console` extra):

```bash
tarnrag config.json          # or: python -m tarnrag.console config.json
```

## Modes

- **`MODE='embedded'`** (default) — runs the whole pipeline in-process over SQLite; each ingest call
  processes to completion. Zero infrastructure.
- **`MODE='distributed'`** — enqueues to pgQueuer over Postgres; run `python run_worker.py` as one or more
  separate consumer processes.

Configuration is environment-driven (pydantic-settings, `GROUP__FIELD` convention, e.g. `EMBEDDING__MODEL`,
`DATABASE__DOCUMENT_URL`); see `.env.example`.

## Docs

Design specs and architecture notes live in [`doc/`](./doc): `FUNCTIONAL_REQUIREMENTS.md` (ingestion),
`ModusQ_RetrievalSubsystemSpec.md` + `retrieval-architecture-design.md` (retrieval), and
`generation-architecture-design.md` (generation). `CLAUDE.md` is the architecture-oriented contributor guide.

## License

MIT — see [LICENSE](./LICENSE).
