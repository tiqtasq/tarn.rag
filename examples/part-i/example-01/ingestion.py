"""Example 01 — Ingestion (embedded mode, SQLite).

The smallest end-to-end ingestion you can run: configure tarnrag for *embedded* mode over a
local SQLite file, ingest a few in-memory documents, and report what was stored.

In embedded mode the whole pipeline (load -> clean -> chunk -> enrich -> embed -> persist) runs
in THIS process — no Postgres, no job queue — so each ingest call returns only once the
documents are fully chunked, embedded, and written to the SQLite store.

Setup (once, from the repo root):

    pip install -e ".[onnx]"       # the tarnrag package + the ONNX embedding runtime
    python scripts/fetch_model.py  # download the embedding model + tokenizer (needs network)

Run:

    python examples/part-i/example-01/ingestion.py

Then query the same store with ``retrieval.py`` (which builds an identical config).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tarnrag import IngestionEngine
from tarnrag.core.config import DatabaseSettings, EmbeddingSettings, Settings

# Resolve every path from THIS file, so the example runs from any working directory.
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = HERE / "rag_docs.db"                          # the SQLite store this example writes
MODEL_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"   # local ONNX model + tokenizer


def build_settings() -> Settings:
    """The minimal embedded/SQLite config.

    IMPORTANT: ``retrieval.py`` must build the SAME settings. Both sides share this one SQLite
    store, and ``RetrievalEngine`` refuses to open an index that was built with a different
    embedding pipeline (it compares an embedding *fingerprint* derived from the model, dimension,
    prefixes, etc.). Keep the ``database`` and ``embedding`` config identical across the two.
    """
    return Settings(
        _env_file=None,           # explicit config only — ignore any ambient .env, for reproducibility
        MODE="embedded",          # run the entire pipeline in-process (no Postgres / queue)
        ID_POLICY="caller",       # we supply our own stable document ids (see DOCUMENTS below)
        EMBEDDING_DIMENSION=384,  # output dimension of all-MiniLM-L6-v2
        database=DatabaseSettings(document_url=f"sqlite:///{DB_PATH}"),
        embedding=EmbeddingSettings(model_dir=str(MODEL_DIR)),
    )


# A few tiny documents to index. Under ID_POLICY="caller" each one needs a stable ``source_id``
# (== its document id). Re-running this script upserts in place — ingestion is idempotent per id.
DOCUMENTS = [
    {
        "source_id": "tank-inspection",
        "content": (
            "Storage tank inspection procedure: visually inspect the shell and roof for "
            "corrosion, measure shell thickness by ultrasonic testing, and check the foundation "
            "for settlement before returning the tank to service."
        ),
    },
    {
        "source_id": "pump-maintenance",
        "content": (
            "Centrifugal pump maintenance: check the mechanical seal for leaks, verify bearing "
            "lubrication, and confirm shaft alignment before restarting the pump."
        ),
    },
    {
        "source_id": "quokka-fact",
        "content": (
            "The quokka is a small macropod about the size of a domestic cat, native to Western "
            "Australia and known for its photogenic, smiling facial expression."
        ),
    },
]


async def main() -> None:
    if not (MODEL_DIR / "model.onnx").exists():
        raise SystemExit(
            f"Embedding model not found under {MODEL_DIR}.\n"
            f"Fetch it once (needs network):  python scripts/fetch_model.py"
        )

    settings = build_settings()
    print(f"Ingesting {len(DOCUMENTS)} documents into {DB_PATH}\n")

    # `create` builds the embedded engine (SQLite repository + in-process worker) and stamps the
    # index's build/identity record. `async with` disposes the DB connection pool on exit.
    async with await IngestionEngine.create(settings) as engine:
        document_ids = await engine.ingest_content(DOCUMENTS)

        # Embedded mode completes the whole pipeline before returning, so status is terminal here.
        for doc_id in document_ids:
            status = await engine.status(doc_id)
            print(
                f"  {doc_id:16}  {status.status:9}  "
                f"chunks={status.chunk_count}  embeddings={status.embedding_count}"
            )

    print(f"\nStored {len(document_ids)} documents. Now run retrieval.py against the same store.")


if __name__ == "__main__":
    asyncio.run(main())
