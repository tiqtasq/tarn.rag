"""Example 01 — Ingestion (embedded mode, SQLite).

The smallest end-to-end ingestion you can run: read the shared sample corpus from
``examples/docs/``, configure tarnrag for *embedded* mode over a local SQLite file, and report
what was stored. Config and shared helpers live in ``examples/common.py``.

In embedded mode the whole pipeline (load -> clean -> chunk -> enrich -> embed -> persist) runs
in THIS process — no Postgres, no job queue — so each ingest call returns only once the
documents are fully chunked, embedded, and written to the SQLite store.

Setup (once, from the repo root):

    pip install -e ".[onnx]"       # the tarnrag package + the ONNX embedding runtime
    python scripts/fetch_model.py  # download the embedding model + tokenizer (needs network)

Run (from the repo root):

    python -m examples.part_i.example_01.ingestion

Then query the same store with the retrieval example (which reuses the same shared config).
"""

from __future__ import annotations

import asyncio

from tarnrag import IngestionEngine

from examples.common import base_settings, corpus, example_db, require_model

# This example's own SQLite store (next to these scripts), and the shared corpus to ingest.
DB_PATH = example_db(__file__)
CORPUS = corpus("corpus-1")
DOC_PATHS = sorted(CORPUS.glob("*.txt"))


async def main() -> None:
    require_model()
    settings = base_settings(DB_PATH)

    # Under ID_POLICY="caller" (the base_settings default) each document needs a stable id; we use
    # the filename stem (e.g. "tank-inspection"). Re-running upserts in place — idempotent per id.
    source_ids = [path.stem for path in DOC_PATHS]
    print(f"Ingesting {len(DOC_PATHS)} documents from {CORPUS} into {DB_PATH}\n")

    # `create` builds the embedded engine (SQLite repository + in-process worker) and stamps the
    # index's build/identity record. `async with` disposes the DB connection pool on exit.
    async with await IngestionEngine.create(settings) as engine:
        document_ids = await engine.ingest_paths(
            [str(path) for path in DOC_PATHS], source_ids=source_ids
        )

        # Embedded mode completes the whole pipeline before returning, so status is terminal here.
        for doc_id in document_ids:
            status = await engine.status(doc_id)
            print(
                f"  {doc_id:16}  {status.status:9}  "
                f"chunks={status.chunk_count}  embeddings={status.embedding_count}"
            )

    print(f"\nStored {len(document_ids)} documents. Now query the same store:")
    print("  python -m examples.part_i.example_01.retrieval")


if __name__ == "__main__":
    asyncio.run(main())
