"""Example 03 — Ingestion for the retrieval-method comparison (embedded mode, SQLite).

Sets up the store that ``example_03/evaluation.py`` sweeps. Uses a small-chunk pipeline (``chunk_size``
80, like example 02) so each short doc splits into several chunks — that gives the retrieval methods
something to rank differently over (with one chunk per doc, every method trivially nails it).

Run (from the repo root, after example 01's one-time setup):

    python -m examples.part_i.example_03.ingestion
    python -m examples.part_i.example_03.evaluation
"""

from __future__ import annotations

import asyncio

from tarnrag import DocumentStatus, IngestionEngine
from tarnrag.core.engine.config import INGESTION_PIPELINE

from examples.common import base_settings, corpus, example_db, require_model

# A small-chunk pipeline — several chunks per short doc (see example 02 for the JSON-pipeline idea).
PIPELINE_SPEC = {
    "class_name": "pipeline",
    "stages": [
        {"class_name": "LoadAndParse"},
        {"class_name": "CleanAndNormalize"},
        {"class_name": "Chunk", "chunker": {"class_name": "recursive", "chunk_size": 80, "overlap": 16}},
        {"class_name": "Embed"},
    ],
}

CORPUS = corpus("corpus-1")
DOC_PATHS = sorted(CORPUS.glob("*.txt"))


async def main() -> list[DocumentStatus]:
    """Ingest the corpus with the small-chunk pipeline; return each document's terminal status."""
    require_model()
    db_path = example_db(__file__)
    settings = base_settings(db_path, components={INGESTION_PIPELINE: PIPELINE_SPEC})

    source_ids = [path.stem for path in DOC_PATHS]
    print(f"Ingesting {len(DOC_PATHS)} documents from {CORPUS} into {db_path}\n")

    async with await IngestionEngine.create(settings) as engine:
        document_ids = await engine.ingest_paths(
            [str(path) for path in DOC_PATHS], source_ids=source_ids
        )
        statuses = [await engine.status(doc_id) for doc_id in document_ids]

    for status in statuses:
        print(
            f"  {status.document_id:16}  {status.status:9}  "
            f"chunks={status.chunk_count}  embeddings={status.embedding_count}"
        )
    print(f"\nStored {len(statuses)} documents in {sum(s.chunk_count for s in statuses)} chunks.")
    print("Now compare retrieval methods:  python -m examples.part_i.example_03.evaluation")
    return statuses


if __name__ == "__main__":
    asyncio.run(main())
