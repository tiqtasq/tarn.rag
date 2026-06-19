"""Example 04 (ingestion) — ingest the corpus for the generation example (offline, SQLite).

The retrieval substrate for example 04's end-to-end generation. Ingestion is entirely offline (the local
ONNX embedder); only the generation step (``generation.py``) needs an LLM. The ingestion pipeline is read
from ``ingestion.json`` next to this script — same JSON-driven composition as example 02.

Run (from the repo root):

    python -m examples.part_i.example_04.ingestion
    python -m examples.part_i.example_04.generation
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tarnrag import DocumentStatus, IngestionEngine
from tarnrag.core.engine.config import INGESTION_PIPELINE

from examples.common import base_settings, corpus, example_db, require_model

HERE = Path(__file__).resolve().parent
PIPELINE_SPEC = json.loads((HERE / "ingestion.json").read_text())
DOC_PATHS = sorted(corpus("corpus-1").glob("*.txt"))


async def main() -> list[DocumentStatus]:
    """Ingest corpus-1 with the JSON-configured pipeline; return each document's status."""
    require_model()
    db_path = example_db(__file__)
    settings = base_settings(db_path, components={INGESTION_PIPELINE: PIPELINE_SPEC})

    print(f"Ingesting {len(DOC_PATHS)} documents (pipeline from ingestion.json) into {db_path}\n")
    async with await IngestionEngine.create(settings) as engine:
        document_ids = await engine.ingest_paths(
            [str(p) for p in DOC_PATHS], source_ids=[p.stem for p in DOC_PATHS]
        )
        statuses = [await engine.status(doc_id) for doc_id in document_ids]

    for s in statuses:
        print(f"  {s.document_id:16}  {s.status:9}  chunks={s.chunk_count}  embeddings={s.embedding_count}")
    print("\nNext:  python -m examples.part_i.example_04.generation")
    return statuses


if __name__ == "__main__":
    asyncio.run(main())
