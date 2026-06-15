"""Example 01 — Retrieval (embedded mode, SQLite).

Query the store that ``ingestion.py`` built. ``RetrievalEngine`` embeds the query with the SAME
pipeline ingestion used (the config is shared via ``examples/common.py``), runs a dense k-NN
search over the SQLite vector index, and returns ranked, provenance-bearing hits.

Run it after the ingestion example has populated the store (same one-time setup — see
``ingestion.py``):

    python -m examples.part_i.example_01.retrieval
"""

from __future__ import annotations

import asyncio

from tarnrag import RetrievalEngine

from examples.common import base_settings, example_db, require_model

DB_PATH = example_db(__file__)  # the same store the ingestion example wrote

QUERIES = [
    "How do I check a storage tank for corrosion?",
    "Which animal is native to Western Australia?",
]


async def main() -> None:
    require_model()
    if not DB_PATH.exists():
        raise SystemExit(
            f"No store at {DB_PATH}. Run the ingestion example first:\n"
            f"  python -m examples.part_i.example_01.ingestion"
        )

    settings = base_settings(DB_PATH)

    # `create` connects the SQLite store and validates compatibility (schema + embedding
    # fingerprint); it raises RetrievalError if the index was built with a different pipeline.
    async with await RetrievalEngine.create(settings) as engine:
        for query in QUERIES:
            print(f"\nQuery: {query!r}")
            # search_text embeds the query and returns up to `top_k` hits, best first.
            results = await engine.search_text(query, top_k=3)
            if not results:
                print("  (no results)")
            # RetrievalResult.score: higher is better. Step A is dense-only, so the score is the
            # negated vector distance (closer match -> nearer to 0); results arrive already ranked.
            for rank, hit in enumerate(results, start=1):
                snippet = hit.text[:88].replace("\n", " ")
                print(f"  {rank}. score={hit.score:+.3f}  document={hit.document_id!r}")
                print(f"     {snippet}...")


if __name__ == "__main__":
    asyncio.run(main())
