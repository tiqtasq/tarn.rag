"""Example 01 — Retrieval (embedded mode, SQLite).

Query the store that ``ingestion.py`` built. ``RetrievalEngine`` embeds the query with the SAME
pipeline ingestion used, runs a dense k-NN search over the SQLite vector index, and returns
ranked, provenance-bearing hits.

Run it after ``ingestion.py`` has populated the store (same one-time setup — see ingestion.py):

    python examples/part-i/example-01/retrieval.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tarnrag import RetrievalEngine
from tarnrag.core.config import DatabaseSettings, EmbeddingSettings, Settings

# These paths MUST match ingestion.py: retrieval opens the very index ingestion wrote.
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = HERE / "rag_docs.db"
MODEL_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"


def build_settings() -> Settings:
    """The same ``database`` + ``embedding`` config as ingestion.py.

    Retrieval only needs those two to match (the id policy, chunking, etc. are ingestion-only):
    it reads from the same SQLite store, and the embedding *fingerprint* recorded at ingestion
    must equal this engine's — so the model, dimension, and prefixes have to be identical.
    """
    return Settings(
        _env_file=None,  # explicit config only — ignore any ambient .env, for reproducibility
        MODE="embedded",
        EMBEDDING_DIMENSION=384,
        database=DatabaseSettings(document_url=f"sqlite:///{DB_PATH}"),
        embedding=EmbeddingSettings(model_dir=str(MODEL_DIR)),
    )


QUERIES = [
    "How do I check a storage tank for corrosion?",
    "Which animal is native to Western Australia?",
]


async def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"No store at {DB_PATH}. Run ingestion.py first.")

    settings = build_settings()

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
