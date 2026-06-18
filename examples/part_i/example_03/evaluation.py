"""Example 03 — Comparing retrieval methods with the eval harness (embedded mode, SQLite).

The payoff of the retrieval seams: **comparing methods is just swapping the RetrievalPipeline spec.**
This sweeps three pipelines — dense, sparse (BM25), and hybrid (RRF over both) — against the SAME index
that ``example_03/ingestion.py`` built, scores each over a small labeled query set (``evalset.json``),
and prints a hit@k / MRR / nDCG@k comparison table.

The corpus is tiny and the queries easy, so the scores run high — the point is the machinery (swap a
spec, re-score), not the benchmark. Relevance is content-based: a hit counts if its chunk text contains
one of a query's gold phrases (so labels survive re-chunking / re-embedding).

Run (after the ingestion example):

    python -m examples.part_i.example_03.ingestion
    python -m examples.part_i.example_03.evaluation
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tarnrag import RetrievalEngine
from tarnrag.eval import EvalReport, EvalSet, format_reports, sweep
from tarnrag.retrieval import RetrievalContext

from examples.common import base_settings, example_db, require_model

EVALSET = Path(__file__).resolve().parent / "evalset.json"

# Comparing retrieval methods = different RetrievalPipeline specs against the same index.
PIPELINES = {
    "dense": {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}]},
    "sparse (bm25)": {
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": "sparse"}],
        "fuser": {"class_name": "identity"},
    },
    "hybrid (rrf)": {
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
        "fuser": {"class_name": "rrf"},
    },
}


async def main() -> dict[str, EvalReport]:
    """Sweep the pipelines over the labeled set against the example_03 store; print the table."""
    require_model()
    db_path = example_db(__file__)  # the store example_03/ingestion.py wrote
    if not db_path.exists():
        raise SystemExit(
            f"No store at {db_path}. Run the ingestion example first:\n"
            f"  python -m examples.part_i.example_03.ingestion"
        )
    settings = base_settings(db_path)
    evalset = EvalSet.from_json(EVALSET)

    # RetrievalEngine.create opens the index (validating the embedding fingerprint); the sweep needs
    # only its store + embedder as the shared context the pipeline specs all run against.
    async with await RetrievalEngine.create(settings) as engine:
        ctx = RetrievalContext(engine.repository, engine.embedder)
        reports = await sweep(PIPELINES, ctx, evalset, k=3)

    print(
        f"Comparing {len(PIPELINES)} retrieval pipelines over {len(evalset.queries)} queries "
        "(k=3, relevance = gold phrase in the chunk text):\n"
    )
    print(format_reports(reports))
    return reports


if __name__ == "__main__":
    asyncio.run(main())
