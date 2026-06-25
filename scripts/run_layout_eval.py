"""Layout-aware retrieval eval on TAT-QA — source-hit@k, dense vs hybrid, segmented by table / text.

    EMBEDDING__MODEL=thenlper/gte-small EMBEDDING__MODEL_DIR=./models/gte-small \
    python scripts/run_layout_eval.py --limit 100

Builds one shared corpus of every TAT-QA table + paragraph (cached), then for each extractive question
measures whether the top-k retrieved chunks include the gold answer-source element — overall and split by
``answer_from`` (table vs text). Runs dense and hybrid (dense + BM25, RRF) back to back so the table-vs-text
gap, and whether hybrid closes it, are visible in one shot.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, get_settings
from tarnrag.eval.benchmark_runner import HYBRID_RETRIEVAL
from tarnrag.eval.layout import build_tatqa_index, format_source_hit, load_tatqa, stream_tatqa, tatqa_source_hit


async def _run(limit: int | None, k: int, concurrency: int) -> None:
    settings = get_settings()
    corpus, queries = load_tatqa(stream_tatqa(limit), limit=limit)
    emb_tag = f"{settings.embedding.model.split('/')[-1]}_{settings.EMBEDDING_DIMENSION}"
    db_path = f"./docs/tatqa_{limit or 'full'}_{emb_tag}.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"building TAT-QA corpus: {len(corpus)} elements -> {db_path} (cached) … ; {len(queries)} extractive queries")
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=db_path)
    try:
        for label, spec in (("DENSE", None), ("HYBRID", HYBRID_RETRIEVAL)):
            if spec is not None:
                settings.components[RETRIEVAL_PIPELINE] = spec
            report = await tatqa_source_hit(queries, repo, embedder, settings=settings, k=k, concurrency=concurrency)
            print("\n" + format_source_hit(report, tag=f"[{label}]"))
    finally:
        await repo.disconnect()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TAT-QA layout-aware retrieval eval (source-hit@k, dense vs hybrid)")
    p.add_argument("--limit", type=int, default=100, help="cap TAT-QA records (corpus + queries); default 100")
    p.add_argument("-k", type=int, default=10, help="top-k retrieved chunks counted for source-hit")
    p.add_argument("--concurrency", type=int, default=8, help="parallel retrievals")
    args = p.parse_args(argv)
    asyncio.run(_run(args.limit, args.k, args.concurrency))


if __name__ == "__main__":
    main()
