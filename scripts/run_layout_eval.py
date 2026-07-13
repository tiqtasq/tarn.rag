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
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import HYBRID_RETRIEVAL
from tarnrag.eval.layout import (
    build_tatqa_index,
    format_numeric,
    load_tatqa_numeric,
    tatqa_numeric,
    format_attribution,
    format_source_hit,
    load_tatqa,
    stream_tatqa,
    tatqa_attribution,
    tatqa_source_hit,
)


async def _run(
    limit: int | None, k: int, concurrency: int, attribution: bool, hybrid: bool, table_view: str,
    rerank: list[str], numeric: bool,
) -> None:
    settings = get_settings()
    rows = stream_tatqa(limit)
    corpus, queries = load_tatqa(rows, limit=limit)
    emb_tag = f"{settings.embedding.model.split('/')[-1]}_{settings.EMBEDDING_DIMENSION}"
    if settings.embedding.contextualize_tables:  # a distinct index — the embed text differs (P1)
        emb_tag += "_ctx"
    db_path = f"./data/tatqa_{limit or 'full'}_{emb_tag}.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"building TAT-QA corpus: {len(corpus)} elements -> {db_path} (cached) … ; {len(queries)} extractive queries")
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=db_path)
    try:
        # Pin the retrieval spec EXPLICITLY either way — a measurement never inherits a default.
        dense_only = {
            "class_name": "retrieval_pipeline",
            "retrievers": [{"class_name": "dense"}],
            "fuser": {"class_name": "identity"},
        }
        settings.components[RETRIEVAL_PIPELINE] = HYBRID_RETRIEVAL if hybrid else dense_only
        if numeric:  # PP-6: the arithmetic slice through table_lookup — deterministic, LLM-free
            nq = load_tatqa_numeric(rows, limit=limit)
            print(f"scoring {len(nq)} arithmetic questions through table_lookup (no LLM) …")
            spec = {"class_name": "table_lookup", "fallback": None, "top_k": k, "row_coverage": 0.5}
            report = await tatqa_numeric(
                nq, repo, embedder, settings=settings, reasoner_spec=spec, k=k, concurrency=concurrency
            )
            print("\n" + format_numeric(report, tag="[HYBRID]" if hybrid else "[DENSE]"))
        elif attribution:  # generation + LLM-judged citations (needs an LLM); retrieval per `hybrid`
            llm = LanguageModel.create(settings.llm)
            print(f"scoring attribution through reader={settings.llm.provider}:{settings.llm.model}"
                  f"{' + hybrid' if hybrid else ''}, table_view={table_view} …")
            overall, by_seg = await tatqa_attribution(
                queries, llm, repo, embedder, settings=settings, table_view=table_view, concurrency=concurrency
            )
            print("\n" + format_attribution(overall, by_seg))
        else:  # retrieval-only source-hit: dense vs hybrid (vs reranked), each leg's spec pinned explicitly.
            legs = [("DENSE", dense_only), ("HYBRID", HYBRID_RETRIEVAL)]
            for rr in rerank:  # P4: a second-pass reranker on top of the hybrid shortlist.
                # The spec is pinned FULLY (top_n included) — a measurement never inherits a default.
                legs.append((
                    f"HYBRID+{rr.upper()}",
                    {**HYBRID_RETRIEVAL, "reranker": {"class_name": rr, "top_n": 20}},
                ))
            for label, spec in legs:
                settings.components[RETRIEVAL_PIPELINE] = spec
                report = await tatqa_source_hit(queries, repo, embedder, settings=settings, k=k, concurrency=concurrency)
                print("\n" + format_source_hit(report, tag=f"[{label}]"))
    finally:
        await repo.disconnect()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TAT-QA layout-aware eval: source-hit@k (dense vs hybrid) or attribution")
    p.add_argument("--limit", type=int, default=100, help="cap TAT-QA records (corpus + queries); default 100")
    p.add_argument("-k", type=int, default=10, help="top-k retrieved chunks counted for source-hit")
    p.add_argument("--concurrency", type=int, default=8, help="parallel retrievals / answers")
    p.add_argument(
        "--attribution", action="store_true",
        help="instead of source-hit: answer each question and have an LLM judge the citations "
             "(grounded_rate = attribution precision) + F1/EM, segmented by table/text — needs an LLM",
    )
    p.add_argument("--hybrid", action="store_true", help="retrieve with dense + BM25 (RRF) instead of dense-only")
    p.add_argument(
        "--table-view", choices=["structured", "text"], default=None,
        help="how the reader + grounding judge see table chunks (REQUIRED with --attribution; "
             "pinned explicitly — a measurement never inherits a default)",
    )
    p.add_argument(
        "--rerank", action="append", choices=["cross_encoder", "llm_judge"], default=[],
        help="add a HYBRID+<reranker> source-hit leg (repeatable; spec pinned explicitly per leg). "
             "cross_encoder needs the local ONNX model; llm_judge needs an LLM key.",
    )
    p.add_argument("--numeric", action="store_true",
                   help="PP-6: run the arithmetic slice through the table_lookup reasoner (no LLM)")
    args = p.parse_args(argv)
    if args.attribution and args.table_view is None:
        p.error("--attribution requires --table-view (pin the view; don't inherit a default)")
    asyncio.run(_run(
        args.limit, args.k, args.concurrency, args.attribution, args.hybrid,
        args.table_view or "structured", args.rerank, args.numeric,
    ))


if __name__ == "__main__":
    main()
