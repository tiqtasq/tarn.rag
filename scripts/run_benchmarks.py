"""Run the multi-hop QA benchmarks (HotpotQA / 2WikiMultiHopQA / MuSiQue, distractor) through tarn.rag's
generation engine and print F1 / EM against MOTHRAG's published numbers.

Stream the datasets from HuggingFace with ``--hf`` (no download), and point the reader at any
OpenAI-compatible endpoint (e.g. a Llama-3.3-70B server, to match MOTHRAG's reader) via env::

    LLM__PROVIDER=openai \
    LLM__MODEL=meta-llama/Llama-3.3-70B-Instruct \
    LLM__API_BASE_URL=https://your-endpoint/v1 \
    LLM__API_KEY=sk-... \
    python scripts/run_benchmarks.py hotpotqa --hf --limit 200 --sweep

Start with a small ``--limit`` to gauge cost/quality (one multi-hop answer is several LLM calls). The
embedder + the GENERATION_PIPELINE (reasoner / grounding) are whatever Settings selects (env / .env). The
comparison is a baseline, not a controlled replication (different reader/embedder; lean reasoners).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import (
    BRIDGE_RETRIEVAL,
    build_corpus_index,
    format_comparison,
    format_sweep,
    run_benchmark,
    run_over_corpus,
    sweep_benchmark,
    sweep_over_corpus,
)
from tarnrag.eval.benchmarks import HF_LOADERS, LOADERS, corpus_from_items


async def _run(
    dataset: str, path: str | None, limit: int | None, hf: bool, sweep: bool, bridge: bool,
    corpus: str, corpus_limit: int | None,
) -> None:
    settings = get_settings()
    if bridge:  # Phase-2 bridge retrieval (multi-query + LLM judge) instead of the lean dense-only default
        settings.components[RETRIEVAL_PIPELINE] = BRIDGE_RETRIEVAL
    llm = LanguageModel.create(settings.llm)
    tag = " + bridge" if bridge else ""
    if corpus == "pool":
        await _run_over_pool(dataset, path, limit, hf, sweep, settings, llm, tag, corpus_limit)
    else:
        await _run_distractor(dataset, path, limit, hf, sweep, settings, llm, tag)


async def _run_distractor(dataset, path, limit, hf, sweep, settings, llm, tag) -> None:
    items = HF_LOADERS[dataset](limit=limit) if hf else LOADERS[dataset](path, limit=limit)
    print(
        f"running {len(items)} {dataset} questions (distractor) "
        f"through reader={settings.llm.provider}:{settings.llm.model}{tag} …"
    )
    if sweep:
        print(format_sweep(dataset, await sweep_benchmark(items, llm, settings=settings)))
    else:
        print(format_comparison({dataset: await run_benchmark(items, llm, settings=settings)}))


async def _run_over_pool(dataset, path, limit, hf, sweep, settings, llm, tag, corpus_limit) -> None:
    # The corpus is built from the first ``corpus_limit`` dev questions (all of them if None) — sized larger
    # than the eval slice so the eval questions' gold passages are in the haystack. A moderate corpus_limit
    # keeps the (per-doc) build tractable while still making retrieval miss-able.
    all_items = HF_LOADERS[dataset](limit=corpus_limit) if hf else LOADERS[dataset](path, limit=corpus_limit)
    corpus = corpus_from_items(all_items)
    eval_items = all_items[:limit] if limit else all_items
    db_path = f"./docs/bench_{dataset}_pool_{corpus_limit or 'full'}.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"building corpus index for {dataset}: {len(corpus)} passages -> {db_path} (cached) …")
    repo, embedder = await build_corpus_index(corpus, settings, db_path=db_path)
    try:
        print(
            f"running {len(eval_items)} {dataset} questions over the shared corpus "
            f"through reader={settings.llm.provider}:{settings.llm.model}{tag} …"
        )
        if sweep:
            print(format_sweep(dataset, await sweep_over_corpus(eval_items, llm, repo, embedder, settings=settings)))
        else:
            print(format_comparison({dataset: await run_over_corpus(eval_items, llm, repo, embedder, settings=settings)}))
    finally:
        await repo.disconnect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a MOTHRAG benchmark through tarn.rag's generation engine")
    parser.add_argument("dataset", choices=sorted(LOADERS), help="which benchmark")
    parser.add_argument("path", nargs="?", help="path to the downloaded dataset file (omit with --hf)")
    parser.add_argument(
        "--hf", action="store_true", help=f"stream from HuggingFace instead of a file ({sorted(HF_LOADERS)})"
    )
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument(
        "--sweep", action="store_true",
        help="sweep the reasoners (single_hop / iterative / decomposition) instead of one configured run",
    )
    parser.add_argument(
        "--bridge", action="store_true",
        help="use the Phase-2 bridge retrieval (multi-query expansion + LLM relevance judge) — needs an LLM",
    )
    parser.add_argument(
        "--corpus", choices=["distractor", "pool"], default="distractor",
        help="retrieval setting: 'distractor' (per-question pool, default) or 'pool' (one shared corpus "
             "built from the dev set — the fullwiki-style setting; built once + cached)",
    )
    parser.add_argument(
        "--corpus-limit", type=int, default=None,
        help="with --corpus pool: build the corpus from only the first N dev questions (a moderate haystack "
             "the per-doc ingest can build in reasonable time); defaults to the whole dev set",
    )
    args = parser.parse_args(argv)
    if args.hf and args.dataset not in HF_LOADERS:
        parser.error(f"--hf supports {sorted(HF_LOADERS)}; {args.dataset!r} needs a file path")
    if not args.hf and not args.path:
        parser.error("provide a dataset file path, or use --hf")
    asyncio.run(
        _run(args.dataset, args.path, args.limit, args.hf, args.sweep, args.bridge, args.corpus, args.corpus_limit)
    )


if __name__ == "__main__":
    main()
