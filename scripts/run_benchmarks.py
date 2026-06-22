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

from tarnrag.core.engine.config import get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import format_comparison, format_sweep, run_benchmark, sweep_benchmark
from tarnrag.eval.benchmarks import HF_LOADERS, LOADERS


async def _run(dataset: str, path: str | None, limit: int | None, hf: bool, sweep: bool) -> None:
    settings = get_settings()
    items = HF_LOADERS[dataset](limit=limit) if hf else LOADERS[dataset](path, limit=limit)
    llm = LanguageModel.create(settings.llm)
    print(
        f"running {len(items)} {dataset} questions "
        f"through reader={settings.llm.provider}:{settings.llm.model} …"
    )
    if sweep:
        reports = await sweep_benchmark(items, llm, settings=settings)
        print(format_sweep(dataset, reports))
    else:
        report = await run_benchmark(items, llm, settings=settings)
        print(format_comparison({dataset: report}))


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
    args = parser.parse_args(argv)
    if args.hf and args.dataset not in HF_LOADERS:
        parser.error(f"--hf supports {sorted(HF_LOADERS)}; {args.dataset!r} needs a file path")
    if not args.hf and not args.path:
        parser.error("provide a dataset file path, or use --hf")
    asyncio.run(_run(args.dataset, args.path, args.limit, args.hf, args.sweep))


if __name__ == "__main__":
    main()
