"""Run a MOTHRAG benchmark (HotpotQA / 2WikiMultiHopQA / MuSiQue, distractor) through tarn.rag's generation
engine and print F1 / EM against MOTHRAG's published numbers.

Download the dataset file first (see ``tarnrag/eval/benchmarks.py`` for sources + formats), then point the
reader at any OpenAI-compatible endpoint (e.g. a Llama-3.3-70B server, to match MOTHRAG's reader) via env::

    LLM__PROVIDER=openai \
    LLM__MODEL=meta-llama/Llama-3.3-70B-Instruct \
    LLM__API_BASE_URL=https://your-endpoint/v1 \
    LLM__API_KEY=sk-... \
    python scripts/eval_mothrag.py hotpotqa /data/hotpot_dev_distractor_v1.json --limit 50

Start with a small ``--limit`` to gauge cost/quality (one multi-hop answer is several LLM calls). The
embedder + the GENERATION_PIPELINE (reasoner / grounding) are whatever Settings selects (env / .env). The
comparison is a baseline, not a controlled replication (different reader/embedder; lean reasoners).
"""

from __future__ import annotations

import argparse
import asyncio

from tarnrag.core.engine.config import get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import format_comparison, run_benchmark
from tarnrag.eval.benchmarks import LOADERS


async def _run(dataset: str, path: str, limit: int | None) -> None:
    settings = get_settings()
    items = LOADERS[dataset](path, limit=limit)
    print(
        f"running {len(items)} {dataset} questions "
        f"through reader={settings.llm.provider}:{settings.llm.model} …"
    )
    llm = LanguageModel.create(settings.llm)
    report = await run_benchmark(items, llm, settings=settings)
    print(format_comparison({dataset: report}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a MOTHRAG benchmark through tarn.rag's generation engine")
    parser.add_argument("dataset", choices=sorted(LOADERS), help="which benchmark")
    parser.add_argument("path", help="path to the downloaded dataset file")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    args = parser.parse_args(argv)
    asyncio.run(_run(args.dataset, args.path, args.limit))


if __name__ == "__main__":
    main()
