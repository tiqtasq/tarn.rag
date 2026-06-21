"""Retrieval eval harness — compare retrieval-pipeline specs (and embed-time variants) on a labeled set.

``sweep`` scores many ``RetrievalPipeline`` specs against one index (the method comparison); embed-time
variants are compared by sweeping against a separately-built index. Relevance is content-based — a result
is relevant iff its text contains a gold phrase — so labels survive re-ingestion (see ``dataset``).
"""

from tarnrag.eval import metrics
from tarnrag.eval.benchmarks import (
    HF_LOADERS,
    LOADERS,
    BenchItem,
    load_2wiki,
    load_hotpotqa,
    load_hotpotqa_hf,
    load_musique,
)
from tarnrag.eval.benchmark_runner import (
    MOTHRAG_PUBLISHED,
    format_comparison,
    format_sweep,
    run_benchmark,
    sweep_benchmark,
)
from tarnrag.eval.dataset import EvalQuery, EvalSet
from tarnrag.eval.generation import (
    GenEvalQuery,
    GenEvalReport,
    GenEvalSet,
    GenQueryReport,
    citation_coverage,
    content_hit,
    evaluate_generation,
    exact_match,
    format_generation_reports,
    sweep_generation,
    token_f1,
)
from tarnrag.eval.harness import (
    EvalReport,
    QueryReport,
    by_query_type,
    evaluate_pipeline,
    format_reports,
    format_segmented,
    sweep,
)

__all__ = [
    "metrics",
    # retrieval eval
    "EvalQuery",
    "EvalSet",
    "EvalReport",
    "QueryReport",
    "evaluate_pipeline",
    "sweep",
    "by_query_type",
    "format_reports",
    "format_segmented",
    # generation eval
    "GenEvalQuery",
    "GenEvalSet",
    "GenEvalReport",
    "GenQueryReport",
    "evaluate_generation",
    "sweep_generation",
    "format_generation_reports",
    "content_hit",
    "token_f1",
    "exact_match",
    "citation_coverage",
    # MOTHRAG benchmark harness
    "BenchItem",
    "load_hotpotqa",
    "load_hotpotqa_hf",
    "load_2wiki",
    "load_musique",
    "LOADERS",
    "HF_LOADERS",
    "run_benchmark",
    "sweep_benchmark",
    "format_comparison",
    "format_sweep",
    "MOTHRAG_PUBLISHED",
]
