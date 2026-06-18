"""Retrieval eval harness — compare retrieval-pipeline specs (and embed-time variants) on a labeled set.

``sweep`` scores many ``RetrievalPipeline`` specs against one index (the method comparison); embed-time
variants are compared by sweeping against a separately-built index. Relevance is content-based — a result
is relevant iff its text contains a gold phrase — so labels survive re-ingestion (see ``dataset``).
"""

from tarnrag.eval import metrics
from tarnrag.eval.dataset import EvalQuery, EvalSet
from tarnrag.eval.harness import (
    EvalReport,
    QueryReport,
    evaluate_pipeline,
    format_reports,
    sweep,
)

__all__ = [
    "metrics",
    "EvalQuery",
    "EvalSet",
    "EvalReport",
    "QueryReport",
    "evaluate_pipeline",
    "sweep",
    "format_reports",
]
