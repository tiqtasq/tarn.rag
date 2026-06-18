"""Retrieval domain — public surface.

``RetrievalEngine`` is the high-level facade (``RetrievalEngine.create()``); ``Query`` /
``RetrievalResult`` are its input/output. The retrieval methods are config-driven Components: the engine
builds a ``Searcher`` from the spec — a ``RetrievalPipeline`` composing ``Retriever`` (dense / sparse),
``Fuser`` (identity / rrf), the optional ``Merger`` (auto-merging) and ``Reranker`` (cross-encoder), or a
``RoutingRetrievalPipeline`` that runs a ``QueryClassifier`` and dispatches to a per-type sub-pipeline.
These are the seams users plug into; the repository, embedder, and cross-encoder are internal.
"""

from tarnrag.contracts import MethodRef, RetrievalResult
from tarnrag.retrieval.classifier import (
    NoOpQueryClassifier,
    QueryClassifier,
    StructuralQueryClassifier,
)
from tarnrag.retrieval.engine import RetrievalEngine, RetrievalError
from tarnrag.retrieval.fuser import Fuser
from tarnrag.retrieval.merger import AutoMerger, Merger
from tarnrag.retrieval.pipeline import RetrievalPipeline
from tarnrag.retrieval.reranker import CrossEncoderReranker, Reranker
from tarnrag.retrieval.retriever import RetrievalContext, Retriever
from tarnrag.retrieval.router import RoutingRetrievalPipeline
from tarnrag.retrieval.searcher import Searcher
from tarnrag.retrieval.types import Query

__all__ = [
    "RetrievalEngine",
    "RetrievalError",
    "Query",
    "RetrievalResult",
    "MethodRef",
    "Searcher",
    "RetrievalPipeline",
    "RoutingRetrievalPipeline",
    "QueryClassifier",
    "NoOpQueryClassifier",
    "StructuralQueryClassifier",
    "Retriever",
    "Fuser",
    "Merger",
    "AutoMerger",
    "Reranker",
    "CrossEncoderReranker",
    "RetrievalContext",
]
