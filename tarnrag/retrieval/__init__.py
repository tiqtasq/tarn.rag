"""Retrieval domain — public surface.

``RetrievalEngine`` is the high-level facade (``RetrievalEngine.create()``); ``Query`` /
``RetrievalResult`` are its input/output. The retrieval methods are config-driven Components: the engine
builds a ``Searcher`` from the spec — a ``RetrievalPipeline`` composing ``Retriever`` (dense / sparse),
``Fuser`` (identity / rrf), the optional ``Merger`` (auto-merging) and ``Reranker`` (cross-encoder), or a
``RoutingRetrievalPipeline`` that runs a ``QueryClassifier`` and dispatches to a per-type sub-pipeline.
These are the seams users plug into; the repository, embedder, and cross-encoder are internal.
"""

from tarnrag.contracts import MethodRef, RetrievalResult
from tarnrag.retrieval.components.classifier import (
    GenericQueryClassifier,
    QueryClassifier,
    StructuralQueryClassifier,
)
from tarnrag.retrieval.engine.engine import RetrievalEngine, RetrievalError
from tarnrag.retrieval.components.fuser import Fuser
from tarnrag.retrieval.components.merger import AutoMerger, Merger
from tarnrag.retrieval.pipeline.pipeline import RetrievalPipeline
from tarnrag.retrieval.engine.retrieval_engine_protocol import RetrievalEngineProtocol
from tarnrag.retrieval.components.reranker import CrossEncoderReranker, Reranker
from tarnrag.retrieval.components.retriever import RetrievalContext, Retriever
from tarnrag.retrieval.pipeline.router import RoutingRetrievalPipeline
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query

__all__ = [
    "RetrievalEngine",
    "RetrievalError",
    "RetrievalEngineProtocol",
    "Query",
    "RetrievalResult",
    "MethodRef",
    "Searcher",
    "RetrievalPipeline",
    "RoutingRetrievalPipeline",
    "QueryClassifier",
    "GenericQueryClassifier",
    "StructuralQueryClassifier",
    "Retriever",
    "Fuser",
    "Merger",
    "AutoMerger",
    "Reranker",
    "CrossEncoderReranker",
    "RetrievalContext",
]
