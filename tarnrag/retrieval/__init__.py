"""Retrieval domain — public surface.

``RetrievalEngine`` is the high-level facade (``RetrievalEngine.create()``); ``Query`` /
``RetrievalResult`` are its input/output. The retrieval methods are config-driven Components composed by
a ``RetrievalPipeline``: ``Retriever`` (dense / sparse), ``Fuser`` (identity / rrf), and the optional
``Merger`` (auto-merging) are the seams users plug into. The repository and embedder are internal.
"""

from tarnrag.contracts import MethodRef, RetrievalResult
from tarnrag.retrieval.engine import RetrievalEngine, RetrievalError
from tarnrag.retrieval.fuser import Fuser
from tarnrag.retrieval.merger import AutoMerger, Merger
from tarnrag.retrieval.pipeline import RetrievalPipeline
from tarnrag.retrieval.retriever import RetrievalContext, Retriever
from tarnrag.retrieval.types import Query

__all__ = [
    "RetrievalEngine",
    "RetrievalError",
    "Query",
    "RetrievalResult",
    "MethodRef",
    "RetrievalPipeline",
    "Retriever",
    "Fuser",
    "Merger",
    "AutoMerger",
    "RetrievalContext",
]
