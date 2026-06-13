"""Retrieval domain — public surface.

``RetrievalEngine`` is the high-level facade (``RetrievalEngine.create()``); ``Query`` /
``RetrievalResult`` are its input/output. The index store and embedder are internal.
"""

from tarnrag.retrieval.engine import RetrievalEngine, RetrievalError
from tarnrag.retrieval.types import MethodRef, Query, RetrievalResult

__all__ = ["RetrievalEngine", "RetrievalError", "Query", "RetrievalResult", "MethodRef"]
