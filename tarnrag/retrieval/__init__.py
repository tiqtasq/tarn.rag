"""Retrieval domain — public surface.

``RetrievalEngine`` is the high-level facade (``RetrievalEngine.create()``); ``Query`` /
``RetrievalResult`` are its input/output. The repository and embedder are internal.
"""

from tarnrag.contracts import MethodRef, RetrievalResult
from tarnrag.retrieval.engine import RetrievalEngine, RetrievalError
from tarnrag.retrieval.types import Query

__all__ = ["RetrievalEngine", "RetrievalError", "Query", "RetrievalResult", "MethodRef"]
