"""The retrieval port — the seam generation depends on (and the C++-port hinge).

Generation never imports the concrete ``RetrievalEngine``; it depends on this ``Protocol``. Today the
Python ``RetrievalEngine`` *structurally* satisfies it (its ``search`` already has this shape — zero new
code); later a thin ``C++RetrievalAdapter(RetrievalEngineProtocol)`` wraps the binding. The data crossing
this seam — ``Query`` in, ``RetrievalResult`` out — is the stable, versioned, serializable cross-language
schema, so the binding is a later *adapter*, not a redesign. Retrieval owns the port (it lives here, not
in ``generation/``), keeping the dependency one-way: ``generation → retrieval``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tarnrag.contracts import RetrievalResult
from tarnrag.retrieval.types import Query


@runtime_checkable
class RetrievalEngineProtocol(Protocol):
    """A query → ranked, provenance-bearing results. ``RetrievalEngine`` satisfies it structurally."""

    async def search(self, query: Query) -> list[RetrievalResult]: ...
