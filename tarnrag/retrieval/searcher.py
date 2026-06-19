"""The Searcher seam — the one thing the engine builds from the ``RETRIEVAL_PIPELINE`` spec.

A ``Searcher`` is a config-driven ``Component`` that turns a query into ranked results over a
``RetrievalContext``. Both the plain ``RetrievalPipeline`` and the ``RoutingRetrievalPipeline`` implement
it, so the engine (and the eval sweep) build ``create_as(spec, Searcher)`` — and that one spec can name
*either* a single pipeline *or* a router, with no new ``Settings`` key. The router's routes are themselves
``Searcher``s, so routing composes (a route is just another pipeline spec, or a nested router).
"""

from __future__ import annotations

from abc import abstractmethod

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import Component
from tarnrag.retrieval.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


class Searcher(Component):
    """Port: turn a ``Query`` into ranked ``RetrievalResult``s over a ``RetrievalContext`` (the index +
    embedder + optional cross-encoder). The abstract base of ``RetrievalPipeline`` and the router."""

    class Config(Component.Config):
        """Base searcher config; concrete searchers pin ``class_name``."""

    @abstractmethod
    async def search(self, query: Query, ctx: RetrievalContext) -> list[RetrievalResult]:
        """The query's ranked results (best first), capped at ``query.top_k``."""
