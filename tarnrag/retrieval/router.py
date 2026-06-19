"""The RoutingRetrievalPipeline — classify a query, dispatch to the per-type-best sub-pipeline.

A ``RoutingRetrievalPipeline`` is a ``Searcher`` (so the engine builds it from the ``RETRIEVAL_PIPELINE``
spec in place of a plain pipeline). It holds a ``QueryClassifier`` + a ``{query_type → Searcher}`` route
map + a ``default`` ``Searcher``. On ``search``: if the query isn't already classified, run the classifier
to populate ``query.query_type`` (+ annotations); then dispatch to ``routes.get(query_type, default)``.
Routes are themselves ``Searcher``s (built recursively), so a route is just another pipeline spec — sparse
for ``lexical``, dense for ``semantic``, a nested router, … — and the route map is the deployment's config,
tuned from the segmented eval.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import ComponentFactory
from tarnrag.retrieval.classifier import QueryClassifier
from tarnrag.retrieval.retriever import RetrievalContext
from tarnrag.retrieval.searcher import Searcher
from tarnrag.retrieval.types import Query

_DEFAULT_SEARCHER: dict[str, Any] = {"class_name": "retrieval_pipeline"}  # dense + identity fuser


class RoutingRetrievalPipeline(Searcher):
    """Classify a query and dispatch it to the route matching its ``query_type`` (else the default)."""

    class Config(Searcher.Config):
        class_name: Literal["routing_retrieval_pipeline"] = "routing_retrieval_pipeline"
        classifier: dict[str, Any] = Field(default_factory=lambda: {"class_name": "generic"})
        routes: dict[str, dict[str, Any]] = Field(default_factory=dict)  # query_type → Searcher spec
        default: dict[str, Any] = Field(default_factory=lambda: dict(_DEFAULT_SEARCHER))

    config: RoutingRetrievalPipeline.Config

    def __init__(self, config: RoutingRetrievalPipeline.Config) -> None:
        super().__init__(config)
        self._classifier: QueryClassifier | None = None
        self._routes: dict[str, Searcher] = {}
        self._default: Searcher | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the classifier + each route + the default through the factory (routes are ``Searcher``s,
        so routing composes — a route may be a plain pipeline or a nested router)."""
        self._classifier = factory.create_as(self.config.classifier, QueryClassifier)
        self._routes = {t: factory.create_as(spec, Searcher) for t, spec in self.config.routes.items()}
        self._default = factory.create_as(self.config.default, Searcher)

    async def search(self, query: Query, ctx: RetrievalContext) -> list[RetrievalResult]:
        self._ensure_children()
        if not query.query_type:  # a caller-supplied query_type wins; otherwise classify
            self._classifier.classify(query, ctx)
        searcher = self._routes.get(query.query_type, self._default)
        return await searcher.search(query, ctx)
