"""EnrichStage — the document-phase enrichment driver (FR-5).

Runs a configured, ordered list of ``Enricher`` components over ``item.document`` (after extraction,
before chunking), each appending typed annotations to the ``StructuredDocument``. The annotations flow
into chunks via the chunker (which aggregates element annotations into ``ChunkProvenance``). The
default is *no* enrichers — a passthrough; users configure their own (NER / topic / …) by spec, like
every other component (e.g. ``{"class_name": "Enrich", "enrichers": [{"class_name": "acronyms"}]}``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from pydantic import Field

from tarnrag.contracts import PipelineItem
from tarnrag.core.components import ComponentFactory
from tarnrag.ingestion.components.enrichment.enricher import Enricher
from tarnrag.ingestion.pipeline import PipelineStage


class EnrichStage(PipelineStage):
    """Run the configured enrichers over ``item.document`` in order — a doc-phase 1→1 transform."""

    class Config(PipelineStage.Config):
        class_name: Literal["Enrich"] = "Enrich"
        # ordered enricher component specs (class_name + options); the config-driven enrichment pipeline.
        enrichers: list[dict[str, Any]] = Field(default_factory=list)

    config: EnrichStage.Config

    def __init__(self, config: EnrichStage.Config) -> None:
        super().__init__(config)
        self._enrichers: list[Enricher] | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the enricher children through the framework factory (the container hook)."""
        self._enrichers = [factory.create_as(spec, Enricher) for spec in self.config.enrichers]

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        self._ensure_children()
        if item.document is not None:
            for enricher in self._enrichers:
                enricher.enrich(item.document)  # mutates the document in place
        yield item.derive()  # content/metadata unchanged; carries the (now-enriched) document forward
