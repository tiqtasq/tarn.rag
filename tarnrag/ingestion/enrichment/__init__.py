"""Enrichment: annotate a loaded document with extra metadata (NER, topic, language, …).

Importing this package registers the built-in enrich stage with the global ``ComponentFactory`` (it
self-registers under its ``class_name``), so ``Pipeline.from_spec`` / ``ComponentFactory`` can build it.
User-supplied enrichers (an ``Enricher`` component family run alongside the extractor) will land here.
"""

from tarnrag.ingestion.enrichment.enrich import EnrichMetadataStage

__all__ = ["EnrichMetadataStage"]
