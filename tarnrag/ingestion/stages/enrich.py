"""EnrichMetadataStage — attach lightweight, content-derived metadata to each chunk."""

from __future__ import annotations

from typing import Any, Literal

from tarnrag.ingestion.pipeline import MapperStage


class EnrichMetadataStage(MapperStage):
    """
    Attach cheap, content-derived metadata. Richer NLP enrichment (NER, noun
    phrases) is a future extension — see the metadata conventions in Core Models.

    Note: the derived fields are not persisted yet — the downstream ChunkMetadataResultSink
    write is a deliberate no-op (§8 chunks have no metadata column).
    """

    class Config(MapperStage.Config):
        class_name: Literal["EnrichMetadata"] = "EnrichMetadata"

    config: EnrichMetadataStage.Config

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return text, {"char_count": len(text), "word_count": len(text.split())}
