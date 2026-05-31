"""EnrichMetadataStage — attach lightweight, content-derived metadata to each chunk."""

from __future__ import annotations

from typing import Any

from app.domains.ingestion.pipeline import MapperStage


class EnrichMetadataStage(MapperStage):
    """Attach cheap, content-derived metadata. Richer NLP enrichment (NER, noun
    phrases) is a future extension — see the metadata conventions in Core Models."""

    def __init__(self, **config: Any):
        super().__init__(name="EnrichMetadata", **config)

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return text, {"char_count": len(text), "word_count": len(text.split())}

    def validate(self) -> None:
        return None
