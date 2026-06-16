"""EmbedStage — vectorize chunks into Embeddings (terminal stage).

Uses the shared ONNX ``OnnxEmbedder`` (passage side), so ingestion embeds with exactly the
pipeline retrieval will replay for queries (§5.3). The embedding identity lives on the stage's
``Config`` as an ``EmbeddingSettings`` (the factory copies ``settings.embedding`` in, so it stays
in sync with the retrieval-side embedder and its fingerprint). The embedder is built lazily via
``OnnxEmbedder.create``; tests inject a fake by setting ``stage._embedder`` (anything with
``embed_passages(list[str]) -> list[list[float]]``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from pydantic import Field

from tarnrag.core.config import EmbeddingSettings
from tarnrag.contracts import Embedding, PipelineItem
from tarnrag.ingestion.pipeline import PipelineStage


class EmbedStage(PipelineStage):
    """
    Terminal stage: yields ``Embedding``s (not ``PipelineItem``s). ``process_batch`` groups items
    into ``embedding.batch_size``-sized embed calls; ``chunk_id`` comes from
    ``metadata['chunk_id']`` (set by ChunkResultSink).
    """

    class Config(PipelineStage.Config):
        class_name: Literal["Embed"] = "Embed"
        embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
        embedding_dimension: int = 384

    config: EmbedStage.Config

    def __init__(self, config: EmbedStage.Config) -> None:
        super().__init__(config)
        self._embedder = None  # lazy; built on first use (or injected by tests)

    def process(self, item: PipelineItem) -> Iterator[Embedding]:
        yield from self.process_batch([item])

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Embedding]:
        """
        Embed items in ``embedding.batch_size`` groups, yielding one ``Embedding`` per chunk
        (``chunk_id`` taken from ``metadata['chunk_id']``).
        """
        embedder = self._get_embedder()
        batch_size = self.config.embedding.batch_size
        for i in range(0, len(items), batch_size):
            sub = items[i : i + batch_size]
            vectors = embedder.embed_passages([it.content for it in sub])
            for it, vec in zip(sub, vectors):
                yield Embedding(
                    chunk_id=it.metadata["chunk_id"],
                    vector=list(vec),
                    model=self.config.embedding.model,
                    dimension=len(vec),
                    metadata={"source_id": it.metadata.get("source_id")},
                )

    def _get_embedder(self):
        if self._embedder is None:
            from tarnrag.core.embedder import OnnxEmbedder

            self._embedder = OnnxEmbedder.create(
                self.config.embedding, self.config.embedding_dimension
            )
        return self._embedder
