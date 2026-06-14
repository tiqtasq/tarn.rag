"""EmbedStage — vectorize chunks into Embeddings (terminal stage).

Uses the shared ONNX ``OnnxEmbedder`` (passage side), so ingestion embeds with exactly the
pipeline retrieval will replay for queries (§5.3). The embedder is built from the same
``EmbeddingSettings`` slice via ``OnnxEmbedder.create`` and lazy-loaded on first use through
``_get_embedder``; tests inject a fake by overriding it (anything with
``embed_passages(list[str]) -> list[list[float]]``).
"""

from __future__ import annotations

from collections.abc import Iterator

from tarnrag.core.config import EmbeddingSettings
from tarnrag.contracts import Embedding, PipelineItem
from tarnrag.ingestion.pipeline import PipelineStage


class EmbedStage(PipelineStage):
    """
    Terminal stage: yields ``Embedding``s (not ``PipelineItem``s). ``process_batch`` groups
    items into ``model_batch_size``-sized embed calls; ``chunk_id`` comes from
    ``metadata['chunk_id']`` (set by ChunkResultSink).
    """

    def __init__(
        self,
        embedding: EmbeddingSettings | None = None,
        embedding_dimension: int = 384,
        *,
        model_batch_size: int = 32,
    ):
        # Set before super().__init__(), which runs validate().
        self.embedding = embedding or EmbeddingSettings()
        self.embedding_dimension = embedding_dimension
        self.model_batch_size = model_batch_size
        self._embedder = None  # lazy; built on first use (or injected by tests)
        super().__init__(name="Embed")

    def process(self, item: PipelineItem) -> Iterator[Embedding]:
        yield from self.process_batch([item])

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Embedding]:
        """
        Embed items in ``model_batch_size`` groups, yielding one ``Embedding`` per chunk
        (``chunk_id`` taken from ``metadata['chunk_id']``).
        """
        embedder = self._get_embedder()
        for i in range(0, len(items), self.model_batch_size):
            sub = items[i : i + self.model_batch_size]
            vectors = embedder.embed_passages([it.content for it in sub])
            for it, vec in zip(sub, vectors):
                yield Embedding(
                    chunk_id=it.metadata["chunk_id"],
                    vector=list(vec),
                    model=self.embedding.model,
                    dimension=len(vec),
                    metadata={"source_id": it.metadata.get("source_id")},
                )

    def _get_embedder(self):
        if self._embedder is None:
            from tarnrag.embedder import OnnxEmbedder

            self._embedder = OnnxEmbedder.create(self.embedding, self.embedding_dimension)
        return self._embedder

    def validate(self) -> None:
        if self.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive")
