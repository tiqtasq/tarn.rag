"""EmbedStage — vectorize chunks into Embeddings (terminal stage).

Uses the shared ONNX ``OnnxEmbedder`` (passage side), so ingestion embeds with exactly the
pipeline retrieval will replay for queries (§5.3). The embedder is lazy-loaded via
``_get_embedder``; tests inject a fake by overriding it (anything with
``embed_passages(list[str]) -> list[list[float]]``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.domains.base.models import Embedding, PipelineItem
from app.domains.ingestion.pipeline import PipelineStage


class EmbedStage(PipelineStage):
    """Terminal stage: yields ``Embedding``s (not ``PipelineItem``s). ``process_batch`` groups
    items into ``model_batch_size``-sized embed calls; ``chunk_id`` comes from
    ``metadata['chunk_id']`` (set by ChunkResultSink)."""

    def __init__(
        self,
        *,
        model_dir: str = "",
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: str = "",
        embedding_dim: int = 384,
        max_length: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
        model_batch_size: int = 32,
        **config: Any,
    ):
        # Set before super().__init__(), which runs validate().
        self.model_dir = model_dir
        self.model_id = model_id
        self.revision = revision
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model_batch_size = model_batch_size
        self._embedder = None  # lazy; not serializable config
        super().__init__(
            name="Embed", model_id=model_id, model_batch_size=model_batch_size, **config
        )

    def process(self, item: PipelineItem) -> Iterator[Embedding]:
        yield from self.process_batch([item])

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Embedding]:
        embedder = self._get_embedder()
        for i in range(0, len(items), self.model_batch_size):
            sub = items[i : i + self.model_batch_size]
            vectors = embedder.embed_passages([it.content for it in sub])
            for it, vec in zip(sub, vectors):
                yield Embedding(
                    chunk_id=it.metadata["chunk_id"],
                    vector=list(vec),
                    model=self.model_id,
                    dimension=len(vec),
                    metadata={"source_id": it.metadata.get("source_id")},
                )

    def _get_embedder(self):
        if self._embedder is None:
            from app.domains.base.embedder import OnnxEmbedder

            self._embedder = OnnxEmbedder(
                self.model_dir,
                model_id=self.model_id,
                revision=self.revision,
                embedding_dim=self.embedding_dim,
                max_length=self.max_length,
                query_prefix=self.query_prefix,
                passage_prefix=self.passage_prefix,
            )
        return self._embedder

    def validate(self) -> None:
        if self.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive")
