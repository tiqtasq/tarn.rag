"""EmbedStage — vectorize chunks into Embeddings (terminal stage)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.domains.base.models import Embedding, PipelineItem
from app.domains.ingestion.pipeline import PipelineStage


class EmbedStage(PipelineStage):
    """Real embedding compute lives here. ``process_batch`` groups items into
    ``model_batch_size``-sized ``encode`` calls; ``chunk_id`` comes from
    ``metadata['chunk_id']`` (set by ChunkResultSink). Terminal stage: yields
    ``Embedding``s (not ``PipelineItem``s), so nothing runs downstream.

    The model is lazy-loaded (sentence-transformers); tests may inject a fake by
    setting ``stage._model`` to anything with ``encode(list[str]) -> list[vector]``.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-minilm-l6-v2",
        model_batch_size: int = 32,
        **config: Any,
    ):
        # Set before super().__init__(), which runs validate().
        self.embedding_model = embedding_model
        self.model_batch_size = model_batch_size
        self._model = None  # lazy; not part of ``config`` (not serializable)
        super().__init__(
            name="Embed",
            embedding_model=embedding_model,
            model_batch_size=model_batch_size,
            **config,
        )

    def process(self, item: PipelineItem) -> Iterator[Embedding]:
        yield from self.process_batch([item])

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Embedding]:
        model = self._get_model()
        for i in range(0, len(items), self.model_batch_size):
            sub = items[i : i + self.model_batch_size]
            vectors = model.encode([it.content for it in sub], convert_to_tensor=False)
            for it, vec in zip(sub, vectors):
                vector = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                yield Embedding(
                    chunk_id=it.metadata["chunk_id"],
                    vector=vector,
                    model=self.embedding_model,
                    dimension=len(vector),
                    metadata={"source_id": it.metadata.get("source_id")},
                )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def validate(self) -> None:
        if self.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive")
