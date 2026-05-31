"""Shared data models flowing through ingestion and (future) retrieval.

These are Pydantic **domain/transfer objects** used at the repository boundary —
in-memory representations, not the stored rows. The persistent schema is the
SQLAlchemy ``Table`` definitions in ``DocumentRepository``, which maps each model
to/from its table (``Document`` <-> ``documents``, etc.).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineItem(BaseModel):
    """A document or chunk flowing through the ingestion pipeline."""

    id: str | None = None  # assigned by the storage layer
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Document(BaseModel):
    """A source document. Mapped to/from the ``documents`` table by the repository."""

    id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Chunk(BaseModel):
    """A text chunk extracted from a document. Mapped to/from the ``chunks`` table."""

    id: str | None = None
    parent_doc_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Embedding(BaseModel):
    """A chunk's dense vector. Mapped to/from the ``embeddings`` table."""

    id: str | None = None
    chunk_id: str
    vector: list[float]
    model: str
    dimension: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
