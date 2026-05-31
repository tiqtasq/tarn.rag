"""Shared data models flowing through ingestion and (future) retrieval."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineItem(BaseModel):
    """A document or chunk flowing through the ingestion pipeline."""

    id: str | None = None  # assigned by the storage layer
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Document(BaseModel):
    """A source document in persistent storage."""

    id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Chunk(BaseModel):
    """A text chunk extracted from a document."""

    id: str | None = None
    parent_doc_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Embedding(BaseModel):
    """A dense vector embedding for a chunk."""

    id: str | None = None
    chunk_id: str
    vector: list[float]
    model: str
    dimension: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
