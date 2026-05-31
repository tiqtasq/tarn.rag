"""Shared Pydantic models for ingestion and (future) retrieval.

Two distinct KINDS of model live here, deliberately kept separate even though some
fields coincide:

1. ``PipelineItem`` is the single, uniform TRANSPORT type the engine flows. Every
   stage is ``PipelineItem -> Iterator[PipelineItem]`` and the worker / DAG / inline
   job payload are all generic over this one type, so the engine stays stage- and
   entity-agnostic. Per-entity structure (doc id, chunk index, ...) rides in the
   loose ``metadata`` bag rather than typed fields.

2. ``Document`` / ``Chunk`` / ``Embedding`` are typed PERSISTENCE records (DTOs) at
   the repository boundary. They are materialised at the ResultSinks to write rows
   and reconstructed on reads; they NEVER flow through the pipeline. The persistent
   schema itself is the SQLAlchemy ``Table``s in ``DocumentRepository``, which maps
   each model to/from its table (``Document`` <-> ``documents``, etc.).

So ``Document`` is NOT a ``PipelineItem`` and does not inherit from it: the field
overlap is incidental (a document is just content + metadata), and it diverges for
``Chunk`` (parent_doc_id, chunk_index) and ``Embedding`` (no content at all) — which
is exactly why ONE uniform transport type is the right abstraction rather than
flowing the three persistence types. (Trade-off: a loosely-typed transport keeps the
engine generic and extensible; the alternative — typed flow, Document -> [Chunk] ->
Embedding — buys type safety at the cost of a non-generic engine.)
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineItem(BaseModel):
    """The single, uniform TRANSPORT type that flows through every stage.

    The engine (stages, worker, DAG, inline job payload) is generic over this one
    type; per-entity structure rides in ``metadata``. Distinct from — and not a base
    of — the persistence DTOs below, which never flow (see the module docstring for
    why the field overlap with Document is intentional).
    """

    id: str | None = None  # assigned by the storage layer
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Document(BaseModel):
    """A source document (persistence DTO; built at a sink, never flows). Mapped
    to/from the ``documents`` table by the repository."""

    id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Chunk(BaseModel):
    """A text chunk (persistence DTO) extracted from a document. Mapped to/from the
    ``chunks`` table."""

    id: str | None = None
    parent_doc_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Embedding(BaseModel):
    """A chunk's dense vector (persistence DTO). Mapped to/from the ``embeddings`` table."""

    id: str | None = None
    chunk_id: str
    vector: list[float]
    model: str
    dimension: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
