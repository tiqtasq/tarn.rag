"""Shared Pydantic models for ingestion and (future) retrieval.

These four models are TWO FORMS of the things being ingested — not four unrelated
types, and not an inheritance hierarchy:

* IN-FLIGHT form -> ``PipelineItem``. The single, uniform type that flows between
  stages. A *document* (at Load/Clean) and a *chunk* (at Chunk/Enrich/Embed) are
  BOTH just ``PipelineItem``s; which one it is rides in ``metadata`` (e.g.
  ``chunk_index``), not in the type. The engine (stages, worker, DAG, inline job
  payload) is generic over this one type, so it stays stage- and entity-agnostic.

* AT-REST form -> ``Document`` / ``Chunk`` / ``Embedding``. Typed persistence records
  (DTOs) materialised at the ResultSinks to write rows (and reconstructed on reads).
  They NEVER flow; the persistent schema is the SQLAlchemy ``Table``s in
  ``DocumentRepository``, which maps each DTO to/from its table.

Worked trace of one ingest — watch the form change:

    after Load    PipelineItem(content="<doc text>", metadata={source_id, doc_id})
    after Chunk   PipelineItem(content="<chunk 3>",  metadata={..., chunk_index: 3})
                      |  ChunkResultSink persists it ->
                      v
                  Chunk(id="c3", parent_doc_id="d1", content="<chunk 3>", chunk_index=3)
    after Embed   Embedding(chunk_id="c3", vector=[...])   # terminal: never flows

So a chunk-in-flight IS a ``PipelineItem``; ``Chunk`` is its stored row. ``Embedding``
is the exception — Embed is the terminal stage, so an embedding is only ever at-rest.

Why two forms, not one: the in-flight form is deliberately loose (one uniform type +
a metadata bag) so the engine is generic and extensible; the at-rest form is typed so
the storage contract is explicit and validated (``store_chunks`` guarantees real
``parent_doc_id`` / ``chunk_index`` fields, not "hopefully in a dict"). The field
overlap (id/content/metadata) is what makes them look alike, but it is coincidental
and diverges (``Chunk`` adds FKs, ``Embedding`` has no content) — hence no shared base
or inheritance. The relations that DO hold are composition (``Chunk.parent_doc_id`` ->
``Document``, ``Embedding.chunk_id`` -> ``Chunk``) and transformation (a sink maps a
``PipelineItem`` into a DTO) — expressed by fields and functions, not a hierarchy.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineItem(BaseModel):
    """
    The IN-FLIGHT form — the single, uniform type that flows through every stage.

    A document or a chunk *in flight* is a PipelineItem (which one rides in
    ``metadata``). The engine (stages, worker, DAG, inline job payload) is generic
    over this one type. Distinct from — and not a base of — the at-rest DTOs below,
    which never flow (see the module docstring for the in-flight/at-rest framing).
    """

    id: str | None = None  # assigned by the storage layer
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Document(BaseModel):
    """
    A source document — AT-REST form (persistence DTO; built at a sink, never
    flows). Mapped to/from the ``documents`` table by the repository.
    """

    id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Chunk(BaseModel):
    """
    A text chunk — at-rest form (persistence DTO) of a chunk that flowed as a
    PipelineItem. Mapped to/from the ``chunks`` table.
    """

    id: str | None = None
    parent_doc_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Embedding(BaseModel):
    """
    A chunk's dense vector — at-rest ONLY (terminal stage output; never flows).
    Mapped to/from the ``embeddings`` table.
    """

    id: str | None = None
    chunk_id: str
    vector: list[float]
    model: str
    dimension: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
