"""Document status read model — composes the job_status projection with the data facts.

Public document status (`pending | in_progress | complete | failed`) needs two *different*
things: the operational **job_status** projection (did anything fail / is it known?) and the
persisted **data facts** (how many chunks/embeddings exist). Both now live on the repository (one
store), but the read model still depends on **two narrow ports** — a ``JobStatusSource`` and a
``DocumentFactsSource`` — not a fat "document storage" object (ISP — each writer/caller depends
only on what it uses, and the facts source can still be pointed elsewhere). Only this thin read
model composes the two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentFacts:
    """
    Persisted-data facts for one document: whether its row exists, and how many
    chunks/embeddings it has.
    """

    present: bool  # the document row exists in the data store
    chunk_count: int
    embedding_count: int


class DocumentFactsSource(ABC):
    """
    Port over the document data store (the repository): status facts, content-hash lookup, and
    the list/delete admin ops.
    """

    @abstractmethod
    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Persisted-data facts for a document (from the repository)."""

    @abstractmethod
    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Public document_ids whose stored ``content_hash`` matches — for content dedup
        (independent of the id policy). Empty if none; possibly several when identical content
        was ingested under different ids."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Remove a document and its derived data (chunks/embeddings/index rows). Returns True
        if a document existed, False if it was unknown."""

    @abstractmethod
    async def list_documents(self) -> list[dict[str, Any]]:
        """Inventory of stored documents — one dict per document with keys ``document_id``,
        ``content_hash``, ``chunk_count``, ``embedding_count``. Order is unspecified."""


class JobStatusSource(ABC):
    """
    Port over the job_status projection (lives on the repository) — read and delete a document's
    per-job rows.
    """

    @abstractmethod
    async def document_jobs(self, document_id: str) -> list[dict[str, Any]]:
        """The per-job rows for a document (the job_status projection)."""

    @abstractmethod
    async def delete_document_jobs(self, document_id: str) -> bool:
        """Remove a document's job_status rows (when deleting a document). Returns True if any
        rows were removed."""


class DocumentStatusReader:
    """
    Derives public status by composing a ``JobStatusSource`` with a ``DocumentFactsSource``.
    """

    def __init__(self, jobs: JobStatusSource, facts: DocumentFactsSource):
        self._jobs = jobs
        self._facts = facts

    async def document_status(
        self, document_id: str, verbose: bool = False
    ) -> dict[str, Any] | None:
        facts = await self._facts.document_facts(document_id)
        job_rows = await self._jobs.document_jobs(document_id)
        if not facts.present and not job_rows:
            return None  # unknown document
        states = {r["status"] for r in job_rows}
        out: dict[str, Any] = {
            "document_id": document_id,
            "status": self._rollup(facts, states),
            "chunk_count": facts.chunk_count,
            "embedding_count": facts.embedding_count,
        }
        if verbose:
            out["jobs"] = job_rows  # debug-only per-job breakdown
        return out

    @staticmethod
    def _rollup(facts: DocumentFacts, states: set[str]) -> str:
        if "failed" in states:
            return "failed"
        if not facts.present:
            return "pending"  # queued, no data persisted yet
        if facts.chunk_count and facts.embedding_count >= facts.chunk_count:
            return "complete"
        return "in_progress"
