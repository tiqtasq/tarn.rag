"""Document status read model — composes the two stores without merging them.

Public document status (`pending | in_progress | complete | failed`) needs two *different*
things: the operational **job_status** projection (did anything fail / is it known?) and the
persisted **data facts** (how many chunks/embeddings exist). Those live in separate stores —
``DocumentRepository`` (job_status) and the §8 ``SqliteIndexStore`` (retrieval data) — and we
keep them separate (the index is a portable product artifact; job_status is operational
exhaust). Only this thin read model sees both; it's injected with two narrow ports, not a fat
"document storage" object (ISP — each writer/caller still depends only on what it uses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentFacts:
    present: bool  # the document row exists in the data store
    chunk_count: int
    embedding_count: int


class DocumentFactsSource(ABC):
    @abstractmethod
    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Persisted-data facts for a document (the repo in the classic path, the §8 index
        in retrieval mode)."""

    @abstractmethod
    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Public document_ids whose stored ``content_hash`` matches — for content dedup
        (independent of the id policy). Empty if none; possibly several when identical content
        was ingested under different ids."""


class JobStatusSource(ABC):
    @abstractmethod
    async def document_jobs(self, document_id: str) -> list[dict[str, Any]]:
        """The per-job rows for a document (the job_status projection)."""


class DocumentStatusReader:
    """Derives public status by composing a ``JobStatusSource`` with a ``DocumentFactsSource``."""

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
