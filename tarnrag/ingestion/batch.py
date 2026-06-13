"""The worker <-> orchestration handshake: a per-batch unit of work (D3/D5).

The worker is compute-only. For each dispatched batch it asks a ``BatchCoordinator``
to ``begin_batch`` (which records 'processing' and picks the per-stage sink), reports
produced results via ``submit``, and then either ``fail``s (compute error) or
``complete``s the unit of work. The worker depends ONLY on these two interfaces — not
on the orchestrator, repository, DAG, queue or ``ResultSink``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tarnrag.ingestion.models import Batch
from tarnrag.ingestion.pipeline import PipelineStage


class BatchContext(ABC):
    """A per-batch unit of work the worker reports to (created by a BatchCoordinator,
    which has already recorded 'processing')."""

    @abstractmethod
    def submit(self, results: list[Any]) -> None:
        """Hand a batch of produced results to the unit of work (buffer-only, sync)."""

    @abstractmethod
    async def fail(self, error: BaseException) -> None:
        """Record that compute failed for this batch (the worker then re-raises)."""

    @abstractmethod
    async def complete(self) -> None:
        """Finalize + persist, record 'completed', and enqueue the downstream jobs.
        RAISES on persistence failure so the queue requeues the job (recovery, D5)."""


class BatchCoordinator(ABC):
    """Provides the worker with the stage to run and starts the unit of work."""

    @abstractmethod
    def get_stage(self, stage_name: str) -> PipelineStage:
        """Return the long-lived, configured pure stage instance for ``stage_name``
        (from the DAG — reused across jobs, so e.g. EmbedStage's model loads once)."""

    @abstractmethod
    async def begin_batch(self, batch: Batch) -> BatchContext:
        """Record 'processing', pick the per-stage sink (by ``batch.stage_name``), and
        return the unit of work for this homogeneous batch."""
