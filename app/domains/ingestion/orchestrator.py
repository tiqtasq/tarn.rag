"""Pipeline DAG + orchestrator — walks the DAG and owns the job lifecycle (D3/D5).

The orchestrator is the ``BatchCoordinator``: it enqueues root jobs, and for each
dispatched batch it hands the worker a ``BatchContext`` that owns persistence + status
+ DAG advance. The worker reports to that context and never holds the orchestrator.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.exceptions import IngestionError
from app.domains.base.chunk_store import ChunkStore
from app.domains.base.models import PipelineItem
from app.domains.base.repository import DocumentRepository
from app.domains.ingestion.batch import BatchContext, BatchCoordinator
from app.domains.ingestion.models import Batch, IngestionJob
from app.domains.ingestion.queue import JobEnqueuer
from app.domains.ingestion.result_sink import ResultSink

logger = logging.getLogger(__name__)


class PipelineDAG:
    """The pipeline as a directed acyclic graph. Sequential edges for now:
    ``stage[i] -> stage[i+1]``."""

    def __init__(self, stages: list):
        self.stages = stages
        self.edges = [(stages[i].name, stages[i + 1].name) for i in range(len(stages) - 1)]

    def get_downstream_stages(self, stage_name: str) -> list[str]:
        """Stage names that depend on this stage (empty for the terminal stage)."""
        return [target for source, target in self.edges if source == stage_name]

    def get_stage(self, stage_name: str):
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        raise ValueError(f"Stage {stage_name} not found")


class PipelineOrchestrator(BatchCoordinator):
    """Walks the DAG and owns the JOB LIFECYCLE (D3/D5).

    The queue owns claiming/retries/acking; the orchestrator enqueues jobs through
    it, and per dispatched batch it issues a ``BatchContext`` (via ``begin_batch``) that
    finalizes the sink, records status, and enqueues downstream jobs — or raises so the
    queue requeues (recovery). Also keeps the document-keyed ``job_status`` projection.
    """

    def __init__(
        self,
        dag: PipelineDAG,
        enqueuer: JobEnqueuer,
        repository: DocumentRepository,
        sink_registry: dict[str, type[ResultSink]],
        observability: Any = None,  # lifecycle metrics (enqueue/status/persistence)
        chunk_store: ChunkStore | None = None,  # sink target; defaults to the repository
    ):
        self.dag = dag
        self.enqueuer = enqueuer
        self.repository = repository  # job_status projection (record_job)
        self.obs = observability
        self.sink_registry = sink_registry
        # Where sinks persist document/chunk/embedding data. Defaults to the repository
        # (unchanged Postgres/SQLite path); the §8 index build passes a SqliteIndexStore.
        self.chunk_store = chunk_store or repository

    async def ingest_documents(self, items: list[PipelineItem]) -> list[str]:
        """Enqueue one root job per document for the first stage. Returns the public
        document_ids (== source_ids); job ids never leave the system."""
        first_stage = self.dag.stages[0].name
        document_ids = []
        for item in items:
            job = self._make_job(item, first_stage)
            await self._enqueue(job)
            document_ids.append(job.document_id)
        if self.obs:
            self.obs.counter("ingest.documents", len(items))
        return document_ids

    def get_stage(self, stage_name: str):
        """The long-lived, configured stage instance from the DAG (reused across jobs)."""
        return self.dag.get_stage(stage_name)

    async def begin_batch(self, batch: Batch) -> BatchContext:
        """Start a unit of work for a dispatched batch: record 'processing', pick the
        per-stage sink, and return the context the worker reports to."""
        await self._record(batch.jobs, "processing")
        sink = self.sink_registry[batch.stage_name](self.chunk_store)
        return _OrchestratorBatchContext(self, batch, sink)

    # ----- internals shared by ingest_documents and the BatchContext -----

    async def _record(
        self, jobs: list[IngestionJob], status: str, error: str | None = None
    ) -> None:
        for job in jobs:
            await self.repository.record_job(
                job.document_id, job.job_id, job.stage_name, status, error
            )

    def _make_job(self, item: PipelineItem, stage_name: str) -> IngestionJob:
        return IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=item.metadata["source_id"],  # public handle; threads via metadata
            item=item,
            stage_name=stage_name,
            created_at=datetime.now(UTC),
        )

    async def _enqueue(self, job: IngestionJob) -> None:
        await self.repository.record_job(
            job.document_id, job.job_id, job.stage_name, "queued"
        )
        await self.enqueuer.enqueue(job)
        if self.obs:
            self.obs.counter("ingest.jobs_enqueued")


class _OrchestratorBatchContext(BatchContext):
    """Per-batch unit of work backed by the orchestrator's repo / DAG / queue. It
    composes the per-stage ResultSink (kept pure) and adds status + DAG advance."""

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        batch: Batch,
        sink: ResultSink,
    ):
        self._orch = orchestrator
        self._batch = batch
        self._sink = sink
        self._produced: list[Any] = []

    def submit(self, results: list[Any]) -> None:
        self._produced.extend(results)
        self._sink.submit(results)

    async def fail(self, error: BaseException) -> None:
        await self._orch._record(self._batch.jobs, "failed", str(error))
        if self._orch.obs:
            self._orch.obs.counter(f"stage.{self._batch.stage_name}.failed")

    async def complete(self) -> None:
        obs = self._orch.obs
        self._sink.close()
        outcome = await self._sink.finalize()
        if not outcome.persisted:
            await self._orch._record(self._batch.jobs, "failed", f"persistence: {outcome.detail}")
            if obs:
                obs.counter(f"stage.{self._batch.stage_name}.persist_failed")
                await obs.log(
                    "error", "persistence failed",
                    stage=self._batch.stage_name, detail=outcome.detail,
                )
            raise IngestionError(
                f"persistence failed for {self._batch.stage_name}: {outcome.detail}"
            )
        await self._orch._record(self._batch.jobs, "completed")
        if obs:
            obs.counter(f"stage.{self._batch.stage_name}.completed")
        # Fan-out (D1): one downstream job per produced item, per next stage.
        for next_stage in self._orch.dag.get_downstream_stages(self._batch.stage_name):
            for item in self._produced:
                await self._orch._enqueue(self._orch._make_job(item, next_stage))
