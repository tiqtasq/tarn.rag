"""Ingestion job model — internal unit of work, never exposed to API clients."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domains.base.models import PipelineItem


class IngestionJob(BaseModel):
    """Internal unit of work enqueued in the JobQueue (pgQueuer in prod).

    Runtime state (queued/processing/.../retries) is owned by the queue; for the
    status API a document-keyed ``job_status`` projection is kept in the repository.
    """

    job_id: str  # logical id (used by the status projection)
    document_id: str  # public document handle (== source_id)
    item: PipelineItem  # in-flight item, INLINE in the payload (D1)
    stage_name: str  # stage to execute
    stage_config: dict[str, Any]  # stage config (to reconstruct the stage)
    created_at: datetime

    model_config = ConfigDict(arbitrary_types_allowed=True)
