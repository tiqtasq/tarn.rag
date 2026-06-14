from datetime import UTC, datetime

import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.jobs import Batch, IngestionJob


def _job(job_id: str, stage_name: str) -> IngestionJob:
    return IngestionJob(
        job_id=job_id,
        document_id="s1",
        item=PipelineItem(content="x", metadata={"source_id": "s1"}),
        stage_name=stage_name,
        created_at=datetime.now(UTC),
    )


def test_batch_derives_stage_from_homogeneous_jobs():
    batch = Batch([_job("j1", "Embed"), _job("j2", "Embed")])
    assert batch.stage_name == "Embed"
    assert [j.job_id for j in batch.jobs] == ["j1", "j2"]


def test_batch_rejects_mixed_stages():
    with pytest.raises(ValueError, match="share one stage_name"):
        Batch([_job("j1", "Chunk"), _job("j2", "Embed")])


def test_batch_rejects_empty():
    with pytest.raises(ValueError, match="at least one job"):
        Batch([])
