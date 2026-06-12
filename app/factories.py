"""Composition factories — build the pipeline (and re-export the sink registry) from
``Settings``. This is *wiring*, not configuration (which lives in ``app/core/config.py``):
the API DI and ``run_worker.py`` import these to assemble the orchestrator.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.domains.ingestion.pipeline import Pipeline
from app.domains.ingestion.result_sink import create_sink_registry  # re-exported
from app.domains.ingestion.stages.chunk import ChunkStage
from app.domains.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from app.domains.ingestion.stages.embed import EmbedStage
from app.domains.ingestion.stages.enrich import EnrichMetadataStage
from app.domains.ingestion.stages.load_parse import LoadAndParseStage

__all__ = ["create_ingestion_pipeline", "create_sink_registry"]


def create_ingestion_pipeline(settings: Settings | None = None) -> Pipeline:
    """Build the ordered ingestion pipeline, configured from ``Settings``. The orchestrator
    wraps ``pipeline.stages`` in a ``PipelineDAG``; sink keys must match each stage ``.name``
    (see ``create_sink_registry``)."""
    settings = settings or get_settings()
    return Pipeline(
        [
            LoadAndParseStage(),
            CleanAndNormalizeStage(),
            ChunkStage(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP),
            EnrichMetadataStage(),
            EmbedStage(
                model_dir=settings.MODEL_DIR,
                model_id=settings.EMBEDDING_MODEL,
                revision=settings.EMBEDDING_MODEL_REVISION,
                embedding_dim=settings.EMBEDDING_DIMENSION,
                max_length=settings.MAX_SEQ_LENGTH,
                query_prefix=settings.EMBEDDING_QUERY_PREFIX,
                passage_prefix=settings.EMBEDDING_PASSAGE_PREFIX,
                model_batch_size=settings.EMBEDDING_BATCH_SIZE,
            ),
        ]
    )
