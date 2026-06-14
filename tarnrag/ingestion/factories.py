"""Composition factories — build the pipeline (and re-export the sink registry) from
``Settings``. This is *wiring*, not configuration (which lives in ``tarnrag/core/config.py``):
the engines' ``create()`` and ``run_worker.py`` import these to assemble the orchestrator.
"""

from __future__ import annotations

from tarnrag.core.config import Settings, get_settings
from tarnrag.ingestion.pipeline import Pipeline
from tarnrag.ingestion.result_sink import create_sink_registry  # re-exported
from tarnrag.ingestion.stages.chunk import ChunkStage
from tarnrag.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.stages.embed import EmbedStage
from tarnrag.ingestion.stages.enrich import EnrichMetadataStage
from tarnrag.ingestion.stages.load_parse import LoadAndParseStage

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
            ChunkStage(chunk_size=settings.chunking.size, overlap=settings.chunking.overlap),
            EnrichMetadataStage(),
            EmbedStage(
                settings.embedding,
                settings.EMBEDDING_DIMENSION,
                model_batch_size=settings.embedding.batch_size,
            ),
        ]
    )
