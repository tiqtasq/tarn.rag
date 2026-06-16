"""Built-in pipeline stages.

Importing this package registers all five stages with the global ``ComponentFactory`` — each stage
self-registers under its ``class_name`` tag on definition, so ``Pipeline.from_spec`` /
``ComponentFactory`` can build them from a spec.
"""

from tarnrag.ingestion.stages.chunk import ChunkStage
from tarnrag.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.stages.embed import EmbedStage
from tarnrag.ingestion.stages.enrich import EnrichMetadataStage
from tarnrag.ingestion.stages.load_parse import LoadAndParseStage

__all__ = [
    "ChunkStage",
    "CleanAndNormalizeStage",
    "EmbedStage",
    "EnrichMetadataStage",
    "LoadAndParseStage",
]
