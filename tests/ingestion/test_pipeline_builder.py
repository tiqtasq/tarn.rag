"""The pipeline builder: a Pipeline is itself a Component spec (``Pipeline.from_spec``), and
``IngestionEngine.build_pipeline`` reads it from Settings.components (Settings fills the default)."""

from typing import Literal

import pytest

from tarnrag.core.components import Component
from tarnrag.core.components.registry import UnknownTagError
from tarnrag.core.config import (
    INGESTION_PIPELINE,
    ChunkingSettings,
    EmbeddingSettings,
    Settings,
)
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.pipeline import Pipeline
from tarnrag.ingestion.embed import EmbedStage


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # explicit config; ignore any ambient .env


def _with_pipeline(spec, **kwargs) -> Settings:
    """Settings carrying a pipeline spec under components[INGESTION_PIPELINE]."""
    return _settings(components={INGESTION_PIPELINE: spec}, **kwargs)


class _NotAStage(Component):  # a registered Component that is not a PipelineStage
    class Config(Component.Config):
        class_name: Literal["_not_a_stage"] = "_not_a_stage"


def test_default_pipeline_is_the_built_in_five_stages():
    pipe = IngestionEngine.build_pipeline(_settings())
    assert isinstance(pipe, Pipeline)
    assert [s.tag for s in pipe.stages] == [
        "LoadAndParse", "CleanAndNormalize", "Chunk", "EnrichMetadata", "Embed",
    ]


def test_default_pipeline_reads_chunking_from_settings():
    pipe = IngestionEngine.build_pipeline(_settings(chunking=ChunkingSettings(size=128, overlap=8)))
    chunk = next(s for s in pipe.stages if s.tag == "Chunk")
    assert (chunk.config.chunk_size, chunk.config.overlap) == (128, 8)


def test_components_spec_overrides_the_default_composition():
    spec = {
        "class_name": "pipeline",
        "stages": [
            {"class_name": "CleanAndNormalize"},
            {"class_name": "Chunk", "chunk_size": 64, "overlap": 8},
        ],
    }
    pipe = IngestionEngine.build_pipeline(_with_pipeline(spec))
    assert [s.tag for s in pipe.stages] == ["CleanAndNormalize", "Chunk"]
    assert pipe.stages[1].config.chunk_size == 64


def test_embedding_identity_comes_from_settings_not_the_spec():
    # A spec that spoofs a different embedding must be overridden by Settings, so the ingest-side
    # fingerprint can't drift from the retrieval-side embedder.
    spec = {
        "class_name": "pipeline",
        "stages": [{"class_name": "Embed", "embedding": {"model": "spoofed/model"}}],
    }
    settings = _with_pipeline(spec, embedding=EmbeddingSettings(model="real/model"))
    embed = IngestionEngine.build_pipeline(settings).stages[0]
    assert isinstance(embed, EmbedStage)
    assert embed.config.embedding.model == "real/model"


def test_settings_fills_the_default_pipeline_without_embedding():
    # Settings makes itself self-complete (the default pipeline is present); its Embed carries no
    # embedding identity — that's injected from Settings at build time.
    spec = _settings().components[INGESTION_PIPELINE]
    assert spec["class_name"] == "pipeline"
    embed = next(s for s in spec["stages"] if s["class_name"] == "Embed")
    assert "embedding" not in embed


def test_from_spec_rejects_a_non_pipeline_top_level_spec():
    with pytest.raises(TypeError, match="not a Pipeline"):
        Pipeline.from_spec({"class_name": "_not_a_stage"})


def test_from_spec_rejects_a_non_stage_child():
    with pytest.raises(TypeError, match="not a PipelineStage"):
        Pipeline.from_spec({"class_name": "pipeline", "stages": [{"class_name": "_not_a_stage"}]})


def test_from_spec_rejects_an_unknown_stage_tag():
    with pytest.raises(UnknownTagError):
        Pipeline.from_spec({"class_name": "pipeline", "stages": [{"class_name": "does-not-exist"}]})
