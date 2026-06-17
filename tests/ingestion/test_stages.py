import pytest
from pydantic import ValidationError

from tarnrag.contracts import PipelineItem
from tarnrag.core.components import ComponentFactory
from tarnrag.core.config import EmbeddingSettings
from tarnrag.ingestion.pipeline import Pipeline
from tarnrag.ingestion.chunking.chunk import ChunkStage
from tarnrag.ingestion.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.embed import EmbedStage
from tarnrag.ingestion.enrichment.enrich import EnrichMetadataStage
from tarnrag.ingestion.extraction.load_parse import LoadAndParseStage


def _item(content, **meta):
    return PipelineItem(content=content, metadata=meta)


def test_load_from_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello from a file")
    out = list(LoadAndParseStage(LoadAndParseStage.Config()).process(_item("", source_path=str(p))))
    assert len(out) == 1
    assert out[0].content == "hello from a file"
    assert out[0].metadata["doc_id"] and out[0].metadata["loaded"] is True


def test_load_passthrough_content():
    out = list(LoadAndParseStage(LoadAndParseStage.Config()).process(_item("already loaded")))
    assert out[0].content == "already loaded"


def test_clean_normalizes_whitespace_and_controls():
    stage = CleanAndNormalizeStage(CleanAndNormalizeStage.Config())
    out = list(stage.process(_item("  a\x07b   c\t\td  ")))
    assert out[0].content == "ab c d"
    assert out[0].metadata["cleaned"] is True


def _recursive(size, overlap):
    """A Chunk stage driving the recursive chunker (the structure-agnostic, size-based strategy)."""
    return ChunkStage(ChunkStage.Config(chunker={"class_name": "recursive", "chunk_size": size, "overlap": overlap}))


def test_chunk_splits_and_indexes():
    stage = _recursive(20, 4)
    text = "abcdefghij " * 8  # ~88 chars -> several chunks
    out = list(stage.process(_item(text, doc_id="d1")))
    assert len(out) > 1
    total = out[0].metadata["total_chunks"]
    assert total == len(out)
    assert [o.metadata["chunk_index"] for o in out] == list(range(total))
    assert all(o.provenance is not None and o.provenance.content_hash for o in out)  # each chunk carries provenance


def test_chunk_validation():
    # The size/overlap validation lives on the recursive chunker's Config (pydantic).
    from tarnrag.ingestion.chunking.recursive import RecursiveCharacterChunker

    with pytest.raises(ValidationError):
        RecursiveCharacterChunker.Config(chunk_size=0)
    with pytest.raises(ValidationError):
        RecursiveCharacterChunker.Config(chunk_size=10, overlap=10)


def test_enrich_counts():
    out = list(EnrichMetadataStage(EnrichMetadataStage.Config()).process(_item("one two three")))
    assert out[0].metadata["word_count"] == 3
    assert out[0].metadata["char_count"] == len("one two three")


class _FakeEmbedder:
    """Deterministic 3-d embedder that records its batch calls."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_passages(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_embed_produces_embeddings_with_model_batching():
    stage = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(batch_size=2)))
    stage._embedder = _FakeEmbedder()
    items = [
        _item(f"chunk {i}", chunk_id=f"c{i}", source_id="s1") for i in range(3)
    ]
    embs = list(stage.process_batch(items))
    assert [e.chunk_id for e in embs] == ["c0", "c1", "c2"]
    assert all(e.dimension == 3 and e.model.endswith("all-MiniLM-L6-v2") for e in embs)
    # batch_size=2 over 3 items -> two embed calls (two-tier batching)
    assert len(stage._embedder.calls) == 2


def test_stage_built_from_dict_spec_via_component_factory():
    """The migration's goal: a stage (and its chunker child) instantiated from a plain dict spec."""
    chunk = ComponentFactory.get().create(
        {"class_name": "Chunk", "chunker": {"class_name": "recursive", "chunk_size": 16, "overlap": 4}}
    )
    assert isinstance(chunk, ChunkStage)
    assert (chunk._chunker.config.chunk_size, chunk._chunker.config.overlap) == (16, 4)  # child built by the factory
    assert chunk.tag == "Chunk"  # the type tag (the sink/metrics key)
    assert chunk.name.startswith("Chunk-")  # unnamed instance -> counter-suffixed unique id
    assert chunk.to_json()["class_name"] == "Chunk"  # round-trips back to a spec


def test_pipeline_composition_doc_to_chunks():
    pipe = Pipeline.from_stages(
        [
            LoadAndParseStage(LoadAndParseStage.Config()),
            CleanAndNormalizeStage(CleanAndNormalizeStage.Config()),
            _recursive(20, 4),
            EnrichMetadataStage(EnrichMetadataStage.Config()),
        ]
    )
    item = _item("  Hello   world.  " * 6, source_id="s1")
    out = list(pipe.run([item]))
    assert len(out) >= 1
    total = out[0].metadata["total_chunks"]
    assert [o.metadata["chunk_index"] for o in out] == list(range(total))
    assert all("char_count" in o.metadata and "doc_id" in o.metadata for o in out)
