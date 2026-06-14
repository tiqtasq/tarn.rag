import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.pipeline import Pipeline
from tarnrag.ingestion.stages.chunk import ChunkStage
from tarnrag.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.stages.embed import EmbedStage
from tarnrag.ingestion.stages.enrich import EnrichMetadataStage
from tarnrag.ingestion.stages.load_parse import LoadAndParseStage


def _item(content, **meta):
    return PipelineItem(content=content, metadata=meta)


def test_load_from_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello from a file")
    out = list(LoadAndParseStage().process(_item("", source_path=str(p))))
    assert len(out) == 1
    assert out[0].content == "hello from a file"
    assert out[0].metadata["doc_id"] and out[0].metadata["loaded"] is True


def test_load_passthrough_content():
    out = list(LoadAndParseStage().process(_item("already loaded")))
    assert out[0].content == "already loaded"


def test_clean_normalizes_whitespace_and_controls():
    out = list(CleanAndNormalizeStage().process(_item("  a\x07b   c\t\td  ")))
    assert out[0].content == "ab c d"
    assert out[0].metadata["cleaned"] is True


def test_chunk_splits_and_indexes():
    stage = ChunkStage(chunk_size=20, overlap=4)
    text = "abcdefghij " * 8  # ~88 chars -> several chunks
    out = list(stage.process(_item(text, doc_id="d1")))
    assert len(out) > 1
    total = out[0].metadata["total_chunks"]
    assert total == len(out)
    assert [o.metadata["chunk_index"] for o in out] == list(range(total))
    assert all(o.metadata["chunk_size"] == len(o.content) for o in out)


def test_chunk_validation():
    with pytest.raises(ValueError):
        ChunkStage(chunk_size=0)
    with pytest.raises(ValueError):
        ChunkStage(chunk_size=10, overlap=10)


def test_enrich_counts():
    out = list(EnrichMetadataStage().process(_item("one two three")))
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
    stage = EmbedStage(model_batch_size=2)
    stage._embedder = _FakeEmbedder()
    items = [
        _item(f"chunk {i}", chunk_id=f"c{i}", source_id="s1") for i in range(3)
    ]
    embs = list(stage.process_batch(items))
    assert [e.chunk_id for e in embs] == ["c0", "c1", "c2"]
    assert all(e.dimension == 3 and e.model.endswith("all-MiniLM-L6-v2") for e in embs)
    # model_batch_size=2 over 3 items -> two embed calls (two-tier batching)
    assert len(stage._embedder.calls) == 2


def test_pipeline_composition_doc_to_chunks():
    pipe = Pipeline(
        [
            LoadAndParseStage(),
            CleanAndNormalizeStage(),
            ChunkStage(chunk_size=20, overlap=4),
            EnrichMetadataStage(),
        ]
    )
    item = _item("  Hello   world.  " * 6, source_id="s1")
    out = list(pipe.run([item]))
    assert len(out) >= 1
    total = out[0].metadata["total_chunks"]
    assert [o.metadata["chunk_index"] for o in out] == list(range(total))
    assert all("char_count" in o.metadata and "doc_id" in o.metadata for o in out)
