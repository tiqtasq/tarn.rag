"""RetrievalEngine (dense-only) over the §8 repository, built with a fake embedder."""

import pytest

from tarnrag.contracts import build_index_meta
from tarnrag.contracts import (
    Chunk, ChunkProvenance, ChunkRecord, Document, Embedding, MethodRef, RetrievalResult,
)
from tarnrag.core.config import RETRIEVAL_PIPELINE, Settings
from tarnrag.retrieval import (
    AutoMerger, Query, RetrievalContext, RetrievalEngine, RetrievalError, RetrievalPipeline,
)
from tarnrag.retrieval.types import Purpose

FINGERPRINT = "fp-123"


class _FakeEmbedder:
    """Embeds a query to a fixed 3-d vector; fingerprint matches the index by default."""

    def __init__(self, query_vec=(1.0, 0.0, 0.0), fingerprint=FINGERPRINT):
        self._vec = list(query_vec)
        self._fp = fingerprint

    def embed_query(self, text):
        return self._vec

    def config_fingerprint(self):
        return self._fp

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": FINGERPRINT}


async def _index(repo):
    """Stamp index_meta and ingest two chunks (+ embeddings) into the repo; return the chunk ids."""
    await repo.write_index_meta(build_index_meta(_FakeEmbedder()))
    _, cids = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1", "title": "T"}),
        [
            Chunk(parent_doc_id="s1", content="storage tank inspection",
                  chunk_index=0, total_chunks=2, metadata={"locator": "§6.4"}),
            Chunk(parent_doc_id="s1", content="quokka marsupial",
                  chunk_index=1, total_chunks=2, metadata={}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="f", dimension=3),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0], model="f", dimension=3),
    ])
    return cids


async def test_open_validates_fingerprint(repo):
    await _index(repo)
    # Matching fingerprint -> opens.
    await RetrievalEngine.open(repo, _FakeEmbedder())
    # Mismatched fingerprint -> refuses.
    with pytest.raises(RetrievalError, match="fingerprint mismatch"):
        await RetrievalEngine.open(repo, _FakeEmbedder(fingerprint="other"))


async def test_search_ranks_nearest_with_provenance(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))

    results = await engine.search(Query(text="tank inspection", top_k=2))
    assert [r.chunk_id for r in results] == cids  # chunk 0 ([1,0,0]) nearest
    top = results[0]
    assert top.text == "storage tank inspection"
    assert top.document_id == "s1"
    assert top.source_kind == "document"
    assert top.license_class == "public_domain"
    assert top.locator == "§6.4"
    assert top.methods == []  # method_chunks empty for now
    assert top.score >= results[1].score  # ranked by score desc
    assert "dense" in top.component_scores


async def test_top_k_truncates(repo):
    await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    assert len(await engine.search(Query(text="x", top_k=1))) == 1


async def test_empty_index_returns_empty(repo):
    await repo.write_index_meta(build_index_meta(_FakeEmbedder()))  # built, but no chunks
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    assert await engine.search(Query(text="x")) == []


async def test_search_text_convenience(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    results = await engine.search_text("tank inspection", top_k=2)
    assert [r.chunk_id for r in results] == cids


async def test_open_refuses_unbuilt_index(repo):
    # No write_index_meta -> the index has not been built yet.
    with pytest.raises(RetrievalError, match="not been built"):
        await RetrievalEngine.open(repo, _FakeEmbedder())


async def test_pipeline_hybrid_fuses_dense_and_sparse(repo):
    cids = await _index(repo)  # cids[0] = "storage tank inspection", cids[1] = "quokka marsupial"
    pipe = RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}, {"class_name": "sparse"}], fuser={"class_name": "rrf"}
        )
    )
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    results = await pipe.search(Query(text="tank inspection", top_k=2), ctx)
    assert results[0].chunk_id == cids[0]  # the tank chunk: top dense AND top sparse
    assert set(results[0].component_scores) == {"dense", "sparse"}  # both retrievers contributed
    assert results[0].provenance is not None  # provenance threaded through the read path


async def test_engine_uses_configured_pipeline_spec(repo):
    cids = await _index(repo)
    settings = Settings(
        _env_file=None,
        components={
            RETRIEVAL_PIPELINE: {
                "class_name": "retrieval_pipeline",
                "retrievers": [{"class_name": "sparse"}],
                "fuser": {"class_name": "identity"},
            }
        },
    )
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(), settings=settings)
    results = await engine.search(Query(text="tank inspection", top_k=5))
    assert [r.chunk_id for r in results] == [cids[0]]  # sparse-only -> just the lexical match
    assert set(results[0].component_scores) == {"sparse"}


async def test_filter_drops_unavailable_and_respects_grounding(repo):
    await repo.write_index_meta(build_index_meta(_FakeEmbedder()))
    _, (a, b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="tank one", chunk_index=0, total_chunks=2,
                  metadata={"available": 0}),  # unavailable
            Chunk(parent_doc_id="s1", content="tank two", chunk_index=1, total_chunks=2,
                  metadata={"ai_grounding_allowed": 0}),  # not grounding-allowed
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=a, vector=[1.0, 0.0, 0.0], model="f", dimension=3),
        Embedding(chunk_id=b, vector=[0.9, 0.1, 0.0], model="f", dimension=3),
    ])
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    # EXECUTION: the unavailable chunk is dropped; the grounding-disallowed one is kept.
    assert [r.chunk_id for r in await engine.search(Query(text="tank", top_k=5))] == [b]
    # GENERATION_GROUNDING: a is unavailable AND b isn't grounding-allowed -> nothing survives.
    assert await engine.search(Query(text="tank", top_k=5, purpose=Purpose.GENERATION_GROUNDING)) == []


def test_scope_filter_matches_methods():
    rec = ChunkRecord(
        chunk_id="c", text="t", document_id="d", source_kind="document",
        standard_id=None, locator=None, license_class="public_domain", methods=[("M1", "v2")],
    )
    assert RetrievalPipeline._passes(rec, Query(text="x"))  # default scope ALL -> in scope
    assert RetrievalPipeline._passes(rec, Query(text="x", scope=[MethodRef("M1")]))  # version-agnostic
    assert RetrievalPipeline._passes(rec, Query(text="x", scope=[MethodRef("M1", "v2")]))  # exact version
    assert not RetrievalPipeline._passes(rec, Query(text="x", scope=[MethodRef("M9")]))  # no match -> out


async def _index_tree(repo):
    """Index a 1-level auto-merging tree: a section parent (ordinal 0, NOT embedded) + two leaf children
    (ordinals 1/2, ``parent_ordinal=0``, embedded). Returns ``(parent_id, leaf1_id, leaf2_id)``."""
    await repo.write_index_meta(build_index_meta(_FakeEmbedder()))
    _, (parent, leaf1, leaf2) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="Safety: wear PPE and inspect the tank.", chunk_index=0,
                  provenance=ChunkProvenance(content_hash="p", header_path=["Safety"], level=1)),
            Chunk(parent_doc_id="s1", content="wear PPE", chunk_index=1, metadata={"parent_ordinal": 0}),
            Chunk(parent_doc_id="s1", content="inspect the tank", chunk_index=2,
                  metadata={"parent_ordinal": 0}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=leaf1, vector=[1.0, 0.0, 0.0], model="f", dimension=3),
        Embedding(chunk_id=leaf2, vector=[0.0, 1.0, 0.0], model="f", dimension=3),
    ])
    return parent, leaf1, leaf2


def _merge_pipeline(**merger):
    return RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}], fuser={"class_name": "identity"},
            merger={"class_name": "auto_merge", **merger},
        )
    )


async def test_auto_merge_collapses_siblings_into_parent(repo):
    parent, _leaf1, _leaf2 = await _index_tree(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 1.0, 0.0)))
    results = await _merge_pipeline().search(Query(text="ppe tank", top_k=5), ctx)
    # Both leaves retrieved and share a parent -> collapsed into the one hydrated section parent.
    assert [r.chunk_id for r in results] == [parent]
    assert results[0].text == "Safety: wear PPE and inspect the tank."
    assert results[0].provenance.header_path == ["Safety"]  # the parent's provenance, not a leaf's
    assert "dense" in results[0].component_scores  # merged from the children


async def test_auto_merge_threshold_not_met_keeps_leaves(repo):
    parent, leaf1, leaf2 = await _index_tree(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 1.0, 0.0)))
    # min_siblings=3 but only two leaves are retrieved -> no merge.
    results = await _merge_pipeline(min_siblings=3).search(Query(text="ppe tank", top_k=5), ctx)
    assert {r.chunk_id for r in results} == {leaf1, leaf2}
    assert parent not in {r.chunk_id for r in results}


async def test_auto_merger_keeps_leaves_when_parent_unavailable():
    """Unit: a parent that is itself unavailable is skipped — its leaves stay (filter already ran)."""
    def _leaf(cid, score):
        return RetrievalResult(
            chunk_id=cid, text=cid, score=score, component_scores={"dense": score},
            document_id="s1", source_kind="document", standard_id=None, locator=None,
            license_class="public_domain",
            provenance=ChunkProvenance(content_hash=cid, parent_chunk_id="p"),
        )

    class _FakeStore:
        async def hydrate(self, ids):
            return [ChunkRecord(
                chunk_id="p", text="parent", document_id="s1", source_kind="document",
                standard_id=None, locator=None, license_class="public_domain", available=False,
            )]

    out = await AutoMerger(AutoMerger.Config()).merge(
        [_leaf("l1", -1.0), _leaf("l2", -2.0)], RetrievalContext(store=_FakeStore(), embedder=None)
    )
    assert [r.chunk_id for r in out] == ["l1", "l2"]
