"""The eval harness: sweep retrieval-pipeline specs over a labeled query set against one index."""

from tarnrag.contracts import Chunk, Document, Embedding
from tarnrag.core.components import ComponentFactory
from tarnrag.eval import EvalQuery, EvalSet, evaluate_pipeline, format_reports, sweep
from tarnrag.retrieval import RetrievalContext, RetrievalPipeline


class _FakeEmbedder:
    """Embeds any query to a fixed 3-d vector (the retrieval tests' fake)."""

    def __init__(self, query_vec=(1.0, 0.0, 0.0)):
        self._vec = list(query_vec)

    def embed_query(self, text):
        return self._vec


async def _index(repo):
    """Ingest two chunks (+ embeddings): cids[0] 'storage tank inspection', cids[1] 'quokka marsupial'."""
    _, cids = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="storage tank inspection", chunk_index=0, total_chunks=2),
            Chunk(parent_doc_id="s1", content="quokka marsupial", chunk_index=1, total_chunks=2),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="f", dimension=3),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0], model="f", dimension=3),
    ])
    return cids


_DENSE = {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}]}
_SPARSE = {
    "class_name": "retrieval_pipeline",
    "retrievers": [{"class_name": "sparse"}],
    "fuser": {"class_name": "identity"},
}


async def test_sweep_scores_each_pipeline_spec(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    evalset = EvalSet([EvalQuery(text="tank inspection", relevant=["tank"])])

    reports = await sweep({"dense": _DENSE, "sparse": _SPARSE}, ctx, evalset, k=5)

    assert set(reports) == {"dense", "sparse"}
    for report in reports.values():
        assert report.n == 1 and report.k == 5
        assert report.hit_at_k == 1.0  # both find the tank chunk for a "tank" query
        assert report.mrr == 1.0  # and rank it first
    table = format_reports(reports)
    assert "dense" in table and "sparse" in table and "hit@k" in table


async def test_evaluate_pipeline_scores_a_miss(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    # Gold phrase appears in no chunk -> nothing relevant -> all metrics zero.
    evalset = EvalSet([EvalQuery(text="tank", relevant=["helicopter"])])
    pipeline = ComponentFactory.get().create_as(_DENSE, RetrievalPipeline)
    report = await evaluate_pipeline(pipeline, ctx, evalset, k=5)
    assert report.hit_at_k == 0.0 and report.mrr == 0.0 and report.ndcg_at_k == 0.0
    assert report.per_query[0].relevances and not any(report.per_query[0].relevances)
