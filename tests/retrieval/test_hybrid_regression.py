"""Drift guard for Option 1's measured gain — hybrid (dense + BM25, RRF) recovers an answer-bearing element
whose exact tokens are present but whose dense embedding is far from the query. This is the *mechanic* behind
the TAT-QA table lift (table cells embed weakly against a natural-language question; BM25 matches the exact
figure), distilled into a deterministic, every-push invariant: if a refactor breaks the sparse arm or the RRF
fusion, hybrid stops rescuing the lexical match and this fails. (The full gte-small table lift is tracked
on-demand by ``scripts/run_layout_eval.py``, which needs corpus scale to be meaningful.)"""

from tarnrag.contracts import Chunk, Document, Embedding, IndexMeta
from tarnrag.retrieval import Query, RetrievalContext, RetrievalPipeline

FINGERPRINT = "hybrid-regression-fp"


class _Embedder:
    """Embeds every query to a vector near the text distractors and far from the 'table' chunk — so the
    table is reachable only lexically (BM25), reproducing the table-retrieval phenomenon deterministically."""

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def config_fingerprint(self):
        return FINGERPRINT

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": FINGERPRINT}


async def _index(repo):
    """A 'table' chunk (exact figure, dense-far) + two text distractors (dense-near, no figure tokens)."""
    await repo.write_index_meta(IndexMeta.build(_Embedder()))
    _, (table, t1, t2) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="Impairment of goodwill 1910 recorded in Q3",
                  chunk_index=0, total_chunks=3),
            Chunk(parent_doc_id="s1", content="Company overview and general background",
                  chunk_index=1, total_chunks=3),
            Chunk(parent_doc_id="s1", content="Notes on market conditions and the outlook",
                  chunk_index=2, total_chunks=3),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=table, vector=[0.0, 0.0, 1.0], model="f", dimension=3),  # far from the query
        Embedding(chunk_id=t1, vector=[1.0, 0.0, 0.0], model="f", dimension=3),     # near the query
        Embedding(chunk_id=t2, vector=[0.9, 0.1, 0.0], model="f", dimension=3),     # near the query
    ])
    return table


def _pipeline(*, hybrid: bool) -> RetrievalPipeline:
    retrievers = [{"class_name": "dense"}, {"class_name": "sparse"}] if hybrid else [{"class_name": "dense"}]
    fuser = {"class_name": "rrf"} if hybrid else {"class_name": "identity"}
    return RetrievalPipeline(RetrievalPipeline.Config(retrievers=retrievers, fuser=fuser))


async def test_hybrid_recovers_a_lexical_match_dense_ranks_below_the_cutoff(repo):
    table = await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_Embedder())
    query = Query(text="goodwill impairment 1910", top_k=2)

    dense = {r.chunk_id for r in await _pipeline(hybrid=False).search(query, ctx)}
    hybrid = await _pipeline(hybrid=True).search(query, ctx)

    assert table not in dense  # dense-only ranks the lexical match below the top-2 cutoff (it's dense-far)
    assert hybrid[0].chunk_id == table  # BM25 rescues it; RRF (dense rank 3 + sparse rank 1) ranks it first
    assert "sparse" in hybrid[0].component_scores  # surfaced by the sparse arm, not dense
