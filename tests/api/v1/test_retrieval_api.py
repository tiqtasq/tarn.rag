"""POST /v1/query over httpx ASGITransport, with the retrieval engine injected."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.v1.dependencies import get_retrieval_engine
from app.domains.base.index_store import SqliteIndexStore
from app.domains.base.models import Chunk, Document, Embedding
from app.domains.retrieval.engine import RetrievalEngine
from app.main import create_app


class _FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def config_fingerprint(self):
        return "fp"

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": "fp"}


@pytest_asyncio.fixture
async def client(tmp_path):
    store = SqliteIndexStore(str(tmp_path / "index.db"), embedding_dim=3).connect()
    store.write_index_meta(_FakeEmbedder())
    await store.store_document(Document(content="d", metadata={"source_id": "s1"}))
    [cid] = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="storage tank inspection",
              chunk_index=0, total_chunks=1, metadata={"locator": "§6.4"})
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0], model="f", dimension=3)
    ])
    engine = RetrievalEngine.open(store, _FakeEmbedder())

    app = create_app()
    app.dependency_overrides[get_retrieval_engine] = lambda: engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    store.close()


async def test_query_returns_ranked_results(client):
    resp = await client.post("/v1/query", json={"text": "tank inspection", "top_k": 3})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results
    top = results[0]
    assert top["text"] == "storage tank inspection"
    assert top["document_id"] == "s1"
    assert top["license_class"] == "public_domain"
    assert top["locator"] == "§6.4"
    assert "dense" in top["component_scores"]


async def test_query_503_when_engine_unavailable(tmp_path):
    # No override + ASGI transport doesn't run the lifespan -> no engine on app.state.
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/v1/query", json={"text": "x"})
    assert resp.status_code == 503
