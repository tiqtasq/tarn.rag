"""Retrieval API — query the §8 index for ranked, provenance-bearing chunks."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.api.v1.dependencies import get_retrieval_engine
from app.api.v1.schemas import QueryRequest, QueryResponse, RetrievalResultOut
from app.domains.retrieval.engine import RetrievalEngine
from app.domains.retrieval.types import Query, RetrievalResult

router = APIRouter(prefix="/v1", tags=["retrieval"])


def _to_out(r: RetrievalResult) -> RetrievalResultOut:
    return RetrievalResultOut(
        chunk_id=r.chunk_id,
        text=r.text,
        score=r.score,
        component_scores=r.component_scores,
        document_id=r.document_id,
        source_kind=r.source_kind,
        standard_id=r.standard_id,
        locator=r.locator,
        license_class=r.license_class,
        methods=[
            {"method_id": m.method_id, "method_version": m.method_version} for m in r.methods
        ],
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    engine: RetrievalEngine = Depends(get_retrieval_engine),
) -> QueryResponse:
    """Embed the query, KNN the index, and return ranked results. The engine is sync; we
    offload it off the event loop."""
    q = Query(text=req.text, top_k=req.top_k, dense_k=req.dense_k)
    results = await asyncio.to_thread(engine.search, q)
    return QueryResponse(results=[_to_out(r) for r in results])
