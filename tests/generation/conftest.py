"""Shared fakes for the generation tests: a retrieval-port double + a provenance-bearing result factory."""

import pytest

from tarnrag.contracts import ChunkProvenance, RetrievalResult
from tarnrag.contracts.structure import Span


class FakeRetrieval:
    """A ``RetrievalEngineProtocol`` double: returns canned results and records the query it received."""

    def __init__(self, results):
        self._results = list(results)
        self.seen = None

    async def search(self, query):
        self.seen = query
        return list(self._results)


@pytest.fixture
def make_result():
    """Factory for a ``RetrievalResult`` with layout provenance (a geometry span + header path)."""

    def _make(chunk_id, text, *, header_path=(), locator=None, start=0, end=None):
        prov = ChunkProvenance(
            geometry=[Span(start=start, end=len(text) if end is None else end)],
            header_path=list(header_path),
            content_hash="h",
        )
        return RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            score=1.0,
            component_scores={},
            document_id="doc",
            source_kind="pdf",
            standard_id=None,
            locator=locator,
            license_class="open",
            provenance=prov,
        )

    return _make


@pytest.fixture
def fake_retrieval():
    """Factory: wrap a list of results in a retrieval-port double (``FakeRetrieval(results)``)."""
    return FakeRetrieval
