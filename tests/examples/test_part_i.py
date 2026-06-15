"""Smoke tests for the Part I examples.

Each test runs the example's own ``main()`` against a temp store (via ``EXAMPLES_DATA_DIR``) and
checks it produces sensible output — so the suite catches API drift in the examples before an
onboarding developer does. Gated on ``requires_model`` (auto-skipped when the model/runtime
aren't present; see ``conftest.py``).

The pattern every Part I example slots into: import its module(s), run ``main()`` under the
isolated store, assert on the returned data.
"""

from __future__ import annotations

import pytest

from examples.part_i.example_01 import ingestion, retrieval

pytestmark = pytest.mark.requires_model


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Redirect each example's SQLite store into a temp dir, so tests never write into the tree."""
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_01_ingest_then_retrieve():
    # Ingestion: the corpus is fully indexed (one chunk + embedding per short doc).
    statuses = await ingestion.main()
    assert len(statuses) == 3
    assert all(s.status == "complete" for s in statuses)
    assert all(s.chunk_count >= 1 and s.embedding_count >= 1 for s in statuses)

    # Retrieval over the same store ranks sensibly: the corrosion query surfaces the tank doc.
    answers = await retrieval.main()
    corrosion_query = retrieval.QUERIES[0]
    assert answers[corrosion_query][0].document_id == "tank-inspection"
