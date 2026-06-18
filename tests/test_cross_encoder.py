"""Real OnnxCrossEncoder over the bundled cross-encoder model — gated on it being fetched.

Populate MODEL_DIR with::

    python scripts/fetch_model.py --repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2

The reranker's *wiring* is unit-tested with a fake scorer (tests/retrieval/test_engine.py); this exercises
the real tokenize-pair → ONNX → relevance-logit path that fake can't.
"""

from pathlib import Path

import pytest

MODEL_DIR = Path("models/ms-marco-MiniLM-L-6-v2")
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(),
    reason="cross-encoder model not fetched (scripts/fetch_model.py "
    "--repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2)",
)


def _cross_encoder():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from tarnrag.core.cross_encoder import OnnxCrossEncoder

    return OnnxCrossEncoder(str(MODEL_DIR), model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_scores_relevant_above_irrelevant():
    ce = _cross_encoder()
    scores = ce.score(
        "How do I inspect a storage tank for corrosion?",
        [
            "Storage tank inspection: visually check the shell and roof for corrosion.",  # relevant
            "The quokka is a small marsupial native to Western Australia.",               # irrelevant
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]  # the relevant query×passage pair scores higher


def test_deterministic_empty_and_identity():
    ce = _cross_encoder()
    assert ce.score("pump", ["check the mechanical seal"]) == ce.score("pump", ["check the mechanical seal"])
    assert ce.score("pump", []) == []  # no passages -> no scores
    assert ce.identity() == "cross-encoder/ms-marco-MiniLM-L-6-v2"


async def test_reranker_reorders_with_the_real_model():
    """The CrossEncoderReranker over the real model lifts a relevant chunk above an irrelevant one,
    flipping a first-pass order that had them the wrong way round."""
    from tarnrag.contracts import RetrievalResult
    from tarnrag.retrieval import CrossEncoderReranker, Query, RetrievalContext

    def _r(cid, text, score):
        return RetrievalResult(
            chunk_id=cid, text=text, score=score, component_scores={"dense": score},
            document_id="s1", source_kind="document", standard_id=None, locator=None,
            license_class="public_domain",
        )

    results = [  # first-pass puts the irrelevant chunk first; the cross-encoder must flip it
        _r("c0", "The quokka is a small marsupial native to Western Australia.", -1.0),
        _r("c1", "Storage tank inspection: check the shell for corrosion.", -2.0),
    ]
    ctx = RetrievalContext(store=None, embedder=None, cross_encoder=_cross_encoder())
    out = await CrossEncoderReranker(CrossEncoderReranker.Config()).rerank(
        Query(text="storage tank corrosion inspection"), results, ctx
    )
    assert [r.chunk_id for r in out] == ["c1", "c0"]  # relevant chunk reranked to the top
    assert "cross_encoder" in out[0].component_scores  # rerank score surfaced alongside the first-pass
