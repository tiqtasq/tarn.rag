"""The MOTHRAG benchmark harness: the HotpotQA-style loader (pure) and an end-to-end run over the real
ingest → retrieve → generate → score loop, driven by a deterministic ``StaticLanguageModel`` (no API key).

The end-to-end test needs the local ONNX embedder (gte-small) for ingestion + retrieval, so it's gated on
the model being fetched — like the other ONNX-backed tests."""

from pathlib import Path

import pytest

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.eval.benchmark_runner import MOTHRAG_PUBLISHED, format_comparison, run_benchmark
from tarnrag.eval.benchmarks import load_hotpotqa

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hotpot_sample.json"
MODEL_DIR = Path("models/gte-small")
requires_model = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(), reason="model not fetched (scripts/fetch_model.py)"
)


def test_load_hotpotqa_parses_passages_answer_and_supporting():
    items = load_hotpotqa(FIXTURE)
    assert len(items) == 3
    q0 = items[0]
    assert q0.query.answer == "Eiffel Tower"
    assert q0.query.answer_contains == ["Eiffel Tower"]  # not yes/no -> drives content-hit
    assert [t for t, _ in q0.passages] == ["Eiffel Tower", "Louvre", "Big Ben"]
    assert q0.passages[0][1].startswith("The Eiffel Tower is a wrought-iron")  # paragraph = joined sentences
    assert q0.query.supporting == ["It is the tallest structure in Paris."]  # gold sentence, resolved by idx
    # a yes/no answer carries no content-hit phrases (token-F1/EM still apply)
    assert items[2].query.answer == "yes" and items[2].query.answer_contains == []
    assert len(load_hotpotqa(FIXTURE, limit=1)) == 1  # limit


def test_mothrag_published_numbers_present():
    assert set(MOTHRAG_PUBLISHED) == {"hotpotqa", "2wiki", "musique"}
    assert MOTHRAG_PUBLISHED["hotpotqa"]["f1"] == 0.781


@requires_model
async def test_run_benchmark_end_to_end_offline():
    """Full distractor loop per question (ingest its passages → retrieve → answer → score), with a canned
    LLM. The canned answer matches q1's gold, so its token-F1/EM are perfect; the others differ."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    items = load_hotpotqa(FIXTURE)
    canned = '{"answer": "Eiffel Tower", "steps": [{"claim": "The tallest structure in Paris.", "cited": [1]}]}'
    report = await run_benchmark(items, StaticLanguageModel(canned))

    assert report.n == 3
    assert report.per_query[0].exact_match is True and report.per_query[0].token_f1 == 1.0  # q1 == canned
    assert report.per_query[1].exact_match is False  # q2 gold "Rome" != canned
    table = format_comparison({"hotpotqa": report})
    assert "hotpotqa" in table and "0.781" in table  # rendered against MOTHRAG's published F1
