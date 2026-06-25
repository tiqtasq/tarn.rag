"""The TAT-QA layout-aware retrieval eval: the pure loader/render logic, and an end-to-end source-hit run
over the real ingest → retrieve loop driven by the model-free ``hash`` embedder (offline, no network)."""

from tarnrag.core.engine.config import Settings
from tarnrag.eval.layout import (
    build_tatqa_index,
    format_source_hit,
    load_tatqa,
    render_table,
    tatqa_source_hit,
)

ROWS = [
    {
        "table": {"uid": "t1", "table": [["", "2019", "2018"], ["Revenue", "$10", "$8"]]},
        "paragraphs": [
            {"uid": "p1", "order": 1, "text": "Revenue grew strongly in 2019."},
            {"uid": "p2", "order": 2, "text": "Operating costs were flat year over year."},
        ],
        "questions": [
            {"question": "What was revenue in 2019?", "answer": ["$10"], "answer_type": "span",
             "answer_from": "table", "rel_paragraphs": []},
            {"question": "How did revenue change?", "answer": ["grew"], "answer_type": "span",
             "answer_from": "text", "rel_paragraphs": ["1"]},
            {"question": "What is total revenue?", "answer": ["18"], "answer_type": "arithmetic",
             "answer_from": "table", "rel_paragraphs": []},  # dropped: not extractive
        ],
    },
]


def test_render_table_linearizes_cells():
    assert render_table([["a", "b"], ["c", "d"]]) == "a | b\nc | d"


def test_load_tatqa_filters_arithmetic_and_labels_sources():
    corpus, queries = load_tatqa(ROWS)
    assert {uid for uid, _ in corpus} == {"t1", "p1", "p2"}  # table + 2 paragraphs, deduped by uid
    assert dict(corpus)["t1"] == " | 2019 | 2018\nRevenue | $10 | $8"  # table rendered (empty first cell)
    assert len(queries) == 2  # the arithmetic question is filtered out
    by_seg = {q.answer_from: q for q in queries}
    assert by_seg["table"].gold_source_ids == ["t1"]
    assert by_seg["text"].gold_source_ids == ["p1"]  # rel_paragraph order "1" -> p1


def test_load_tatqa_limit_caps_records():
    corpus, queries = load_tatqa(ROWS + ROWS, limit=1)
    assert {uid for uid, _ in corpus} == {"t1", "p1", "p2"}  # only the first record


async def test_source_hit_runs_over_the_index(tmp_path):
    settings = Settings(_env_file=None, embedding={"provider": "hash"})
    corpus, queries = load_tatqa(ROWS)
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=str(tmp_path / "t.db"))
    try:
        report = await tatqa_source_hit(queries, repo, embedder, settings=settings, k=3, concurrency=2)
        assert report.n == 2 and 0.0 <= report.source_hit <= 1.0
        assert set(report.by_segment) == {"table", "text"}  # segmented by answer_from
        assert report.by_segment["table"][0] == 1 and report.by_segment["text"][0] == 1  # one query each
        assert "OVERALL" in format_source_hit(report, tag="[DENSE]")
    finally:
        await repo.disconnect()
