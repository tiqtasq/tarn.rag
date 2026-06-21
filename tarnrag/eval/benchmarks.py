"""Loaders for the multi-hop QA benchmarks MOTHRAG published — HotpotQA, 2WikiMultiHopQA, MuSiQue.

Each loader turns a downloaded dataset file into ``BenchItem``s: a ``GenEvalQuery`` (question + gold answer
+ gold supporting sentences) paired with the question's candidate **passages**. This is the *distractor*
setting — every question carries its own ~10 (HotpotQA / 2Wiki) or ~20 (MuSiQue) paragraphs, a few gold +
the rest distractors — which ``benchmark_runner`` ingests per question and runs the generation engine over.

The datasets aren't shipped (sizable, separately licensed); download them and point the loader at the file:

- HotpotQA distractor dev: ``hotpot_dev_distractor_v1.json``  (https://hotpotqa.github.io/)
- 2WikiMultiHopQA dev:     ``dev.json``  (same HotpotQA-style schema)
- MuSiQue (answerable) dev: ``musique_ans_v1.0_dev.jsonl``  (JSON Lines)

Scoring is token-F1 / exact-match (SQuAD-style) over ``GenEvalQuery.answer`` — the metric MOTHRAG reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tarnrag.eval.generation import GenEvalQuery


@dataclass
class BenchItem:
    """One benchmark question: the labeled ``GenEvalQuery`` + the ``(title, text)`` passages to ingest."""

    query: GenEvalQuery
    passages: list[tuple[str, str]]


def _yes_no(answer: str) -> bool:
    return answer.strip().lower() in {"yes", "no", "noanswer"}


def _hotpot_item(r: dict) -> BenchItem:
    """One HotpotQA/2Wiki distractor record → ``BenchItem``. The record's ``context`` is
    ``[[title, [sent, …]], …]`` and ``supporting_facts`` is ``[[title, sent_idx], …]`` (the shape both the
    raw JSON and the HF rows are normalized to). Each paragraph becomes a passage; the gold supporting
    *sentences* (resolved by index) drive citation coverage."""
    sents_by_title = {title: sents for title, sents in r["context"]}
    passages = [(title, " ".join(sents)) for title, sents in r["context"]]
    supporting = [
        sents_by_title[title][idx]
        for title, idx in r.get("supporting_facts", [])
        if title in sents_by_title and 0 <= idx < len(sents_by_title[title])
    ]
    answer = r["answer"]
    return BenchItem(
        query=GenEvalQuery(
            text=r["question"],
            answer=answer,
            answer_contains=[] if _yes_no(answer) else [answer],
            supporting=supporting,
        ),
        passages=passages,
    )


def load_hotpotqa(path: str | Path, *, limit: int | None = None) -> list[BenchItem]:
    """HotpotQA / 2WikiMultiHopQA distractor JSON (same schema): a list of
    ``{question, answer, supporting_facts: [[title, sent_idx], …], context: [[title, [sent, …]], …]}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_hotpot_item(r) for r in raw[: limit or len(raw)]]


# 2WikiMultiHopQA uses the HotpotQA distractor schema.
load_2wiki = load_hotpotqa


def load_hotpotqa_hf(split: str = "validation", *, limit: int | None = None) -> list[BenchItem]:
    """HotpotQA distractor from HuggingFace (``hotpotqa/hotpot_qa``, config ``distractor``) — the reliable
    source, since the canonical ``curtis.ml.cmu.edu`` host is frequently down. Streams the split (so a
    ``limit`` fetches only what it needs) and normalizes the column-oriented HF rows to the JSON shape
    ``_hotpot_item`` consumes. Needs the ``datasets`` package (the ``benchmarks`` extra)."""
    import itertools

    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    rows = itertools.islice(ds, limit) if limit else ds
    return [_hotpot_item(_hf_hotpot_record(r)) for r in rows]


def _hf_hotpot_record(r: dict) -> dict:
    """A HuggingFace ``hotpot_qa`` row (column-oriented ``context``/``supporting_facts``) → the row-oriented
    record shape ``_hotpot_item`` expects."""
    return {
        "question": r["question"],
        "answer": r["answer"],
        "context": list(zip(r["context"]["title"], r["context"]["sentences"])),
        "supporting_facts": list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])),
    }


def load_musique(path: str | Path, *, limit: int | None = None) -> list[BenchItem]:
    """MuSiQue (answerable) JSON Lines: one object per line —
    ``{question, answer, answerable, paragraphs: [{title, paragraph_text, is_supporting}, …]}``.
    Unanswerable items (in the full set) map to ``should_abstain``; supporting paragraphs drive coverage."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    items: list[BenchItem] = []
    for ln in lines[: limit or len(lines)]:
        r = json.loads(ln)
        paragraphs = r.get("paragraphs", [])
        passages = [(p.get("title", ""), p.get("paragraph_text", "")) for p in paragraphs]
        supporting = [p.get("paragraph_text", "") for p in paragraphs if p.get("is_supporting")]
        answerable = r.get("answerable", True)
        answer = r.get("answer", "") or ""
        items.append(
            BenchItem(
                query=GenEvalQuery(
                    text=r["question"],
                    answer=answer,
                    answer_contains=[] if _yes_no(answer) else ([answer] if answer else []),
                    should_abstain=not answerable,
                    supporting=supporting,
                ),
                passages=passages,
            )
        )
    return items


# dataset name -> file loader (takes a downloaded path), for the CLI.
LOADERS = {"hotpotqa": load_hotpotqa, "2wiki": load_2wiki, "musique": load_musique}

# dataset name -> HuggingFace loader (no local file needed), for `--hf`.
HF_LOADERS = {"hotpotqa": load_hotpotqa_hf}
