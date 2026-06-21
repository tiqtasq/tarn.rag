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


def load_hotpotqa(path: str | Path, *, limit: int | None = None) -> list[BenchItem]:
    """HotpotQA / 2WikiMultiHopQA distractor JSON (same schema): a list of
    ``{question, answer, supporting_facts: [[title, sent_idx], …], context: [[title, [sent, …]], …]}``.
    Each paragraph becomes a passage; the gold supporting *sentences* drive citation coverage."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[BenchItem] = []
    for r in raw[: limit or len(raw)]:
        sents_by_title = {title: sents for title, sents in r["context"]}
        passages = [(title, " ".join(sents)) for title, sents in r["context"]]
        supporting = [
            sents_by_title[title][idx]
            for title, idx in r.get("supporting_facts", [])
            if title in sents_by_title and 0 <= idx < len(sents_by_title[title])
        ]
        answer = r["answer"]
        items.append(
            BenchItem(
                query=GenEvalQuery(
                    text=r["question"],
                    answer=answer,
                    answer_contains=[] if _yes_no(answer) else [answer],
                    supporting=supporting,
                ),
                passages=passages,
            )
        )
    return items


# 2WikiMultiHopQA uses the HotpotQA distractor schema.
load_2wiki = load_hotpotqa


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


# dataset name -> loader, for the CLI.
LOADERS = {"hotpotqa": load_hotpotqa, "2wiki": load_2wiki, "musique": load_musique}
