"""Layout-aware retrieval eval on TAT-QA — Option #2 (differentiators), PR-1.

TAT-QA pairs a financial **table** with associated **paragraphs**, and labels each question with where its
answer lives (``answer_from`` ∈ table / text / table-text). That label is ground truth for a question the
MOTHRAG benchmarks can't ask: **does retrieval surface the answer-bearing *element* — and does it find
*tables* as well as it finds text?** We build one shared corpus of every table + paragraph (so retrieval has
to pick the right element out of thousands, not ~6), then measure **source-hit@k** (did the top-k include the
gold answer-source element), segmented by ``answer_from``. Comparing dense vs hybrid (``--hybrid``) shows
whether BM25's exact-token matching helps *table* retrieval specifically (cell tokens embed weakly).

Extractive only: arithmetic / count answers have no retrievable source span, so they're filtered out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tarnrag.core.engine.config import DatabaseSettings, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.types import Query
from tarnrag.storage.repository import DocumentRepository

_EXTRACTIVE = frozenset({"span", "multi-span"})  # arithmetic/count have no retrievable source span


@dataclass
class TatQaQuery:
    """One extractive TAT-QA question: the text, gold answer span(s), the gold answer-source element ids
    (table uid and/or paragraph uids), and the ``answer_from`` segment the source-hit is reported under."""

    question: str
    answers: list[str]
    gold_source_ids: list[str]
    answer_from: str  # "table" | "text" | "table-text"


def render_table(grid: list[list[str]]) -> str:
    """Linearize a TAT-QA table grid (rows of cells) to text — cells joined by ``|`` per row — preserving the
    exact cell tokens (so BM25 can match them) and the row structure."""
    return "\n".join(" | ".join(c for c in row) for row in grid)


def load_tatqa(rows: list[dict], *, limit: int | None = None) -> tuple[list[tuple[str, str]], list[TatQaQuery]]:
    """Map TAT-QA records (``{table, paragraphs, questions}``) → a deduped corpus of ``(uid, text)`` elements
    (each table rendered, each paragraph) and the extractive ``TatQaQuery`` list. ``limit`` caps records.
    Pure (no network) — pass ``rows`` from :func:`stream_tatqa` or a fixture."""
    corpus: dict[str, str] = {}
    queries: list[TatQaQuery] = []
    for rec in rows[:limit] if limit else rows:
        table = rec["table"]
        tuid = table["uid"]
        corpus[tuid] = render_table(table["table"])
        para_uid_by_order = {}
        for p in rec["paragraphs"]:
            corpus[p["uid"]] = p["text"]
            para_uid_by_order[str(p["order"])] = p["uid"]
        for q in rec["questions"]:
            if q.get("answer_type") not in _EXTRACTIVE:
                continue
            af = q.get("answer_from", "")
            gold: set[str] = set()
            if af in ("table", "table-text"):
                gold.add(tuid)
            if af in ("text", "table-text"):
                gold.update(para_uid_by_order[str(o)] for o in q.get("rel_paragraphs", []) if str(o) in para_uid_by_order)
            if not gold:
                continue
            queries.append(
                TatQaQuery(question=q["question"], answers=list(q.get("answer", [])), gold_source_ids=sorted(gold), answer_from=af)
            )
    return list(corpus.items()), queries


def stream_tatqa(limit: int | None = None) -> list[dict]:
    """Stream TAT-QA validation records from HuggingFace (integration-only; the rest of the module is pure)."""
    from datasets import load_dataset

    split = f"validation[:{limit}]" if limit else "validation"
    return list(load_dataset("next-tat/TAT-QA", split=split))


@dataclass
class SourceHitReport:
    """source-hit@k overall and per ``answer_from`` segment (n + hit-rate each)."""

    k: int
    n: int
    source_hit: float
    by_segment: dict[str, tuple[int, float]] = field(default_factory=dict)  # segment -> (n, hit-rate)


async def build_tatqa_index(
    corpus: list[tuple[str, str]], settings: Settings, *, db_path: str
) -> tuple[DocumentRepository, Embedder]:
    """Ingest each ``(uid, text)`` element with ``source_id == uid`` (so ``document_id`` maps a retrieved
    chunk back to its element) into a persistent, **cached** store. Mirrors ``build_corpus_index``: a
    full store is reused (skip the slow embed), a partial one is rebuilt."""
    settings.ID_POLICY = "caller"  # ingest each element under its uid, so document_id maps back for source-hit
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    repo = await DocumentRepository.create(
        DatabaseSettings(document_url=f"sqlite:///{db_path}"), settings.EMBEDDING_DIMENSION
    )
    await IngestionEngine.ensure_index_meta(repo, embedder)
    ingest = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
    if len(await ingest.list_documents()) == len(corpus):  # already indexed — reuse
        return repo, embedder
    for summary in await ingest.list_documents():
        await ingest.delete_document(summary.document_id)
    await ingest.ingest_content(
        [{"source_id": uid, "content": text, "title": uid, "source_type": "text"} for uid, text in corpus]
    )
    return repo, embedder


async def tatqa_source_hit(
    queries: list[TatQaQuery],
    repo: DocumentRepository,
    embedder: Embedder,
    *,
    settings: Settings,
    k: int = 10,
    concurrency: int = 8,
) -> SourceHitReport:
    """For each query, retrieve top-``k`` over the shared corpus and score a **source-hit** = any retrieved
    chunk's ``document_id`` is a gold answer-source element. Reported overall + segmented by ``answer_from``.
    Bounded ``concurrency`` (independent read-only retrievals)."""
    retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    sem = asyncio.Semaphore(concurrency)

    async def _hit(q: TatQaQuery) -> bool:
        async with sem:
            results = await retrieval.search(Query(text=q.question, top_k=k))
        return any(r.document_id in set(q.gold_source_ids) for r in results)

    hits = await asyncio.gather(*(_hit(q) for q in queries))
    by_segment: dict[str, tuple[int, float]] = {}
    for seg in sorted({q.answer_from for q in queries}):
        seg_hits = [h for h, q in zip(hits, queries) if q.answer_from == seg]
        by_segment[seg] = (len(seg_hits), sum(seg_hits) / len(seg_hits)) if seg_hits else (0, 0.0)
    return SourceHitReport(
        k=k, n=len(queries), source_hit=(sum(hits) / len(hits)) if hits else 0.0, by_segment=by_segment
    )


def format_source_hit(report: SourceHitReport, *, tag: str = "") -> str:
    """Render the source-hit report: overall + a row per answer_from segment (table / text / table-text)."""
    lines = [
        f"TAT-QA layout-aware retrieval{(' ' + tag) if tag else ''}  (source-hit@{report.k}, n={report.n})",
        f"{'segment':<14}{'n':>6}{'source-hit':>12}",
        "-" * 32,
    ]
    for seg, (n, hit) in report.by_segment.items():
        lines.append(f"{seg:<14}{n:>6}{hit:>12.3f}")
    lines.append(f"{'OVERALL':<14}{report.n:>6}{report.source_hit:>12.3f}")
    return "\n".join(lines)
