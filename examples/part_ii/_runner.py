"""The thin teaching shell for the Part II examples — the narration around a YAML config.

Each example *is* a YAML config (run it directly with ``python -m tarnrag.console cfg.yaml``); this runner
is the scripted story around it. An example's ``run.py`` opens a :class:`~tarnrag.TarnRag` over its config,
ingests the shared corpus once, then runs a *good* and a *bad* probe through retrieval (showing the
``explain`` breakdown) and — in the generation act — through ``ask``, marking each probe HIT/MISS against a
gold phrase so the failure a step demonstrates, and the next step's fix, is **shown**, not asserted.

The rendering is the console's own static factory methods (``Console.create_trace_view`` /
``create_results_table`` / ``create_answer_view``) — one renderer, two surfaces: the REPL and these
examples. Needs the ``console`` extra (rich).
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.rule import Rule

from tarnrag import TarnRag
from tarnrag.console import Console as View

_out = RichConsole()


def banner(title: str, *, shows: str, fails: str | None = None, fixed_next: str | None = None) -> None:
    """Frame a step: what it demonstrates, what still fails here, and what the next step changes to fix
    it. The three lines are the spine of the ladder — every step has a ``shows``; an intermediate step
    also has a ``fails`` and the ``fixed_next`` that repairs it."""
    lines = [f"[bold]shows[/]  {shows}"]
    if fails:
        lines.append(f"[yellow]fails[/]  {fails}")
    if fixed_next:
        lines.append(f"[green]next [/]  {fixed_next}")
    _out.print(Panel("\n".join(lines), title=title, title_align="left", border_style="cyan"))


async def ingest_corpus(tarn: TarnRag, *paths: str | Path) -> None:
    """Ingest the corpus into the session's store (re-ingesting is idempotent) and print a one-line
    summary, surfacing any issues the facade reported (a path that matched nothing, a failed document)."""
    outcome = await tarn.ingest([str(p) for p in paths])
    chunks = sum(d.chunk_count for d in outcome.value)
    _out.print(f"[dim]ingested {len(outcome.value)} documents · {chunks} chunks[/]")
    for issue in outcome.report.issues:
        _out.print(f"[yellow]{issue.severity.value}[/] {issue.subject}: {issue.message}")


async def show_retrieval(
    tarn: TarnRag, query: str, *, gold: str | None = None, label: str | None = None, top_k: int = 8
) -> bool | None:
    """Run ``explain`` for one probe and render its inner workings (per-retriever candidates, the ranking
    at each stage, component scores, routing). When a ``gold`` phrase is given, print a HIT/MISS verdict
    — a hit is content-based: the phrase appears in some top-``k`` passage — and return it (``None`` when
    no gold is given). This is the label-free check the whole ladder turns on."""
    _out.print(Rule(_probe_title(query, label), align="left", style="cyan"))
    trace = (await tarn.explain(query, top_k=top_k)).value
    _out.print(View.create_trace_view(trace))
    if gold is None:
        return None
    ok = any(gold.lower() in result.text.lower() for result in trace.results)
    verdict = "[green]HIT [/]" if ok else "[red]MISS[/]"
    _out.print(f"{verdict} gold phrase {gold!r} in the top {top_k}")
    return ok


async def show_answer(tarn: TarnRag, query: str, *, label: str | None = None) -> None:
    """Run ``ask`` for one probe and render the grounded answer + its proof tree (needs an LLM key —
    ``ANTHROPIC_API_KEY``). Used by the generation act of the ladder."""
    _out.print(Rule(_probe_title(query, label), align="left", style="cyan"))
    result = (await tarn.ask(query)).value
    _out.print(View.create_answer_view(result))


def _probe_title(query: str, label: str | None) -> str:
    return f"[bold]{label}[/]  {query}" if label else query
