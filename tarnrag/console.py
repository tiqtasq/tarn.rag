"""An interactive console for tarn.rag — ingest, query (retrieval), and ask (retrieval + generation).

Start it with one JSON config:

    python -m tarnrag.console path/to/config.json

The config is a ``Settings`` document — ``database`` / ``embedding`` / ``llm`` plus the
ingestion / retrieval / generation pipeline specs under ``components`` (a sample is at
``examples/console.config.json``). The API key is read from the environment (``ANTHROPIC_API_KEY``), not
the config. Then type commands at the ``tarn>`` prompt::

    ingest <path> ...    ingest (or RE-ingest) files; a directory ingests the files in it. The document
                         id is the filename stem, so re-ingesting a file replaces it.
    docs                 list ingested documents (id, chunks, embeddings)
    delete <id>          delete a document and everything derived from it
    retrieve <query>     retrieval only — the ranked passages
    explain <query>      retrieval with its inner workings — each retriever's candidates before fusion,
                         the ranking at every pipeline stage (with component scores), and any routing
    ask <query>          retrieval + generation — the grounded answer + its proof tree
    help                 show the commands
    quit                 exit

This is purely the **UI**: a rich parse-and-print REPL that owns all output and delegates the work to a
:class:`tarnrag.TarnRag` through its high-level methods. Rendering needs the ``console`` extra
(``pip install '.[console]'``).
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console as RichConsole, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from tarnrag.contracts import RetrievalResult
from tarnrag.report import Report, Severity
from tarnrag.retrieval.types import RetrieverCandidates, SearchTrace
from tarnrag.tarnrag import TarnRag

_out = RichConsole()

_COMMANDS = [
    ("ingest <path> ...", "ingest (or re-ingest) files; a directory ingests the files in it"),
    ("docs", "list the ingested documents"),
    ("delete <id>", "delete a document and everything derived from it"),
    ("retrieve <query>", "retrieval only — the ranked passages"),
    ("explain <query>", "retrieval with its inner workings — per-retriever candidates, per-stage ranking"),
    ("ask <query>", "retrieval + generation — the grounded answer + its proof tree"),
    ("help", "show this list"),
    ("quit", "exit"),
]


class Console:
    """A rich REPL over a :class:`~tarnrag.TarnRag` session — all output lives here. Construct with
    an (opened) ``TarnRag`` and call :meth:`run`; the console never touches the engines directly, only the
    facade's high-level methods (``ingest`` / ``docs`` / ``delete`` / ``retrieve`` / ``ask``)."""

    def __init__(self, tarn: TarnRag) -> None:
        self._tarn = tarn

    async def run(self) -> None:
        """Read commands at the ``tarn>`` prompt until EOF / quit."""
        _out.print(f"[bold]tarn.rag console[/]  [dim]{escape(self._tarn.settings.database.document_url)}[/]")
        _out.print("Type [cyan]help[/], or [cyan]quit[/] to exit.\n")
        handlers = {
            "help": self._do_help,
            "ingest": self._do_ingest,
            "docs": self._do_docs,
            "delete": self._do_delete,
            "retrieve": self._do_retrieve,
            "explain": self._do_explain,
            "ask": self._do_ask,
        }
        while True:
            try:
                line = (await asyncio.to_thread(_out.input, "[bold cyan]tarn>[/] ")).strip()
            except (EOFError, KeyboardInterrupt):
                _out.print()
                break
            if not line:
                continue
            command, _, arg = line.partition(" ")
            command = command.lower()
            if command in ("quit", "exit", "q"):
                break
            handler = handlers.get(command)
            if handler is None:
                _out.print(f"[red]unknown command[/] {command!r} — type [cyan]help[/]")
                continue
            try:
                await handler(arg.strip())
            except Exception as exc:  # keep the session alive on any command error
                _out.print(f"[red]error:[/] {escape(str(exc))}")
        _out.print("[dim]bye[/]")

    async def _do_help(self, _arg: str) -> None:
        table = Table(title="commands", show_edge=False, title_justify="left", header_style="bold")
        table.add_column("command", style="cyan", no_wrap=True)
        table.add_column("description", style="dim")
        for command, description in _COMMANDS:
            table.add_row(command, description)
        _out.print(table)

    async def _do_ingest(self, arg: str) -> None:
        if not arg:
            _out.print("usage: ingest <path> ...", style="dim")
            return
        outcome = await self._tarn.ingest(arg.split())
        if outcome.value:
            table = Table(show_edge=False, header_style="bold")
            table.add_column("document", style="cyan")
            table.add_column("status")
            table.add_column("chunks", justify="right")
            table.add_column("embeddings", justify="right")
            for s in outcome.value:
                color = {"complete": "green", "failed": "red"}.get(s.status, "yellow")
                table.add_row(
                    s.document_id, f"[{color}]{s.status}[/]", str(s.chunk_count), str(s.embedding_count)
                )
            _out.print(table)
        else:
            _out.print("[yellow]nothing ingested[/]")
        _render_report(outcome.report)

    async def _do_docs(self, _arg: str) -> None:
        outcome = await self._tarn.docs()
        if not outcome.value:
            _out.print("[yellow]no documents — ingest some first[/]")
        else:
            table = Table(show_edge=False, header_style="bold")
            table.add_column("document", style="cyan")
            table.add_column("chunks", justify="right")
            table.add_column("embeddings", justify="right")
            for d in outcome.value:
                table.add_row(d.document_id, str(d.chunk_count), str(d.embedding_count))
            _out.print(table)
            _out.print(f"[dim]{len(outcome.value)} document(s)[/]")
        _render_report(outcome.report)

    async def _do_delete(self, arg: str) -> None:
        if not arg:
            _out.print("usage: delete <document-id>", style="dim")
            return
        outcome = await self._tarn.delete(arg)
        if outcome.value:
            _out.print(f"[green]deleted[/] {escape(arg)}")
        else:
            _out.print(f"[yellow]no such document:[/] {escape(arg)}")
        _render_report(outcome.report)

    async def _do_retrieve(self, arg: str) -> None:
        if not arg:
            _out.print("usage: retrieve <query>", style="dim")
            return
        outcome = await self._tarn.retrieve(arg)
        if not outcome.value:
            _out.print("[yellow]no results — ingest some documents first[/]")
        else:
            _out.print(results_table(outcome.value))
        _render_report(outcome.report)

    async def _do_explain(self, arg: str) -> None:
        if not arg:
            _out.print("usage: explain <query>", style="dim")
            return
        outcome = await self._tarn.explain(arg)
        if not outcome.value.results:
            _out.print("[yellow]no results — ingest some documents first[/]")
        else:
            _out.print(trace_view(outcome.value))
        _render_report(outcome.report)

    async def _do_ask(self, arg: str) -> None:
        if not arg:
            _out.print("usage: ask <query>", style="dim")
            return
        outcome = await self._tarn.ask(arg)
        result = outcome.value
        if result.abstained:
            _out.print(Panel(escape(result.answer), title="abstained", border_style="yellow"))
        else:
            border = "green" if result.grounded else "yellow"
            label = "grounded" if result.grounded else "not fully grounded"
            _out.print(
                Panel(escape(result.answer), title=f"answer  ([{border}]{label}[/])", border_style=border)
            )
            tree = Tree("[bold]proof[/]")
            for step in result.proof:
                mark = "[green]✓[/]" if step.grounded else "[red]✗[/]"
                node = tree.add(f"{mark} {escape(step.claim)}")
                for c in step.citations:
                    node.add(f"[dim]cite[/] {escape(_cite(c))}")
            _out.print(tree)
        _render_report(outcome.report)


def results_table(
    results: list[RetrievalResult], *, prev_order: list[str] | None = None, title: str | None = None
) -> Table:
    """A ranked-results table: ``#``, the fused ``score``, one column per **component score** present
    (dense / sparse / cross_encoder / …), the document, and a passage snippet. When ``prev_order`` (the
    chunk-id order of the previous stage) is given, a ``Δ`` column shows how each row *moved* — what
    merging or reranking did. Shared by the ``retrieve`` table and ``explain``'s per-stage tables, and
    reusable by the example runner."""
    components: list[str] = []
    for r in results:
        for key in r.component_scores:
            if key not in components:
                components.append(key)
    table = Table(
        title=title, show_edge=False, header_style="bold", title_justify="left", title_style="green"
    )
    table.add_column("#", justify="right", style="dim")
    if prev_order is not None:
        table.add_column("Δ", justify="center")
    table.add_column("score", justify="right")
    for key in components:
        table.add_column(key, justify="right", style="dim")
    table.add_column("document", style="cyan")
    table.add_column("passage")
    for rank, r in enumerate(results, 1):
        row = [str(rank)]
        if prev_order is not None:
            row.append(_movement(r.chunk_id, rank - 1, prev_order))
        row.append(f"{r.score:.3f}")
        row += [f"{r.component_scores[k]:.3f}" if k in r.component_scores else "" for k in components]
        row += [escape(r.document_id), escape(_snippet(r.text, 60))]
        table.add_row(*row)
    return table


def _movement(chunk_id: str, index: int, prev_order: list[str]) -> str:
    """How a result moved from the previous stage: ``▲n`` up, ``▼n`` down, ``·`` unchanged, ``＋`` newly
    appeared (e.g. an auto-merged section parent that wasn't a retrieved leaf)."""
    if chunk_id not in prev_order:
        return "[green]＋[/]"
    delta = prev_order.index(chunk_id) - index
    if delta > 0:
        return f"[green]▲{delta}[/]"
    if delta < 0:
        return f"[red]▼{-delta}[/]"
    return "[dim]·[/]"


def _candidates_table(rc: RetrieverCandidates, by_id: dict[str, RetrievalResult]) -> Table:
    """One retriever's pre-fusion candidates — rank, the raw engine score (distance for dense, BM25 for
    sparse), and the document/passage (resolved from the hydrated stage results when available)."""
    table = Table(
        title=rc.key, show_edge=False, header_style="dim", title_justify="left", title_style="cyan"
    )
    table.add_column("rank", justify="right", style="dim")
    table.add_column("raw", justify="right")
    table.add_column("document", style="cyan")
    table.add_column("passage")
    for c in rc.candidates:
        hit = by_id.get(c.chunk_id)
        document = escape(hit.document_id) if hit else escape(c.chunk_id)
        passage = escape(_snippet(hit.text, 50)) if hit else "[dim]—[/]"
        table.add_row(str(c.rank), f"{c.raw_score:.3f}", document, passage)
    return table


def trace_view(trace: SearchTrace) -> Group:
    """Render a :class:`~tarnrag.retrieval.types.SearchTrace` as the full explain breakdown: the query
    (and routing decision), each retriever's candidates before fusion, then the ranking at every pipeline
    stage (``fused`` → ``merged`` → ``reranked`` → ``final``) with component scores and the inter-stage
    movement. Pure rendering — it returns a renderable; the caller prints it (UI layer, not the facade)."""
    by_id = {r.chunk_id: r for stage in trace.stages for r in stage.results}
    parts: list = [Text.assemble(("query: ", "bold"), trace.query.text)]
    if trace.routing is not None:
        query_type, route = trace.routing
        parts.append(Text(f"routed: query_type={query_type or '∅'} → route {route!r}", style="magenta"))
    if trace.per_retriever:
        parts.append(Text("\nretrievers (before fusion)", style="bold"))
        parts += [_candidates_table(rc, by_id) for rc in trace.per_retriever]
    parts.append(Text("\npipeline stages", style="bold"))
    prev_order: list[str] | None = None
    for stage in trace.stages:
        parts.append(results_table(stage.results, prev_order=prev_order, title=stage.name))
        prev_order = [r.chunk_id for r in stage.results]
    return Group(*parts)


def _render_report(report: Report) -> None:
    """Show the issues a facade call reported — nothing when the report is empty (all went well)."""
    for issue in report.issues:
        color = "red" if issue.severity is Severity.ERROR else "yellow"
        subject = f"{escape(issue.subject)}: " if issue.subject else ""
        _out.print(f"[{color}]{issue.severity.value}:[/] {subject}{escape(issue.message)}")


def _snippet(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _cite(citation) -> str:
    return " > ".join(citation.header_path) or citation.locator or citation.document_id


async def _main(config_path: str) -> None:
    async with TarnRag(config_path) as tarn:
        await Console(tarn).run()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m tarnrag.console <config.json>")
    asyncio.run(_main(argv[0]))


if __name__ == "__main__":
    main()
