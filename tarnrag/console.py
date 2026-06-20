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

from rich.console import Console as RichConsole
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from tarnrag.report import Report, Severity
from tarnrag.tarnrag import TarnRag

_out = RichConsole()

_COMMANDS = [
    ("ingest <path> ...", "ingest (or re-ingest) files; a directory ingests the files in it"),
    ("docs", "list the ingested documents"),
    ("delete <id>", "delete a document and everything derived from it"),
    ("retrieve <query>", "retrieval only — the ranked passages"),
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
            table = Table(show_edge=False, header_style="bold")
            table.add_column("#", justify="right", style="dim")
            table.add_column("score", justify="right")
            table.add_column("document", style="cyan")
            table.add_column("passage")
            for rank, r in enumerate(outcome.value, 1):
                table.add_row(str(rank), f"{r.score:.3f}", r.document_id, escape(_snippet(r.text)))
            _out.print(table)
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
