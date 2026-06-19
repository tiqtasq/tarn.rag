"""An interactive console for tarn.rag — ingest, query (retrieval), and ask (retrieval + generation).

Start it with one JSON config:

    python -m tarnrag.console path/to/config.json

The config is a ``Settings`` document — ``database`` / ``embedding`` / ``llm`` plus the
ingestion / retrieval / generation pipeline specs under ``components`` (a sample is at
``examples/console.config.json``). The API key is read from the environment (``ANTHROPIC_API_KEY``), not
the config. Then type commands at the ``tarn>`` prompt::

    ingest <path> [<path> ...]   ingest (or RE-ingest) files; a directory ingests the files in it.
                                 The document id is the filename stem, so re-ingesting a file replaces it.
    docs                         list the ingested documents (id, chunks, embeddings)
    delete <id>                  delete a document and everything derived from it
    retrieve <query>             retrieval only — the ranked passages
    ask <query>                  retrieval + generation — the grounded answer + its proof tree
    help                         show this list
    quit                         exit

The command methods (``ingest`` / ``retrieve`` / ``ask`` / ``docs`` / ``delete``) return data and are
unit-tested directly; ``run()`` is the thin parse-and-print REPL on top of them.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from tarnrag import DocumentStatus, DocumentSummary, IngestionEngine, RetrievalEngine
from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import GENERATION_PIPELINE, Settings
from tarnrag.core.resources.embedder import build_embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation import GenerationEngine, GenerationPipeline, GenerationResult

_HELP = """commands:
  ingest <path> [<path> ...]   ingest (or re-ingest) files; a directory ingests the files in it
  docs                         list the ingested documents
  delete <id>                  delete a document and everything derived from it
  retrieve <query>             retrieval only — the ranked passages
  ask <query>                  retrieval + generation — the grounded answer + its proof tree
  help                         show this list
  quit                         exit"""


def load_settings(config_path: str | Path) -> Settings:
    """Build ``Settings`` from a JSON config file (ambient ``.env`` ignored — the file is authoritative;
    OS env vars still supplement, e.g. the LLM key)."""
    config = json.loads(Path(config_path).read_text())
    return Settings(_env_file=None, **config)


class Console:
    """The tarn.rag session: an ingestion engine + a retrieval engine over one shared repository, and a
    lazily-built generation engine. Construct with a ``Settings``; ``open()`` before use, ``close()`` after
    (or use it as an async context manager). ``llm`` may be injected (tests / a custom backend)."""

    def __init__(self, settings: Settings, llm: LanguageModel | None = None) -> None:
        self.settings = settings
        self._injected_llm = llm
        self._ingestion: IngestionEngine | None = None
        self._retrieval: RetrievalEngine | None = None
        self._generation: GenerationEngine | None = None

    async def open(self) -> Console:
        """Build the engines. The ingestion engine owns the repository (and stamps the index metadata);
        the retrieval engine *shares* that repository — one connection, and it sees ingests live."""
        self._ingestion = await IngestionEngine.create(self.settings)
        embedder = build_embedder(self.settings.embedding, self.settings.EMBEDDING_DIMENSION)
        self._retrieval = RetrievalEngine(self._ingestion.repository, embedder, self.settings)
        return self

    async def close(self) -> None:
        if self._ingestion is not None:
            await self._ingestion.aclose()  # owns the shared repository; retrieval just borrows it

    async def __aenter__(self) -> Console:
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ---------------- commands (return data; the REPL prints it) ----------------

    async def ingest(self, paths: list[str]) -> list[DocumentStatus]:
        """Ingest (or re-ingest) the given files / directories. The document id is each file's stem, so
        re-ingesting the same file upserts in place (under ``ID_POLICY='caller'``)."""
        files = _expand(paths)
        if not files:
            return []
        source_ids = [p.stem for p in files] if self.settings.ID_POLICY == "caller" else None
        doc_ids = await self._ingestion.ingest_paths([str(p) for p in files], source_ids=source_ids)
        return [s for s in [await self._ingestion.status(d) for d in doc_ids] if s is not None]

    async def docs(self) -> list[DocumentSummary]:
        return await self._ingestion.list_documents()

    async def delete(self, document_id: str) -> bool:
        return await self._ingestion.delete_document(document_id)

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        return await self._retrieval.search_text(query, top_k=top_k)

    async def ask(self, query: str) -> GenerationResult:
        """Retrieval + generation: a grounded answer with a proof tree. Needs an LLM."""
        return await self._gen().answer_text(query)

    def _gen(self) -> GenerationEngine:
        """The generation engine, built once over the shared retrieval engine + the configured LLM."""
        if self._generation is None:
            llm = self._injected_llm or self._build_llm()
            spec = self.settings.components.get(GENERATION_PIPELINE) or {"class_name": "generation_pipeline"}
            pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
            self._generation = GenerationEngine(self._retrieval, llm, pipeline)
        return self._generation

    def _build_llm(self) -> LanguageModel:
        if not (self.settings.llm.api_key or os.environ.get("ANTHROPIC_API_KEY")):
            raise RuntimeError(
                "generation needs an LLM key — set ANTHROPIC_API_KEY (the provider/model are in the config)"
            )
        return LanguageModel.create(self.settings.llm)

    # ---------------- the REPL ----------------

    async def run(self) -> None:
        """Read commands at the ``tarn>`` prompt until EOF / quit."""
        print(f"tarn.rag console — {self.settings.database.document_url}.  Type 'help', or 'quit' to exit.")
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
                line = (await asyncio.to_thread(input, "tarn> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            command, _, arg = line.partition(" ")
            command = command.lower()
            if command in ("quit", "exit", "q"):
                break
            handler = handlers.get(command)
            if handler is None:
                print(f"unknown command {command!r} — type 'help'")
                continue
            try:
                await handler(arg.strip())
            except Exception as exc:  # keep the session alive on any command error
                print(f"error: {exc}")
        print("bye")

    async def _do_help(self, _arg: str) -> None:
        print(_HELP)

    async def _do_ingest(self, arg: str) -> None:
        if not arg:
            print("usage: ingest <path> [<path> ...]")
            return
        statuses = await self.ingest(arg.split())
        for s in statuses:
            print(f"  {s.document_id:20}  {s.status:9}  chunks={s.chunk_count}  embeddings={s.embedding_count}")
        print(f"({len(statuses)} document(s))")

    async def _do_docs(self, _arg: str) -> None:
        summaries = await self.docs()
        for d in summaries:
            print(f"  {d.document_id:20}  chunks={d.chunk_count}  embeddings={d.embedding_count}")
        print(f"({len(summaries)} document(s))")

    async def _do_delete(self, arg: str) -> None:
        if not arg:
            print("usage: delete <document-id>")
            return
        print("deleted" if await self.delete(arg) else f"no such document: {arg}")

    async def _do_retrieve(self, arg: str) -> None:
        if not arg:
            print("usage: retrieve <query>")
            return
        results = await self.retrieve(arg)
        if not results:
            print("  (no results — ingest some documents first)")
        for rank, r in enumerate(results, 1):
            print(f"  {rank}. [{r.score:.3f}] {r.document_id}: {_snippet(r.text)}")

    async def _do_ask(self, arg: str) -> None:
        if not arg:
            print("usage: ask <query>")
            return
        result = await self.ask(arg)
        if result.abstained:
            print(f"  [abstained] {result.answer}")
            return
        print(f"  {result.answer}    (grounded={result.grounded})")
        for step in result.proof:
            mark = "grounded" if step.grounded else "UNSUPPORTED"
            cites = ", ".join(_cite(c) for c in step.citations) or "(no citations)"
            print(f"    - [{mark}] {step.claim}  <- {cites}")


def _expand(paths: list[str]) -> list[Path]:
    """Resolve each path to file(s): a directory contributes the files it directly contains."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            print(f"  skipping (not found): {raw}")
    return files


def _snippet(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _cite(citation) -> str:
    return " > ".join(citation.header_path) or citation.locator or citation.document_id


async def _main(config_path: str) -> None:
    async with Console(load_settings(config_path)) as console:
        await console.run()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m tarnrag.console <config.json>")
    asyncio.run(_main(argv[0]))


if __name__ == "__main__":
    main()
