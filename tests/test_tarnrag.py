"""The TarnRag facade — drive its high-level methods directly; plus a smoke check that the rich Console
renders over it without touching the engines.

Gated on the ONNX model being fetched (ingest + retrieve need the embedder); the LLM is injected, so no
API key is needed. Mirrors tests/test_embedder.py's skip pattern.
"""

import json
from pathlib import Path

import pytest

from examples.common import MODEL_DIR, corpus
from tarnrag.core.engine.config import Settings
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.report import Severity
from tarnrag.tarnrag import TarnRag

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(), reason="model not fetched (scripts/fetch_model.py)"
)


def _settings(tmp_path: Path) -> Settings:
    """The sample console config, pointed at a temp store + the fetched model."""
    config = json.loads(Path("examples/console.config.json").read_text())
    config["database"]["document_url"] = f"sqlite:///{tmp_path}/console.db"
    config["embedding"]["model_dir"] = str(MODEL_DIR)
    return Settings(_env_file=None, **config)


def _canned_llm() -> StaticLanguageModel:
    """Routes reasoner replies on the question line; grounding prompts (the fact-checker) say grounded."""

    def _reply(prompt):
        if prompt.system and "fact-checker" in prompt.system:
            return json.dumps({"verdicts": [True, True, True, True]})
        question = prompt.user.split("\n", 1)[0].lower()
        if "pump" in question:
            return json.dumps(
                {"answer": "Check the mechanical seal and bearing lubrication.",
                 "steps": [{"claim": "service the mechanical seal and bearing lubrication", "cited": [1]}]}
            )
        if "capital" in question:
            return json.dumps({"answer": "Paris.", "steps": [{"claim": "the capital of france is paris", "cited": [1]}]})
        return json.dumps({"answer": "I don't know.", "steps": []})

    return StaticLanguageModel(_reply)


async def test_tarnrag_ingest_retrieve_ask_delete(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    doc_paths = sorted(str(p) for p in corpus("corpus-1").glob("*.txt"))

    async with TarnRag(_settings(tmp_path), llm=_canned_llm()) as tarn:
        # ingest — the document id is each file's stem; a clean run reports no issues
        ingested = await tarn.ingest(doc_paths)
        assert ingested.report.ok
        statuses = ingested.value
        assert len(statuses) == 3 and all(s.status == "complete" for s in statuses)
        assert {s.document_id for s in statuses} == {"pump-maintenance", "quokka", "tank-inspection"}
        assert len((await tarn.docs()).value) == 3

        # retrieve — the pump question surfaces the pump doc
        hits = (await tarn.retrieve("how should I service a pump")).value
        assert hits and any(h.document_id == "pump-maintenance" for h in hits)

        # ask — an answerable question is grounded; an off-topic one abstains (the cascade + policy)
        answered = (await tarn.ask("How should I service a pump before restarting it?")).value
        assert not answered.abstained and answered.grounded and answered.answer
        off_topic = (await tarn.ask("What is the capital of France?")).value
        assert off_topic.abstained

        # re-ingest one file (same stem -> upsert, still 3 docs) + delete one
        again = await tarn.ingest([doc_paths[0]])
        assert len(again.value) == 1 and len((await tarn.docs()).value) == 3
        assert (await tarn.delete("quokka")).value is True
        assert len((await tarn.docs()).value) == 2


async def test_ingest_reports_missing_paths_instead_of_skipping(tmp_path):
    """The core report contract: a path that matches nothing is surfaced in the report, never dropped
    silently — while the files that do exist still ingest."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    real = sorted(str(p) for p in corpus("corpus-1").glob("*.txt"))[0]

    async with TarnRag(_settings(tmp_path), llm=_canned_llm()) as tarn:
        outcome = await tarn.ingest(["does/not/exist.txt", real])
        # the real file ingested...
        assert len(outcome.value) == 1 and outcome.value[0].status == "complete"
        # ...and the missing path is reported (warning), not skipped silently
        assert not outcome.report.ok
        assert [i.subject for i in outcome.report.issues] == ["does/not/exist.txt"]
        assert outcome.report.issues[0].severity is Severity.WARNING

        # all-missing -> nothing ingested, but the issues are still reported
        none = await tarn.ingest(["nope-a.txt", "nope-b.txt"])
        assert none.value == []
        assert {i.subject for i in none.report.issues} == {"nope-a.txt", "nope-b.txt"}


async def test_open_injects_one_shared_repository_and_embedder(tmp_path):
    """P1 — TarnRag is the composition root: it builds the repository + embedder once and injects the
    *same* objects into both engines (no duplicate embedder, no reaching into a sub-engine)."""
    async with TarnRag(_settings(tmp_path)) as tarn:
        # one repository, built by TarnRag and injected into both engines
        assert tarn._repository is tarn._ingestion.repository
        assert tarn._repository is tarn._retrieval.repository
        # one embedder, injected into retrieval (not rebuilt at the facade level)
        assert tarn._embedder is tarn._retrieval.embedder
        # close() releases the store TarnRag owns
        assert tarn._repository is not None


async def test_console_renders_over_the_facade(tmp_path):
    """The rich Console is a thin UI over TarnRag: its handlers run end to end (no engine access of their
    own) and stay alive when a command errors."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    pytest.importorskip("rich")
    from tarnrag.console import Console

    async with TarnRag(_settings(tmp_path), llm=_canned_llm()) as tarn:
        console = Console(tarn)
        # a missing path mixed in -> the report-rendering path runs too
        files = " ".join(str(p) for p in corpus("corpus-1").glob("*.txt"))
        await console._do_ingest(f"does/not/exist.txt {files}")
        await console._do_docs("")
        await console._do_retrieve("how should I service a pump")
        await console._do_ask("How should I service a pump before restarting it?")
        await console._do_ask("What is the capital of France?")  # the abstain branch
        await console._do_delete("quokka")
        await console._do_delete("nope")  # missing id -> handled, not raised


def test_tarnrag_loads_settings_from_a_path(tmp_path):
    """Constructing with a path (str / Path) loads the JSON config into ``settings`` — no engines opened."""
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"EMBEDDING_DIMENSION": 384, "database": {"document_url": "sqlite:///x.db"}}))
    settings = TarnRag(cfg).settings
    assert settings.EMBEDDING_DIMENSION == 384 and "x.db" in settings.database.document_url
