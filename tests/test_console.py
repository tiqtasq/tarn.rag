"""The interactive console: drive its command methods directly (the REPL loop is a thin wrapper).

Gated on the ONNX model being fetched (ingest + retrieve need the embedder); the LLM is injected, so no
API key is needed. Mirrors tests/test_embedder.py's skip pattern.
"""

import json
from pathlib import Path

import pytest

from examples.common import MODEL_DIR, corpus
from tarnrag.console import Console, load_settings
from tarnrag.core.engine.config import Settings
from tarnrag.core.resources.llm import StaticLanguageModel

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


async def test_console_ingest_retrieve_ask_delete(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    doc_paths = sorted(str(p) for p in corpus("corpus-1").glob("*.txt"))

    async with Console(_settings(tmp_path), llm=_canned_llm()) as console:
        # ingest — the document id is each file's stem
        statuses = await console.ingest(doc_paths)
        assert len(statuses) == 3 and all(s.status == "complete" for s in statuses)
        assert {s.document_id for s in statuses} == {"pump-maintenance", "quokka", "tank-inspection"}
        assert len(await console.docs()) == 3

        # retrieve — the pump question surfaces the pump doc
        hits = await console.retrieve("how should I service a pump")
        assert hits and any(h.document_id == "pump-maintenance" for h in hits)

        # ask — an answerable question is grounded; an off-topic one abstains (the cascade + policy)
        answered = await console.ask("How should I service a pump before restarting it?")
        assert not answered.abstained and answered.grounded and answered.answer
        off_topic = await console.ask("What is the capital of France?")
        assert off_topic.abstained

        # re-ingest one file (same stem -> upsert, still 3 docs) + delete one
        again = await console.ingest([doc_paths[0]])
        assert len(again) == 1 and len(await console.docs()) == 3
        assert await console.delete("quokka") is True
        assert len(await console.docs()) == 2


def test_load_settings_reads_json(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"EMBEDDING_DIMENSION": 384, "database": {"document_url": "sqlite:///x.db"}}))
    settings = load_settings(cfg)
    assert settings.EMBEDDING_DIMENSION == 384 and "x.db" in settings.database.document_url
