"""The LanguageModel seam: the StaticLanguageModel fake, the Anthropic backend (stubbed), and the selector.

The Anthropic path is exercised without the SDK or a key by stubbing the one ``_create`` seam — the same
honest gating the API embedders use for ``_post``.
"""

import types

import pytest

from tarnrag.core.engine.config import LLMSettings
from tarnrag.core.resources.llm import LanguageModel, Prompt, StaticLanguageModel
from tarnrag.core.resources.llm_api import AnthropicLanguageModel


def _fake_response(text, *, stop="end_turn", in_tok=10, out_tok=3):
    """A stand-in for the SDK's Message: text content blocks + stop reason + usage."""
    block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return types.SimpleNamespace(content=[block], stop_reason=stop, usage=usage)


async def test_static_language_model_canned_and_callable():
    assert (await StaticLanguageModel("hi").complete(Prompt(user="x"))).text == "hi"
    out = await StaticLanguageModel(lambda p: p.user.upper()).complete(Prompt(user="abc"))
    assert out.text == "ABC" and out.stop_reason == "static"
    assert StaticLanguageModel(name="r").identity() == "static:r"


async def test_anthropic_shapes_request_and_parses_reply():
    lm = AnthropicLanguageModel(model="claude-sonnet-4-6", api_key="k")
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("the answer")

    lm._create = fake_create
    out = await lm.complete(Prompt(user="hi", system="sys", max_tokens=42))

    assert out.text == "the answer" and out.stop_reason == "end_turn"
    assert out.usage == {"input_tokens": 10, "output_tokens": 3}
    assert captured["model"] == "claude-sonnet-4-6" and captured["max_tokens"] == 42
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert lm.identity() == "anthropic:claude-sonnet-4-6"


async def test_anthropic_omits_system_when_absent():
    lm = AnthropicLanguageModel(model="m", api_key="k")
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("x")

    lm._create = fake_create
    await lm.complete(Prompt(user="hi"))  # no system instruction
    assert "system" not in captured


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        AnthropicLanguageModel(model="m")._key()


def test_create_selects_anthropic():
    lm = LanguageModel.create(LLMSettings(api_key="k"))
    assert isinstance(lm, AnthropicLanguageModel) and lm.identity() == "anthropic:claude-sonnet-4-6"


def test_create_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        LanguageModel.create(types.SimpleNamespace(provider="bogus"))  # bypass the Literal validation
