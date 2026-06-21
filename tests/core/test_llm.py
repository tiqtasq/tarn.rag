"""The LanguageModel seam: the StaticLanguageModel fake, the Anthropic backend (stubbed), and the selector.

The Anthropic path is exercised without the SDK or a key by stubbing the one ``_create`` seam — the same
honest gating the API embedders use for ``_post``.
"""

import types

import pytest

from tarnrag.core.engine.config import LLMSettings
from tarnrag.core.resources.llm import LanguageModel, Prompt, StaticLanguageModel
from tarnrag.core.resources.llm_api import AnthropicLanguageModel, OpenAILanguageModel


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


async def test_openai_shapes_request_and_parses_reply():
    lm = OpenAILanguageModel(model="meta-llama/Llama-3.3-70B-Instruct", api_key="k", base_url="https://host/v1")
    captured = {}

    async def fake_post(url, body, headers):
        captured.update(url=url, body=body, headers=headers)
        return {
            "choices": [{"message": {"content": "the answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    lm._post = fake_post
    out = await lm.complete(Prompt(user="hi", system="sys", max_tokens=42))

    assert out.text == "the answer" and out.stop_reason == "stop"
    assert out.usage == {"input_tokens": 10, "output_tokens": 3}
    assert captured["url"] == "https://host/v1/chat/completions"
    assert captured["body"]["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert captured["body"]["max_tokens"] == 42
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert lm.identity() == "openai:meta-llama/Llama-3.3-70B-Instruct"


async def test_openai_omits_system_and_auth_when_absent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    lm = OpenAILanguageModel(model="m", base_url="http://localhost:8000/v1")  # keyless local endpoint
    captured = {}

    async def fake_post(url, body, headers):
        captured.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": "x"}}]}

    lm._post = fake_post
    await lm.complete(Prompt(user="hi"))  # no system instruction, no key
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "Authorization" not in captured["headers"]  # keyless local server


def test_openai_defaults_base_url():
    assert OpenAILanguageModel(model="gpt-4o-mini")._base_url == "https://api.openai.com/v1"


async def _no_sleep(*_a, **_k):
    pass


async def test_openai_retries_transient_errors(monkeypatch):
    import httpx

    monkeypatch.setattr("asyncio.sleep", _no_sleep)  # no real backoff wait
    lm = OpenAILanguageModel(model="m", api_key="k")
    calls = {"n": 0}

    async def flaky_post(url, body, headers):
        calls["n"] += 1
        if calls["n"] < 3:  # 429 twice, then succeed
            req = httpx.Request("POST", url)
            raise httpx.HTTPStatusError("rate limited", request=req, response=httpx.Response(429, request=req))
        return {"choices": [{"message": {"content": "ok"}}]}

    lm._post = flaky_post
    out = await lm.complete(Prompt(user="hi"))
    assert out.text == "ok" and calls["n"] == 3


async def test_openai_does_not_retry_client_errors(monkeypatch):
    import httpx

    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    lm = OpenAILanguageModel(model="m", api_key="k")
    calls = {"n": 0}

    async def bad_post(url, body, headers):
        calls["n"] += 1
        req = httpx.Request("POST", url)
        raise httpx.HTTPStatusError("bad request", request=req, response=httpx.Response(400, request=req))

    lm._post = bad_post
    with pytest.raises(httpx.HTTPStatusError):
        await lm.complete(Prompt(user="hi"))
    assert calls["n"] == 1  # 4xx is not retried


def test_create_selects_anthropic():
    lm = LanguageModel.create(LLMSettings(api_key="k"))
    assert isinstance(lm, AnthropicLanguageModel) and lm.identity() == "anthropic:claude-sonnet-4-6"


def test_create_selects_openai():
    lm = LanguageModel.create(
        LLMSettings(provider="openai", model="meta-llama/Llama-3.3-70B-Instruct", api_base_url="http://h/v1")
    )
    assert isinstance(lm, OpenAILanguageModel)
    assert lm.model == "meta-llama/Llama-3.3-70B-Instruct" and lm._base_url == "http://h/v1"


def test_create_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        LanguageModel.create(types.SimpleNamespace(provider="bogus"))  # bypass the Literal validation
