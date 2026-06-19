"""LLM backends — Anthropic (Claude) via the official SDK, behind ``LanguageModel.create``.

``AnthropicLanguageModel`` is a ``LanguageModel`` (a ``Resource``, like the embedders) selected via
``LLMSettings.provider`` by ``LanguageModel.create``. The official ``anthropic`` SDK is imported **lazily**
(the ``generation`` extra), and the one network call goes through a single stubbable seam — ``_create`` —
so request-shaping and response-parsing are unit-tested by injecting a fake client, without the SDK or a
key (the same honest gating as the API embedders' ``_post``). New providers add a backend here + an entry
in ``LanguageModel.create``'s provider map.
"""

from __future__ import annotations

import os
from typing import Any

from tarnrag.core.resources.llm import Completion, LanguageModel, Prompt


class AnthropicLanguageModel(LanguageModel):
    """Claude via the Anthropic Messages API. Maps a ``Prompt`` (system + user + knobs) to one
    ``messages.create`` call and concatenates the text blocks of the reply."""

    PROVIDER = "anthropic"
    API_KEY_ENV = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
        client: Any = None,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.timeout = timeout
        self._client = client  # injected in tests; built lazily from the SDK otherwise

    def identity(self) -> str:
        return f"{self.PROVIDER}:{self.model}"

    async def complete(self, prompt: Prompt) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": prompt.max_tokens or self.default_max_tokens,
            "temperature": prompt.temperature,
            "messages": [{"role": "user", "content": prompt.user}],
        }
        if prompt.system is not None:
            kwargs["system"] = prompt.system
        resp = await self._create(**kwargs)
        return Completion(
            text=self._text_of(resp),
            stop_reason=getattr(resp, "stop_reason", None),
            usage=self._usage_of(resp),
        )

    # ---------------- the one stubbable seam ----------------

    async def _create(self, **kwargs: Any) -> Any:
        """The single network call; tests stub this to avoid the SDK + network."""
        return await self._sdk().messages.create(**kwargs)

    def _sdk(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "AnthropicLanguageModel needs the anthropic SDK — install the 'generation' extra "
                    "(pip install '.[generation]')"
                ) from e
            opts: dict[str, Any] = {"api_key": self._key(), "timeout": self.timeout}
            if self._base_url:
                opts["base_url"] = self._base_url
            self._client = anthropic.AsyncAnthropic(**opts)
        return self._client

    def _key(self) -> str:
        key = self._api_key or os.environ.get(self.API_KEY_ENV, "")
        if not key:
            raise ValueError(
                f"{self.PROVIDER} LLM needs an API key — set LLM__API_KEY or {self.API_KEY_ENV}"
            )
        return key

    @staticmethod
    def _text_of(resp: Any) -> str:
        """Concatenate the ``text`` of the reply's text blocks (the SDK returns a list of content blocks;
        tool/other blocks are ignored). Tolerates object- or dict-shaped blocks."""
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "".join(parts)

    @staticmethod
    def _usage_of(resp: Any) -> dict[str, int]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }
