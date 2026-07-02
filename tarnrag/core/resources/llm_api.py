"""LLM backends behind ``LanguageModel.create`` — Anthropic (Claude, official SDK) and any
OpenAI-compatible ``/chat/completions`` endpoint (httpx).

Each is a ``LanguageModel`` (a ``Resource``, like the embedders) selected via ``LLMSettings.provider``. The
heavy client is imported **lazily** (the ``generation`` / ``openai`` extras), and the one network call goes
through a single stubbable seam — ``_create`` (Anthropic) / ``_post`` (OpenAI-compatible) — so request-shaping
and response-parsing are unit-tested by stubbing that seam, without the SDK or a key (the same honest gating
as the API embedders' ``_post``). New providers add a backend here + an entry in ``LanguageModel.create``'s
provider map.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from tarnrag.core.resources.llm import Completion, LanguageModel, Prompt


def _sampling(prompt: Prompt, default_max_tokens: int, default_temperature: float) -> tuple[int, float]:
    """The effective ``(max_tokens, temperature)`` for one request — a **three-level fallback**, first
    set wins:

    1. the ``Prompt``'s explicit value — for a caller that truly needs a per-call override;
    2. the backend's configured default — ``LLMSettings.max_tokens`` / ``.temperature``
       (``LLM__MAX_TOKENS`` / ``LLM__TEMPERATURE``), threaded in by ``LanguageModel.create``;
    3. the library fallback (1024 / 0.0) — the ``LLMSettings`` field defaults (mirrored by the backend
       constructor defaults for direct construction).

    ``None`` — not falsiness — is the "unset" sentinel, so an explicit ``temperature=0.0`` still wins over
    a configured 0.7. Shared by both backends so neither drifts back to pinning the ``Prompt`` value
    unconditionally (the bug where the configured sampling settings were silently ignored)."""
    return (
        prompt.max_tokens if prompt.max_tokens is not None else default_max_tokens,
        prompt.temperature if prompt.temperature is not None else default_temperature,
    )


class AnthropicLanguageModel(LanguageModel):
    """Claude via the Anthropic Messages API. Maps a ``Prompt`` (system + user + knobs) to one
    ``messages.create`` call and concatenates the text blocks of the reply. Transient-failure retries are
    delegated to the SDK (``max_retries`` — exponential backoff, ``Retry-After`` honored), configured in
    ``_sdk_options`` for parity with the OpenAI backend's ``_post_retrying``."""

    PROVIDER = "anthropic"
    API_KEY_ENV = "ANTHROPIC_API_KEY"
    # SDK-managed retry attempts (429 / 5xx / connection errors, Retry-After honored) — the SDK's default
    # of 2 is too few to ride out sustained rate-limiting on a batch/eval run; parity with
    # OpenAILanguageModel.RETRY_ATTEMPTS.
    MAX_RETRIES = 8

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
        max_tokens, temperature = _sampling(prompt, self.default_max_tokens, self.default_temperature)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
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
            self._client = anthropic.AsyncAnthropic(**self._sdk_options())
        return self._client

    def _sdk_options(self) -> dict[str, Any]:
        """The ``AsyncAnthropic`` constructor options — extracted so the retry/timeout wiring is testable
        without the SDK installed. ``max_retries`` delegates transient-failure handling to the SDK
        (exponential backoff, ``Retry-After`` honored) — the Anthropic counterpart of the OpenAI backend's
        ``_post_retrying``."""
        opts: dict[str, Any] = {
            "api_key": self._key(),
            "timeout": self.timeout,
            "max_retries": self.MAX_RETRIES,
        }
        if self._base_url:
            opts["base_url"] = self._base_url
        return opts

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


class OpenAILanguageModel(LanguageModel):
    """Any OpenAI-compatible ``/chat/completions`` endpoint via httpx — OpenAI itself, or a self-/3rd-party
    hosted server (vLLM, Together, Groq, OpenRouter, …) selected with ``api_base_url`` (this is how you point
    the reader at a Llama-3.3-70B endpoint to match a MOTHRAG-style protocol). Maps a ``Prompt`` to one chat
    request through the single async ``_post`` seam; the bearer key is omitted when none is set (keyless
    local servers). ``httpx`` is imported lazily (the ``openai`` extra)."""

    PROVIDER = "openai"
    API_KEY_ENV = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})  # transient: rate-limit / overloaded / 5xx
    RETRY_ATTEMPTS = 8  # enough to ride out sustained rate-limiting (429s) on a batch run
    RETRY_MAX_DELAY = 30.0  # cap per-wait (exponential backoff or a server Retry-After)

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
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.timeout = timeout
        self._client = client  # injected in tests; built lazily (httpx.AsyncClient) otherwise

    def identity(self) -> str:
        return f"{self.PROVIDER}:{self.model}"

    async def complete(self, prompt: Prompt) -> Completion:
        messages: list[dict[str, str]] = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        max_tokens, temperature = _sampling(prompt, self.default_max_tokens, self.default_temperature)
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = await self._post_retrying(f"{self._base_url}/chat/completions", body, self._headers())
        choice = (data.get("choices") or [{}])[0]
        return Completion(
            text=(choice.get("message") or {}).get("content") or "",
            stop_reason=choice.get("finish_reason"),
            usage=self._usage_of(data.get("usage")),
        )

    async def _post_retrying(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """``_post`` with backoff on transient failures (429 / 5xx / timeouts). Over a long batch (e.g. the
        eval sweep — thousands of calls) a transient error is near-certain, so retry rather than lose the
        run. On a 429 the server's ``Retry-After`` is honored (wait exactly as told, not a guess); otherwise
        exponential backoff (capped). Client errors (4xx) and the final attempt propagate."""
        import httpx

        delay = 1.0
        for attempt in range(self.RETRY_ATTEMPTS):
            last = attempt + 1 == self.RETRY_ATTEMPTS
            try:
                return await self._post(url, body, headers)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in self.RETRY_STATUS or last:
                    raise
                wait = self._retry_after(e.response, delay)
            except (httpx.TimeoutException, httpx.TransportError):
                if last:
                    raise
                wait = delay
            await asyncio.sleep(wait)
            delay = min(delay * 2, self.RETRY_MAX_DELAY)
        raise RuntimeError("unreachable: the retry loop returns or raises")  # pragma: no cover

    @classmethod
    def _retry_after(cls, response: Any, default: float) -> float:
        """The server's ``Retry-After`` (seconds), capped at ``RETRY_MAX_DELAY`` — honored on 429 so we wait
        exactly as told instead of guessing; falls back to the exponential ``default`` when absent/unparseable."""
        raw = response.headers.get("retry-after")
        if raw:
            try:
                return min(float(raw), cls.RETRY_MAX_DELAY)
            except ValueError:
                pass
        return default

    # ---------------- the one stubbable seam ----------------

    async def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """The single network call; tests stub this to avoid httpx + the network."""
        resp = await self._http().post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "the openai LLM backend needs httpx — install the 'openai' extra (pip install '.[openai]')"
                ) from e
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        """JSON headers, with a bearer key when one is configured (env or settings); omitted for keyless
        local endpoints (e.g. a vLLM server)."""
        headers = {"Content-Type": "application/json"}
        key = self._api_key or os.environ.get(self.API_KEY_ENV, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @staticmethod
    def _usage_of(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
