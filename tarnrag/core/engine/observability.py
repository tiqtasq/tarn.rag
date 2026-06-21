"""Observability interface (metrics + logging) and a no-op implementation.

Core logic must work with ``observability=None`` (every call site guards ``self.obs``);
``NoOpObservability`` is the dev/test stand-in when you want a non-None instance with no
effect. Real adapters (Prometheus, structured logging) are future work and live behind this
ABC — the domain never imports them. Obs is held by the **worker** (compute metrics) and the
**orchestrator** (lifecycle: enqueue / status / persistence); stages stay pure (D6).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tarnrag.core.engine.config import ObservabilitySettings


class Observability(ABC):
    """
    Metrics + logging port. ``counter``/``gauge`` are sync (cheap); ``log`` is async.
    """

    @staticmethod
    def create(settings: ObservabilitySettings) -> Observability | None:
        """Build the configured observability adapter, or ``None`` when disabled (core logic guards every
        ``self.obs`` call, so ``None`` = off). The adapter is selected by ``settings.type``; only
        ``NoOpObservability`` ships today, so an *enabled* observability currently installs the no-op until
        a real adapter (Prometheus, structured logging) is registered here — by design, not an oversight."""
        if not settings.enabled:
            return None
        # When real adapters land, dispatch on settings.type here; the no-op is the only one for now.
        return NoOpObservability()

    @abstractmethod
    async def log(self, level: str, message: str, **context: Any) -> None:
        """Log a message with structured context. Level: debug | info | warning | error."""

    @abstractmethod
    def counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""

    @abstractmethod
    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        """Time the wrapped block; records elapsed seconds as ``{name}.seconds`` (gauge).
        Concrete (built on ``gauge``) so every adapter — and the no-op — gets it for free."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.gauge(f"{name}.seconds", time.perf_counter() - start, tags=tags)


class NoOpObservability(Observability):
    """
    No-op implementation — for development/testing (or when obs is enabled but no real
    adapter is configured yet). All methods do nothing.
    """

    async def log(self, level: str, message: str, **context: Any) -> None:
        pass

    def counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        pass

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        pass
