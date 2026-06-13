"""Worker process entry — the consumer side for ``MODE='distributed'``.

Builds an :class:`IngestionEngine` from settings and runs its consumer loop. Run multiple
instances for parallelism — pgQueuer owns claiming / SKIP LOCKED / retries. Requires
``MODE=distributed`` (with ``QUEUE_DB_URL`` set); in ``embedded`` mode ingestion runs
in-process and there is no separate worker.

    python run_worker.py
"""

from __future__ import annotations

import asyncio
import logging

from app.domains.ingestion.engine import IngestionEngine

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    engine = await IngestionEngine.create()  # MODE (and wiring) come from Settings
    await engine.run_worker()


if __name__ == "__main__":
    asyncio.run(main())
