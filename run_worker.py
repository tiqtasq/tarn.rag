"""Worker process entry — the consumer side for ``MODE='distributed'``.

Runs the ingestion consume loop. Start multiple instances for parallelism — pgQueuer owns
claiming / SKIP LOCKED / retries. Requires ``MODE=distributed`` (with the Postgres
``DATABASE__QUEUE_URL`` set); in ``embedded`` mode ingestion runs in-process and there is no
separate worker.

    python run_worker.py
"""

from __future__ import annotations

import asyncio
import logging

from app.domains.ingestion import run_worker

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    asyncio.run(run_worker())
