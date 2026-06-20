# tarnrag examples

Runnable, self-contained examples that teach the `tarnrag` architecture one piece at a time.

- **Part I** (`part_i/`) — minimal examples over **SQLite / embedded mode** (zero infra). Each
  shows one aspect of the pipeline: ingestion + retrieval, chunking, id policy, enrichment, …
- **Part II** (`part_ii/`) — production concerns: deployment, evaluation, fine-tuning *(coming)*.

## Setup (once, from the repo root)

```bash
pip install -e ".[onnx]"       # the tarnrag package + the ONNX embedding runtime
python scripts/fetch_model.py  # download the embedding model + tokenizer (needs network)
```

## Running an example

Examples are a Python package; run them as modules **from the repo root**:

```bash
python -m examples.part_i.example_01.ingestion   # ingest the sample corpus into a local SQLite store
python -m examples.part_i.example_01.retrieval   # query that store

python -m examples.part_i.example_02.ingestion   # same, but the pipeline is configured from pipeline.json
python -m examples.part_i.example_02.retrieval   # query the JSON-configured store

python -m examples.part_i.example_03.ingestion   # index the corpus with a small-chunk pipeline
python -m examples.part_i.example_03.evaluation  # compare retrieval methods on a labeled set (eval harness)
```

Running with `-m` puts the repo root on the import path, so `examples` and `tarnrag` both resolve —
no `PYTHONPATH` or `sys.path` tweaks. (Directory names use underscores because they are module names.)

## Interactive console

For an interactive session instead of scripts, start the REPL with one JSON config:

```bash
python -m tarnrag.console examples/console.config.json
```

Then `ingest <path>` (or a directory; re-ingesting a file replaces it), `docs`, `delete <id>`,
`retrieve <query>` (passages only), and `ask <query>` (retrieval + generation — answer + proof tree; needs
`ANTHROPIC_API_KEY`). The config is a `Settings` document: the database, embedding, and llm settings plus
the retrieval/generation pipeline specs under `components`.

The console is just a UI; the work lives in `tarnrag.TarnRag`, a facade over the three engines that you
can drive directly in your own code. Each call returns an `Outcome` — its `value` plus a `report` of any
non-fatal issues (empty when all went well), so nothing is printed or silently skipped:

```python
from tarnrag import TarnRag

async with TarnRag("examples/console.config.json") as tarn:
    ingested = await tarn.ingest(["examples/docs/corpus-1", "typo.txt"])
    for issue in ingested.report.issues:
        print(f"{issue.severity.value}: {issue.subject}: {issue.message}")  # warning: typo.txt: not found
    answer = await tarn.ask("How should I service a pump before restarting it?")
    print(answer.value.answer)
```

## Layout

```
examples/
├── common.py            # shared helpers used by every example
├── docs/                # sample corpora, ingested by path
│   └── corpus-1/        # add more as docs/corpus-2/, docs/<name>/ …
├── part_i/              # SQLite teaching examples
│   ├── example_01/      # minimal ingestion + retrieval (the default pipeline)
│   │   ├── ingestion.py
│   │   └── retrieval.py
│   ├── example_02/      # the ingestion pipeline configured from JSON
│   │   ├── pipeline.json
│   │   ├── ingestion.py
│   │   └── retrieval.py
│   └── example_03/      # comparing retrieval methods with the eval harness
│       ├── evalset.json
│       ├── ingestion.py
│       └── evaluation.py
└── part_ii/             # production / eval / fine-tuning (coming)
```

## Conventions (`common.py`)

The incidental setup every example repeats lives in `examples/common.py`, so each example file keeps
only the part it is teaching:

| Helper | What it gives you |
|--------|-------------------|
| `base_settings(db_path, **overrides)` | Default embedded/SQLite `Settings`; override just the knob you are demonstrating, e.g. `base_settings(db, chunking=ChunkingSettings(size=128))`. |
| `example_db(__file__)` | A `rag_docs.db` next to the example — each example gets its **own** store, so they run independently and in any order. |
| `corpus("corpus-1")` | Path to a named corpus under `docs/`. |
| `require_model()` | Friendly error if the embedding model has not been fetched. |

An example's ingestion and retrieval **must** build from the same `db_path` + `embedding` config —
retrieval validates an embedding *fingerprint* before opening the index. Reusing
`base_settings(example_db(__file__))` on both sides guarantees it.

## Adding things

- **A corpus:** drop files in `examples/docs/<name>/` and select it with `corpus("<name>")`.
- **An example:** create `examples/part_i/example_NN/` with an `__init__.py` and your script(s), and
  import what you need from `examples.common`.

Generated stores (`*.db`) and `__pycache__/` are gitignored.
