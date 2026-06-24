# Example 00 · Ingest the shared corpus

**Setup step.** Builds the `base` store — the SQLite index every later example queries. Ingest once
here; the retrieval and generation steps then only swap their *config* over this same store.

## What it shows

This builds the **complete baseline** — ingestion **and** the dense retrieval and `single_hop` generation
pipelines — so every console command (`ingest` / `explain` / `ask`) is fully specified, not left to a
default. Later steps vary one pipeline at a time from this baseline.

The ingestion DAG, made fully explicit in [`config.yaml`](config.yaml):

```
each .md file → LoadAndParse (markdown) → CleanAndNormalize → Chunk (recursive 400/80) → Embed → store
```

`corpus-2`'s 12 Markdown documents become ~19 overlapping chunks, each embedded with `gte-small` into one
SQLite file (`examples/part_ii/store/base.db`). The `recursive` chunker makes a **flat** set of chunks (no
section-parent tree) — the right default for the first retrieval examples; the structure-aware chunker that
builds parents shows up only at the auto-merge step.

## Run it

Interactively (the hero path — the config *is* the example):

```bash
python -m tarnrag.console examples/part_ii/example_00/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> status
tarn> explain how do I prevent a pump from cavitating?
tarn> ask how should I service a centrifugal pump before restarting it?
```

Or as a script (ingests, then prints one `explain` breakdown):

```bash
python -m examples.part_ii.example_00.run
```

Both need the embedding model: `python scripts/fetch_model.py` (once).

`ask` additionally needs the generation reader (configured in `config.yaml`): install the LLM backend
(`pip install '.[openai]'`) and put your OpenAI key in the **`OPENAI_LLM_KEY`** env var (the config names
that var, so the key never lives in a file). Retrieval/`explain` need neither — the LLM is built lazily on
the first `ask`.

## The "no defaults" rule

Every value in `config.yaml` is set explicitly — nothing relies on a library default. So the file alone
tells you exactly what the pipeline does, and its results won't drift if a default ever changes. Each
later example's config is likewise self-contained; the **diff** between two configs is that step's lesson.

→ Next: **Example 01** queries this store with dense-only retrieval — and meets its first failure.
