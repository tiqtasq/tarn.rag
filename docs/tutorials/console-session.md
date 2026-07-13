# A console session

The interactive console is the fastest end-to-end tour of tarn.rag: ingest files, inspect the
corpus, query it, and get grounded answers — all from a `tarn>` prompt, no code.

## Setup

The console ships with the `console` extra (included in `all`):

```bash
pip install "tarn-rag[all]"          # or minimally: "tarn-rag[onnx,parsers,console]"
python scripts/fetch_model.py        # the local ONNX embedding model (once)
export ANTHROPIC_API_KEY=sk-ant-…    # only needed for the `ask` command
```

The console also loads a `.env` (searched from the current directory upward) at startup, so the key
can live there instead.

## Launch

The console takes exactly one argument — a config file (a `Settings` document; the repo ships a
sample). There is no `--help` flag; the argument is always treated as the config path.

```bash
tarnrag examples/console.config.json     # or: python -m tarnrag.console <config>
```

You should see the banner and prompt:

```
tarn.rag console  sqlite:///./console.db
Type help, or quit to exit.  (↑/↓ for history)

tarn>
```

The config file is authoritative; the ambient environment only supplies what it omits (like the
API key).

## Work the corpus

Ingest the sample corpus that ships with the repo — a directory ingests the files in it:

```
tarn> ingest examples/docs/corpus-1/
```

Document ids are the filename stems, so re-ingesting a file replaces it. Check what landed:

```
tarn> docs        # each document: id, chunk count, embedding count
tarn> status      # corpus rollup — counts + document-length stats
```

Now query. `retrieve` is retrieval only — the ranked passages with their provenance:

```
tarn> retrieve how do I inspect a tank?
```

`explain` runs the same retrieval but shows its inner workings: each retriever's candidates before
fusion, and the ranking at every pipeline stage with component scores:

```
tarn> explain how do I inspect a tank?
```

`ask` adds the generation layer — the grounded answer plus its proof tree and evidence:

```
tarn> ask how do I inspect a tank?
```

Clean up and leave:

```
tarn> delete <id>
tarn> quit        # Ctrl-D / Ctrl-C also exit
```

The full command set is in the [console command reference](../reference/console-commands.md).

## Where to go next

The console is purely a UI over the `TarnRag` facade — everything it does is three method calls in
the [getting-started tutorial](getting-started.md). When the defaults stop being enough, the
[retrieval-pipeline how-to](../how-to/swap-the-retrieval-pipeline.md) shows what to put in the
config's `components` section.
