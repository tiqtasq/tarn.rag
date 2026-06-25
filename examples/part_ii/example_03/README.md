# Example 03 · Cross-encoder reranking

One change from Example 02: a **cross-encoder `reranker`** on the `retrieval_pipeline` (plus the `rerank`
model block it needs). `diff examples/part_ii/example_02/config.yaml examples/part_ii/example_03/config.yaml`:

```yaml
  retrieval_pipeline:
    ...
    fuser: { class_name: rrf }
    reranker: { class_name: cross_encoder }   # ← added
```

It reads the **same base store** (no re-ingest), and needs the reranker model:

```bash
python scripts/fetch_model.py --repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2
```

## What it shows

A bi-encoder (dense) embeds the query and passage *separately*; a **cross-encoder** scores the `(query,
passage)` pair *together*, so it judges relevance far better — but only over the handful of fused
candidates (too slow for the whole corpus). Re-scoring those candidates gets the **best of both**:

| query | dense (01) | hybrid (02) | **rerank (03)** |
|-------|------------|-------------|------------------|
| paraphrase "rotary fluid machine…" | ✅ | ❌ | ✅ **recovered** |
| `XQ-9920-A` (part number) | ❌ | ✅ | ✅ **kept** |

But there's a new limitation: the reranker ranks individual **chunks**. Ask *"all the steps in the
compressor startup procedure?"* and the single best chunk holds only the first half of the steps — the
rest live in a sibling chunk. The answer is **fragmented**.

## Run it (script)

```bash
python -m examples.part_ii.example_03.run
```

## Run it interactively (console)

Start the console on this example's config (run from the repo root; the cross-encoder loads lazily on the
first query):

```bash
python -m tarnrag.console examples/part_ii/example_03/config.yaml
```

This reads the **base store** Example 00 built. If you haven't built it yet, ingest the corpus once (it's
idempotent — re-ingesting just refreshes it):

```text
tarn> ingest examples/docs/corpus-2
```

Each `explain` below prints the full breakdown — the per-retriever candidates, then the `fused` →
**`reranked`** → `final` stages with scores and a `Δ` (movement) column, then a details table.

**1 · the paraphrase the reranker recovers.** Type:

```text
tarn> explain service a rotary fluid machine before powering it up
```

Look at the **`reranked`** stage's `Δ` column: `pump-maintenance` jumps **up** (`▲`) from where RRF had
buried it — the cross-encoder recognizes it as the real answer and pulls it into the top few. (For
contrast, run the very same query under Example 02's config —
`python -m tarnrag.console examples/part_ii/example_02/config.yaml` — which has *no* reranker, and
`pump-maintenance` sits far down the list.)

**2 · the part-number win, kept.** Type:

```text
tarn> explain XQ-9920-A
```

`pump-maintenance` is rank **1** — sparse's exact match on the part number, carried straight through the
rerank.

**3 · the new problem: fragmentation.** Type:

```text
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
```

The top result is a short (~300-char) `compressor-startup` chunk. Read its `passage`: it has the first
steps (set the valves, pre-lubricate) but **not** the rest — no "loading sequence". `recursive` chunking
sliced the section, so the single best chunk is only a fragment. Example 04 fixes this.

Type `quit` (or Ctrl-D) to exit.

→ Next: **Example 04** re-ingests the corpus with the **structure-aware** chunker (which builds a section
tree) and adds **auto-merge** — so when several sibling chunks of one section are retrieved, they're
consolidated into the whole coherent section, and the full procedure comes back as one passage.
