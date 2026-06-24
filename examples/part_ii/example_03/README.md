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

## Run it

```bash
python -m examples.part_ii.example_03.run
```

→ Next: **Example 04** re-ingests the corpus with the **structure-aware** chunker (which builds a section
tree) and adds **auto-merge** — so when several sibling chunks of one section are retrieved, they're
consolidated into the whole coherent section, and the full procedure comes back as one passage.
