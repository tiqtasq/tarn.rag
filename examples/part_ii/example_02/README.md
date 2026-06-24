# Example 02 · Hybrid retrieval (dense + sparse + RRF)

One change from the base config: the `retrieval_pipeline` now runs a **dense** *and* a **sparse (BM25)**
retriever and fuses them with **Reciprocal Rank Fusion**. That single delta is the lesson —
`diff examples/part_ii/example_00/config.yaml examples/part_ii/example_02/config.yaml`:

```yaml
  retrieval_pipeline:
    retrievers:
      - class_name: dense
      - class_name: sparse        # ← added
    fuser: { class_name: rrf }    # ← was identity
```

It reads the **same base store** (no re-ingest).

## What it shows

Track the two probes from Example 01 — they swap:

| Probe | Query | Example 01 (dense) | Example 02 (hybrid) |
|-------|-------|--------------------|---------------------|
| opaque part number | `XQ-9920-A` | ❌ MISS (rank 5) | ✅ **HIT** (rank 1) — sparse's exact match, fused in |
| paraphrase | "service a rotary fluid machine…" | ✅ HIT (rank 3) | ❌ **MISS** (rank 6+) — sparse boosts surface-word matches, demoting the semantic answer |

Hybrid fixes exact-term queries but **isn't free**: RRF blends in sparse's ranking, so a pure-semantic
query can get demoted by lexical noise. The `explain` breakdown shows it — for the paraphrase, the sparse
column lights up on the wrong documents and pulls them above the right one.

## Run it

```bash
python -m examples.part_ii.example_02.run
```

…or `explain` both queries over `example_02/config.yaml` in the console.

→ Next: **Example 03** adds a **cross-encoder reranker** that re-scores the fused candidates by true
query–passage relevance — recovering the paraphrase that hybrid demoted, while keeping the part-number win.
So: paraphrase → dense ✓ → hybrid ✗ → rerank ✓; part number → dense ✗ → hybrid ✓ → rerank ✓.
