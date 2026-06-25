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

## Run it (script)

```bash
python -m examples.part_ii.example_02.run
```

## Run it interactively (console)

Start the console on this example's config (run from the repo root):

```bash
python -m tarnrag.console examples/part_ii/example_02/config.yaml
```

This reads the **base store** Example 00 built. If you haven't built it yet, ingest the corpus once (it's
idempotent):

```text
tarn> ingest examples/docs/corpus-2
```

Each `explain` now shows **two** per-retriever candidate tables — `dense` *and* `sparse` — then the
`fused` → `final` stages (no reranker yet). That second retriever is the whole lesson; watch how RRF
combines the two rankings.

**1 · the opaque part number, now found.** Type:

```text
tarn> explain XQ-9920-A
```

In the per-retriever tables, the **`sparse`** retriever ranks `pump-maintenance` **#1** (an exact BM25 match
on the part number) while **`dense`** has it down at ~5 — there's no meaning to embed. RRF fuses the two, so
`pump-maintenance` lands at rank **1** in the `final` table. (Under Example 01's dense-only config this
query missed entirely.)

**2 · the paraphrase, now demoted.** Type:

```text
tarn> explain service a rotary fluid machine before powering it up
```

Here it's the opposite: **`dense`** ranks `pump-maintenance` **#3** (it gets the paraphrase), but **`sparse`**
ranks it **#12** — no shared words — and instead boosts surface-word matches (`compressor-startup`,
`valve-maintenance`, …). RRF blends them, so `pump-maintenance` slips to rank **6** in the `final` table. The
lexical noise demoted the semantic answer — fusion isn't free. Example 03's reranker undoes exactly this.

Type `quit` (or Ctrl-D) to exit.

→ Next: **Example 03** adds a **cross-encoder reranker** that re-scores the fused candidates by true
query–passage relevance — recovering the paraphrase that hybrid demoted, while keeping the part-number win.
So: paraphrase → dense ✓ → hybrid ✗ → rerank ✓; part number → dense ✗ → hybrid ✓ → rerank ✓.
