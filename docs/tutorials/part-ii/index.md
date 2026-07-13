# Part II — the failure-driven ladder

Where [Part I](../part-i/index.md) teaches the machinery on a corpus small enough to verify by eye,
Part II drives that machinery until it **breaks** — on purpose.

Every rung of the ladder adds exactly **one** config knob, and shows two queries: one that now works,
and one that **fails**. The next rung's config repairs that failure — and usually introduces a new
one. The corpus (`examples/docs/corpus-2/`) is engineered so each failure is reproducible rather than
anecdotal.

The point is not that any one config is correct. It is that every retrieval decision is a *trade*, and
the only way to know what you traded away is to look at the query it broke.

## The ladder

Runnable programs live in [`examples/part_ii/`](../../../examples/part_ii); each carries a README with
a full console walkthrough of its good and bad cases.

| # | Adds | Fixes | Breaks |
|---|------|-------|--------|
| [00](../../../examples/part_ii/example_00/README.md) | Ingest the shared corpus | — | *(the base store every later rung reads)* |
| [01](../../../examples/part_ii/example_01/README.md) | Dense-only retrieval | Paraphrases | Exact part numbers (`XQ-9920-A`) — nothing to embed |
| [02](../../../examples/part_ii/example_02/README.md) | Hybrid (dense + sparse + RRF) | The part number | The paraphrase — lexical noise demotes it |
| [03](../../../examples/part_ii/example_03/README.md) | Cross-encoder reranking | The paraphrase, *keeping* the part number | Cost, latency |
| [04](../../../examples/part_ii/example_04/README.md) | Structure-aware chunking + auto-merge | Answers split across chunk boundaries | — |
| [05](../../../examples/part_ii/example_05/README.md) | Query routing *(Act A capstone)* | Per-query-type method selection | — |
| [06](../../../examples/part_ii/example_06/README.md) | Minimal generation *(Act B opener)* | Passages → an answer | Ungrounded claims |
| [07](../../../examples/part_ii/example_07/README.md) | Grounding check + abstain | Confident wrong answers | — |

## How it is built

Three conventions, chosen deliberately (see `examples/part_ii/PLAN.md`):

- **The console is the hero.** Each example *is* a `Settings` config you can run interactively:
  `python -m tarnrag.console examples/part_ii/example_02/config.yaml`. The `run.py` script narrates
  the good/bad cases, but the config is the artifact.
- **Configs are YAML, fully explicit, and heavily commented.** Nothing relies on a Python default —
  every stage and component parameter is enumerated. The acceptance test is: *if a library default
  changes, these examples still produce the same results.* (Part I tutorial 3 shows why that matters:
  its numbers survived the day hybrid retrieval became the default, precisely because it named every
  spec in full.)
- **Ingest once.** Retrieval and generation rungs are pure config swaps over the one store example 00
  built. Only chunking, enrichment, or extractor changes force a re-ingest, and those rungs say so.

## Status

The prose tutorials for Part II have not been written yet — the per-example READMEs linked above are
the current documentation, and they are detailed: each walks through the console session, the score
breakdown from `explain`, and exactly which rank moved and why.

Start with [example 00](../../../examples/part_ii/example_00/README.md) to build the store, then walk
the ladder in order. The failures only make sense in sequence.

---

[← Tutorials](../index.md) · [← Part I](../part-i/index.md)
