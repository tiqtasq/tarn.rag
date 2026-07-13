# S1 scope — query understanding as the shipped default

Scoping for roadmap item **S1** (`doc/strategic-roadmap.md`, sequencing step 4). Companion to
`doc/phases.md` (the measurement log this work will extend). Ground rules carried over: one lever
per PR, measured before/after with the in-repo harnesses.

**What S1 delivers:** a documented "quality" retrieval profile built on
`routing_retrieval_pipeline` + the structural classifier — lexical queries → a sparse-weighted
hybrid, semantic queries → hybrid + LLM query expansion (when an LLM is configured) — with a HyDE
retriever added to the sweep before the profile's contents are finalized. The selling point stays
what the roadmap named: the routing decision is visible in the trace (`SearchTrace.routing` +
the classifier's annotations), which most stacks can't show.

A load-bearing property discovered while scoping: **every S1 lever is query-time only.** Nothing
rides the embedding fingerprint, so the entire sweep runs against one index — no re-ingest for any
leg. That makes S1 cheap to measure relative to its expected value.

---

## Inventory — already built, no work needed

| piece | where | state |
|---|---|---|
| `RoutingRetrievalPipeline` | `retrieval/pipeline/router.py` | classify → dispatch; routes are recursive `Searcher` specs; records `trace.routing`; caller-supplied `query_type` wins |
| `StructuralQueryClassifier` | `retrieval/components/classifier.py` | `lexical`/`semantic` by query *form*; shares `looks_like_identifier` with the P3 sparse-query builder, so what it labels lexical is exactly what FTS phrase-requires |
| `MultiQueryRetriever` | `retrieval/components/retriever.py` | LLM expansion → per-variant dense KNN → RRF; degrades to plain dense with no `ctx.llm` |
| LLM injection | `retrieval/engine/engine.py` | `LanguageModel.create(settings.llm)` already lands in `RetrievalContext.llm` (lazy — no key/network unless a bridge component calls it) |
| Sweep + segmentation | `eval/harness.py` (`sweep`, `by_query_type`), `eval/layout.py` | the P4 methodology: TAT-QA source-hit@10, n=334, every leg's spec pinned |
| Profile delivery | `README.md` § "The quality profile" | the P4 precedent: profiles are documented YAML specs, not code |

## Gap 1 — the HyDE retriever *(PR-1, new component)*

A fourth member of the `Retriever` family, `class_name: hyde`: ask `ctx.llm` for a short
*hypothetical answer* to the query, embed that text, and dense-KNN it — retrieving by
answer-to-passage similarity instead of question-to-passage similarity.

Design decisions:

- **Embed the hypothesis with `embed_passages`, not `embed_query`.** The hypothetical text is a
  fake *document*, so it belongs in the passage space. Free on symmetric gte-small (no prefixes);
  correct on prefixed/asymmetric models (bge-class), where using the query prefix would be a bug.
- **Always include the original query** (embedded as a query), RRF-fused with the hypothesis
  list(s) — same shape as `MultiQueryRetriever._fuse`, same deterministic tie-break
  (score desc, chunk_id asc). HyDE-only retrieval is brittle when the LLM hallucinates off-domain;
  fusing with the literal query bounds the downside.
- **Graceful degradation** mirroring `multi_query`: no `ctx.llm`, or an unusable completion ⇒
  plain dense retrieval over the original query. Transport errors still propagate (also
  `multi_query` parity) — a bad API key should fail loud, not silently degrade retrieval.
- **Config:** `num_hypotheses: int = 1` (>1 samples several hypotheses and fuses all lists),
  `rrf_k: int = 60`, `max_words: int` cap folded into the prompt. Costs `num_hypotheses` LLM
  call(s) per query — opt-in, never in the lean default.

Size: ~100–150 lines in `retriever.py` + unit tests (mirror `tests/retrieval/test_bridge.py`:
stub LLM, stub store; assert passage-space embedding, fusion with the original query, no-LLM
degradation, LLM-garbage degradation).

## Gap 2 — weighted RRF *(PR-2, tiny)*

The roadmap's "lexical → **sparse-weighted** hybrid" has no mechanism today: `RRFFuser` sums
`1/(k + rank)` uniformly. Add an optional per-retriever weight:

```yaml
fuser: {class_name: rrf, weights: {sparse: 2.0}}   # score = Σ weight_r / (k + rank)
```

- Keys are the retriever keys the pipeline already produces (`_retriever_keys()`: configured
  `name`, else `class_name`, `#n`-deduped); unlisted retrievers default to `1.0`, so every
  existing spec stays byte-identical (the drift-guard from Option 4 must not move).
- Alternative considered and rejected: routing lexical → sparse-*only*. Hybrid was never lost on
  any measured segment (`doc/phases.md`, Options 1–2); keep the dense arm, weight it down.

Size: ~15 lines in `fuser.py` + tests (weighted ordering, default-weight backward compatibility).

## The sweep *(PR-3 — decides what the profile contains)*

Datasets: **TAT-QA** source-hit@10 (n=334, the P4 setup) + the **Wikipedia pool benchmark**, both
segmented with `by_query_type` so the lexical/semantic split is visible per leg. Legs, each spec
pinned:

1. shipped hybrid (baseline) — and hybrid + CE reranker (the P4 profile) as the quality baseline
2. hybrid with `multi_query` replacing the dense arm (global, unrouted)
3. hybrid with `hyde` replacing the dense arm (global, unrouted)
4. sparse-weighted hybrid (weights swept coarsely, e.g. sparse ∈ {1.5, 2, 3}) — judged on the
   lexical-classified slice
5. the routed composition: structural classifier; lexical → the winning weighted hybrid;
   semantic → the winner of legs 2/3; default → shipped hybrid
6. leg 5 + the CE reranker (P4 composes orthogonally; confirm, don't assume)

Cautions from the measured history:

- The ±0.02–0.03 nondeterminism band (`doc/phases.md`, P1 PR-2) applies doubly to LLM legs —
  repeat runs before believing small deltas.
- **HyDE and multi_query are overlapping recall levers** (both rewrite the query). The sweep picks
  one for the semantic route; stacking both is a separate, later question and needs its own leg if
  ever proposed.
- Multi-hop/pool composition (sub-questions through hybrid, pooled-evidence rerank) is **P7**, not
  S1 — the pool benchmark is measured here only to catch regressions, not chased.

## Draft profile (to be confirmed by the sweep)

```yaml
components:
  retrieval_pipeline:
    class_name: routing_retrieval_pipeline
    classifier: {class_name: structural}
    routes:
      lexical:
        class_name: retrieval_pipeline
        retrievers: [{class_name: dense}, {class_name: sparse}]
        fuser: {class_name: rrf, weights: {sparse: 2.0}}
      semantic:
        class_name: retrieval_pipeline
        retrievers: [{class_name: hyde}, {class_name: sparse}]   # or multi_query — sweep decides
        fuser: {class_name: rrf}
    default:
      class_name: retrieval_pipeline
      retrievers: [{class_name: dense}, {class_name: sparse}]
      fuser: {class_name: rrf}
```

The lean default (`config.py`) does not change; the profile ships as README documentation next to
the P4 quality profile, with the reranker line added per leg-6's verdict.

## PR breakdown

| PR | content | measurement |
|---|---|---|
| PR-1 | `HydeRetriever` + tests | none (component alone; no default touched) |
| PR-2 | `RRFFuser.weights` + tests | none (default weight 1.0 ⇒ byte-identical) |
| PR-3 | the sweep, the `phases.md` entry, the README profile | TAT-QA + pool, segmented, legs 1–6 |

## Out of scope for S1

- P7 (multi-hop pool composition) — deliberately separate, same roadmap tier.
- Intent-based routing (PP-1's taxonomy) — the router accepts any classifier, but calibrating
  six intent routes is its own measured effort; S1 routes on the structural pair only.
- An LLM query classifier — the structural heuristic is deterministic and C++-portable; keep it.
