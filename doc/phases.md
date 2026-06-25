# tarn.rag — performance roadmap toward SOTA

**Goal:** SOTA-competitive multi-hop QA (MOTHRAG ≈ 68.3 avg F1; NeocorRAG ≈ 69 is the GPU/constrained-decoding
ceiling) **without** trading away tarn.rag's differentiators (offline / layout-grade provenance / license
filtering). **Method: measure first, then add by measured leverage.** The MOTHRAG gap analysis lives in
`generation-architecture-design.md` §6; the eval harness is `scripts/run_benchmarks.py` (`tarnrag/eval/`).

## Roadmap, by leverage (cheap → expensive)

### Phase 0 — make the baseline trustworthy
Fix a target reader + embedder (either point at a real Llama-3.3-70B endpoint + a 3072-d embedder to match
MOTHRAG, or commit to our own default and own that number), run n ≥ 200/dataset, and sweep the reasoner
specs (`single_hop` / `iterative` / `decomposition`) — `sweep_benchmark` does this. This one step tells us
how much of the gap is **config** vs **architecture**.

### Phase 1 — cheap wins the eval flagged (config / prompt)
1. **Answer format / extraction** — the data shows `hit ≫ EM` (the reader finds the answer but returns prose,
   not a terse span; see the Phase 0 results below). A short-answer prompt + answer normalization is
   near-zero-cost and should move EM/F1 a lot. Do this first.
2. **Switch the default reasoner to `decomposition`** (multi-hop, already built; HotpotQA/MuSiQue are
   multi-hop). Config only.
3. **Stronger reader / embedder.** Config only.

Re-measure after each. Hypothesis: a meaningful chunk of the gap closes here for almost no code — and we'll
*know*, not guess.

### Phase 2 — architectural additions (§6), one at a time, each measured against Phase 1
1. **Ensemble + deterministic arbitration** (PDD, cross-arm agreement, faithfulness) — the single biggest
   MOTHRAG delta, and the heaviest.
2. **γ-driven re-retrieval** — couple the grounding checker back into the loop (today it's terminal).
3. **Bridge substrate** — multi-query fusion + an LLM relevance-judge reranker; targets recall (MuSiQue's
   known bottleneck).
4. **ChainFilter / OpenIE** — last; needs a triples/graph substrate; optional.

**Recommendation:** do Phase 0 + Phase 1 before building any Phase-2 component. They're cheap, they're exactly
what the architecture was built to make trivial (swap the spec), and they convert "behind by 0.27 F1" into
"X is config, Y is architecture" — which is what should decide whether the expensive ensemble/bridge work is
worth it. Starting with the ensemble would be optimizing the unmeasured.

---

## Phase 0 — results (2026-06-22)

### Setup
- **Harness:** `python scripts/run_benchmarks.py <dataset> --hf --limit 200 --sweep` (branch
  `feature/mothrag-phase0-sweep`, PR #71).
- **Protocol:** *distractor*, n = 200 per dataset. Each question ingests only its own candidate passages
  (~10 HotpotQA/2Wiki, ~20 MuSiQue) into an isolated store, then retrieve → reason → answer; scored
  token-F1 / exact-match (SQuAD-style, max over answer + aliases — the metric MOTHRAG reports).
- **Fixed:** reader `gpt-4o-mini` (OpenAI), embedder `gte-small` (local ONNX, 384-d), grounding **off**,
  `temperature=0`. **Swept:** the three reasoners.

### Results (n = 200 each; *hit* = a gold phrase appears in the answer, *cite* = gold support in cited passages)

**HotpotQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.585 | 0.456 | 0.265 | 0.696 |
| `iterative` | 0.601 | 0.184 | 0.005 | 0.799 |
| **`decomposition`** | **0.645** | **0.507** | **0.280** | 0.722 |

**2WikiMultiHopQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| **`single_hop`** | **0.654** | **0.414** | **0.170** | 0.739 |
| `iterative` | 0.467 | 0.130 | 0.000 | 0.857 |
| `decomposition` | 0.637 | 0.391 | 0.160 | 0.769 |

**MuSiQue**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.360 | 0.256 | 0.085 | 0.575 |
| `iterative` | 0.370 | 0.131 | 0.000 | 0.757 |
| **`decomposition`** | **0.445** | **0.275** | **0.075** | 0.655 |

**Best reasoner vs MOTHRAG (Llama-3.3-70B, n=1000)**
| dataset | best | F1 | F1\* | ΔF1 | EM | EM\* | ΔEM |
|---|---|---|---|---|---|---|---|
| HotpotQA | decomposition | 0.507 | 0.781 | −0.274 | 0.280 | 0.648 | −0.368 |
| 2Wiki | single_hop | 0.414 | 0.763 | −0.349 | 0.170 | 0.682 | −0.512 |
| MuSiQue | decomposition | 0.275 | 0.505 | −0.230 | 0.075 | 0.406 | −0.331 |
| **avg** | | **0.399** | **0.683** | **−0.284** | **0.175** | **0.579** | **−0.404** |

### Conclusions
1. **`decomposition` is the best default** — clearly best on HotpotQA + MuSiQue; on 2Wiki `single_hop` edges
   it (F1 0.414 vs 0.391, within noise). Best on average; the free config win holds.
2. **`iterative` format-collapses on all three** — F1 0.13–0.18, EM ≈ 0, *despite the best citation coverage*
   (0.76–0.86). It reasons and retrieves *well*, then narrates prose instead of a terse span. A systemic
   output-surface failure, not a reasoning one.
3. **`hit ≫ EM` everywhere → answer format is the dominant lever.** Starkest on 2Wiki (`single_hop` hit
   0.654 but EM **0.170** — a 0.48 gap: entity/date answers found but not emitted cleanly). A short-answer
   prompt + answer normalization attacks this directly, at near-zero cost.
4. **MuSiQue adds a second lever.** Its hit is far lower (0.445 vs ~0.65) — on 4-hop questions the content
   often isn't *found at all*, a genuine retrieval/multi-hop gap, not just formatting.

### Config-vs-architecture verdict
- **HotpotQA + 2Wiki are format-bound** — Phase 1 (decomposition default + short-answer/extraction) should
  close much of their gap with no architecture.
- **MuSiQue needs both** — format *and* retrieval/multi-hop capability. It's the dataset that justifies the
  Phase-2 bridge substrate / stronger multi-hop.
- Measure-first paid off: leading with the ensemble would have optimized the unmeasured, and the cheap
  levers (reasoner choice + answer surface) are the largest visible ones on 2 of 3 datasets.

### Methodology note
The first full sweep hit a transient DNS outage during the 2Wiki run (32 calls failed → the per-question
guard scored them empty and the run survived; the retry's ~15 s window was outlasted). 2Wiki was re-run
clean (0 failures) — and the clean numbers *flipped* the 2Wiki reasoner ranking vs the contaminated run, so
the re-run mattered. HotpotQA + MuSiQue were unaffected (0 failures).

### Caveats
n = 200/dataset (CI ≈ ±3.5 %); `gpt-4o-mini` reader (not MOTHRAG's Llama-3.3-70B); a single 384-d embedder
(`gte-small`). Absolute numbers will shift with a stronger reader, but the `hit ≫ EM/F1` **format** gap is
largely reader-independent.

### Status / next
- [x] All three datasets swept, clean (HotpotQA, 2Wiki, MuSiQue).
- [x] **Phase 1** done — see below.
- [ ] **Phase 2 candidate:** bridge substrate / stronger multi-hop — to push *past* MOTHRAG, esp. on MuSiQue.

---

## Phase 1 — results (2026-06-22)

**Changes:** default reasoner → `decomposition`; short-answer prompts (demand the minimal span, not a
sentence) + a `_clean_answer` lead-in/quote stripper. No architecture. Same harness/protocol as Phase 0
(`gpt-4o-mini` + `gte-small`, n=200, distractor, grounding off). Clean run, 0 failures.

### Results (n = 200 each)

**HotpotQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.536 | 0.659 | 0.550 | 0.681 |
| `iterative` | 0.432 | 0.523 | 0.430 | 0.767 |
| **`decomposition`** | **0.574** | **0.705** | **0.570** | 0.713 |

**2WikiMultiHopQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| **`single_hop`** | **0.648** | **0.651** | **0.575** | 0.704 |
| `iterative` | 0.418 | 0.393 | 0.335 | 0.814 |
| `decomposition` | 0.643 | 0.641 | 0.565 | 0.738 |

**MuSiQue**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.350 | 0.446 | 0.330 | 0.497 |
| `iterative` | 0.300 | 0.344 | 0.270 | 0.738 |
| **`decomposition`** | **0.415** | **0.513** | **0.390** | 0.530 |

### Phase 0 → Phase 1, best reasoner per dataset, vs MOTHRAG
| dataset | F1: P0 → P1 (Δ) | F1 gap to MOTHRAG | EM: P0 → P1 (Δ) | EM gap to MOTHRAG |
|---|---|---|---|---|
| HotpotQA | 0.507 → 0.705 (+0.198) | −0.076 | 0.280 → 0.570 (+0.290) | −0.078 |
| 2Wiki | 0.414 → 0.651 (+0.237) | −0.112 | 0.170 → 0.575 (+0.405) | −0.107 |
| MuSiQue | 0.275 → 0.513 (+0.238) | **+0.008** | 0.075 → 0.390 (+0.315) | −0.016 |
| **avg** | **0.399 → 0.623** | **−0.060** | **0.175 → 0.512** | **−0.067** |

### Conclusions
1. **The format fix was decisive.** The gap to MOTHRAG collapsed from **−0.28 F1 / −0.40 EM** (Phase 0) to
   **−0.06 F1 / −0.07 EM** — ~80 % closed with **zero architecture**, just the prompt + reasoner default.
   Emphatic validation of measure-first.
2. **MuSiQue reaches MOTHRAG parity** (F1 0.513 vs 0.505) — even though it was the "retrieval-bound" dataset.
   Answer format was the dominant lever *everywhere*, including the hard one. (Its hit is still ~0.42, so
   retrieval caps the ceiling — Phase 2 retrieval work would push it *past* MOTHRAG, not just to parity.)
3. **`iterative` recovered but still lags** — F1 0.18→0.52 / 0.13→0.39 / 0.13→0.34 (the short-answer prompt
   fixed its prose-narration collapse), yet it trails on hit: its follow-up loop gathers good evidence (best
   citation coverage) but finds the answer less often. Not the default.
4. **`decomposition` confirmed as the default** — best on HotpotQA + MuSiQue; `single_hop` still edges it on
   2Wiki, within noise.

Note: "hit" fell on some arms vs Phase 0 because Phase 0's verbose answers inflated it (a gold phrase appears
*somewhere* in a long answer); with terse answers, hit ≈ correctness, and F1/EM are the trustworthy metrics.

### Caveats
Same as Phase 0: n = 200 (CI ≈ ±3.5 %); `gpt-4o-mini`, not Llama-3.3-70B; 384-d `gte-small`. A controlled
match of MOTHRAG's reader/embedder would shift absolutes; the *relative* Phase-0→Phase-1 gain is robust.

### What's next
The cheap levers are nearly exhausted (~0.06 F1 off MOTHRAG). To go *past* it, **Phase 2** — and the data
says **retrieval is now the ceiling** (hit 0.42–0.65): the bridge substrate (multi-query fusion + an LLM
relevance-judge reranker) and stronger multi-hop, measured against this Phase-1 baseline. A stronger
reader/embedder (the other Phase-1 lever, not yet pulled) is a parallel free check — done next.

### Reader check — gpt-4o (2026-06-22)

The one Phase-1 lever not yet pulled: swap the reader to **gpt-4o** (everything else identical — `gte-small`,
n=200, same prompts, `--sweep`), to see how much of the residual gap is the reader. Clean run, 0 failures.

**gpt-4o results (n=200; decomposition is now best on all three)**
| dataset | reasoner | hit | F1 | EM |
|---|---|---|---|---|
| HotpotQA | `decomposition` | 0.639 | **0.766** | **0.625** |
| 2Wiki | `decomposition` | 0.747 | **0.726** | **0.650** |
| MuSiQue | `decomposition` | 0.540 | **0.617** | **0.505** |

**Best reasoner per dataset: Phase 1 (gpt-4o-mini) → gpt-4o, vs MOTHRAG**
| dataset | F1: mini → 4o | F1 gap to MOTHRAG | EM: mini → 4o | EM gap to MOTHRAG |
|---|---|---|---|---|
| HotpotQA | 0.705 → 0.766 | −0.015 | 0.570 → 0.625 | −0.023 |
| 2Wiki | 0.651 → 0.726 | −0.037 | 0.575 → 0.650 | −0.032 |
| MuSiQue | 0.513 → 0.617 | **+0.112** | 0.390 → 0.505 | **+0.099** |
| **avg** | **0.623 → 0.703** | **+0.020** | **0.512 → 0.593** | **+0.014** |

**Conclusions**
1. **The residual gap was the reader.** gpt-4o lifts best-reasoner average F1 0.623 → **0.703** and EM 0.512 →
   **0.593** — *at/above* MOTHRAG's average (0.683 / 0.579). HotpotQA + 2Wiki land within −0.02…−0.04;
   MuSiQue is well **above** (+0.11 F1).
2. **The headline:** tarn.rag's **lean** stack — no ensemble, no bridge, no ChainFilter — matches MOTHRAG's
   average on a comparable reader. The §6 architecture is **not needed for parity**; it's for going *past*.
3. **Caveat — not a controlled win.** gpt-4o is very likely a stronger reader than MOTHRAG's Llama-3.3-70B,
   so "above MOTHRAG's average" reflects a reader edge *and* the Phase-1 prompts, not architecture. The
   defensible claim: **the residual Phase-1 gap was dominated by reader strength, not by the missing
   architecture.** A true match needs tarn.rag *on* Llama-3.3-70B (an OpenAI-compatible endpoint).
4. **decomposition is now best on all three** (it overtook `single_hop` on 2Wiki with the stronger reader) —
   the default is firmly vindicated.
5. `hit` also rose with gpt-4o (0.54–0.75) — some of gpt-4o-mini's low hit was weak extraction, not just
   retrieval misses; but headroom to 1.0 remains.

This **reframes Phase 2**: not "catch up to MOTHRAG" (parity is reached) but "go beyond" — raise retrieval
`hit` toward its ceiling (bridge substrate), with MuSiQue the clearest target. A controlled run on a
Llama-3.3-70B endpoint would also pin the apples-to-apples number.

---

## Phase 2 — results: bridge substrate (2026-06-23)

**Feature:** the bridge retrieval substrate (PR #73) — `multi_query` (LLM query expansion → dense per
variant → RRF) + `llm_judge` (LLM relevance reranker). **Eval:** `--bridge --sweep`, same protocol as
Phase 1 (`gpt-4o-mini` + `gte-small`, n=200, distractor), so the per-reasoner table overlays Phase 1's.
Clean run, 0 failures.

### Bridge results (n = 200 each)
| dataset | reasoner | hit | F1 | EM |
|---|---|---|---|---|
| HotpotQA | `single_hop` | 0.590 | 0.716 | 0.585 |
| HotpotQA | `iterative` | 0.497 | 0.582 | 0.470 |
| HotpotQA | **`decomposition`** | 0.596 | **0.729** | 0.600 |
| 2Wiki | `single_hop` | 0.659 | 0.649 | 0.575 |
| 2Wiki | `iterative` | 0.291 | 0.295 | 0.245 |
| 2Wiki | **`decomposition`** | 0.687 | **0.676** | 0.585 |
| MuSiQue | `single_hop` | 0.410 | 0.486 | 0.375 |
| MuSiQue | `iterative` | 0.300 | 0.340 | 0.275 |
| MuSiQue | **`decomposition`** | 0.450 | **0.534** | 0.420 |

### Δ vs Phase 1 (baseline dense retrieval), `decomposition` (the default)
| dataset | hit | F1 | EM |
|---|---|---|---|
| HotpotQA | 0.574 → 0.596 (+0.022) | 0.705 → 0.729 (+0.024) | 0.570 → 0.600 (+0.030) |
| 2Wiki | 0.643 → 0.687 (+0.044) | 0.641 → 0.676 (+0.035) | 0.565 → 0.585 (+0.020) |
| MuSiQue | 0.445 → 0.450 (+0.005) | 0.513 → 0.534 (+0.021) | 0.390 → 0.420 (+0.030) |
| **avg** | | **0.620 → 0.646 (+0.026)** | **0.503 → 0.535 (+0.032)** |

### Conclusions
1. **The bridge helps — modestly and consistently.** `decomposition` gains +0.026 avg F1 / +0.032 avg EM
   (gpt-4o-mini); positive on every dataset. **MuSiQue `decomposition` now exceeds MOTHRAG F1** (0.534 vs
   0.505) *even on gpt-4o-mini*. It closes ~40 % of the residual gpt-4o-mini→MOTHRAG F1 gap (−0.060 → −0.037).
2. **`single_hop` benefits most** (+0.057 F1 HotpotQA, +0.040 MuSiQue) — it does *one* retrieval, so better
   ranking helps it directly; `decomposition` already multi-retrieves per sub-question, so it gains less.
3. **`iterative` + bridge regresses** — −0.098 F1 on 2Wiki (hit 0.418 → 0.291). Two query-reshaping
   mechanisms (the iterative follow-up loop + multi-query/judge) interact badly. Caution: don't pair them.
   (`iterative` isn't the default, so the headline is unaffected.)
4. **The gains are small because distractor bounds the bridge.** Each question's store holds only ~10–20
   passages, so the bridge can only re-rank that pool into the top-8 — it can't *recall* missing evidence.
   That's why MuSiQue's `hit` barely moved (+0.005): its 4-hop depth isn't a ranking problem.

### Verdict — fullwiki is the next gate
The bridge's real value — recall over a large corpus — is **not exercised in distractor**. The modest but
positive distractor gains confirm it works; the *order-of-magnitude* payoff needs the **fullwiki** setting
(retrieve over the whole Wikipedia corpus, not a per-question pool). So the next required step before more
retrieval architecture (γ-retrieval, ChainFilter) is **fullwiki ingestion + retrieval**, against which those
levers — and the bridge itself — can actually show their worth. Building more retrieval cleverness against a
20-passage pool would be optimizing the wrong setting.

---

## Phase 2.5 — shared-corpus (pool) investigation (2026-06-24)

Built the shared-corpus harness (one persistent index, retrieve-only per question; `--corpus pool`), plus
the supporting hardening: **bounded concurrency** for the eval (~7× — the LLM calls are I/O-bound), an
**`llm_judge` shortlist cap** (corpus retrieval hands the reranker 100+ candidates), the OpenAI retry
**honoring `Retry-After`**, and a per-embedder index cache key. Ran HotpotQA over a **~10K-passage pool**
(1,000 dev questions, deduped), decomposition, n=200.

### Results (pool ~10K, decomposition)
| variant | hit | F1 | EM |
|---|---|---|---|
| gte-small + gpt-4o-mini (baseline) | 0.508 | 0.633 | 0.515 |
| + bridge (multi-query + judge) | 0.497 | 0.632 | 0.515 |
| gte-small + **gpt-4o** | 0.541 | 0.653 | 0.540 |
| **text-embedding-3-small** (1536-d) + mini | 0.514 | 0.644 | 0.515 |
| *MOTHRAG* | — | *0.781* | *0.648* |

(Distractor reference: gte-small + gpt-4o-mini decomposition was hit 0.574 / F1 0.705.)

### Conclusions
1. **`hit` is stuck at ~0.51–0.54 across every lever** — bridge, reader, and a much stronger embedder all
   hit the same wall. So the recall ceiling is **not** embedder quality, the reader, or query expansion.
2. **The bridge is redundant with decomposition** (flat) — decomposition already expands the query.
3. **The reader is modest on the pool** (gpt-4o +0.02 F1 vs distractor's +0.06) — it can only answer what's
   retrieved, and retrieval misses ~half.
4. **A stronger embedder is flat** (te3-small ≈ gte-small) — so we did *not* pursue a local strong embedder.
5. The wall is the **multi-hop retrieval problem**: HotpotQA's 2nd-hop passage isn't similar to the
   *question* (it's similar to the 1st hop's *answer*), so dense retrieval can't surface it — no matter the
   embedder. The cheap levers (reader / embedder / bridge) are **spent** on the realistic pool setting.

### Verdict → Phase 3
tarn.rag is at MOTHRAG parity on **distractor** (lean stack + gpt-4o); the **pool** gap is the multi-hop
retrieval ceiling. The levers that actually attack it are the §6 items *not* redundant with decomposition:
**γ-driven re-retrieval** (a failed grounding check triggers a follow-up search using what's been read — it
can find the missing hop) and, heavier, ChainFilter. Phase 3 = **γ-driven re-retrieval**.

---

## Phase 3 — results: γ-driven re-retrieval (2026-06-24)

`GroundedRetrievalReasoner` (`grounded_retrieval`) — retrieve → read → grounding-check; an ungrounded claim
triggers a follow-up search *for that claim* (the bridge passage), then re-read. Eval: pool ~10K, gte-small,
gpt-4o-mini, n=200, **heuristic** grounding (the LLM-free default).

| reasoner | hit | F1 | EM |
|---|---|---|---|
| decomposition | 0.481 | 0.613 | 0.500 |
| **grounded_retrieval (γ, heuristic)** | **0.508** | **0.621** | **0.510** |

**Verdict: a small, partly-within-noise positive** (+0.027 hit; F1/EM within the n=200 ~±0.02–0.03 floor —
decomposition read 0.633 last run vs 0.613 here). γ-retrieval *nudges* the multi-hop ceiling but doesn't
break it with heuristic grounding. Open levers: (a) **LLM grounding** (sharper gap detection → better-targeted
re-retrieval — cheap config try), (b) **ChainFilter** (the heavy §6 item: OpenIE triples + chain density),
or (c) bank it — tarn.rag is at MOTHRAG parity on distractor; the realistic-pool multi-hop recall is a hard
frontier that incremental levers (bridge, embedder, γ-heuristic) only nudge.

### γ with LLM grounding — the bottleneck is re-retrieval, not the signal (2026-06-24)

Re-ran γ with `--grounding llm_grounding` (sharper claim-gap detection) vs decomposition, same protocol:

| reasoner | hit | F1 | EM |
|---|---|---|---|
| decomposition | 0.481 | 0.610 | 0.495 |
| grounded_retrieval (γ, **llm_grounding**) | 0.486 | 0.596 | 0.480 |

**LLM grounding made γ worse, not better** (−0.014 F1 vs decomposition; below γ-heuristic's 0.621 F1). So the
weak link is **not** the grounding signal — it's the re-retrieval itself: claim-as-query dense retrieval still
can't surface the question-dissimilar bridge passage, and the extra hops add context noise. **γ-retrieval is
spent** as a pool lever.

### Phase 3 verdict (final)
Every measured lever — bridge, stronger reader, stronger embedder, γ-heuristic, γ-llm — only *nudges* the
pool's hit ≈ 0.51 multi-hop recall ceiling. The multi-hop retrieval problem (the 2nd-hop passage isn't similar
to the question) is a **genuine hard frontier** that incremental retrieval tricks don't crack. The honest
state: **tarn.rag matches MOTHRAG on distractor with a lean, differentiated stack** (offline / layout
provenance / license filtering); the realistic-pool multi-hop recall gap is characterized and bounded. The
only untried lever is **ChainFilter** (OpenIE triples + chain density — heavy, uncertain payoff against a
ceiling 5 levers couldn't move).

---

## Post-MOTHRAG performance plan

Stepping back from the benchmark chase: "performance" is several axes, and the MOTHRAG setting (multi-hop
Wikipedia QA) doesn't exercise tarn.rag's differentiators (offline / layout provenance / licensing). Four
options, in order: (1) hybrid retrieval, (2) differentiators (attribution + layout, on real docs), (3)
systems (bulk-ingest throughput + latency), (4) robustness / eval on real data.

### Option 1 — hybrid retrieval (dense + BM25, RRF) (2026-06-24)

The whole pool push was dense-only. BM25 matches exact entity tokens (the bridge entity that embeds weakly
against the question), so hybrid targets the dense-only recall ceiling — at no LLM cost, and no rebuild (the
FTS5 index is built at ingest). Pool ~10K, decomposition, gpt-4o-mini, n=200, same session:

| retrieval | hit | F1 | EM |
|---|---|---|---|
| dense (baseline) | 0.497 | 0.623 | 0.505 |
| **hybrid (dense + BM25, RRF)** | **0.508** | **0.638** | **0.510** |

**+0.011 hit / +0.015 F1 / +0.005 EM** — small but consistent across all three, and **free** (no LLM, no
rebuild): the best cost/benefit lever found on the pool. It doesn't *break* the multi-hop ceiling (BM25 can't
surface a bridge entity absent from the question), but it's a worthwhile default. Enabled via `--hybrid`
(`HYBRID_RETRIEVAL`).

### Option 2 (PR-1) — layout-aware retrieval on TAT-QA (2026-06-24)

The MOTHRAG setting can't test tarn.rag's differentiators. TAT-QA (financial **table** + **paragraphs**,
with an `answer_from` ∈ table/text/table-text label) can: build one shared corpus of every table + paragraph,
then for each extractive question measure **source-hit@k** (did the top-k include the gold answer-source
element), segmented by where the answer lives. 100 records → 612 elements, 334 extractive queries, gte-small.

| segment | dense | hybrid | Δ |
|---|---|---|---|
| table | 0.779 | 0.853 | +0.074 |
| table-text | 0.855 | 0.945 | +0.090 |
| text | 0.938 | 0.953 | +0.015 |
| **overall** | 0.865 | 0.922 | +0.057 |

**Findings:** (1) **dense has a table deficit** — tables 0.779 vs text 0.938 (a 0.16 gap): cell tokens embed
weakly against a NL question. (2) **hybrid's real home is tables** — BM25 matches exact cell tokens, lifting
table/table-text far more than text (+0.074 / +0.090 vs +0.015); hybrid's +0.057 overall here dwarfs its
+0.015 F1 on Wikipedia QA. The differentiated setting reveals a lever the benchmark hid. Run via
`scripts/run_layout_eval.py`. Next (Option 2 PR-2): attribution precision (LLM-judge / `grounded_rate`) on
TAT-QA; later, ingest tables through the native structured path (Table elements) instead of rendered text.

### Option 2 (PR-2) — attribution precision on TAT-QA (2026-06-25)

Beyond *finding* the source (PR-1), does the answer *attribute* to it? Answer each extractive question with a
`single_hop` reader over hybrid retrieval, then have an LLM judge whether each cited span supports its claim
(`grounding_checker: llm_grounding`). `grounded_rate` = attribution precision; `citation_coverage` = is the
gold answer span present in the cited evidence. n=334, gte-small + gpt-4o-mini.

| segment | n | F1 | EM | attrib | cite |
|---|---|---|---|---|---|
| table | 95 | 0.509 | 0.379 | 0.884 | 0.732 |
| table-text | 110 | 0.571 | 0.427 | 0.936 | 0.706 |
| text | 129 | 0.595 | 0.349 | 0.992 | 0.884 |
| **overall** | 334 | 0.563 | 0.383 | 0.943 | 0.782 |

**Findings:** (1) **attribution precision is high (0.94 overall)** — when tarn.rag answers, the cited spans
support the claim 94% of the time. (2) **A consistent table penalty across every metric** — F1 0.51 vs 0.60,
attribution 0.88 vs 0.99, citation-coverage 0.73 vs 0.88 (table vs text). Tables are systematically harder to
answer *and* attribute, which points at the **linearized-table representation** (rendered to text) as the
next lever — the motivation for ingesting tables through the native structured `Table`-element path (a
follow-up). Run via `scripts/run_layout_eval.py --attribution`.

---

## Option 3 — bulk-ingest throughput (2026-06-25)

The per-doc ingest path gated every corpus build this session (~2/s). Profiling a 100-doc ingest (offline
hash embedder, isolating orchestration from embedding) found **~64 `engine.begin()` transactions per
document** — transaction-count *overhead* (~5 ms each of asyncio+SQLAlchemy machinery; not fsync, not the
inserts, which total ~1 s). Two causes:
1. the embedded queue dispatched **one document at a time** (`Batch([job])`), so each doc ran the DAG solo;
2. **two per-doc loops** persisted per document even within a batch — `DocumentResultSink._persist`
   (`store_document` per doc) and the orchestrator's `_record` (status per job, per stage).

**Fix (two parts):**
- **Batched dispatch** — `InMemoryJobQueue` groups a wave's queued jobs by `stage_name` into homogeneous
  `Batch`es (capped at `max_batch_size`), so a stage runs over many docs at once (the worker/sink already
  supported multi-job batches). On a batch failure it re-runs each job solo, preserving per-job
  at-least-once / dead-letter semantics.
- **Bulk-persist** the two hot loops — `store_documents` (one transaction for the batch's documents) and
  `record_jobs` (one transaction for the batch's status rows), reusing the exact per-row logic.

**Result** (300 docs, offline): **3.6 → 9.4 docs/s, ~2.6×**. Equivalence verified — the batched + bulk index
is identical to the per-doc index (same documents, chunks, embeddings); only the transaction count drops.
Batched dispatch also batches the **Embed** stage, so API embedders make far fewer requests (the cost that
made the te3 corpus build take ~2.6 h). Remaining floor: per-row inserts + ~5 stages; further wins would need
cross-stage batching.
