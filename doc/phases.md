# tarn.rag — performance roadmap toward SOTA

**Goal:** SOTA-competitive multi-hop QA (MOTHRAG ≈ 68.3 avg F1; NeocorRAG ≈ 69 is the GPU/constrained-decoding
ceiling) **without** trading away tarn.rag's differentiators (offline / layout-grade provenance / license
filtering). **Method: measure first, then add by measured leverage.** The MOTHRAG gap analysis lives in
`generation-architecture-design.md` §6; the eval harness is `scripts/eval_mothrag.py` (`tarnrag/eval/`).

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

## Phase 0 — results (2026-06-21)

### Setup
- **Harness:** `python scripts/eval_mothrag.py hotpotqa --hf --limit 200 --sweep` (branch
  `feature/mothrag-phase0-sweep`, PR #71).
- **Protocol:** HotpotQA *distractor*, n = 200. Each question ingests only its own ~10 candidate passages
  into an isolated store, then retrieve → reason → answer; scored token-F1 / exact-match (SQuAD-style — the
  metric MOTHRAG reports).
- **Fixed:** reader `gpt-4o-mini` (OpenAI), embedder `gte-small` (local ONNX, 384-d), grounding **off**,
  `temperature=0`. **Swept:** the three reasoners.

### Results (n = 200, HotpotQA)

| reasoner | hit | F1 | EM | grounded | citation |
|---|---|---|---|---|---|
| `single_hop` | 0.590 | 0.456 | 0.265 | 1.000 | 0.686 |
| `iterative` | 0.579 | 0.183 | 0.010 | 1.000 | 0.804 |
| **`decomposition`** | **0.661** | **0.510** | **0.290** | 1.000 | 0.732 |
| *MOTHRAG (Llama-3.3-70B, n=1000)* | — | *0.781* | *0.648* | — | — |

*hit = a gold answer phrase appears anywhere in the answer; citation = fraction of gold supporting phrases
inside the cited passages.*

### Conclusions
1. **`decomposition` wins** — best on every answer metric. Making it the default is a free **+0.05 F1 /
   +0.03 EM** over `single_hop`.
2. **Answer format is the biggest lever, not architecture.** `decomposition` *finds* the answer 66 % of the
   time (hit) but emits an exact match only 29 % (EM), and F1 (0.51) trails hit. The content is there; the
   answer string isn't clean → a short-answer prompt + extraction attacks this directly.
3. **`iterative` confirms it** — best citation coverage (0.804) and decent hit (0.579), yet F1 collapses to
   0.18 / EM 0.01: it reasons and retrieves *well*, then narrates prose instead of a terse span. A pure
   output-surface failure, not a reasoning one.

### Config-vs-architecture verdict
A substantial part of the ~0.27 F1 / ~0.36 EM gap to MOTHRAG is **config/format-addressable** (reasoner
choice + answer surface) *before* any §6 architecture. Measure-first paid off: building the ensemble first
would have optimized the unmeasured.

### Caveats
n = 200 (CI ≈ ±3.5 %); **HotpotQA only**; `gpt-4o-mini` reader (not MOTHRAG's Llama-3.3-70B); a single 384-d
embedder. Absolute numbers will shift with a stronger reader, but the `hit ≫ EM/F1` **format** gap is largely
reader-independent.

### Status / next
- [x] HotpotQA reasoner sweep (above).
- [ ] Wire **2Wiki + MuSiQue** HF loaders and sweep all three — MuSiQue (4-hop, recall-bound) may show the
  lever there is *retrieval*, not format, which would reprioritize Phase 2 toward the bridge substrate.
- [ ] **Phase 1:** default reasoner → `decomposition`; short-answer prompt + extraction; re-measure.
