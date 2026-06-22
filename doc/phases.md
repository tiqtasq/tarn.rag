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

## Phase 0 — results (2026-06-22)

### Setup
- **Harness:** `python scripts/eval_mothrag.py <dataset> --hf --limit 200 --sweep` (branch
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
- [ ] **Phase 1:** default reasoner → `decomposition`; short-answer prompt + answer extraction; re-measure
  (expect the largest EM gains on HotpotQA/2Wiki).
- [ ] **Phase 2 candidate:** bridge substrate / stronger multi-hop — justified primarily by MuSiQue's low hit.
