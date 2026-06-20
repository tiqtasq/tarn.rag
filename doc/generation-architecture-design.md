# Generation architecture — design

The design for **Goal 3**: an end-to-end **multi-hop generation layer with evidence**, added *on top of*
the existing retrieval engine. This doc fixes the seams before any code, the way
`doc/retrieval-architecture-design.md` did for the retrieval track.

## 0. Requirements & constraints (locked)

1. **Production multi-hop QA with evidence.** Given a question + the corpus, produce an *answer* plus a
   **proof tree** of inspectable reasoning steps, each citing the passage spans it relies on. Multi-hop =
   assemble a chain of evidence across passages (iterate: retrieve → reason → re-retrieve).
2. **Dual-mode, permanently.** Retrieval-only stays a first-class product (it is the thing getting a
   future **C++ port**). Generation is *additive* — `IngestionEngine` / `RetrievalEngine` keep working
   untouched; a new `GenerationEngine` sits beside them.
3. **Strict one-way dependency: `generation → retrieval`, never the reverse.** No *generation* concept
   (reasoner, proof-tree assembly, …) may leak into `tarnrag/retrieval/` or `tarnrag/ingestion/`. The
   retrieval core stays **self-contained + LLM-free *by default*** — which is what keeps it portable to
   C++ and valuable as a retrieval-only product. The LLM itself is a shared **`core` `Resource`** (§2.2),
   not a generation-only thing, so retrieval *may opt in* to it (an LLM-based `Router`) without ever
   depending on `generation/`. "LLM-free" is the default, not a prohibition; an LLM-based component the
   C++ port can equally implement (an LLM call is just HTTP — *not* a portability barrier).
4. **Lean now, MOTHRAG-extensible later.** Build the simplest production pipeline that works, on the
   **Component framework**, so MOTHRAG-parity is reached later by *adding components + swapping the spec*,
   framework intact — exactly as "compare retrieval methods = swap the `RetrievalPipeline` spec" works
   today. (See §6 for the mapping.)
5. **Keep the C++ path open cheaply.** Generation depends on a **retrieval *port*** (an interface), not
   the concrete engine; the retrieval **result contracts are treated as a stable, versioned,
   serializable cross-language schema**. So an eventual C++ retrieval engine is a later *adapter*, not a
   redesign — without deciding the binding mechanism (pybind11 / C-ABI / IPC) now.
6. **REST stays out of tarn.rag.** `GenerationEngine` is the library API; **tiqtasq.backend** wraps it in
   REST. (Same posture as today.)

## 1. Layered architecture

```
  tiqtasq.backend  (REST — out of scope here)
        │  wraps
        ▼
  tarnrag/generation/        ← NEW layer (Python; LLM orchestration)
        │  depends on (one-way)
        ▼
  RetrievalEngineProtocol (port) ──┬── RetrievalEngine            (Python-native, today)
                            └── C++RetrievalAdapter         (later; wraps the binding)
        ▲
  tarnrag/retrieval/ , tarnrag/ingestion/   ← unchanged; C++-portable core
```

`tarnrag/generation/` is a new top-level package next to `ingestion/` and `retrieval/`. It **consumes** a
retrieval port and an LLM; it is consumed by tiqtasq's REST layer.

## 2. The two seams that cross layer boundaries

### 2.1 The retrieval port (the C++-port hinge)

Generation never imports the concrete `RetrievalEngine`. It depends on a `Protocol`:

```python
class RetrievalEngineProtocol(Protocol):                       # in tarnrag/retrieval/ (retrieval owns it)
    async def search(self, query: Query) -> list[RetrievalResult]: ...
```

- **Python deployment:** `RetrievalEngine` *structurally* satisfies it (its `search` already has this
  shape — see `retrieval/engine/engine.py`). Zero new code.
- **C++ deployment (later):** a thin Python `C++RetrievalAdapter(RetrievalEngineProtocol)` wraps the binding.

The data crossing this seam — `Query` (in) and `RetrievalResult` / `ChunkProvenance` (out) — is the
**cross-language schema**. It must stay serializable, flat-ish, **versioned**, and free of Python-only
behavior. We are already close: these are dataclasses, the provenance has a JSON codec
(`storage/repository/chunk_provenance.py`), and the index carries `SCHEMA_VERSION`. The discipline is:
*treat `RetrievalResult` as a stable wire/ABI contract and version it.*

Boundary properties (apply regardless of binding mechanism):
- **Synchronous + offloaded.** A C++ call is sync; the adapter awaits it via `asyncio.to_thread` — the
  pattern we already use for the embedder / cross-encoder.
- **Coarse-grained.** Multi-hop makes *many* retrieval calls; each crossing the boundary has cost. The
  `Reasoner` keeps calls chunky (one `search` returns what a hop needs; batch sub-questions) rather than
  chatty round-trips.

### 2.2 The LLM seam

The reader/decomposer LLM is a new **`Resource`** (an engine-built, injected model — exactly like
`Embedder` / `CrossEncoder`, *not* a `Component`), and it lives in **`core/`** — *not* in `generation/`:

```python
class LanguageModel(Resource):                          # tarnrag/core/resources/llm.py
    async def complete(self, prompt: Prompt) -> Completion: ...   # text in / text (+ optional structured) out
    def identity(self) -> str: ...
```

Provider-pluggable behind the same selector pattern as `build_embedder` (Anthropic / OpenAI / …),
lazily-loaded client, faked in tests by injection. (Default provider: TBD — Claude is the leading
candidate given the stack.)

It lives in `core/` (not `generation/`) so that **any** layer that opts in can inject it — generation's
`Reasoner` (its primary consumer) **and** an optional LLM-based retrieval `Router`. A retrieval router
using the LLM depends on `core` (`retrieval → core`), so the one-way `generation → retrieval` rule is
untouched. Generation is the LLM's heavy user; retrieval's *default* path uses no LLM at all.

## 3. The generation pipeline (a container Component, sibling of `RetrievalPipeline`)

`GenerationPipeline` is a config-driven container `Component` (same machinery as the ingestion `Pipeline`
and `RetrievalPipeline`), built from a `GENERATION_PIPELINE` spec in `Settings.components`. Its seams (all
`Component`s — pure strategies; resources injected at call time via a `GenerationContext`):

| Seam | Job | Lean (slice) | LLM? |
|---|---|---|---|
| **`Router`** | classify the query → choose the retrieval strategy (and later the reasoning path) | one heuristic classifier → a retrieval spec | no |
| **`Reasoner`** | retrieve → read → (optionally iterate / decompose) → candidate answer + reasoning steps | one single-hop reader; then one iterative reasoner | yes |
| **`GroundingChecker`** | verify the answer's steps against the retrieved evidence; trigger re-retrieval or a fallback | a simple check + best-grounded fallback | yes/no |
| **`EvidenceAssembler`** | build the proof tree from the cited chunks' provenance | map citations → spans/boxes/header-paths | no |

```python
@dataclass
class GenerationContext:        # the runtime resources, injected at call time
    retrieval: RetrievalEngineProtocol # the port (Python engine or C++ adapter)
    llm: LanguageModel          # the reader/decomposer

class GenerationPipeline(Component):
    async def answer(self, query: Query, ctx: GenerationContext) -> GenerationResult: ...
    # route → reason (retrieve↔read loop, budgeted) → ground-check → assemble proof tree
```

**Where the `Router` lives — a deliberate call.** Routing is a **retrieval-layer** concern (it benefits
retrieval-only users and the C++ port, and it's driven by the segmentation evidence we built), so slice 1
lives entirely in `tarnrag/retrieval/` — the `generation/` package isn't created until slice 2. Its
shape (all on the existing Component framework):

**The `Searcher` seam.** Extract the one method `RetrievalPipeline` exposes —
`async search(query, ctx) -> list[RetrievalResult]` — as a tiny abstract base `Searcher(Component)`.
`RetrievalPipeline` becomes a `Searcher`; so does the router. The engine builds
`create_as(RETRIEVAL_PIPELINE, Searcher)`, so that one spec can name **either** a plain pipeline **or** a
router — no new `Settings` key.

**The `Query` contract carries the classification.** Classification is *integrated into the contract*, the
same way enrichment output lands on a chunk (`Annotation`), not buried inside the router:

```python
@dataclass
class Query:
    text: str
    ...
    query_type: str = ""                 # cheap route key — the headline label the router dispatches on
    annotations: list[Annotation] = []   # the rich, extensible channel (reuses the enrichment Annotation)
```

The denormalized `query_type` (route key) + the rich `annotations` (full detail) mirror the chunk's
`license_class` column alongside its `annotations`: cheap to route on, with the complete extraction
carried along for downstream consumers (the eval harness, logging, later the `Reasoner`). Reusing
`Annotation` is free leverage: the `deterministic` flag means an LLM classifier's findings are flagged
(never silently trusted), and `span` can mark which substring of the query is an identifier.

**The `QueryClassifier` seam — optional & configurable, no hard-wired taxonomy.** A `Component` whose job
is to populate `query.query_type` (+ `annotations`), with an `annotate` helper mirroring
`Enricher.annotate` (producer auto-filled):

- **`NoOpQueryClassifier`** — the **default**; classifies nothing, so an unconfigured router falls through
  to its `default` route (≡ a single pipeline). Routing is opt-in.
- **`StructuralQueryClassifier`** — the first *real*, **domain-independent** classifier: a heuristic over
  the query's *form*, not its subject matter (interrogative form, function-word ratio, exact-match cues
  like quotes/identifiers/acronyms, length) → `query_type ∈ {lexical, semantic}` + a rich annotation of
  what it found. Deterministic, LLM-free, C++-portable. Labels/thresholds/word-lists are config (the
  form-independent cues — identifiers/quotes/length — carry weight regardless of language). Domain-
  *specific* taxonomies are deferred — added later as more classifier components on this seam.
- (later) an **LLM-based** classifier on the same seam, using the `core` `LanguageModel` — depends on
  `core` not `generation`, so the one-way rule holds, and the C++ port can implement it too (an LLM call
  is just HTTP, not a portability barrier). The default is heuristic for self-containedness / determinism
  / cost, **not** because LLM routing is unportable.

**The router** — `RoutingRetrievalPipeline(Searcher)` — holds a `classifier` + a `routes: {query_type →
Searcher spec}` map + a `default` Searcher. `search` runs the classifier (populating `query_type`), then
dispatches to `routes.get(query_type, default)`. Routes are themselves `Searcher`s (built recursively, so
a route is just another pipeline spec — sparse for `lexical`, dense for `semantic`, …), and the route map
is the deployment's config, tuned from the segmented eval. The **eval harness is the judge**: it already
segments by `query_type`, so a router spec is swept alongside the single pipelines and must match-or-beat
the best per type. (The same seam is reusable by the generation layer.)

## 4. Evidence: proof trees on *our* provenance (the differentiator)

MOTHRAG cites passage spans; we already emit **layout-grade** provenance at ingestion (char spans + PDF
page boxes + header paths + table cells). So our proof trees are richer for free:

```python
@dataclass
class Citation:                 # one cited span of evidence
    chunk_id: str; document_id: str; locator: str | None
    geometry: Geometry          # char spans (+ page boxes) → highlightable, from RetrievalResult.provenance

@dataclass
class ProofStep:
    claim: str                  # one reasoning step / intermediate answer
    citations: list[Citation]   # the evidence it rests on

@dataclass
class GenerationResult:         # the GenerationEngine's output (and tiqtasq's REST payload)
    answer: str
    proof: list[ProofStep]      # inspectable reasoning, each step → cited spans
    evidence: list[RetrievalResult]
    grounded: bool              # did the grounding check pass within budget?
    abstained: bool = False     # refusal path (off in F1-style configs)
```

`GenerationResult` is produced by Python generation and consumed by tiqtasq (JSON/REST), so it must be
**serializable + versioned** — but it does *not* cross the C++ binding (only the retrieval contracts do).

## 5. The facade

`GenerationEngine` — the library entry point, sibling of `IngestionEngine` / `RetrievalEngine`:

```python
engine = await GenerationEngine.create(settings)        # builds RetrievalEngineProtocol + LanguageModel + the
                                                        # GenerationPipeline from GENERATION_PIPELINE
result = await engine.answer(Query(text="…"))           # GenerationResult (answer + proof tree + evidence)
```

`create()` builds the retrieval service (a `RetrievalEngine` today; a C++ adapter later), the LLM
resource, and the pipeline from the spec, then delegates `answer` to it. tiqtasq.backend wraps this.

## 6. MOTHRAG-parity is additive (framework intact)

Lean ships one component per seam; MOTHRAG-parity = **add components + a richer `GENERATION_PIPELINE`
spec**, no framework change. The plan held: slices 1–5 are now built (§7), and the remaining MOTHRAG
deltas are exactly the *additive* components named below — no base-class edits were needed to get here, and
none are needed to finish.

### 6.1 Current state vs MOTHRAG (re-reviewed against `doc/main.pdf`, 2026-06-20)

Status against the MOTHRAG method (§3 of the paper). **✅ built · ◑ partial · ❌ missing.**

| MOTHRAG feature (paper §) | tarnrag today | Status |
|---|---|---|
| Reasoning **arms**: direct / decomposition / iterative (§3.1) | `SingleHopReasoner` / `DecompositionReasoner` / `IterativeReasoner` exist as **swappable** reasoners | ◑ the three voices exist, but as *alternatives* (pick one via spec), **not** run together |
| **Four-arm ensemble pool**, fixed N=4 (§3.1) | — no ensemble; one reasoner runs per request | ❌ |
| **PDD** (Pool-Duplicate Dispatch — double-weight the γ-checked iterative voice) (§3.1) | — | ❌ (needs the ensemble first) |
| **Deterministic arbitration** over candidates: γ (1.0) + cross-arm agreement (0.5) + faithfulness (0.3) (§3.5) | — no arbitrator; no agreement/faithfulness signals | ❌ |
| **Bridge retrieval substrate**: multi-query ANN fusion + a tripartite **LLM relevance judge** conditioned on bridge evidence, reshaping every retrieval round (§3.2) | RRF fusion *across configured retrievers* + a **cross-encoder** reranker; no multi-query expansion, no LLM judge, no bridge conditioning | ◑ fusion/rerank seams exist; the bridge *mechanism* is missing |
| **ChainFilter**: chain-density rerank over OpenIE triples (§3.3) | — no OpenIE extraction, no chain-density filter | ❌ |
| **γ grounding verifier** + proof tree (§3.4) | `GroundingChecker` (heuristic / LLM / cascading), 3-valued `Verdict`, proof tree | ✅ — and our proof tree is **richer** (§4: char spans + PDF page boxes + header paths + table cells) |
| **γ-failure → iterative re-retrieval** loop (§3.4) | `IterativeReasoner` loops on the *model's own* "done" judgment; the γ checker runs **after** reasoning and does not feed back into retrieval | ◑ the loop exists, but it isn't γ-driven |
| **γ-cap fallback** (return best-grounded within budget) (§3.4) | abstention policy: `min_grounded`, best-grounded flag, optional refusal | ✅ (analog) |
| **Query-type gating** (input-feature classifier; never dataset identity) (§3.2, §3.6) | `RoutingRetrievalPipeline` + `Structural`/`Generic` `QueryClassifier` — **retrieval-layer routing** | ◑ routing is built, but it isn't yet used to *gate arms / filters* (no ensemble/ChainFilter to gate) |
| **Abstention pathway** (present, disabled in the F1 config) (§5) | abstention policy with `abstain` flag (off by default) | ✅ |
| **Eval harness** (paper-grade F1/EM over the benchmark triple) | retrieval harness (`hit@k`/MRR/nDCG, segmented) + generation harness (token-F1/EM, grounded rate, abstention accuracy, citation coverage) | ✅ harness exists; **not yet run on HotpotQA / 2Wiki / MuSiQue** |
| Provider mix: Llama-3.3-70B reader, Gemini embedder, Claude/Gemini judges (§3.6) | embedder: onnx/openai/voyage/gemini; LLM: **anthropic-only** (`LLMSettings.provider = Literal["anthropic"]`) | ◑ multi-provider embedder; single-provider reader/judge |

### 6.2 What we'd need to add / improve for MOTHRAG parity (priority order)

1. **`EnsembleReasoner` + deterministic arbitration (the headline gap).** A container `Reasoner` that runs
   the existing arms in parallel and an `Arbitrator` that picks the winner by fixed weights — γ (1.0) +
   cross-arm **agreement** (0.5) + **faithfulness** (0.3). This is the single biggest delta: today the arms
   are alternatives, not an ensemble. It pulls in two new signals (agreement needs ≥2 candidates;
   faithfulness is a new scorer distinct from grounding) and **PDD** (register the γ-checked iterative
   candidate twice). All on the existing Component seam — no framework change.
2. **γ-driven re-retrieval.** Couple the `GroundingChecker` back into the loop so a failed step *triggers*
   re-retrieval (MOTHRAG §3.4), rather than the `IterativeReasoner` looping only on the model's self-report.
   Likely a new reasoner (or a pipeline that interleaves reason ↔ ground ↔ retrieve) — the proof tree +
   `Verdict` plumbing it needs already exists.
3. **Bridge retrieval substrate.** (a) A **multi-query** retriever/fuser (expand one query into several ANN
   queries, fuse) and (b) an **LLM relevance judge** reranker that scores candidate utility conditioned on
   the query + retrieved bridge evidence (premium/economy tiers = swap the LLM spec). Both are retrieval-
   layer `Retriever`/`Fuser`/`Reranker` components; the `RetrievalContext` already injects the resources.
   Optional (extra LLM cost) — outside the lean self-contained default.
4. **ChainFilter + an OpenIE substrate.** Chain-density reranking needs OpenIE triples; extraction belongs
   at **ingestion** (a graph/triples enricher → a new store or annotation), then a retrieval-layer
   `Reranker` scores chain density. The heaviest item (a new extraction model + a graph substrate; see
   `rag-design-building-blocks.md` Table 2). Optional.
5. **Arm/filter gating by `query_type`.** Once the ensemble + ChainFilter exist, reuse the existing
   `QueryClassifier` to gate which arms/filters apply per query class (MOTHRAG excludes the iterative arms
   from bridge reshaping on one class). The classifier seam is built; only the gating wiring is new.
6. **Run the benchmark eval + broaden providers (validation, not architecture).** Wire HotpotQA / 2Wiki /
   MuSiQue into the generation harness to *measure* the gap, and add a second reader/judge provider behind
   `LanguageModel.create` (Gemini/an OpenAI-style chat API) to match MOTHRAG's model-agnostic posture.

Items 3 and 4 are **optional components outside the lean default core** — they bring extra dependencies (an
LLM call for query expansion / the judge; an OpenIE model for chain density). A deployment opts into them on
the same seams, in *either* language (the C++ port can make those calls too); the default lean retrieval
core omits them, and chain-extraction may instead move into ingestion. The invariant is "lean +
self-contained *by default*," not "no external calls ever."

### 6.3 Where tarnrag already exceeds MOTHRAG

Not everything is catch-up. tarnrag's **layout-grade provenance** (char spans + PDF page boxes + header
paths + table-cell geometry) makes its proof trees strictly richer than MOTHRAG's passage-span citations
(§4), and its **retrieval core is offline / self-hosted / C++-portable** (local ONNX embedder + sqlite-vec)
where MOTHRAG is API-only by design. tarnrag also carries **license/scope filtering** (the ModusQ
requirement) that MOTHRAG has no analog for. Parity work should not regress these differentiators.

## 7. Build slices

**Status (2026-06-20): slices 1–5 are all implemented.** The remaining MOTHRAG-parity work is the §6.2
additive components. One naming delta vs. the plan below: the default classifier shipped as
`GenericQueryClassifier` (not `NoOpQueryClassifier`) — it tags every query with one constant `query_type`
so a router still guarantees a classification annotation while falling through to its `default` route.

1. **Query-type routing (retrieval layer, no LLM).** ✅ `Query` gains `query_type` + `annotations`; a
   `Searcher` seam (engine builds a plain pipeline *or* a router from `RETRIEVAL_PIPELINE`); a
   `QueryClassifier` seam (`GenericQueryClassifier` default + a domain-independent `StructuralQueryClassifier`);
   and a `RoutingRetrievalPipeline` that classifies → dispatches to the per-type-best sub-pipeline.
   Portable; benefits retrieval-only; validated by the segmented eval harness. *(The only slice touching
   retrieval — nothing lands in `generation/` yet.)*
2. **Generation MVP.** ✅ The LLM `Resource` seam + `GenerationPipeline` + a single-hop `Reasoner` + the
   `EvidenceAssembler` (proof tree from provenance) + the `GenerationEngine` facade + the
   `RetrievalEngineProtocol` port. End-to-end: question → answer + evidence.
3. **Grounding + abstention.** ✅ A `GroundingChecker` (heuristic / LLM / cascading) that verifies the proof
   tree against the evidence, with a best-grounded fallback (γ-cap analog) and an optional refusal path.
4. **Multi-hop.** ✅ An iterative `Reasoner` (retrieve ↔ read loop, budgeted) + a decomposition `Reasoner`
   (split into sub-queries) — the multi-hop capability, over our retrieval substrate. *(Caveat: the loop is
   model-driven, not yet γ-driven — see §6.2 item 2.)*
5. **Generation eval harness.** ✅ Answer-quality (token-F1 / EM / content-hit) + grounded-rate + abstention
   accuracy + citation coverage over a labeled set (`eval/generation.py`) — the analog of the retrieval
   harness, to compare generation pipeline specs.

(Post-5, MOTHRAG-parity is the §6.2 additive components, measured against the slice-5 harness.)

## 8. Open decisions

- **Default LLM provider / model** (the seam is provider-agnostic; default TBD — likely Claude).
- **Binding mechanism for the eventual C++ adapter** (pybind11 vs C-ABI+ctypes vs out-of-process service)
  — *deliberately deferred*; §5/§2.1 keep it open.
- **Reasoner prompting/decomposition strategy** for slices 2/4 (single read vs decomposition-first).
- **Abstention policy** (when to refuse vs return best-grounded).
- **Domain-specific query taxonomy** — *deliberately deferred*. Slice 1 ships a domain-independent
  classifier (query *form*); a domain taxonomy (e.g. exact-clause-lookup vs conceptual for ModusQ) is a
  later classifier component on the same seam, route map tuned from the segmented eval.
