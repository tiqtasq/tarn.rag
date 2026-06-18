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
  RetrievalService (port) ──┬── RetrievalEngine            (Python-native, today)
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
class RetrievalService(Protocol):                       # in tarnrag/retrieval/ (retrieval owns it)
    async def search(self, query: Query) -> list[RetrievalResult]: ...
```

- **Python deployment:** `RetrievalEngine` *structurally* satisfies it (its `search` already has this
  shape — see `retrieval/engine.py`). Zero new code.
- **C++ deployment (later):** a thin Python `C++RetrievalAdapter(RetrievalService)` wraps the binding.

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
class LanguageModel(Resource):                          # tarnrag/core/llm.py
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
    retrieval: RetrievalService # the port (Python engine or C++ adapter)
    llm: LanguageModel          # the reader/decomposer

class GenerationPipeline(Component):
    async def answer(self, query: Query, ctx: GenerationContext) -> GenerationResult: ...
    # route → reason (retrieve↔read loop, budgeted) → ground-check → assemble proof tree
```

**Where the `Router` lives — a deliberate call.** The `Router` is a **retrieval-layer seam** (a routing
`RetrievalPipeline` variant): query features → which retrieval strategy. Putting it there means
**retrieval-only** users get routing too, and it's driven by the segmentation evidence we built. It admits
two kinds of implementation:

- a **heuristic** router (rules/features over the query) — LLM-free, deterministic, trivially portable;
  **slice 1 ships this**, so the retrieval core stays LLM-free *by default*;
- an **LLM-based** router (classify the query with the `core` `LanguageModel`) — an *optional* component
  on the same seam. It depends on `core` (not `generation`), so the one-way rule holds; and the **C++
  port can implement it too** — an LLM call is just HTTP, not a portability barrier. The reason to default
  to the heuristic one is self-containedness / determinism / cost, **not** that LLM routing is unportable.

(An LLM-based router is also reusable by the generation layer; the seam is shared.)

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
engine = await GenerationEngine.create(settings)        # builds RetrievalService + LanguageModel + the
                                                        # GenerationPipeline from GENERATION_PIPELINE
result = await engine.answer(Query(text="…"))           # GenerationResult (answer + proof tree + evidence)
```

`create()` builds the retrieval service (a `RetrievalEngine` today; a C++ adapter later), the LLM
resource, and the pipeline from the spec, then delegates `answer` to it. tiqtasq.backend wraps this.

## 6. MOTHRAG-parity is additive (framework intact)

Lean ships one component per seam; MOTHRAG-parity = **add components + a richer `GENERATION_PIPELINE`
spec**, no framework change:

| MOTHRAG feature | Added later as |
|---|---|
| direct / decomposition / iterative **arms** | extra `Reasoner` components |
| **PDD + deterministic arbitration** ensemble | an `Ensemble` container `Reasoner` (runs several, votes) |
| **γ grounding verifier** + retry loop | a richer `GroundingChecker` behind the same seam |
| **bridge substrate / multi-query fusion** | retrieval-layer `Fuser` / `Retriever` components |
| **ChainFilter** (OpenIE chain density) | a retrieval-layer `Reranker`/filter component |
| **query-type gating** | the `Router` (slice 1) |

The two retrieval-layer items (multi-query, chain-density) are **optional components outside the lean
default core** — they bring extra dependencies (an LLM call for query expansion; an OpenIE model for chain
density). A deployment opts into them on the same seams, in *either* language (the C++ port can make those
calls too); the default lean retrieval core omits them, and chain-extraction may instead move into
ingestion. The invariant is "lean + self-contained *by default*," not "no external calls ever."

## 7. Build slices

1. **Query-type routing (retrieval layer, no LLM).** A routing `RetrievalPipeline` that classifies a query
   (heuristic, segmentation-informed) and dispatches to the per-type-best sub-pipeline. Portable;
   benefits retrieval-only. *(Foundation; the only slice touching retrieval.)*
2. **Generation MVP.** The LLM `Resource` seam + `GenerationPipeline` + a single-hop `Reasoner` + the
   `EvidenceAssembler` (proof tree from provenance) + the `GenerationEngine` facade + the
   `RetrievalService` port. End-to-end: question → answer + evidence. No multi-hop yet.
3. **Grounding + abstention.** A `GroundingChecker` that verifies the proof tree against the evidence,
   with a best-grounded fallback (γ-cap analog) and an optional refusal path.
4. **Multi-hop.** An iterative `Reasoner` (retrieve ↔ read loop, budgeted) + question decomposition into
   sub-queries — the actual multi-hop capability, over our retrieval substrate.
5. **Generation eval harness.** Answer-quality + grounding/citation metrics over a labeled set — the
   analog of the retrieval eval harness, to compare generation pipeline specs.

(Post-5, MOTHRAG-parity is the §6 additive components, measured against the slice-5 harness.)

## 8. Open decisions

- **Default LLM provider / model** (the seam is provider-agnostic; default TBD — likely Claude).
- **Binding mechanism for the eventual C++ adapter** (pybind11 vs C-ABI+ctypes vs out-of-process service)
  — *deliberately deferred*; §5/§2.1 keep it open.
- **Reasoner prompting/decomposition strategy** for slices 2/4 (single read vs decomposition-first).
- **Abstention policy** (when to refuse vs return best-grounded).
