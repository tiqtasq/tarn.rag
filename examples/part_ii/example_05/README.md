# Example 05 · Query routing (Act A capstone)

Act A's capstone steps back and **compares** the retrieval methods, then routes. One change from the base
config: the `retrieval_pipeline` is now a `routing_retrieval_pipeline` —
`diff examples/part_ii/example_00/config.yaml examples/part_ii/example_05/config.yaml`:

```yaml
  retrieval_pipeline:
    class_name: routing_retrieval_pipeline
    classifier: { class_name: structural }
    routes:
      lexical:  { ... retrievers: [sparse] }    # exact terms / identifiers
      semantic: { ... retrievers: [dense]  }    # paraphrases
    default:    { ... dense + sparse + rrf }     # unknown → hybrid
```

It reads the **same base store**.

## What it shows

Earlier examples showed dense and sparse pulling in opposite directions. Scored on a small labeled set
(`evalset.yaml`, 4 semantic + 4 lexical queries), that split is stark — **a different method wins on a
different *kind* of query**:

| hit@k by type | lexical | semantic |
|---------------|---------|----------|
| dense | 0.75 | **1.00** |
| sparse | **1.00** | 0.75 |
| hybrid | 1.00 | 0.75 |
| **routed** | **1.00** | **1.00** |

So routing's ceiling — the best method *on each type* — is **1.0 on both**, beating every fixed pipeline
(all 0.875). A **`StructuralQueryClassifier`** then reproduces the labels with **none of its own** (7/8 on
this set), so the routing happens automatically in production — no labels at query time.

> The scoreboard uses the labeled `query_type` so the router dispatches on the gold labels (*oracle*
> routing) — that isolates "does routing help?" from classifier accuracy, which is reported separately.

## Run it (script)

```bash
python -m examples.part_ii.example_05.run
```

Prints the scoreboard, the classifier-vs-labels check, then two routed `explain` breakdowns.

## Run it interactively (console)

Start the console on this example's config (run from the repo root):

```bash
python -m tarnrag.console examples/part_ii/example_05/config.yaml
```

It reads the **base store** from Example 00 (if you haven't built it: `tarn> ingest examples/docs/corpus-2`).
Now every query is **classified and routed**, and the `explain` breakdown shows the decision on a
`routed:` line at the top.

**1 · a lexical query routes to sparse.** Type:

```text
tarn> explain XQ-9920-A
```

The top of the breakdown reads `routed: query_type=lexical → route 'lexical'`, and the only per-retriever
table is **`sparse`** — the dense retriever isn't even run. `pump-maintenance` (the part) is rank 1.

**2 · a semantic query routes to dense.** Type:

```text
tarn> explain how do I keep a big metal container from rusting over time
```

Now it reads `routed: query_type=semantic → route 'semantic'`, the only per-retriever table is **`dense`**,
and `tank-corrosion` comes back on top — found by meaning, where a keyword search would struggle.

Type `quit` (or Ctrl-D) to exit.

→ That closes **Act A** (retrieval). Next, **Act B** turns retrieved passages into grounded answers:
**Example 06** is minimal generation (`ask`), and Example 07 adds the grounding check + abstain.
