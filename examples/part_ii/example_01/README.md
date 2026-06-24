# Example 01 · Dense-only retrieval

The simplest retrieval: a single **dense** retriever — vector KNN over the embedded chunks. This step
reuses the base store **and** config from [Example 00](../example_00) (`retrieval_pipeline: dense + identity
fuser`); there's no config change yet — the first config *delta* is Example 02's hybrid.

## What it shows

Dense embeddings match by **meaning**. That's their strength — and their blind spot.

| Probe | Query | Dense |
|-------|-------|-------|
| ✅ **good** | *"service a rotary fluid machine before powering it up"* — a paraphrase of `pump-maintenance` with **no shared words** | **HIT** (rank 3): found by meaning. A keyword search misses it entirely (BM25 ranks it 12th). |
| ❌ **bad** | `XQ-9920-A` — an opaque **part number** | **MISS**: a bare identifier has no meaning to embed, so the right doc ranks 5th — out of the top 4. |

The `explain` breakdown makes the *why* visible: for the paraphrase the dense retriever's candidates surface
`pump-maintenance` near the top; for the part number it's pushed down, because the chunk's embedding is
"about pump maintenance", not about the string `XQ-9920-A`.

## Run it

```bash
python -m examples.part_ii.example_01.run
```

…or `explain` both queries interactively over the base config:

```bash
python -m tarnrag.console examples/part_ii/example_00/config.yaml
tarn> explain service a rotary fluid machine before powering it up
tarn> explain XQ-9920-A
```

→ Next: **Example 02** adds a **sparse (BM25)** retriever — exact-term matching, the natural complement to
dense — and fuses the two with RRF. The part-number query starts hitting, while the paraphrase still works.
