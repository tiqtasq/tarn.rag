# Example 04 · Structure-aware chunking + auto-merge

The first step that changes the **representation**, so it re-ingests into its **own store**
(`structured.db`, not the shared base store). Three changes from Example 03 —
`diff examples/part_ii/example_03/config.yaml examples/part_ii/example_04/config.yaml`:

```yaml
  database:
    document_url: sqlite:///./examples/part_ii/store/structured.db   # ← separate store
  ...
    - class_name: Chunk
      chunker: { class_name: structure_aware, max_chars: 450 }       # ← was recursive
  ...
    merger: { class_name: auto_merge }                               # ← added
```

## What it shows

`recursive` chunks are flat, overlapping windows — so a long section is sliced into pieces (Example 03's
fragmentation). `structure_aware` instead splits on the document's headings into a tree of **leaf chunks +
section parents**: a chunk over a whole section, emitted where a section groups ≥2 children.

| | Example 03 (recursive) | Example 04 (structure-aware) |
|---|---|---|
| *"all the steps in the startup procedure?"* (top-1) | ❌ a **fragment** — the first step-group only | ✅ the whole **section** (parent chunk) — complete |

And **auto-merge**: when several sibling *leaf* fragments of one section are retrieved, they're collapsed
into their parent. For *"compressor startup and shutdown"* that turns 5 redundant `compressor-startup`
pieces into the consolidated section — leaving room for **5 distinct documents** in the top 6 instead of
being dominated by one. Watch the `merged` stage in the `explain` breakdown.

> Parents are themselves directly searchable (there's no level filter), so the *fix* above is really the
> structure-aware representation; auto-merge then keeps a section's retrieved leaves from crowding the
> top-k.

## Run it

```bash
python -m examples.part_ii.example_04.run
```

→ Next: **Example 05** closes Act A with **query routing** — classify each query and dispatch it to the
per-type-best retrieval method, plus a scoreboard comparing dense / sparse / hybrid / routed.
