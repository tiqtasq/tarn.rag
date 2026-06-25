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

## Run it (script)

```bash
python -m examples.part_ii.example_04.run
```

## Run it interactively (console)

Start the console on this example's config (run from the repo root):

```bash
python -m tarnrag.console examples/part_ii/example_04/config.yaml
```

This example uses its **own** store (`structured.db`), so build it first — *this* is the re-ingest, with
the structure-aware chunker:

```text
tarn> ingest examples/docs/corpus-2
tarn> status
```

`status` reports ~**21 chunks** (vs the base store's ~19) — the extra chunks are the **section parents**
the structure-aware chunker adds.

**1 · the fragmentation from Example 03 is fixed.** Type:

```text
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
```

The top result is now a long (~700-char) `compressor-startup` passage — the **whole "startup" section** —
and its text contains *every* step, including "loading sequence". (Run the same query under Example 03's
config and the top result is a ~300-char fragment that stops after the first steps.)

**2 · auto-merge consolidates fragments.** Type:

```text
tarn> explain compressor startup and shutdown
```

Notice a new **`merged`** stage in the breakdown — it doesn't appear in Examples 01–03. Auto-merge collapses
the retrieved sibling chunks of one section into their parent: in that stage's `Δ` column a section parent
appears (marked `＋`) and the redundant leaves drop out. The `final` table then holds **one** consolidated
`compressor-startup` section plus a spread of *other* documents (compressor-models, lubrication-spec, …),
instead of being dominated by one document's pieces.

Type `quit` (or Ctrl-D) to exit.

→ Next: **Example 05** closes Act A with **query routing** — classify each query and dispatch it to the
per-type-best retrieval method, plus a scoreboard comparing dense / sparse / hybrid / routed.
