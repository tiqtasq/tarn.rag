# Example 06 · Minimal generation (Act B opener)

Act A retrieved passages; **Act B reads them into an answer.** One change from the base config — the reader
model is `gpt-4o` (`diff examples/part_ii/example_00/config.yaml examples/part_ii/example_06/config.yaml`):

```yaml
  llm:
    model: gpt-4o   # was gpt-4o-mini — which intermittently declines answerable questions, even at temp 0
```

The `generation_pipeline` (single_hop reasoner + provenance assembler) was already pinned in the base; here
we finally exercise it with `ask`. It reads the **same base store**.

## What it shows

`single_hop` does one retrieve→read pass and returns the answer plus a **proof tree** — which chunk backed
each claim:

| | question | result |
|---|---|---|
| ✅ grounded | *"How do I service a centrifugal pump before starting it?"* | `check the mechanical seal for leaks` — cites *Centrifugal pump maintenance* |
| ⚠️ limitation | *"…maximum allowable working temperature of the cartridge mechanical seal?"* | `not provided` — **but still reported `grounded=True`, `abstained=False`** |

The limitation is the point of the opener: the minimal pipeline **trusts the reader**. There's no grounding
verification and no abstain, so a question the corpus can't answer comes back as a terse non-answer that the
system still labels grounded — it can't tell *"I answered"* from *"I couldn't."* **Example 07** adds the
grounding check + abstain.

> gpt-4o is honest and extractive — it answers when the passages support it and says "not provided"
> otherwise; it does not fabricate. (gpt-4o-mini, by contrast, intermittently declined the answerable
> question even at temperature 0, which is why this example pins gpt-4o.)

## Run it (script)

```bash
python -m examples.part_ii.example_06.run
```

## Run it interactively (console)

`ask` needs the LLM key in `OPENAI_LLM_KEY` (loaded from the repo-root `.env`) and `pip install '.[openai]'`.
Start the console on this example's config (run from the repo root):

```bash
python -m tarnrag.console examples/part_ii/example_06/config.yaml
```

It reads the **base store** from Example 00 (if you haven't built it: `tarn> ingest examples/docs/corpus-2`).

**1 · a grounded answer.** Type:

```text
tarn> ask how do I service a centrifugal pump before starting it?
```

The panel shows the answer — `check the mechanical seal for leaks` — and a **proof tree** beneath it: a `✓`
claim with a `cite` to *Centrifugal pump maintenance*. That's the chunk the answer is grounded in.

**2 · the limitation.** Type:

```text
tarn> ask what is the maximum allowable working temperature of the cartridge mechanical seal?
```

The corpus describes the cartridge mechanical seal but states no temperature, so the answer is honestly
`not provided` — yet the panel header still reads **`answer (grounded)`** and the pipeline does **not**
abstain. Nothing flags that the question went unanswered. That missing signal is what Example 07 adds.

Type `quit` (or Ctrl-D) to exit.

→ Next: **Example 07** adds a **grounding check + abstain** — verify each claim against its evidence and
refuse when the answer isn't supported.
