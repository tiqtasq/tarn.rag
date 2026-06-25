# Example 07 · Grounding check + abstain

Example 06 *trusted* the reader. This adds **verification** to the generation_pipeline — the only change
(`diff examples/part_ii/example_06/config.yaml examples/part_ii/example_07/config.yaml`):

```yaml
    grounding_checker: { class_name: llm_grounding }   # verify each claim against its cited evidence
    abstain: true                                      # refuse rather than return an unsupported answer
    min_grounded: 1.0                                  # every claim must be grounded
    refusal: "I don't have enough grounded evidence to answer that confidently."
```

It reads the **same base store**.

## What it shows

- **Verification (live).** A `grounding_checker` re-reads each claim against its cited evidence and stamps
  it grounded/ungrounded. On the answerable question the claim holds up, so the proof's `✓` is now *earned*
  (Example 06 simply assumed it), and the question is still answered.
- **Abstain (constructed).** The `abstain` policy refuses outright when fewer than `min_grounded` of the
  claims are grounded, returning `refusal` instead of an unsupported answer.

> **Why the abstain demo is constructed.** gpt-4o is honest: it answers when the passages support it and
> says *"not provided"* otherwise — it does **not** fabricate, so there's no reproducible way to make a live
> `ask` abstain. So, exactly as the library's own grounding tests do, the script plants one unsupported
> claim — *"the pump carries a ten-year warranty"*, cited to a maintenance passage that says nothing about a
> warranty — and runs it through a grounding+abstain pipeline (`heuristic_grounding`, content-word overlap,
> no LLM → deterministic). The check finds the claim disjoint from its evidence and the pipeline **refuses**.

## Run it (script)

```bash
python -m examples.part_ii.example_07.run
```

Prints the live verified answer, then the constructed abstain (the refusal replacing the fabricated answer).

## Run it interactively (console)

`ask` needs the LLM key in `OPENAI_LLM_KEY` (repo-root `.env`) + `pip install '.[openai]'`. Start the
console on this config (from the repo root):

```bash
python -m tarnrag.console examples/part_ii/example_07/config.yaml
```

It reads the **base store** from Example 00 (if you haven't built it: `tarn> ingest examples/docs/corpus-2`).

```text
tarn> ask how do I service a centrifugal pump before starting it?
```

You get the same grounded answer as Example 06 — `check the mechanical seal for leaks` with a `✓` proof
step citing *Centrifugal pump maintenance*. The difference is invisible but real: `llm_grounding` ran a
second pass to **verify** that claim against the passage before the answer was returned, and had it *not*
held up, the `abstain` policy would have replaced the answer with the refusal. Seeing that refusal needs an
unsupported claim, which the honest reader won't produce — so run the **script** above for the constructed
illustration of the abstain.

Type `quit` (or Ctrl-D) to exit.

→ Next: **Example 08** — multi-hop: decompose a question whose answer spans two documents, which the
single retrieve→read pass can't gather on its own.
