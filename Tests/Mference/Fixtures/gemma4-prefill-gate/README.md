# Gemma prefill equivalence gate: saved answers

`answers.json` holds the text the two Gemma checkpoints produced for the frozen
community prompts `medium-review` and `long-synthesis`
(`docs/benchmark-prompts/real-generation-v1/`). `GemmaPrefillEquivalenceGateTests`
teacher-forces the control and the candidate prefill through the same answers,
so each checkpoint is scored on both its own text and the other checkpoint's.

| Key | Checkpoint |
| --- | --- |
| `medium-review.qat`, `long-synthesis.qat` | Gemma 4 QAT (`gemma4qat.gturbo`) |
| `medium-review.original`, `long-synthesis.original` | Gemma 4 (`gemma4.gturbo`) |

Sampled with the release `MferenceCLI --messages-file <prompt> --max-new 1024
--max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`, seed 20260722
for `medium-review` and 20260723 for `long-synthesis`, min-P 0 and neutral
penalties, on an Apple M2 in September 2026. The answers
are inputs, not expectations: regenerating them changes which tokens the gate
scores, not what it requires.
