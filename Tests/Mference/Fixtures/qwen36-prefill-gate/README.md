# Qwen 3.6 prefill equivalence gate: saved answers

`answers.json` holds the text Qwen 3.6 35B-A3B (`qwen36.gturbo`) produced for
the frozen community prompts `medium-review` and `long-synthesis`
(`docs/benchmark-prompts/real-generation-v1/`). `Qwen36PrefillEquivalenceGateTests`
teacher-forces the control and the candidate prefill through these answers and
through two Gemma 4 answers to the same prompts, so the model is scored on both
its own text and another model's.

| Key | Checkpoint |
| --- | --- |
| `medium-review.qwen36`, `long-synthesis.qwen36` | Qwen 3.6 (`qwen36.gturbo`) |
| `medium-review.gemma4`, `long-synthesis.gemma4` | Gemma 4 (`gemma4.gturbo`) |

Sampled with the release `MferenceCLI --messages-file <prompt> --max-new 1024
--max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --verify
full-sha256`, seed 20260722 for `medium-review` and 20260723 for
`long-synthesis`, min-P 0 and neutral penalties, on an Apple M1 Max in
November 2026 (MD5 `1555ee40…` and `0a5f63a4…`, the community-protocol
baselines). The answers are inputs, not expectations: regenerating them changes
which tokens the gate scores, not what it requires.

The two Gemma 4 answers were sampled with the same sampling flags and seeds on an
Apple M2 in September 2026 (they were the `.original` keys of the former Gemma
prefill gate's fixture, copied here unchanged when that gate was removed).
