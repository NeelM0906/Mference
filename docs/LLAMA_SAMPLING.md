# Sampling compatibility with llama.cpp

The user requested llama.cpp's sampling defaults for every Mference model,
plus Min-P and matching repetition, presence and frequency penalty behavior.
Thinking, context allocation, output limits, model weights and cache policy
are outside this change.

The implementation reference is llama.cpp revision
[`b23701f77d47dad9de834d59ebfcbe25c9e8b46f`](https://github.com/ggml-org/llama.cpp/tree/b23701f77d47dad9de834d59ebfcbe25c9e8b46f),
inspected on 2026-09-19:

- [Default values and chain order](https://github.com/ggml-org/llama.cpp/blob/b23701f77d47dad9de834d59ebfcbe25c9e8b46f/common/common.h).
- [Chain construction](https://github.com/ggml-org/llama.cpp/blob/b23701f77d47dad9de834d59ebfcbe25c9e8b46f/common/sampling.cpp).
- [Top-P, Min-P and penalty calculations](https://github.com/ggml-org/llama.cpp/blob/b23701f77d47dad9de834d59ebfcbe25c9e8b46f/src/llama-sampler.cpp).

## Defaults and calculations

| Parameter | CLI, server and GenerationConfig default |
|---|---:|
| Temperature | 0.8 |
| Top-K | 40 |
| Top-P | 0.95 |
| Min-P | 0.05 |
| Repetition penalty | 1.0 |
| Presence penalty | 0.0 |
| Frequency penalty | 0.0 |
| Penalty history window | 64 tokens |

These are llama.cpp's built-in fallback values. llama.cpp can override its
defaults using GGUF metadata; Mference does not load GGUF or adopt such
metadata automatically. Explicit Mference request/CLI values override defaults.
The values are declared once, in `GenerationConfig.init`; the CLI and server
read `GenerationConfig.defaults` only for a parameter the caller omitted.

1. On post-softcap model logits, count occurrences in the most recent
   `repeat_last_n` prompt/generated tokens. `0` disables all penalties; `-1`
   uses the complete effective history. For each seen token, divide positive
   logits by repetition penalty or multiply nonpositive logits by it; then
   subtract `count * frequency_penalty + presence_penalty`.
2. Keep the largest K logits, and apply Top-P to probabilities normalized over
   those K candidates, including the token that crosses the cumulative bound.
3. Retain candidates with probability at least `min_p * peak_probability`.
   The peak always survives; zero disables Min-P.
4. Apply temperature to the survivors and draw. Temperature zero selects the
   argmax after penalties; the truncation filters cannot remove that maximum.

The existing Top-64 reduction also serves K below 64, avoiding a slow
per-candidate vocabulary scan for the new default K=40. Both that path and the
general kernel use the same filter order. Min-P also works with Top-K disabled
when Top-P is 1. Full-vocabulary Top-P remains unsupported and is rejected;
supported nonzero K remains 1...256. Presence/frequency retain the API's finite
-2...2 range. Arbitrary sampler-chain customization is not added.

The previous Mference contract used Top-P before Top-K, a whole-history
presence/repetition window, no frequency penalty, and zero-only Min-P. Tests
asserting those policies must change for this explicit user-requested contract
change. The old softcap inverse/clamp also could not represent boosted logits
above the cap; direct GPU penalties remove that mismatch.

Sampling mathematics and defaults do not imply identical generated text or
seed streams across engines. Mference retains its own RNG and FP16 probability
storage, and checkpoint quantization/runtime arithmetic can differ.

## Validation status

Validation ran on the dirty working tree based on commit
`4534be371e4c472c5fecf11db5dee9104aaff074`, Mac14,2 / Apple M2 / 16 GiB,
macOS 26.6.2 (25G83), Apple Swift 6.4. The real-checkpoint HTTP and CLI
checks of this record ran on a family since retired (see
[Retired model families](RETIRED_FAMILIES.md)), so they are not repeated here.
Before model execution, memory pressure showed
61% free and disk had 9.6 GiB available. No checkpoint was downloaded or copied.

The first prerequisite check found the user's existing server (PID 4034).
The user stopped it; the subsequent owner check returned no matches. Therefore
no pre-edit runtime red test was available. Expected sampling-policy changes
come from the pinned upstream contract, not regenerated golden outputs.

Focused command:

```bash
Scripts/test.sh --filter 'LlamaSamplingTests|SamplerTests|SampleTests|SampleTopK64Tests|LogitSoftcapSoftmaxTests|RawCompletionLoopTests|CLIArgumentsTests|OpenAIValidationTests|HTTPServerTests'
```

**Passed**, exit 0: 84 core/CLI tests in 7 suites (17.470 seconds), plus 86
server tests in 7 suites (9.131 seconds). Full output:
`/tmp/mference-llama-sampling-focused-fixed.log`. This includes actual Metal
kernel compilation/execution, independent post-softcap penalty probabilities,
Min-P disabled/boundary/tied-peak behavior, K=40 and optimized/general parity,
Top-P's crossing token, reset/resume with prompt/generated frequency counts,
and JSON/SSE HTTP fields in single-model/library modes. All six chat dialects
receive the shared defaults.

The first focused run found one CLI help formatting defect: a punctuation mark
attached to the alias target looked like an extra option. The help text was
corrected; the exact existing assertion passed unchanged on rerun. Its original
output is `/tmp/mference-llama-sampling-focused.log` (exit 1).

`swift build -c release` passed, exit 0, `Build complete! (469,21 sec)`;
output: `/tmp/mference-llama-sampling-build.log`. The final
`swift build -c release` refresh also passed, exit 0,
`Build complete! (122,83 sec)`; output:
`/tmp/mference-llama-sampling-final-build.log`.

Broader command: `Scripts/test.sh --skip-build`, **failed**, exit 1; full output:
`/tmp/mference-llama-sampling-full.log`.

- Core: 986 tests / 185 suites, 351.135 seconds, one failing Maple assertion
  plus an existing optional-fixture issue for a since-retired family.
- Server: 163 tests / 23 suites, passed, 8.491 seconds.
- Repack: 169 tests / 23 suites, passed, 60.544 seconds.
- Vendored Jinja compatibility: 835 tests / 15 suites, passed, 2.180 seconds.

The Maple reproducer is `MapleQKNormRoPETests.swift:74`: expected max absolute
error at most `0.0078125`, observed `0.0234375`. It calls the Maple Q/K wrapper
and `maple_attention.metal` directly; those two sources and the test have no
working-tree diff. The exact failure was already recorded in earlier work and
in the earlier Qwen sampling baseline.
It remains unresolved and the full package gate is not green. The optional-fixture
issue was a missing optional checkpoint assertion whose tests were removed
with their family on 2026-11-18.

A standalone `xcrun -sdk macosx metal -std=metal3.2 -c
Sources/Mference/Metal/Sampling/logit.metal -o /tmp/mference-llama-logit.air`
check exited 1 because the optional Metal Toolchain is absent. Nothing was
installed; the passing runtime Metal tests above supply shader verification.
These are correctness checks, not community performance measurements.


## Outcome accounting

- **Passed:** all-family defaults in core, CLI and server; Min-P implementation;
  repetition/presence/frequency formulas and 64-token window; final release
  build and whitespace check.
- **Failed, pre-existing:** full-package gate due to the Maple assertion above;
  the optional-fixture issue above was a reported known issue.
- **Unverified:** generated-sequence equivalence with llama.cpp and real-weight
  runs for other installed model families. Neither is claimed by the shared
  policy/defaults proof. Different engines retain different numerical/RNG paths.

No model-run safety protocol deviation occurred. No benchmark comparison or
performance ceiling is claimed. `lesson_capture=none`: the verified work did
not establish a recurring or unusually severe reusable failure pattern; no
lesson file was written and composition is not recommended.
