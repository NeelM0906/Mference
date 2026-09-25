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

The later, separate Gemma QAT checkpoint is an explicit exception: its CLI and
server paths use the verified installed generation settings (temperature 1, Top-K 64,
Top-P 0.95, Min-P off and neutral penalties). Explicit controls still override
that profile. The sampler calculations and original checkpoint defaults stay
unchanged; see [QAT controls](RUNTIME_CONTROLS.md#gemma-qat).

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
macOS 26.6.2 (25G83), Apple Swift 6.4. The existing Gemma install is
`$HOME/llm-models/gemma4.gturbo`; its receipt's manifest hash and all
37 recorded file sizes passed. Before model execution, memory pressure showed
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
  plus the existing optional FlashNext fixture issue.
- Server: 163 tests / 23 suites, passed, 8.491 seconds.
- Repack: 169 tests / 23 suites, passed, 60.544 seconds.
- Vendored Jinja compatibility: 835 tests / 15 suites, passed, 2.180 seconds.

The Maple reproducer is `MapleQKNormRoPETests.swift:74`: expected max absolute
error at most `0.0078125`, observed `0.0234375`. It calls the Maple Q/K wrapper
and `maple_attention.metal` directly; those two sources and the test have no
working-tree diff. The exact failure was already recorded during the Gemma
thinking work and in the earlier Qwen sampling baseline.
It remains unresolved and the full package gate is not green. The FlashNext
fixture issue is the existing missing optional checkpoint assertion in
`FlashNextToyInstallTests.swift:17`.

A standalone `xcrun -sdk macosx metal -std=metal3.2 -c
Sources/Mference/Metal/Sampling/logit.metal -o /tmp/mference-llama-logit.air`
check exited 1 because the optional Metal Toolchain is absent. Nothing was
installed; the passing runtime Metal tests above supply shader verification.
These are correctness checks, not community performance measurements.


## Real Gemma entry-point checks

After the serial suite and release build exited, the owner check had no
matches and memory pressure showed 66% free. The server command was:

```bash
.build/release/MferenceServer --model $HOME/llm-models/gemma4.gturbo --port 18080 --max-context 4096
python3 /tmp/mference-llama-sampling-smoke.py
```

**Passed**, client exit 0: four real HTTP requests, JSON and SSE each with
omitted sampling fields and with `min_p=0.25`, `frequency_penalty=0.5`,
`presence_penalty=0.25`, `repeat_penalty=1.1`, `repeat_last_n=64`. Each request
used seed 777, a 16-token cap, and the prompt `Reply with exactly READY.`.
All returned `READY.` with 18 prompt / 3 completion tokens. SSE ended in
`[DONE]` and included usage. The server was bound only to loopback and exited
0 after the test session received Ctrl-C. No user-owned process was stopped.

Complete server timing lines:

```text
[2026-09-19T08:41:54Z] request chatcmpl-4e454755b9e24f41bf491405bfb03942 completed in 11.0s prompt=18 cached=0 completion=3 finish=stop prefill=10.637s
[2026-09-19T08:41:58Z] request chatcmpl-51be85122b9d46c3afc9251c7b09f1c2 completed in 3.5s prompt=18 cached=0 completion=3 finish=stop prefill=3.087s
[2026-09-19T08:42:01Z] request chatcmpl-1f63d5b4302a4b02a7a45054cfd95697 completed in 3.1s prompt=18 cached=0 completion=3 finish=stop prefill=2.782s
[2026-09-19T08:42:05Z] request chatcmpl-3731fff9c3104c42993d4ca74c9f3909 completed in 4.0s prompt=18 cached=0 completion=3 finish=stop prefill=3.635s
```

Full artifacts: `/tmp/mference-llama-sampling-server.log`,
`/tmp/mference-llama-sampling-smoke.log`, and
`/tmp/mference-llama-sampling-smoke-results.json`.
These short requests exercised fresh-prefill paths (`cached=0`); cached history
and resume penalty counts were verified by the focused raw-completion tests.

After confirming the server had exited, memory pressure showed 54% free.
The CLI smoke command was:

```bash
.build/release/MferenceCLI --model $HOME/llm-models/gemma4.gturbo --prompt 'The capital of France is' --max-new 8 --max-context 1024 --min-p 0.25 --frequency-penalty 0.5 --presence-penalty 0.25 --repeat-penalty 1.1 --repeat-last-n 64 --seed 777 --verify trusted-receipt
```

**Passed execution**, exit 0: parsed the new flags and generated eight tokens
through the real shared sampler. This raw-completion smoke checks wiring and
execution, not answer quality. Full stdout/stderr is
`/tmp/mference-llama-sampling-cli.log`; complete footer:

```text
[stop=maxTokens prefill=6tok/1.67s new=8tok decode=1.51s tok/s=5.311]
```

## Outcome accounting

- **Passed:** all-family defaults in core, CLI and server; Min-P implementation;
  repetition/presence/frequency formulas and 64-token window; real Gemma JSON,
  SSE and CLI execution; final release build and whitespace check.
- **Failed, pre-existing:** full-package gate due to the Maple assertion above;
  the optional FlashNext fixture remains a reported known issue.
- **Unverified:** generated-sequence equivalence with llama.cpp and real-weight
  runs for other installed model families. Neither is claimed by the shared
  policy/defaults proof. Different engines retain different numerical/RNG paths.

No model-run safety protocol deviation occurred. No benchmark comparison or
performance ceiling is claimed. `lesson_capture=none`: the verified work did
not establish a recurring or unusually severe reusable failure pattern; no
lesson file was written and composition is not recommended.
