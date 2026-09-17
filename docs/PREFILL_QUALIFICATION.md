# Prefill qualification matrix

This is an evidence map, not a declaration that every hardware/context
combination is qualified. An executed `.chunked` report must contain the actual
number of batched tokens, zero replay, and the observed chunk sizes. A factory's
requested mode is not evidence. Explicit sequential reference modes remain
available for correctness comparisons.

The matrix exposed a production bug in the shared Gemma/Qwen runner: its
depth-1, eight-expert tiles rejected the supported eight-slot cache because
they required sixteen slots. Tile width now follows the actual budget (four
experts at eight slots), preserving the overlap depth and grouped batched
execution. Resident and larger-budget configurations keep their existing width.

## Automated coverage

| Family / path | Production-dispatch evidence | Limits |
| --- | --- | --- |
| Gemma, bounded / resident experts | `ProductionPrefillContractTests`: ragged cold/warm appends, multiple chunks, sliding-ring boundary, reset and decode handoff | Small loader fixture; execution/state reproducibility, not independent model-quality parity |
| Qwen 3.6, bounded / resident experts | Same matrix plus `QwenRunnerTests` and existing hybrid-state parity suites | Small fixture is not smaller-RAM hardware qualification |
| Qwen 3.8 dense / paged / spilled KV | `Qwen38ForwardRunnerTests`, `Qwen38PagedKVParityTests`, `Qwen38BlockedPrefillTests`; observed counts alongside existing numerical/continuation checks | Installed fine-tunes need their own numerical gate |
| Swift-Qwen | Phase 2 PR #33 includes source/template, installed-reference, numerical and MTP gates | Candidate only: two installed prefill mean-error probes missed the declared threshold; no default promotion |
| Flash-Next, bounded / resident | `FlashNextChunkedPrefillTests`: exact logits/state, warm appends, six continuation rows and zero replay | Synthetic INT4 router fixture does not qualify every real router precision or hardware profile |
| GLM, bounded / resident | Phase 3 PR #34: cutover/warm-appends/partial-chunks matrix, exact streamed/resident logits and cancellation/reset | Existing completed GLM installation not found; real streamed qualification remains blocked |
| DeepSeek, bounded / resident | Phase 4 PR #35: below/across/above sparse cutover, cache slots 8/16/resident, exact state and cancellation/reset; opt-in installed gate | Installed results and limitations live in `DEEPSEEK_V4_FLASH.md` |
| MiniCPM5 dense / paged / spilled KV | `MiniCPM5ForwardRunnerTests`, `MiniCPM5PagedKVTests`; observed counts alongside existing golden/continuation checks | Small fixture, not a new hardware recommendation |
| Maple | `MapleForwardRunnerTests` asserts batched execution and compares continuation logits | Existing sparse/zero fixture does not replace a broad installed-model gate |
| Inkling | `InklingGenerationRegressionTests.shortExplanationHasNoExclamationBurst` now also asserts actual batched prefill and zero replay | Env-gated existing install; not run by ordinary CI; a small full-runner fixture and wider boundary matrix remain needed |

PR references identify separate reviewable changes targeting `main`, not changes
already merged into it. The all-family invariant remains **not fully qualified**
until the outstanding cells have evidence, including OS/GPU fallback paths,
partial-chunk cancellation across the remaining runners, and real checkpoints.
Do not hide a missing combination behind a new unsupported-context error or
rename host-side full-model replay as batching.

## Roadmap status

- Phase 1: actual execution and memory diagnostics landed on `main`.
- Phase 2: Swift-Qwen implementation and initial qualification in PR #33;
  numerical-gate investigation and broader quality/hardware qualification remain.
- Phase 3: bounded GLM prefill implementation in PR #34; real install required
  to finish its qualification.
- Phase 4: DeepSeek sparse-cutover implementation in PR #35; its own report
  records the measured correctness status, separately from throughput.
- Phase 5: execution-contract coverage extended here; the gaps above remain
  explicit. A green ordinary CI run does not execute env-gated real-model tests.
- Phase 6: Flash-Next/GLM optimization and matched performance experiments are
  not complete. No unmeasured speedup or default change is claimed.
- Phase 7: the Phase 2 frozen 12-task screen and UI checks are an initial
  screen, not the final warmup-plus-three-run comparison, broad quality claim,
  or support table for hardware not tested here.

The missing GLM installation and untested smaller-memory hardware are external
qualification requirements. Kernel optimizations must pass correctness and
repeatable matched measurements before promotion. Never infer 24 GB support
from a 256 GB host with a reduced expert-slot setting.
