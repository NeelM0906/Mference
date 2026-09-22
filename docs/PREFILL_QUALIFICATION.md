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
| Gemma, bounded / resident experts | `ProductionPrefillContractTests`: ragged cold/warm appends, multiple chunks, sliding-ring boundary, cancellation/reset and decode handoff; installed resident/eight-slot gate at `ba14ff4` matches all full logits across 1,024-token window and eight continuation steps, zero replay | [Installed evidence](RELEASE_VALIDATION_2026-09-20.md); not independent upstream quality parity or fresh physical low-RAM qualification |
| Qwen 3.6, bounded / resident experts | Same synthetic matrix plus `QwenRunnerTests`/hybrid-state parity; installed resident/eight-slot gate at `ba14ff4` matches ragged append heads and eight full continuation rows, with cancellation/dirty rejection/exact reset | [Installed evidence](RELEASE_VALIDATION_2026-09-20.md); reduced slots are not smaller-RAM hardware qualification |
| Qwen 3.8 dense / paged / spilled KV | `Qwen38ForwardRunnerTests`, `Qwen38PagedKVParityTests`, `Qwen38BlockedPrefillTests`; observed counts and numerical/continuation checks; `7505e82` adds actual cancellation after GPU writes in all three backends, dirty rejection and exact reset/next logits | Installed fine-tunes need their own numerical gate; fixture recovery is not physical low-RAM qualification |
| Swift-Qwen | PR #33 source/template, installed-reference and MTP gates; `d2b84f1` and four-row tiled `ee0bfb9` pass the installed numerical/state/MTP retest without changing tolerances | Candidate only; default-effort community attempts truncate without visible answers; broader quality/latency/hardware evidence required before promotion |
| Flash-Next, bounded / resident | Synthetic state gates, installed short A/B, plus `c8f3298`: actual TensorOps encodings with 1,024-token chunks; exact resident/16-slot logits across 2,048-token cutover, eight continuation rows and cancellation/reset | [September 20 evidence](RELEASE_VALIDATION_2026-09-20.md); wider contexts, old-OS real fallback and physical hardware remain separate |
| GLM, bounded / resident | PR #34 matrix plus installed gates at `ed69598` and post-ragged-read-fix `84bfefc`: exact resident/16-slot full logits at 33/2047/2051/2083 tokens and eight continuation steps, zero replay, cancellation/dirty rejection/exact reset | [Latest installed evidence](RELEASE_VALIDATION_2026-09-20.md#installed-glm-recheck-after-ragged-tile-correction); not longer-context, smaller-hardware or throughput qualification |
| DeepSeek, bounded / resident | Phase 4 PR #35: below/across/above sparse cutover, cache slots 8/16/resident, exact state and cancellation/reset; opt-in installed gate | Installed results and limitations live in `DEEPSEEK_V4_FLASH.md` |
| MiniCPM5 dense / paged / spilled KV | `MiniCPM5ForwardRunnerTests`, `MiniCPM5PagedKVTests`; observed counts and golden/continuation checks; `7505e82` exercises actual warm-append cancellation/reset in all three backends and compares exact next logits | Representative full-selection/5-page-spill fixture; not every sparse budget/context or new hardware qualification |
| Maple | Existing fixture plus installed 16/8-slot factory gate at `e549ea6`: exact full logits across 512-token window, eight continuation steps, cancellation/dirty rejection/reset, zero replay | [September 20 evidence](RELEASE_VALIDATION_2026-09-20.md); resident-expert mode is not implemented by this runner; not every context/hardware |
| Inkling | Nonzero CI fixture at `3462477`: resident/eight-slot exact boundary, continuation and recovery results plus bounded sequential-reference error; installed resident/16-slot factory gate at `e549ea6`: exact full logits across 512-token window, eight continuation steps and cancellation/dirty rejection/reset, zero replay | [September 20 evidence](RELEASE_VALIDATION_2026-09-20.md); installed weights remain env-gated; wider hardware/context qualification is separate |

PRs #33–36 merged into `main` on September 18 (`c67e857`); PR #37 merged on
September 20 (`049bdcd`). Further qualification in
[PR #38](https://github.com/NeelM0906/Mference/pull/38) merged September 21
(`4ff63ff`), the [0.1.0 source-release baseline](RELEASE_GATE_0.1.0.md).
The all-family invariant remains **not fully qualified**
until the outstanding cells have evidence, including OS/GPU fallback paths,
partial-chunk cancellation across the remaining runners, and real checkpoints.
Do not hide a missing combination behind a new unsupported-context error or
rename host-side full-model replay as batching.

September 20 evidence correction: `d4f5dd4` fixes unaligned resident GPU
mappings and the Flash-Next synthetic fixture. Earlier synthetic NaN-bit
equality is not numerical parity. The replacement finite/nonzero gate asserts
bounded FP16-scale error and exact greedy continuation; rollback remains
byte-exact. Installed aligned-checkpoint evidence is separate. See the
[detailed correction](RELEASE_VALIDATION_2026-09-20.md#resident-gpu-alignment-and-target-state-follow-up).

## Roadmap status

- Phase 1: actual execution and memory diagnostics landed on `main`.
- Phase 2: Swift-Qwen implementation and initial qualification merged in PR #33;
  numerical retest passes. The [matched-low screen](families/QWEN_MATCHED_QUALIFICATION_2026-09-18.md)
  passes 54/60 cases versus base 59/60 with 6.77% fewer completion tokens;
  the [five-seed xhigh screen](QWEN_SOURCE_EFFICIENCY_V1.md) passes 293/300
  attempts versus base 297/300 with 14.992% fewer completion tokens. A parser
  compatibility issue is fixed separately. The default-policy screen passes
  Swift 180/180 versus base 177/180 with 20.088% fewer completion tokens; its
  policies differ and the base failures share one punctuation ambiguity.
  Broader quality/performance/hardware qualification remains.
- Phase 3: bounded GLM prefill implementation merged in PR #34; pinned install
  and current resident/16-slot sparse-cutover/continuation/recovery gate now
  complete on the 256 GiB M3 Ultra; broader hardware/performance remains open.
- Phase 4: DeepSeek sparse-cutover implementation merged in PR #35; its own report
  records the measured correctness status, separately from throughput.
- Phase 5: execution-contract coverage extended here; the gaps above remain
  explicit. A green ordinary CI run does not execute env-gated real-model tests.
- Phase 6: Flash-Next/GLM optimization and matched performance experiments are
  not complete. [Flash-Next resident and 16-slot measurements](RELEASE_PERFORMANCE_2026-09-20.md)
  now pass completed-answer gates and show lower prefill-plus-decode time,
  but resident short/medium prefill regresses. The separate 16-slot revision
  has modest 0.5%/1.2%/2.3% median generation-time reductions; short-case and
  all decode ranges overlap. GLM's new default-profile warmups truncate on
  short/medium cases and memory preflight stops before the long case; no
  completed-answer comparison follows from them. GLM comparisons and native MTP
  integration remain open. No unmeasured speedup or default change is claimed.
- Phase 7: real UI streaming, tool loops, history, cancellation and model-switch
  recovery have been tested; a [support table](RELEASE_SUPPORT.md) distinguishes
  established paths from candidates. The separately frozen 60-case screen has
  run for matched-low base/Swift only, not every release profile. Neither this
  screen nor the earlier 12-task screen or UI checks is
  a broad quality claim or support evidence for hardware not tested here.

The GLM installation blocker is resolved. Per the September 22 release-scope
decision, physical 16/24-GiB qualification is not a blocker for this release;
those profiles remain unqualified. Kernel optimizations must pass correctness and
repeatable matched measurements before promotion. Never infer 24 GB support
from a 256 GB host with a reduced expert-slot setting.

## Phase 5 validation record

Code `f2f45f5`; Mac Studio Mac15,14, M3 Ultra (32 CPU cores), 256 GiB;
macOS 26.3 (25D125); Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3`).

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh \
  --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'ProductionPrefillContractTests|PrefillRoutedTileSchedulerTests|QwenRunnerTests|FlashNextChunkedPrefillTests|Qwen38PagedKVParityTests|Qwen38BlockedPrefillTests|MiniCPM5PagedKVTests'
```

Exit 0; build `5.40s`;
`Test run with 31 tests in 7 suites passed after 8.826 seconds.`
Gemma/Qwen eight-slot and resident full-logit results match after warm appends
and decode; reset reproduces them. All observed prefill tokens are batched.
Development failures exposed the eight-slot rejection, a loader fixture's
unsupported top-2 geometry, and its deliberately non-finite norm byte pattern.
The runtime budget bug is fixed; historical loader-fixture defaults remain
unchanged and inference tests request top-8 plus finite unit norms explicitly.

The existing Inkling installation then ran alone after OS/toolchain, disk,
memory-pressure and process checks (788 GiB free disk, memory-pressure free
percentage 98). All 48 receipt-file sizes matched before launch; the loader
used default full-SHA verification.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
MFERENCE_INKLING_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/inklingsmall.gturbo \
Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter shortExplanationHasNoExclamationBurst
```

Exit 0; build `11.60s`. Full footer:

```text
Test shortExplanationHasNoExclamationBurst() passed after 87.056 seconds.
Suite InklingGenerationRegressionTests passed after 87.056 seconds.
Test run with 1 test in 1 suite passed after 87.056 seconds.
```

This runs the existing greedy short-explanation regression (120-token cap)
with the default 16-slot streamed profile. It passes the no-exclamation-burst
and prose-length assertions and now also confirms zero prefill replay. No
download, model copy, cache purge or profiling was used. It is a debug-build
correctness test, not a community performance measurement or a broader
quality/hardware recommendation.

## Combined PR validation

The local `codex/roadmap-validation` branch combines PRs #33–36 without merging
them into `main`. On the same Mac Studio/toolchain recorded above, revision
`f60a74b` passed the full serial package suite:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Exit 0; build `19.04s`; full footer:

```text
Test run with 1227 tests in 216 suites passed after 254.968 seconds with 1 known issue.
```

The known issue is the existing absent optional Flash-Next toy checkpoint.
Ordinary package runs do not enable installed-model environment gates. Real
Inkling results are above; the separate resident and 16-slot installed
DeepSeek cutover/continuation gates are recorded in
[the DeepSeek validation record](DEEPSEEK_V4_FLASH.md) in PR #35.

The combined release build ran on `d769a7e`, whose production source is
identical to `f60a74b` (only qualification documentation changed):

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Exit 0; full footer: `Build complete! (59.29s)`.

The dedicated scratch path reuses the compatible Xcode build directory; the
default `.build` contained artifacts from a different toolchain and was not
purged. No model download/copy, profiling or benchmark protocol was used.
These are build/correctness results, not throughput measurements.

Swift 6.1 CI subsequently exposed a type-checker timeout in the new test's
single token-generation expression. Commit `ce2d0a9` replaces that expression
with an explicitly typed loop and explicitly types the execution report; no
production code or numerical limits changed. The full serial command above
was repeated on combined revision `e0c02ba5` after that fix, with 788 GiB free
disk, memory-pressure free percentage 98 and no existing model owner:

```text
Build complete! (11.45s)
Test run with 1227 tests in 216 suites passed after 255.092 seconds with 1 known issue.
```

Exit 0; the same optional-fixture known issue remains. Markdown validation
also passed for all 64 files, and the Open WebUI adapter/task-screen/reference
script unit suites passed (3/3/4 tests respectively). The reference script's
tests require its declared `uv run` environment; system Python lacked MLX, so
the successful rerun used that documented runner. This did not perform model
inference or download weights.
