# Prefill qualification matrix

This is an evidence map, not a declaration that every hardware/context
combination is qualified. An executed `.chunked` report must contain the actual
number of batched tokens, zero replay, and the observed chunk sizes. A factory's
requested mode is not evidence. Explicit sequential reference modes remain
available for correctness comparisons.

The matrix exposed a production bug in the shared Qwen 3.6 MoE runner: its
depth-1, eight-expert tiles rejected the supported eight-slot cache because
they required sixteen slots. Tile width now follows the actual budget (four
experts at eight slots), preserving the overlap depth and grouped batched
execution. Resident and larger-budget configurations keep their existing width.

## Automated coverage

| Family / path | Production-dispatch evidence | Limits |
| --- | --- | --- |
| Qwen 3.6, bounded / resident experts | Same synthetic matrix plus `QwenRunnerTests`/hybrid-state parity; installed resident/eight-slot gate at `ba14ff4` matches ragged append heads and eight full continuation rows, with cancellation/dirty rejection/exact reset | [Installed evidence](RELEASE_VALIDATION_2026-09-20.md); reduced slots are not smaller-RAM hardware qualification |
| Qwen 3.8 dense / paged / spilled KV | `Qwen38ForwardRunnerTests`, `Qwen38PagedKVParityTests`, `Qwen38BlockedPrefillTests`; observed counts and numerical/continuation checks; `7505e82` adds actual cancellation after GPU writes in all three backends, dirty rejection and exact reset/next logits | Installed fine-tunes need their own numerical gate; fixture recovery is not physical low-RAM qualification |
| MiniCPM5 dense / paged / spilled KV | `MiniCPM5ForwardRunnerTests`, `MiniCPM5PagedKVTests`; observed counts and golden/continuation checks; `7505e82` exercises actual warm-append cancellation/reset in all three backends and compares exact next logits | Representative full-selection/5-page-spill fixture; not every sparse budget/context or new hardware qualification |
| Maple | Existing fixture plus installed 16/8-slot factory gate at `e549ea6`: exact full logits across 512-token window, eight continuation steps, cancellation/dirty rejection/reset, zero replay | [September 20 evidence](RELEASE_VALIDATION_2026-09-20.md); resident-expert mode is not implemented by this runner; not every context/hardware |

PRs #33–36 merged into `main` on September 18 (`c67e857`); PR #37 merged on
September 20 (`049bdcd`). Further qualification in
[PR #38](https://github.com/NeelM0906/Mference/pull/38) merged September 21
(`4ff63ff`), the [0.1.0 source-release baseline](RELEASE_GATE_0.1.0.md).
The all-family invariant remains **not fully qualified**
until the outstanding cells have evidence, including OS/GPU fallback paths,
partial-chunk cancellation across the remaining runners, and real checkpoints.
Do not hide a missing combination behind a new unsupported-context error or
rename host-side full-model replay as batching. Rows for families since retired
were removed with them ([Retired model families](RETIRED_FAMILIES.md)).

September 20 evidence correction: `d4f5dd4` fixes unaligned resident GPU
mappings and a synthetic fixture. Earlier synthetic NaN-bit
equality is not numerical parity. The replacement finite/nonzero gate asserts
bounded FP16-scale error and exact greedy continuation; rollback remains
byte-exact. Installed aligned-checkpoint evidence is separate. See the
[detailed correction](RELEASE_VALIDATION_2026-09-20.md#resident-gpu-alignment-and-target-state-follow-up).

## Roadmap status

- Phase 1: actual execution and memory diagnostics landed on `main`.
- Phases 2, 3, 4 and 6 covered families that have since been retired
  ([Retired model families](RETIRED_FAMILIES.md)).
- Phase 5: execution-contract coverage extended here; the gaps above remain
  explicit. A green ordinary CI run does not execute env-gated real-model tests.
- Phase 7: real UI streaming, tool loops, history, cancellation and model-switch
  recovery have been tested; a [support table](RELEASE_SUPPORT.md) distinguishes
  established paths from candidates. The separately frozen 60-case screen has
  run for the matched-low profiles only, not every release profile. Neither this
  screen nor the earlier 12-task screen or UI checks is
  a broad quality claim or support evidence for hardware not tested here.

Smaller-memory hardware remains an external qualification requirement. Kernel optimizations must pass correctness and
repeatable matched measurements before promotion. Never infer 24 GB support
from a 256 GB host with a reduced expert-slot setting.

## Phase 5 validation record

The records below are dated; their commands and footers are kept as run, and
may name families that have since been retired.

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
Ordinary package runs do not enable installed-model environment gates.

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
