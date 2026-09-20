# Release qualification — September 20, 2026

Continuation of [September 18](RELEASE_VALIDATION_2026-09-18.md).
The previous head `4a100e8` passed macOS 15 / Swift 6.1, macOS 26, docs and
security checks ([CI run](https://github.com/NeelM0906/Mference/actions/runs/35384943818)).
PR #37 merged on September 20 at `049bdcd74f3f43fd858a5e7cd88a9e134b4cc701`.
The additional September 20 work follows that merge in
[PR #38](https://github.com/NeelM0906/Mference/pull/38), on
`codex/roadmap-qualification`. No source release is published.

The pre-loader head `15713e7` passed macOS 15 / Swift 6.1, macOS 26, docs and
security checks ([CI run](https://github.com/NeelM0906/Mference/actions/runs/35539493702)).
Its completed-answer resident performance record is
[reported separately](RELEASE_PERFORMANCE_2026-09-20.md), including the
short/medium-prefill regressions. Newer code requires its own CI result.

## Native Flash-Next MTP loader qualification

Code `49deb89`; same Mac Studio, OS/toolchain and safety protocol below.
This adds a strict sidecar loader, not enabled speculative decoding. The
synthetic fixture carries a separate one-layer expert pool and all 29 resident
tensors. Coverage includes absent sidecars, pre-folded norm pass-through,
exact cached pre-FC folding, mixed matrix dtypes, malformed dimensions and
companion ranges, bad pool geometry/path/count, truncation and SHA mismatch.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextMTPWeightsTests|FlashNextResidentLoadTests|FlashNextMTPInputFusionTests|FlashNextCheckpointTests' \
  > /tmp/mference-mtp-loader-20260920.log 2>&1
```

Exit 0; `Build complete! (14.38s)`;
`Test run with 24 tests in 4 suites passed after 3.415 seconds.`
The installed gate was disabled in this first invocation.

The existing completed INT8-router Flash-Next install was then checked:
57 receipt file sizes matched, 98% memory free, 619 GiB disk available, no
other model/test owner. No download, model copy, profiling or cache purge.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter FlashNextMTPWeightsTests \
  > /tmp/mference-mtp-installed-loader-20260920.log 2>&1
```

Exit 0; complete timing footer:

```text
Build complete! (1.41s)
Test installedSidecarLoadsWithNativeDtypes() passed after 2.237 seconds.
Suite FlashNextMTPWeightsTests passed after 3.556 seconds.
Test run with 7 tests in 1 suite passed after 3.556 seconds.
```

All 29 resident entries resolve with expected geometry, including the actual
INT8 router/shared gate and INT4 FC projection. The auxiliary pool's SHA-256
passes and its 512-expert, 2,768,896-byte-stride layout is validated. The test
does not open all trunk experts, allocate a draft KV/cache, run a native draft
or establish acceptance/performance. Full draft-runner/reference/verification
gates remain in [MTP status](FLASHNEXT_MTP_STATUS.md).

Before the subsequent alignment/target-state changes, code `49deb89` also
passed the full serial suite: exit 0, build 1.41s, **1,271 tests in 228 suites
after 270.633s, one known issue** (absent optional Flash-Next toy checkpoint).
Command: `env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs`,
log `/tmp/mference-full-mtp-loader-20260920.log`. The all-product release build
at `d029137` exited 0, `Build complete! (58.03s)`; command
`env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs`,
log `/tmp/mference-release-mtp-loader-20260920.log`.

## Resident GPU alignment and target-state follow-up

**Correction to earlier synthetic evidence:** the Flash-Next synthetic fixture
had an unaligned resident index. `ResidentBuffer` passed the shifted mapping
address directly to Metal's `bytesNoCopy` API. CPU reads could look correct
while GPU reads used incorrect bytes: the audit observed a zero embedding and
NaNs after PLE. Earlier byte-equality checks could compare identical NaN bits;
they did **not** establish numerical prefill parity. The independently
installed-model gates, which use production-aligned installs, are separate
evidence and are not replaced by these synthetic results.

`d4f5dd4` wraps the page-aligned mapping base, carries its logical byte offset
into every weight/scale/bias `TensorView`, and includes page overhead when
splitting at the device buffer limit. No weight files are rewritten and no
resident-weight copy is introduced. The fixture now follows production index
alignment. A real GPU compute test compares every payload byte at offsets
0/137 and across a forced 32 KiB buffer limit; CPU-only tests are insufficient.

With finite, nonzero fixture activations, chunked and sequential reductions
are not bit-exact. The original short/40-token probes measured worst logit
error 0.0078125, relative error 0.000677–0.000679, matching top choices. The
replacement gate explicitly requires finite/nonzero rows, error no greater
than `max(abs(reference)) / 512` (two FP16 relative-precision units), and exact
greedy choices over eight more steps. This is a correction of invalid synthetic
evidence, **not preservation of the previous zero-tolerance claim**. The existing
installed-model 5% bound is unchanged. Same-execution rollback, snapshot
ownership and reset still require byte equality.

`5d260d7` captures an owned copy of the last committed **full, unmixed** target
HC bundle, with its processed-token count. It covers decode, ragged chunked
prefill, warm append/scratch resize, rejected drafts, reset and dirty-state
rejection. Ordinary generation retains a view and performs no additional GPU
copy. Checkpoints preserve the bundle as well as recurrent/PLE state. This
does not implement a native drafter or prove chunked speculative verification.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'ResidentBufferTests|FlashNextCheckpointTests|FlashNextChunkedPrefillTests|FlashNextMTPWeightsTests|FlashNextResidentLoadTests' \
  > /tmp/mference-resident-alignment-v3-20260920.log 2>&1
```

Exit 0; build 6.01s; **34 tests in five suites passed after 5.968s**. Same environment,
98% memory free and 619 GiB disk; no other model owner or downloads. Initial
hidden-state tests exposed the NaNs; fixing alignment then exposed finite
rounding differences in the previous invalid bit-parity assertions. One test
macro rejected a key-path predicate and was changed to its equivalent closure.
Those failed attempts remain in `/tmp/mference-target-hidden-20260920.log`,
`/tmp/mference-target-hidden-audit-20260920.log`, and
`/tmp/mference-resident-alignment{,-v2}-20260920.log`. Diagnostic captures ran
only in correctness tests, never in the performance protocol.

The first full-suite rerun then stopped with exit 1 / signal 5 at the Qwen
fixture's GDN two-byte-alignment precondition. Seven other synthetic/parity
builders also used raw, non-page-aligned index lengths. They now follow the
production writer's 16 KiB index alignment. The direct unaligned-buffer GPU
test remains, so aligning the fixtures does not remove coverage of the bug.
A further test compares every weight/scale/bias view through forced 64 KiB
resident chunks against the original single mapping.

Fixture correction `b4325a4`: the first padding edit to Inkling's custom
`Data` builder shrank its padded region during string-table replacement;
the second full attempt exited 1 / signal 5. Restricting replacement to the
actual string bytes fixes that fixture-writing error. No runtime tolerance
changed. The focused follow-up passed **29 tests in five suites after 18.577s**,
build 5.95s, exit 0:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'ResidentBufferTests|InklingPrefillContractTests|QwenRunnerTests|DecodeOverlapTests' \
  > /tmp/mference-aligned-fixtures-20260920.log 2>&1
```

Full-attempt logs are `/tmp/mference-full-alignment-20260920.log` and
`/tmp/mference-full-alignment-v2-20260920.log`.

The fresh full serial suite at `b4325a4` passes: exit 0, build 0.15s, **1,275 tests in
228 suites after 271.169 seconds, one known issue** (the same absent optional
Flash-Next reference fixture). No installed-model environment gate was set.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  > /tmp/mference-full-alignment-v3-20260920.log 2>&1
```

The installed Flash-Next recheck then passed at the same code revision. Fresh
preflight: 98% memory free, 619 GiB disk, no model/test owner, all 57 receipt
file sizes verified. The loader used full SHA-256 verification.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'InstalledPrefillBoundaryTests|FlashNextMTPWeightsTests|FlashNextCheckpointTests|ResidentBufferTests' \
  > /tmp/mference-installed-alignment-20260920.log 2>&1
```

Exit 0; complete build and installed/aggregate timing footer:

```text
Build complete! (1.44s)
Test installedSidecarLoadsWithNativeDtypes() passed after 2.204 seconds.
Test memoryProfilesMatchAcrossBoundaries(name:) with 5 test cases passed after 92.996 seconds.
Suite InstalledPrefillBoundaryTests passed after 92.997 seconds.
Test run with 19 tests in 4 suites passed after 98.162 seconds.
```

Only the Flash-Next installed gate was enabled; four other family parameters
returned without loading weights. Resident and 16-slot modes pass sparse-boundary,
eight-step continuation and cancellation/reset checks. Ordinary and rejected-draft
rollback restore the finite, nonzero full HC bundle byte-for-byte; rejected-draft
rollback also reproduces all eight full logit rows exactly. Actual sidecar dtypes
and pool SHA pass again. This does not run a native drafter or qualify its speed.
All release products were rebuilt from `b4325a4` after that test completed:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs \
  > /tmp/mference-release-alignment-20260920.log 2>&1
```

Exit 0; complete footer: `Build complete! (55.74s)`. Fresh remote CI is still
required; earlier green CI does not qualify this newer loader path.

## Environment and protocol

Mac Studio Mac15,14; M3 Ultra, 32 CPU cores, 256 GiB; macOS 26.3 (25D125);
Swift 6.3.3 (`swiftlang-6.3.3.1.3`), developer directory
`/Applications/Xcode.app/Contents/Developer`. AC power, low-power mode off.
Preflight: 97–98% memory free, 619 GiB available disk, no existing model owner.
Receipt file sizes were checked before every installed-model launch; the
loader then used full SHA-256 verification. No downloads/copies of weights,
cache purges, worktrees, profiling or experimental controls. The temporary
dependency workspace required resolution/rebuild using the pinned package lock.
The user-owned untracked execution-plan document is untouched.

These are correctness tests, not the community speed protocol. Only one model
is loaded at a time; one arm returns before the next arm loads. Documentation
and preparation of later tests may occur alongside them; no performance
comparison is inferred from their wall time. Ordinary CI without environment
gates returns before loading installed weights; skipped arms are not passes
for those checkpoints.

## Maple and Inkling installed window-boundary checks

Test implementation `e549ea6`; production code unchanged from `4a100e8`.
Both runs invoke the actual factory and production chunked-prefill runner,
not a hand-built stand-in. Each arm checks 33/511/515/547-token heads around
the 512-token window boundary, ragged warm appends, eight greedy continuation
steps, finite valid logits and masked padded vocabulary. All appends report
their full token count batched and zero replay. Cancellation after layer 3
rejects dirty continuation/decode; reset reproduces exact clean prefix logits.

Maple compares its supported 16-slot and 8-slot modes, with 64-token chunks.
Inkling compares resident and 16-slot modes, with 128-token chunks. **All four
boundary rows, all eight continuation rows and all eight greedy choices match
exactly between each pair of memory profiles.** This is not independent
upstream model-quality parity or qualification of every context/hardware.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_MAPLE_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/maple.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter InstalledPrefillBoundaryTests \
  > /tmp/mference-maple-boundary-v3-20260920.log 2>&1
```

Exit 0; complete build/test timing footer:

```text
Build complete! (5.76s)
Test memoryProfilesMatchAcrossBoundaries(name:) with 3 test cases passed after 40.310 seconds.
Suite InstalledPrefillBoundaryTests passed after 40.310 seconds.
Test run with 1 test in 1 suite passed after 40.310 seconds.
```

Only Maple's gate was enabled; the other two parameter cases returned without
loading weights. Initial test development failed compilation due to an error
case requiring an associated value, then wrongly requested unsupported Maple
resident experts, then called continuation preparation on an empty prefix.
The test was corrected to exercise the supported 16/8 slots and prepare only
nonempty continuations; no production behavior or numerical tolerance changed.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_INKLING_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/inklingsmall.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter InstalledPrefillBoundaryTests \
  > /tmp/mference-inkling-boundary-20260920.log 2>&1
```

Exit 0; complete build/test timing footer:

```text
Build complete! (3.95s)
Test memoryProfilesMatchAcrossBoundaries(name:) with 3 test cases passed after 201.820 seconds.
Suite InstalledPrefillBoundaryTests passed after 201.820 seconds.
Test run with 1 test in 1 suite passed after 201.820 seconds.
```

Only Inkling's gate was enabled. Flash-Next counters and later evaluation
tooling were edited while this already-built correctness test ran; they did
not alter its executable. Reduced slots on this host do not certify a smaller
physical Mac. A small nonzero Inkling full-runner CI fixture remains separate.

## Flash-Next installed long chunks and sparse boundary

Commit `c8f3298876c1bfd36b9a7b812011b7e1b5ba78d0`; same environment and
preflight, with the completed INT8-router install, strict verification and no
other model owner. Factory-dispatched production prefill uses 1,024-token
chunks, 2,115 maximum context, and heads at 33/2047/2051/2083 tokens. The long
append reports chunks `[1024, 990]`, all batched and zero replay. Later appends
cross the 2,048-token indexer budget. Actual TensorOps encodings were observed
in both modes: resident 240, 16-slot streamed 4,548 (different tile granularity,
not a speed or work-efficiency comparison). Portable toy geometry separately
asserted zero TensorOps encodings.

**All four full boundary logit rows and eight continuation rows match exactly
between resident and streamed modes; eight greedy choices match.** Both modes
pass cancellation after partially advanced GPU state, dirty-state rejection,
and exact reset. No numerical tolerance was relaxed. These results qualify
this context/chunk configuration, not every larger context or an old-OS
installed-model fallback.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'InstalledPrefillBoundaryTests|FlashNextChunkedPrefillTests' \
  > /tmp/mference-flashnext-boundary-20260920.log 2>&1
```

Exit 0; complete build and installed-suite/aggregate timing footer:

```text
Build complete! (12.82s)
Test memoryProfilesMatchAcrossBoundaries(name:) with 3 test cases passed after 93.611 seconds.
Suite InstalledPrefillBoundaryTests passed after 93.611 seconds.
Test run with 7 tests in 2 suites passed after 95.569 seconds.
```

Only Flash-Next's installed gate was enabled in this invocation. The additional
six tests are the portable synthetic prefill suite, not extra checkpoints.

## Further work

The [five-seed source-policy comparison](QWEN_SOURCE_EFFICIENCY_V1.md) is
predeclared separately from the frozen low-budget/default-policy screen.
Native Flash-Next MTP, matched release performance and remaining hardware
coverage still require their own implementation/evidence; these checks do not
close the entire roadmap.

## Full regression and release build before the five-seed run

Runtime `c8f3298`, documentation head `8c46ad3`, same environment as above.
Commands ran serially, with no installed-model environment gate enabled:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  > /tmp/mference-full-20260920.log 2>&1
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs \
  > /tmp/mference-release-build-20260920.log 2>&1
```

Both exited 0. Full test footer:

```text
Build complete! (3.79s)
Test run with 1253 tests in 224 suites passed after 268.488 seconds with 1 known issue.
```

The known issue remains the absent optional Flash-Next toy checkpoint. Release
build footer: `Build complete! (86.09s)`. Installed boundary gates above ran
separately; they are not implicitly covered by the full package invocation.
Later MTP checkpoint work must be tested separately; it is not in this binary.

## Checkpoint, input-fusion, Inkling and boolean-parser regression

Commit `3462477e170e1ee289b40ddd48d33a0384af17af`; same machine/toolchain,
98% memory free, 619 GiB available disk, no other model owner. No installed
weights are needed: fixtures generate small nonzero test weights locally.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextCheckpointTests|FlashNextMTPInputFusionTests|InklingPrefillContractTests|QwenToolCallParserTests|ChatMLDecoderTests|FlashNextChunkedPrefillTests' \
  > /tmp/mference-focused-v4-20260920.log 2>&1
```

Exit 0; complete timing footer:

```text
Build complete! (5.94s)
Test run with 40 tests in 6 suites passed after 3.831 seconds.
```

- Flash-Next rollback restores exact full logits after rejected drafts, PLE
  EOS-history changes, pooled-index boundaries and interrupted decode/prefill.
  Foreign, stale and discarded-branch snapshots are rejected. The scratch
  resize test uses supported 32-to-64-token chunks.
- Native MTP input fusion matches its independent scalar oracle exactly
  (`maxAbs=0`) for BF16/INT4 at hidden sizes 64 and 2,560. This tests only input
  fusion, not an integrated or independently qualified native MTP decoder.
- The nonzero Inkling fixture compares resident and eight-slot production
  batching across a 32-token window and relative-bias/log-scaling boundaries.
  Nine full boundary/continuation rows are bit-identical between profiles and
  after reset. Against sequential execution, maximum absolute error is
  `1.5258789e-05`, minimum cosine `0.9999999783216431`; the predeclared bounds
  remain 0.002 and 0.999. Cancellation, dirty rejection and recovery pass.
- Qwen's XML parser recognizes case-insensitive true/false only for an explicit
  boolean schema. Declared strings, ambiguous schemas and unrecognized values
  are unchanged; token-by-token streaming uses the same contract. It does not
  evaluate Python literals or silently coerce arbitrary malformed arguments.

Development runs initially aborted on unsupported 8- then 16-token fixture
chunks (`PrefillRuntimeConfig.swift:247: Precondition failed: unsupported
prefill chunk size`, exit 1). The supported minimum is 32. The next run failed
with one fixture manifest `keyNotFound(moeIntermediateSize)` error (exit 1,
40 tests / six suites / 3.144 seconds); the field spelling was corrected.
No runtime guard or numerical tolerance was relaxed. The test command also
rebuilt debug dependencies when replacing the earlier manual compiler flags.

## Five-seed Swift-Qwen token/quality screen

The [separate source-policy report](QWEN_SOURCE_EFFICIENCY_V1.md) records all
600 completed requests, settings, failures and raw-evidence hash. Base passed
297/300 and Swift 293/300; Swift used 14.992% fewer completion tokens. The
fixed binary predates the boolean-parser correction. Do not rescore this run
or infer new whole-corpus quality from a targeted parser recheck.

## Installed Flash-Next rollback recheck

Commit `3462477`; same machine and preflight (98% memory free, 619 GiB disk),
all 57 receipt sizes verified, strict SHA-256 loader. No other model owner.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'InstalledPrefillBoundaryTests|FlashNextChunkedPrefillTests' \
  > /tmp/mference-flashnext-rollback-20260920.log 2>&1
```

Exit 0; complete build and installed-suite/aggregate timing footer:

```text
Build complete! (1.43s)
Test memoryProfilesMatchAcrossBoundaries(name:) with 3 test cases passed after 93.365 seconds.
Suite InstalledPrefillBoundaryTests passed after 93.365 seconds.
Test run with 7 tests in 2 suites passed after 95.316 seconds.
```

Only the Flash-Next installed parameter was enabled. All previous long-chunk,
2,048-token sparse-boundary, TensorOps, continuation and cancellation/reset
assertions pass again. In **both resident and 16-slot modes**, restoring a
checkpoint after a rejected four-token draft (including EOS) reproduces all
eight subsequent full logit rows exactly. This closes the installed rollback
component gate, not native MTP integration, acceptance or speed qualification.
Documentation and CPU-only script checks ran concurrently; not a speed test.

## Full regression after the new runtime/parser changes

Code `3462477`, documentation head `13a0367`; same hardware/toolchain and
98%-free / 619-GiB preflight. No installed-model gates or other model owner.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  > /tmp/mference-full-v2-20260920.log 2>&1
```

Exit 0; complete timing footer:

```text
Build complete! (1.43s)
Test run with 1264 tests in 227 suites passed after 269.850 seconds with 1 known issue.
```

The known issue is still the absent optional Flash-Next toy checkpoint.
This run includes the new synthetic checkpoint, fusion, Inkling and parser
tests. It does not replace the separate installed-checkpoint gates above.
Launcher tests (8), UI-adapter tests (5), and evaluation-script tests (16)
also pass. Markdown validation checks 74 files; the source archive checker
validates 845 entries. Source archive validation does not bundle local weights
or claim an actual end-user installation on another physical Mac.

Release products were then rebuilt serially from `13a0367` (same production
code as the full test run):

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs \
  > /tmp/mference-release-v2-20260920.log 2>&1
```

Exit 0; complete footer: `Build complete! (58.02s)`.

## Default-policy screen and eight-slot installed regression

The [default-policy screen](QWEN_SOURCE_EFFICIENCY_V1.md)
completed all 480 requests: Swift 180/180 measured passes, base 177/180, with
20.088% fewer Swift completion tokens. Policies differ; all three base failures
share one ambiguous punctuation item. Both model-server sessions were stopped
after their final request, before the next model test.

Commit `ba14ff4a2a2173088c36b48ca58d202fbaf639fd` adds opt-in installed coverage
for the eight-slot Gemma/Qwen prefill fix already on main. Same hardware and
toolchain; preflight 98% memory free, 619 GiB disk, no model owner; all 37 Gemma
and 47 Qwen receipt-file sizes verified, then full SHA-256 loading. Each arm
returns before the next loads; no simultaneous model owners.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_GEMMA4_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/gemma4.gturbo \
  MFERENCE_QWEN36_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen36.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter InstalledPrefillBoundaryTests \
  > /tmp/mference-gemma-qwen36-boundary-20260920.log 2>&1
```

Exit 0; complete timing footer:

```text
Build complete! (13.19s)
Test memoryProfilesMatchAcrossBoundaries(name:) with 5 test cases passed after 59.968 seconds.
Suite InstalledPrefillBoundaryTests passed after 59.969 seconds.
Test run with 1 test in 1 suite passed after 59.969 seconds.
```

Only Gemma and Qwen 3.6 gates are enabled. Gemma uses 128-token chunks and
heads 33/1023/1027/1059 across its 1,024-token window; Qwen uses 64-token
chunks and ragged heads 33/127/131/163. **All four boundary and eight
continuation full logit rows match exactly between resident and eight-slot
modes for both checkpoints.** All tokens are batched with zero replay.
Cancellation after partially advanced state, dirty reuse rejection and exact
reset pass in all four arms. These are correctness checks, not speed or
physical eight-GB-Mac qualification. Documentation editing continued during
the already-built tests; no performance inference is drawn from their timing.

CI for PR head `0832751` also passed both macOS 15 / Swift 6.1 and macOS 26,
docs and security checks ([run](https://github.com/NeelM0906/Mference/actions/runs/35538016328)).
Subsequent test/document/comment-only changes require a fresh final-head CI
run; this historical success is not silently relabeled as a result for them.
