# Release qualification — September 20, 2026

Continuation of [September 18](RELEASE_VALIDATION_2026-09-18.md).
The previous head `4a100e8` passed macOS 15 / Swift 6.1, macOS 26, docs and
security checks ([CI run](https://github.com/NeelM0906/Mference/actions/runs/35384943818)).
PR #37 merged on September 20 at `049bdcd74f3f43fd858a5e7cd88a9e134b4cc701`.
The additional September 20 work follows that merge in
[PR #38](https://github.com/NeelM0906/Mference/pull/38), on
`codex/roadmap-qualification`. No source release is published.

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
