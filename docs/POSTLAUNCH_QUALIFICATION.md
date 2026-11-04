# Post-launch qualification — September 21, 2026

This work does not retag or replace the published v0.1.0 source archive.
Performance qualification, checkpoint-quality promotion and native MTP
enablement are separate gates. Per the September 22 release-scope decision,
physical 16/24-GiB qualification is **not a blocker for this release**. Those
profiles remain unqualified; no claim of fit or measured performance is added.

Follow-up evidence: the [September 22 record](RELEASE_VALIDATION_2026-09-22.md)
closes the tested independent installed native-MTP numerical comparison and
records the larger-budget GLM Max profile. Historical failures and limits in
this September 21 record remain intact; the follow-up does not retroactively
pass the original default-budget runs or enable accelerated native MTP.

## Machine and protocol

Mac Studio Mac15,14, Apple M3 Ultra (32 CPU cores), 256 GiB; macOS 26.3
(25D125), Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`),
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. AC power,
Low Power Mode off, sleep disabled. Initial checks: 97–98% memory free,
614 GiB available disk, no model/test/MLX process. Completed Swift, base Qwen
and GLM receipt file sizes checked before execution. Inference uses default
full-SHA verification. No weights downloaded/copied, caches purged, profiling,
experimental controls or concurrent builds/tests during performance runs.
Light source editing continued without rebuilding the measured executable.

Measured executable: `/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI`,
built from `be12fc61e40581a88d83052042ac8acc03421319`, whose production source
is identical to harness commit `353f3f5caa16eaed8aee1b6fc17e6ff722e54ee2`.
Release build command:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --product MferenceCLI
```

Exit 0; `Build of product 'MferenceCLI' complete! (76.36s)`.
Executable SHA-256:
`e826b8b17b3dd17b9e572145ea1bfc5c07f59d961b461b02a9167b8577fe8d6a`.
The new internal verifier described below is not in this measured executable.
After all performance runs stopped, the measured executable and resource
bundles were preserved at `/tmp/mference-postlaunch-measured.E5hCEV/` before
building the updated source. No model weights were copied.

Frozen community prompts/seeds and sampling are retained. Differences are
explicitly labeled below: these opt-in profiles do **not** replace the failed
source-default benchmark or establish a quality-preserving default change.
Each complete profile requires one discarded warmup and three fresh-process
measured repetitions per case, end-of-turn, visible output and manual reading.
Prefill plus decode excludes process startup/loading; `/usr/bin/time` additionally
records whole-process wall time and memory statistics. Mapped/GPU allocations
are not fully described by CPU RSS, so RSS is not a smaller-Mac fit certificate.

`run-benchmark.sh` now rechecks memory/disk/ownership before every process,
refuses to overwrite an evidence directory, rejects empty visible answers, and
records the executable hash, expanded commands, preflights and exit codes.
The initial five mock-based regression tests passed without loading a model;
the fixed-settling follow-up below expands that set to eight.

## Swift low effort, original 1,024-token allowance: rejected

Exact command:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI \
  ./run-benchmark.sh swift-low-20260921 scratch/swiftqwen38.gturbo 3 \
  --reasoning-effort low
```

Protocol deviation: explicit source-supported low effort, not default xhigh.
Both CLI processes exited 0; the harness exited 1 on the second warmup.
No measured repetitions or long-case warmup ran. Complete timing footers:

```text
short warmup [stop=endOfTurn prefill=90tok/2.06s new=847tok decode=22.76s tok/s=37.220]
medium warmup [stop=maxTokens prefill=454tok/5.79s new=1024tok decode=27.99s tok/s=36.585]
```

This establishes that switching effort alone does not make this allowance
adequate. The failure remains in `benchmark-results/swift-low-20260921/`.

## Swift low effort, 4,096-token allowance

Separate profile, selected after the rejected warmup; not a rerun of the
unchanged community protocol. Exact command:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI \
  ./run-benchmark.sh swift-low-4096-20260921 scratch/swiftqwen38.gturbo 3 \
  --reasoning-effort low --max-new 4096 --max-context 8192
```

Explicit deviations: low effort, 4,096 output allowance, 8,192 context and
three measured repetitions. The CLI uses the final occurrence of budget/context
options; expanded per-run commands preserve both the runner defaults and these
overrides. Swift MTP remains off by its unchanged default.

**Completed:** all 12 processes exited 0 and reached end-of-turn. The harness
exited 0; 9/9 measured answers completed. Output bytes match the warmup exactly
for every repetition of each case. Evidence:
`benchmark-results/swift-low-4096-20260921/` (per-run commands, exits,
preflights, stdout/stderr and machine record).

| Case (prompt/new tokens) | Median prefill, s | Median decode, tok/s | Median prefill + decode, s (range) | Median whole-process wall, s |
| --- | ---: | ---: | ---: | ---: |
| Short (90/847) | 2.11 | 37.052 | 24.97 (24.96–24.99) | 31.66 |
| Medium (454/1237) | 5.83 | 36.073 | 40.12 (40.08–40.25) | 46.81 |
| Long (2968/1036) | 34.09 | 33.058 | 65.43 (65.35–65.56) | 72.16 |

No comparison to base Qwen or causal optimization speedup follows from this
single-profile result. The source-default xhigh/1,024 gate remains failed.

Complete footers (0 is warmup, 1–3 measured; all exits 0):

```text
short 0 [stop=endOfTurn prefill=90tok/2.08s new=847tok decode=22.89s tok/s=37.010]
short 1 [stop=endOfTurn prefill=90tok/2.10s new=847tok decode=22.89s tok/s=37.009]
short 2 [stop=endOfTurn prefill=90tok/2.12s new=847tok decode=22.84s tok/s=37.086]
short 3 [stop=endOfTurn prefill=90tok/2.11s new=847tok decode=22.86s tok/s=37.052]
medium 0 [stop=endOfTurn prefill=454tok/5.78s new=1237tok decode=34.32s tok/s=36.042]
medium 1 [stop=endOfTurn prefill=454tok/5.83s new=1237tok decode=34.29s tok/s=36.073]
medium 2 [stop=endOfTurn prefill=454tok/5.81s new=1237tok decode=34.44s tok/s=35.923]
medium 3 [stop=endOfTurn prefill=454tok/5.83s new=1237tok decode=34.25s tok/s=36.117]
long 0 [stop=endOfTurn prefill=2968tok/34.07s new=1036tok decode=31.33s tok/s=33.063]
long 1 [stop=endOfTurn prefill=2968tok/34.09s new=1036tok decode=31.26s tok/s=33.147]
long 2 [stop=endOfTurn prefill=2968tok/34.09s new=1036tok decode=31.34s tok/s=33.058]
long 3 [stop=endOfTurn prefill=2968tok/34.12s new=1036tok decode=31.44s tok/s=32.956]
```

All three warmup answers were read. They finish without loops, but are not a
broad quality pass: the cache-design answer does not establish safe lifetime
for outstanding views or bound in-flight allocations, and the synthesis's
cache-purge recommendation conflicts with this project's measurement rules.
No generated instruction was executed. A completed answer is not necessarily
a correct answer or a reason to promote this checkpoint over base Qwen.

## GLM low-effort resident: completed short warmup, safety stop

Exact command, same frozen executable and host as above:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI \
  MIN_FREE_PCT=75 MFERENCE_GLM5_REASONING_EFFORT=low \
  ./run-benchmark.sh glm-low-resident-20260921 scratch/glm53flash.gturbo 3 \
  --expert-cache-slots resident
```

Protocol deviations: vendor-supported low effort instead of Max, explicit
resident policy, three planned measured repetitions. Sampling, context 4,096,
output allowance 1,024 and prompts/seeds match the frozen protocol. Initial
memory check was 98%, completed receipt had 49 matching file sizes, and model
loading used full SHA-256. The short warmup exited 0 with a complete, manually
read, non-repeating answer. Full footer and process time:

```text
[stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.43s tok/s=25.683]
109.96 real 61.63 user 45.85 sys
171843846144 maximum resident set size
172400765688 peak memory footprint
```

Before the medium warmup, the per-process check stopped the harness (exit 1):

```text
ABORT: memory preflight failed before medium-review: 38% free
```

**No medium/long warmups, measured repetitions, bounded GLM runs or subsequent
local model tests were launched.** No apps were killed or caches purged; the
75% resident headroom requirement was not lowered. A later read-only snapshot
found no model owner, 98% free memory and 0.56 MiB swap used. This suggests a
transient post-exit headroom report, not proof of a persistent leak; it does
not retroactively pass the failed preflight or authorize ignoring it.

Evidence: `benchmark-results/glm-low-resident-20260921/`, plus the orchestration
error above. The runner was subsequently hardened to retain failed preflights,
batch exit status and failure text as files as well as terminal output.
**That attempt did not qualify matched GLM performance.** A successful
warmup is neither a median nor an improvement over the historical baseline.

## GLM low-effort resident with fixed settling: completion passed

A separate attempt preserves the failed attempt above. Harness commit
`fba8196b09424100071ab08a9c25419f31f2eb79` adds an opt-in fixed idle interval
**before** the next preflight; it never retries a failed check. Default is zero,
accepted range 0–60 seconds. All eight mock safety tests pass (7.983s, exit 0),
including immediate abort after a failed check despite a configured interval.

Same host/toolchain/power configuration as above; 613 GiB free disk. Release
CLI rebuilt once with the earlier build command: exit 0, 3.70s incremental,
log `/tmp/mference-gate-revision-build.log`. Source `fba8196` was tracked-clean
at build/start (the user's untracked execution plan was preserved). Binary
SHA-256 `29d9eefeb8686b43d7a26f9af0ec748cf9d66a4aa154cfa85b799ed690646f4e`.
Later native-proposal source edits are **not** in this executable. No local
builds, tests, downloads, profiling or experimental controls ran during timing;
light editing and source inspection continued. All 49 installed receipt file
sizes were checked; default full-SHA verification remained on.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI \
  BENCH_SETTLE_SECONDS=10 MIN_FREE_PCT=75 MFERENCE_GLM5_REASONING_EFFORT=low \
  ./run-benchmark.sh glm-low-resident-settled-20260921 scratch/glm53flash.gturbo 3 \
  --expert-cache-slots resident
```

Deviations from the frozen community protocol: low effort, explicit resident
policy, three measured repetitions, fixed ten-second inter-process settling.
Prompts, seeds, sampling, 1,024 output allowance and 4,096 context are unchanged.
All 12 preflights reported 98% free memory against the unchanged 75% minimum.
All 12 CLI processes and the harness exited 0; 9/9 measured answers completed.
Per-case stdout matches byte-for-byte across warmup and repetitions. All three
unique answers were read: complete, no looping or repeated blocks. This is a
completion check, not an independent factual/technical accuracy evaluation.
The long answer discusses the frozen historical project, not current support.

| Case (prompt/new tokens) | Median prefill, s | Median decode, tok/s | Median prefill + decode, s (range) | Median whole-process wall, s |
| --- | ---: | ---: | ---: | ---: |
| Short (60/653) | 1.23 | 25.662 | 26.68 (26.64–26.70) | 109.73 |
| Medium (420/841) | 3.95 | 24.195 | 38.71 (38.67–38.72) | 121.86 |
| Long (2792/709) | 29.90 | 19.499 | 66.26 (66.13–66.28) | 149.33 |

Complete footers (0 warmup, 1–3 measured; all exits 0):

```text
short 0 [stop=endOfTurn prefill=60tok/1.25s new=653tok decode=25.44s tok/s=25.672]
short 1 [stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.45s tok/s=25.662]
short 2 [stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.41s tok/s=25.694]
short 3 [stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.47s tok/s=25.635]
medium 0 [stop=endOfTurn prefill=420tok/3.96s new=841tok decode=34.58s tok/s=24.318]
medium 1 [stop=endOfTurn prefill=420tok/3.95s new=841tok decode=34.72s tok/s=24.220]
medium 2 [stop=endOfTurn prefill=420tok/3.95s new=841tok decode=34.77s tok/s=24.186]
medium 3 [stop=endOfTurn prefill=420tok/3.95s new=841tok decode=34.76s tok/s=24.195]
long 0 [stop=endOfTurn prefill=2792tok/29.91s new=709tok decode=36.34s tok/s=19.513]
long 1 [stop=endOfTurn prefill=2792tok/29.91s new=709tok decode=36.37s tok/s=19.494]
long 2 [stop=endOfTurn prefill=2792tok/29.90s new=709tok decode=36.36s tok/s=19.499]
long 3 [stop=endOfTurn prefill=2792tok/29.89s new=709tok decode=36.24s tok/s=19.562]
```

Evidence: `benchmark-results/glm-low-resident-settled-20260921/`, including
expanded commands, all preflights/exits, system record, answers and complete
`time -l` output. This does not pass source-default Max/1,024, bounded-memory
performance or physical smaller-Mac qualification. The executable and resource
bundles were preserved at `/tmp/mference-glm-measured.Dr7tT9/` before subsequent
builds; the preserved CLI hash is unchanged. No model files were copied.

### Matched GLM baseline

The preserved baseline is production source `c67e857`, source-equivalent
`d769a7e` build, CLI SHA-256
`621d9b41f09836e8d23fc231597f316b052b86b1420daa5deafea1341602dfb2`.
All 26 preserved Metal resources were byte-checked against `c67e857`.
The benchmark harness/workspace HEAD was `fba8196` at baseline start; this is
**not** the baseline executable's source. The working edits were internal
native-proposal code, tests and later documentation, not rebuilt resources.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-release-baseline.olcDV9/MferenceCLI \
  BENCH_SETTLE_SECONDS=10 MIN_FREE_PCT=75 MFERENCE_GLM5_REASONING_EFFORT=low \
  ./run-benchmark.sh glm-low-resident-baseline-settled-20260921 scratch/glm53flash.gturbo 3 \
  --expert-cache-slots resident
```

Same host, install, prompts, seeds, sampling, effort, capacity and cache policy.
Additional ordering limitation: the entire candidate batch ran first, then
the baseline batch, not randomized/alternating order; a fixed ten-second pause
preceded the fresh baseline preflight. No cache purge or concurrent local
build/test/download. Remote CI, light source editing and read-only checks
continued. All baseline preflights were 98% free; all 12 processes and its
harness exited 0. All 24 answers are byte-identical within each case across
versions, warmups and repetitions. Both batches pass completion.

| Case | Baseline generation, s (range) | Candidate generation, s (range) | Baseline / candidate median whole-process wall, s |
| --- | ---: | ---: | ---: |
| Short | 26.63 (26.59–26.66) | 26.68 (26.64–26.70) | 109.69 / 109.73 |
| Medium | 38.73 (38.66–38.77) | 38.71 (38.67–38.72) | 121.70 / 121.86 |
| Long | 68.78 (68.67–68.83) | 66.26 (66.13–66.28) | 151.62 / 149.33 |

Long median generation is **3.7% lower**, with non-overlapping measured ranges;
median prefill falls 32.47 to 29.90 seconds (7.9%). Decode medians are
25.709/24.257/19.527 baseline versus 25.662/24.195/19.499 candidate tok/s;
this is not a decode-speed improvement. Short/medium generation ranges overlap,
so do not claim a meaningful gain there. Whole-process loading/integrity costs
are included only in the separate wall-time column. These measurements are not
performance ceilings or evidence for bounded/smaller-hardware/default-Max use.

Complete baseline footers (0 warmup, 1–3 measured; all exits 0):

```text
short 0 [stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.57s tok/s=25.541]
short 1 [stop=endOfTurn prefill=60tok/1.25s new=653tok decode=25.41s tok/s=25.697]
short 2 [stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.36s tok/s=25.752]
short 3 [stop=endOfTurn prefill=60tok/1.23s new=653tok decode=25.40s tok/s=25.709]
medium 0 [stop=endOfTurn prefill=420tok/4.05s new=841tok decode=34.65s tok/s=24.269]
medium 1 [stop=endOfTurn prefill=420tok/4.07s new=841tok decode=34.70s tok/s=24.238]
medium 2 [stop=endOfTurn prefill=420tok/4.05s new=841tok decode=34.61s tok/s=24.301]
medium 3 [stop=endOfTurn prefill=420tok/4.06s new=841tok decode=34.67s tok/s=24.257]
long 0 [stop=endOfTurn prefill=2792tok/32.47s new=709tok decode=36.35s tok/s=19.506]
long 1 [stop=endOfTurn prefill=2792tok/32.45s new=709tok decode=36.22s tok/s=19.574]
long 2 [stop=endOfTurn prefill=2792tok/32.47s new=709tok decode=36.36s tok/s=19.501]
long 3 [stop=endOfTurn prefill=2792tok/32.47s new=709tok decode=36.31s tok/s=19.527]
```

Evidence: `benchmark-results/glm-low-resident-baseline-settled-20260921/`.

## Native MTP reference verifier

`FlashNextMTPGreedyVerifier` is internal and disconnected from production
generation. It uses the actual target decoder and generation sampler, not
batched-prefill approximation or raw-logit argmax. It commits only a matching
proposal prefix plus the target correction/bonus. On failure it restores both
target and primer and the visible logits; failed recovery resets the pair.
It is a correctness reference, not a speed optimization or native drafter
qualification. See [the remaining MTP gates](FLASHNEXT_MTP_STATUS.md).

Synthetic bounded/resident tests and the installed 16-slot test pass. The same
35-token cold prefix is used for plain and verified paths; the toy crosses its
compressed-indexer boundary and prefill chunk. All-rejected, each partially
accepted prefix, all-accepted plus bonus, subsequent continuation, budget,
stop-token and context exhaustion cases compare exact full-logit rows. Faults
after a verified token and inside the target-to-primer consumer both recover
the starting logits/cursors and reproduce plain continuation. Arbitrary
sampling, stop strings, server prefix reuse/model switching and an accelerated
verifier are not covered by this component gate.

Completed Flash-Next INT8-router install: 57 receipt file sizes checked;
manifest `3fed7dfb9f94c26d2fb33ba0ead016bb4f34edc0e286b579331cafa76e509897`.
Preflight 98% free memory, 613 GiB disk, no other model owner. Command:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter FlashNextMTPGreedyVerifierTests
```

Implementation commit `b62b400` (before documentation-only merge of main).
Exit 0. Full timing footer, log `/tmp/mference-postlaunch-installed-verifier.log`:

```text
Build complete! (7.55s)
Test acceptedRejectedAndInterruptedRoundsMatchPlainTarget(resident:) with 2 test cases passed after 1.808 seconds.
Test installedAcceptedRejectedAndInterruptedRoundsMatchPlainTarget() passed after 42.120 seconds.
Suite FlashNextMTPGreedyVerifierTests passed after 43.929 seconds.
Test run with 2 tests in 1 suite passed after 43.929 seconds.
```

These are debug correctness timings, not throughput measurements. The earlier
toy-only version passed in 1.785 seconds (build 14.25s), before adding the
installed gate. No threshold was loosened; the existing unrounded-FP32 native
draft issue remains unchanged.

## Native proposal integration follow-up

Runtime commit `445cf2cb9f663b3cc7a3dd4f1a3f6325e617ce60` adds real native
proposals to the internal verifier. The target-selected seed is followed by
drafts using the preceding draft's full HC bundle; the speculative draft branch
is restored before verification. Separate draft scratch preserves the target
head. Acceptance statistics exclude the target-selected seed. No production
generation call site or default is changed.

On the same host, after all timing processes exited: 98% free memory, 613 GiB
free disk, no model owner; all 57 Flash-Next receipt file sizes checked, with
the unchanged manifest recorded above. Exact command:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextMTPGreedyVerifierTests|FlashNextMTPPrimingTests'
```

Exit 0; log `/tmp/mference-native-proposal-installed.log`. Full suite footer:

```text
Build complete! (10.87s)
Test acceptedRejectedAndInterruptedRoundsMatchPlainTarget(resident:) with 2 test cases passed after 2.109 seconds.
Test installedAcceptedRejectedAndInterruptedRoundsMatchPlainTarget() passed after 46.562 seconds.
Suite FlashNextMTPGreedyVerifierTests passed after 48.672 seconds.
Test installedTargetRowsPrimeNativeDraft() passed after 29.863 seconds.
Suite FlashNextMTPPrimingTests passed after 31.575 seconds.
Test run with 9 tests in 2 suites passed after 80.247 seconds.
```

Actual proposals repeat exactly without advancing committed cursors or
altering the target head. An injected interruption after a speculative row
restores the branch and permits identical retry. Multi-round verification
matches eight ordinary target tokens and the full head after every round;
actual-proposal stopping, context clamping, reset and invalid counts are checked.
Toy bounded/resident runs each accept 5/5 actual draft guesses; the installed
artificial-prefix probe accepts **0/10**. These are fixture diagnostics, not
natural-language quality or speed measurements. The unchanged primer gate also
checks 43 installed target rows across cold/warm/decode boundaries.

An additional installed chat-template probe uses a fixed two-sentence wetlands
question, greedy target sampling, 16 expert slots, three native guesses per
round and a 64-token limit. It compares all target tokens and each round's full
head, using the tokenizer's actual stop IDs. Its first suite run exits 0:

```text
Build complete! (7.28s)
Test acceptedRejectedAndInterruptedRoundsMatchPlainTarget(resident:) with 2 test cases passed after 2.097 seconds.
Test installedAcceptedRejectedAndInterruptedRoundsMatchPlainTarget() passed after 44.537 seconds.
[installed chat MTP probe] prefix=27 target=51 drafted=48 accepted=20 target-seeds-excluded; exact tokens/full heads; diagnostic only, not completed-answer performance
Test installedChatProposalsMatchPlainTarget() passed after 41.724 seconds.
Suite FlashNextMTPGreedyVerifierTests passed after 88.359 seconds.
Test run with 3 tests in 1 suite passed after 88.360 seconds.
```

Command is the one above with `--filter FlashNextMTPGreedyVerifierTests`; log
`/tmp/mference-native-chat-installed.log`. Preflight again 98%, 613 GiB, no
owner. The answer reaches a real stop at token 51, rather than its 64-token
ceiling. Acceptance of 20/48 on one chosen short prompt is **not** a broad
acceptance result or evidence that sequential verification accelerates output.
Independent native parity, the existing unrounded-FP32 issue, accelerated
verification, server integration and completed-answer speed remain open.

Test commit `4be2533cee38bc39634d132190a11acd9d992b1c` then makes a natural
stop within 64 tokens mandatory and prints the answer for inspection. The same
command with `--filter installedChatProposalsMatchPlainTarget` passes after a
fresh 98%-free/no-owner/613-GiB check (exit 0):

```text
Build complete! (6.12s)
[installed chat MTP probe] prefix=27 target=51 drafted=48 accepted=20 target-seeds-excluded; exact tokens/full heads; diagnostic only, not completed-answer performance
Test installedChatProposalsMatchPlainTarget() passed after 41.751 seconds.
Suite FlashNextMTPGreedyVerifierTests passed after 41.751 seconds.
Test run with 1 test in 1 suite passed after 41.752 seconds.
```

Log `/tmp/mference-native-chat-stop-installed.log`. The answer was read: two
complete non-repeating sentences explaining wave/surge damping and vegetation.
This inspection is not an independent factual accuracy evaluation. The test
uses one short selected prompt, a 32-token prefill chunk and debug execution;
it is not the frozen performance protocol, a full generation API, or a
same-weight upstream drafter comparison.

All-product release build on runtime `445cf2c` exits 0 (54.88s):
`env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs`;
log `/tmp/mference-proposal-release-build.log`. These are build/debug correctness
times, not performance measurements. CI on `445cf2c` passes macOS 15 (20m40s),
macOS 26 (16m20s), documentation/archive and secret scanning; installed-model
gates are local, not inferred from CI's ungated runs.

## Physical hardware boundary

This host supplies only 256-GiB M3 Ultra evidence. Neither limited expert slots,
CPU RSS nor a software memory cap qualifies physical 16/24-GiB hardware.
No smaller Mac is attached to this task. This is explicitly outside the current
release-blocking scope, per the September 22 user decision. A future claim of
smaller-Mac qualification still requires actual
target-hardware install/first-use, the frozen benchmark, pressure/swap checks,
streaming and cancellation/recovery, and declared model/context/cache settings.
Do not claim every checkpoint fits every smaller tier: dense Swift weights
alone occupy about 15.4 GB before runtime/context and OS memory.
