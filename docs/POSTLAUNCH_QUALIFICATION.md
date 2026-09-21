# Post-launch qualification — September 21, 2026

This work does not retag or replace the published v0.1.0 source archive.
Performance qualification, checkpoint-quality promotion, native MTP enablement
and physical smaller-Mac qualification are separate gates.

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
Five mock-based regression tests pass without loading a model.

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

## Physical hardware boundary

This host supplies only 256-GiB M3 Ultra evidence. Neither limited expert slots,
CPU RSS nor a software memory cap qualifies physical 16/24-GiB hardware.
No smaller Mac is attached to this task. Qualification still requires actual
target-hardware install/first-use, the frozen benchmark, pressure/swap checks,
streaming and cancellation/recovery, and declared model/context/cache settings.
Do not claim every checkpoint fits every smaller tier: dense Swift weights
alone occupy about 15.4 GB before runtime/context and OS memory.
