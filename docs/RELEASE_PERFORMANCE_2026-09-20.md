# Release performance experiments — September 20, 2026

## Flash-Next resident: matched completed-answer measurement

All 24 fresh-process runs exited 0, reached `stop=endOfTurn`, and produced
byte-identical stdout within each case across both versions and all repeats.
The three unique answers were read: they finish coherently without repeated
blocks, and contain 419/401/466 whitespace-separated words. This passes the
[community benchmark](COMMUNITY_BENCHMARKS.md) completion gate, not a broad
quality gate. For example, the cache review's per-object size limit alone does
not establish an aggregate memory bound. Generated assessments of the frozen
historical project document are not current Mference support statements.

Medians of three measured repetitions; parentheses give min–max. One warmup
per case and version is excluded. Generation time is the sum of the reported
prefill and decode durations **within each run**; it excludes process startup,
model loading and integrity verification and is not end-to-end CLI latency.

| Case (prompt / new tokens) | Baseline prefill, s | Candidate prefill, s | Baseline generation, s | Candidate generation, s | Median generation reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short (62 / 521) | 2.44 (2.43–2.44) | 5.31 (5.29–5.41) | 25.92 (25.85–26.48) | 24.85 (24.41–25.43) | 4.1% |
| Medium (426 / 557) | 4.90 (4.88–5.10) | 5.56 (5.51–5.58) | 29.53 (29.00–29.73) | 27.03 (26.89–27.53) | 8.5% |
| Long (2940 / 635) | 15.68 (14.63–17.65) | 10.31 (9.67–12.71) | 47.00 (46.81–50.09) | 41.69 (41.15–44.99) | 11.3% |

| Case | Baseline decode tok/s | Candidate decode tok/s |
| --- | ---: | ---: |
| Short | 22.193 (21.675–22.248) | 26.639 (26.020–27.279) |
| Medium | 22.614 (22.594–23.114) | 25.966 (25.349–26.050) |
| Long | 19.730 (19.572–20.272) | 20.169 (19.672–20.233) |

**Trade-off:** short prefill takes 2.18 times as long and medium prefill is
13.5% slower. Long prefill is 34.2% faster by median. All three measured
generation-time ranges are disjoint, but long-case decode ranges overlap;
do not claim a separate long-decode improvement. These are local measurements,
not ceilings or guarantees on another host, context or memory profile.

This compares the combined changes since `c67e857`, not one isolated kernel.
The new resident route-grouping path binds a whole expert slab, whereas the
older path binds selected expert views. Full-buffer residency can move page
cost into prefill; that is a code-based hypothesis, **not an isolated causal
measurement**. Warm filesystem state, fixed baseline-first ordering and changing
long-case prefill times also limit attribution. No cache purge or profiling
was used. Do not describe the decode improvement as an unconditional TTFT win.

### Provenance and exact commands

Mac Studio Mac15,14, M3 Ultra (32 CPU cores), 256 GiB RAM; macOS 26.3
(25D125); Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`), developer
directory `/Applications/Xcode.app/Contents/Developer`. AC power, Low Power
Mode off, sleep disabled. Initial memory-pressure free percentage 98, disk
619 GiB. Completed install: all 57 receipt file sizes checked before the batch;
each process uses default strict SHA-256 verification. Per-launch memory,
disk and model/test/installer-owner checks passed.

- Baseline source `c67e857`, source-equivalent preserved `d769a7e` release build.
  `Sources`, package files and all 26 bundled Metal sources were verified
  against that tree. Binary SHA-256:
  `621d9b41f09836e8d23fc231597f316b052b86b1420daa5deafea1341602dfb2`.
- Candidate release `15713e729f3d01ec9b25d887da046420d8b1b896`; all-product
  release build exited 0, `Build complete! (52.02s)`, log
  `/tmp/mference-release-v3-20260920.log`. Binary SHA-256:
  `0a505cfeb841fe86ccf369b820ffe11825ca1cb172626ade6c9fe77875d72b23`.
  After measurement, this executable and resource bundles were preserved at
  `/tmp/mference-release-resident-qualified.I9Ynva` before any newer release
  rebuild. No model weights were copied.
- Existing `scratch/qwen38flashnext-r8.gturbo`, INT8 routers, resident experts.
  Manifest SHA-256:
  `3fed7dfb9f94c26d2fb33ba0ead016bb4f34edc0e286b579331cafa76e509897`.
- Frozen `real-generation-v1`: short-explanation seed 20260721, medium-review
  seed 20260722, long-synthesis seed 20260723. Temperature 0.2, top-k 64,
  top-p 0.95, context 4096, max-new 1024, default reasoning, native MTP disabled.
  No `MFERENCE_*` environment overrides, profiling or experimental controls.
- Protocol extensions: explicit resident expert mode and three measured
  repeats per case/arm, alternating baseline then candidate, with repetition
  zero discarded for each. Case-major order. No downloads, builds, tests or
  second model owner during timing. Light source/doc edits and read-only
  GitHub checks continued; neither executable/resource bundle changed. The
  later native-MTP loader edits are **not** part of the measured candidate.
- Initial worktree clean except the user-owned untracked execution plan.
  Final orchestration exit 0. Exact per-run commands, preflights, input hashes,
  machine record, stdout/stderr, exits and summary:
  `/tmp/mference-release-bench-flash-resident.wfk30e`.

Executed command form (each frozen case/seed pair above, repetitions 0–3):

```bash
/tmp/mference-release-baseline.olcDV9/MferenceCLI --model scratch/qwen38flashnext-r8.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots resident
/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI --model scratch/qwen38flashnext-r8.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots resident
```

Stdout SHA-256, identical across all eight runs per case:

```text
short  a371ede16188ea9bc9060c190a398effd8c0c9c1ce46e44853558df287c8144c
medium fe050777cd1677142c97fc74038e2d5c9630e3bd9c5e10fbb5c55e4e8b782de2
long   13ca296544dcc9e62e7ca85e2c67ee450f2e2aed2b11fe508a13446295fc484d
```

### Complete timing footers

Every process exited 0. Repetition 0 is warmup; 1–3 are measured.

```text
short 0 baseline [stop=endOfTurn prefill=62tok/2.53s new=521tok decode=23.89s tok/s=21.804]
short 0 candidate [stop=endOfTurn prefill=62tok/5.45s new=521tok decode=20.01s tok/s=26.034]
short 1 baseline [stop=endOfTurn prefill=62tok/2.44s new=521tok decode=24.04s tok/s=21.675]
short 1 candidate [stop=endOfTurn prefill=62tok/5.41s new=521tok decode=20.02s tok/s=26.020]
short 2 baseline [stop=endOfTurn prefill=62tok/2.44s new=521tok decode=23.48s tok/s=22.193]
short 2 candidate [stop=endOfTurn prefill=62tok/5.29s new=521tok decode=19.56s tok/s=26.639]
short 3 baseline [stop=endOfTurn prefill=62tok/2.43s new=521tok decode=23.42s tok/s=22.248]
short 3 candidate [stop=endOfTurn prefill=62tok/5.31s new=521tok decode=19.10s tok/s=27.279]
medium 0 baseline [stop=endOfTurn prefill=426tok/5.05s new=557tok decode=24.71s tok/s=22.542]
medium 0 candidate [stop=endOfTurn prefill=426tok/5.50s new=557tok decode=21.64s tok/s=25.736]
medium 1 baseline [stop=endOfTurn prefill=426tok/5.10s new=557tok decode=24.63s tok/s=22.614]
medium 1 candidate [stop=endOfTurn prefill=426tok/5.51s new=557tok decode=21.38s tok/s=26.050]
medium 2 baseline [stop=endOfTurn prefill=426tok/4.88s new=557tok decode=24.65s tok/s=22.594]
medium 2 candidate [stop=endOfTurn prefill=426tok/5.58s new=557tok decode=21.45s tok/s=25.966]
medium 3 baseline [stop=endOfTurn prefill=426tok/4.90s new=557tok decode=24.10s tok/s=23.114]
medium 3 candidate [stop=endOfTurn prefill=426tok/5.56s new=557tok decode=21.97s tok/s=25.349]
long 0 baseline [stop=endOfTurn prefill=2940tok/17.67s new=635tok decode=32.73s tok/s=19.401]
long 0 candidate [stop=endOfTurn prefill=2940tok/13.49s new=635tok decode=32.55s tok/s=19.506]
long 1 baseline [stop=endOfTurn prefill=2940tok/17.65s new=635tok decode=32.44s tok/s=19.572]
long 1 candidate [stop=endOfTurn prefill=2940tok/12.71s new=635tok decode=32.28s tok/s=19.672]
long 2 baseline [stop=endOfTurn prefill=2940tok/14.63s new=635tok decode=32.18s tok/s=19.730]
long 2 candidate [stop=endOfTurn prefill=2940tok/10.31s new=635tok decode=31.38s tok/s=20.233]
long 3 baseline [stop=endOfTurn prefill=2940tok/15.68s new=635tok decode=31.32s tok/s=20.272]
long 3 candidate [stop=endOfTurn prefill=2940tok/9.67s new=635tok decode=31.48s tok/s=20.169]
```

Streamed Flash-Next and matched GLM measurements remain separate outstanding
gates. This resident result does not qualify physical smaller-memory hardware.
