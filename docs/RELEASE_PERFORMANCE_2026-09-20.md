# Release performance experiments — September 20, 2026

## GLM default-profile warmups: rejected completion gate, safety stop

**Not a completed-answer performance result.** Four resident warmups ran:
short explanation and medium review, baseline then candidate for each. Every
process exited 0 but stopped at `maxTokens`, with 1,024 generated tokens and
zero bytes of visible stdout. Empty stdout equality does not establish equal
reasoning, answers, quality or token sequences. The candidate prints the
existing actionable token-budget warning. No measured repetitions ran.

Before launching the long-synthesis baseline, the harness's resident-memory
preflight reported **53% memory free**, below its 75% headroom requirement for
this roughly 181-GB installed model. The harness exited 2 without starting
that process. Long synthesis and the bounded profile were not run. A later
read-only process check found no model/test/installer owner; no application
was terminated, no cache purged and no model deleted/reinstalled. No further
model run was attempted after the failed safety check.

Host: Mac Studio Mac15,14, M3 Ultra (32 CPU cores), 256 GiB; macOS 26.3
(25D125), Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3`). Local date September 20;
the four launch times are September 21 00:00:17/00:02:22/00:04:29/00:06:39 UTC.
AC power, Low Power Mode off, sleep off. Initial disk 619 GiB; all 49 receipt
file sizes matched. Each completed run's preflight reported 98% memory free
and no other model owner. Default full-SHA verification; no effort override,
MTP, profiling or experimental controls. No local build/test/download ran
concurrently. Light documentation, read-only source/web review and remote-CI
checks continued; this interrupted warmup set supports no speed inference.

Candidate: `28ce980cf8f4c5d7d650dafb1460aaa5c4ed0fdf`, release build 56.83s.
Its preserved CLI is `/tmp/mference-release-glm-qualified.2S9zMK/MferenceCLI`,
SHA-256 `07c01f8859bf7a030ad008d02081ac9aae6de09d2e5109f903f784a2a0a6608f`.
All 26 preserved Metal resources were byte-checked against that commit.
Baseline: `c67e857` production source, source-equivalent `d769a7e` build,
`/tmp/mference-release-baseline.olcDV9/MferenceCLI`, SHA-256
`621d9b41f09836e8d23fc231597f316b052b86b1420daa5deafea1341602dfb2`.
Neither binary nor its resources was rebuilt during the attempt.

Raw commands, footers, stdout, per-run exit codes, preflights and system/hash
records: `/tmp/mference-glm-default-warmups.8viocX`.
Manifest SHA-256: `bee37795db54e06f2af66225c864b6cfc10d14156c893f476e24df6017e56287`.
Frozen prompt hashes, short/medium/long respectively:

```text
c57da2677143657be55e03797c1aabe8e012d61c1a71bf9345acdd575117d1e1
23add7976db2069c9927affd13c2f5120508702a1c4915ea88b7de565bdd4b33
b12e4a71493ad151d7a85b91e878ba784b116ef65de9fcb1b1b1c05198b537b8
```

Exact orchestration (the final argument selects warmups only):

```sh
bash /tmp/mference-release-compare.sh glm53flash resident \
  /tmp/mference-glm-default-warmups.8viocX \
  /tmp/mference-release-baseline.olcDV9/MferenceCLI \
  c67e857-source-equivalent-d769a7e-build \
  /tmp/mference-release-glm-qualified.2S9zMK/MferenceCLI \
  28ce980cf8f4c5d7d650dafb1460aaa5c4ed0fdf warmup
```

Each CLI command uses `--model scratch/glm53flash.gturbo`,
`--messages-file docs/benchmark-prompts/real-generation-v1/<case>.json`,
`--max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`,
`--expert-cache-slots resident`, and seed 20260721/20260722 for short/medium.
These settings retain the frozen community protocol. **Protocol incompleteness:**
the planned long warmups and all three measured repetitions were not executed;
the warmups already fail the completion criterion and the safety stop prevented
the next launch. They must not be presented as a completed benchmark.

Complete timing footers (each underlying CLI exits 0):

```text
short-explanation baseline [stop=maxTokens prefill=60tok/1.25s new=1024tok decode=40.84s tok/s=25.076]
short-explanation candidate [stop=maxTokens prefill=60tok/1.23s new=1024tok decode=40.87s tok/s=25.058]
medium-review baseline [stop=maxTokens prefill=420tok/4.05s new=1024tok decode=42.70s tok/s=23.980]
medium-review candidate [stop=maxTokens prefill=420tok/3.96s new=1024tok decode=42.82s tok/s=23.915]
```

Candidate stderr also contains, before each footer:

```text
note: token limit reached before a visible answer. Increase --max-new (and --max-context if needed).
```

Complete orchestration error (exit 2):

```text
Refusing benchmark: memory check: The system has 274877906944 (16777216 pages with a page size of 16384).
System-wide memory free percentage: 53%
```

The [historical GLM report](families/GLM53_FLASH.md#measured-results) already
distinguishes the template-default Max reasoning from its separately disclosed
Low-effort measurements. No new Low-effort comparison ran, and no defaults or
checkpoint bytes were changed. Any such comparison needs its own explicitly
labeled protocol. Current matched GLM completed-answer performance
and INT8 throughput optimization remain open.

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

## Flash-Next 16-slot: separate matched measurement

Candidate `b4325a4c9669594f3478aa85491313e8a4dd7777`, not the resident
experiment's `15713e7`. All 24 processes exited 0 and reached
`stop=endOfTurn`. All eight outputs per case have the same SHA-256 as the
resident outputs above. The three unique answers were read again: complete,
non-repeating, with the same limited-quality caveats.

Medians and min–max across three measured repetitions, excluding one warmup
per case/arm. Generation time remains **prefill plus decode**, excluding
process startup and work completed during initial model loading. **Unlike
resident mode, bounded experts are opened lazily: first-touch expert SHA-256
verification inside prefill is included in these footers.** These are matched
fresh-process first-request measurements, not isolated kernel time or warm
server requests. Do not subtract an inferred hash cost or compare these
prefill durations directly with resident mode as a kernel-only difference.

| Case (prompt / new tokens) | Baseline prefill, s | Candidate prefill, s | Baseline generation, s | Candidate generation, s | Median generation reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short (62 / 521) | 27.11 (27.10–27.20) | 26.85 (26.82–27.00) | 64.04 (63.98–64.07) | 63.70 (63.69–64.05) | 0.5% |
| Medium (426 / 557) | 28.75 (28.63–28.77) | 28.06 (28.03–28.11) | 68.93 (68.69–69.07) | 68.12 (68.07–68.32) | 1.2% |
| Long (2940 / 635) | 39.59 (39.46–39.60) | 37.14 (37.08–37.15) | 93.93 (93.88–94.08) | 91.78 (91.47–91.84) | 2.3% |

| Case | Baseline decode tok/s | Candidate decode tok/s |
| --- | ---: | ---: |
| Short | 14.104 (14.098–14.164) | 14.126 (14.061–14.141) |
| Medium | 13.870 (13.816–13.903) | 13.905 (13.851–13.910) |
| Long | 11.689 (11.627–11.697) | 11.623 (11.609–11.674) |

The short generation ranges overlap: the 0.5% median difference is not a
robust completed-answer improvement. Medium and long ranges are disjoint
here, but the reductions are modest (1.2% and 2.3%). Long prefill falls 6.2%;
decode ranges overlap in every case, and the long decode median is slightly
slower. This does **not** establish a streamed decode speedup or inherit the
resident experiment's larger gains.

### Streamed provenance and commands

Same M3 Ultra / 256 GiB, macOS 26.3 (25D125), Swift 6.3.3 toolchain and AC/
normal-power settings as above. Initial preflight: 98% memory free, 619 GiB
disk, no model/test/installer owner, all 57 install receipt sizes verified.
Each fresh process uses default full-SHA verification; per-launch process,
memory and disk checks passed. The manifest, prompts, seeds, context, token
allowance and sampling match the resident experiment; the explicit cache
policy is now **16 slots**. Native MTP remains off.

Baseline executable/provenance are unchanged. The candidate all-product
release build exited 0, `Build complete! (55.74s)`, log
`/tmp/mference-release-alignment-20260920.log`. Candidate executable SHA-256:
`44304df747f3ce8651c0de42ec1eff812a2e7c5309007c166900c70f8cdc51d1`.
All 26 bundled Metal sources were checked against `b4325a4` during the batch;
neither executable nor resources changed. After measurement, only binaries/
resource bundles were preserved at
`/tmp/mference-release-streamed-qualified.fGHHDY`; no model weights copied.

Protocol extensions/limitations: case-major order, baseline then candidate
alternation, three measured repetitions, no cache purge, profiling,
experimental controls, downloads, concurrent builds/tests or other model
owner. Light source/doc edits, commits and read-only GitHub checks continued.
Those later router/reference/native-draft/GLM fixes are **not** measured here.
Filesystem/cache state was not reset or controlled as a cold-SSD experiment.
A 16-slot run on 256 GiB does not qualify a physical smaller-memory Mac.

Executed orchestration (exit 0):

```bash
bash /tmp/mference-release-compare.sh qwen38flashnext-r8 16 /tmp/mference-release-bench-flash-streamed.jJkIRX /tmp/mference-release-baseline.olcDV9/MferenceCLI c67e857-source-equivalent-d769a7e-build /tmp/mference-phase1-build.sXnNTs/release/MferenceCLI b4325a4c9669594f3478aa85491313e8a4dd7777
```

Each case/seed pair above, repetitions 0–3, uses this CLI command form for both
executables:

```bash
/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI --model scratch/qwen38flashnext-r8.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots 16
```

Raw stdout/stderr, exact commands, exits, preflights, hashes and machine record:
`/tmp/mference-release-bench-flash-streamed.jJkIRX`. The strict 24-record
summary is `summary.json` there. Full timing footers follow; repetition 0 is
warmup, 1–3 measured, every exit is 0.

```text
short 0 baseline [stop=endOfTurn prefill=62tok/27.30s new=521tok decode=38.44s tok/s=13.554]
short 0 candidate [stop=endOfTurn prefill=62tok/26.84s new=521tok decode=36.83s tok/s=14.146]
short 1 baseline [stop=endOfTurn prefill=62tok/27.10s new=521tok decode=36.94s tok/s=14.104]
short 1 candidate [stop=endOfTurn prefill=62tok/26.82s new=521tok decode=36.88s tok/s=14.126]
short 2 baseline [stop=endOfTurn prefill=62tok/27.11s new=521tok decode=36.96s tok/s=14.098]
short 2 candidate [stop=endOfTurn prefill=62tok/26.85s new=521tok decode=36.84s tok/s=14.141]
short 3 baseline [stop=endOfTurn prefill=62tok/27.20s new=521tok decode=36.78s tok/s=14.164]
short 3 candidate [stop=endOfTurn prefill=62tok/27.00s new=521tok decode=37.05s tok/s=14.061]
medium 0 baseline [stop=endOfTurn prefill=426tok/29.39s new=557tok decode=41.86s tok/s=13.306]
medium 0 candidate [stop=endOfTurn prefill=426tok/28.07s new=557tok decode=40.24s tok/s=13.842]
medium 1 baseline [stop=endOfTurn prefill=426tok/28.63s new=557tok decode=40.06s tok/s=13.903]
medium 1 candidate [stop=endOfTurn prefill=426tok/28.06s new=557tok decode=40.06s tok/s=13.905]
medium 2 baseline [stop=endOfTurn prefill=426tok/28.77s new=557tok decode=40.16s tok/s=13.870]
medium 2 candidate [stop=endOfTurn prefill=426tok/28.11s new=557tok decode=40.21s tok/s=13.851]
medium 3 baseline [stop=endOfTurn prefill=426tok/28.75s new=557tok decode=40.32s tok/s=13.816]
medium 3 candidate [stop=endOfTurn prefill=426tok/28.03s new=557tok decode=40.04s tok/s=13.910]
long 0 baseline [stop=endOfTurn prefill=2940tok/43.13s new=635tok decode=56.64s tok/s=11.212]
long 0 candidate [stop=endOfTurn prefill=2940tok/37.07s new=635tok decode=54.71s tok/s=11.606]
long 1 baseline [stop=endOfTurn prefill=2940tok/39.59s new=635tok decode=54.29s tok/s=11.697]
long 1 candidate [stop=endOfTurn prefill=2940tok/37.08s new=635tok decode=54.39s tok/s=11.674]
long 2 baseline [stop=endOfTurn prefill=2940tok/39.46s new=635tok decode=54.62s tok/s=11.627]
long 2 candidate [stop=endOfTurn prefill=2940tok/37.15s new=635tok decode=54.63s tok/s=11.623]
long 3 baseline [stop=endOfTurn prefill=2940tok/39.60s new=635tok decode=54.33s tok/s=11.689]
long 3 candidate [stop=endOfTurn prefill=2940tok/37.14s new=635tok decode=54.70s tok/s=11.609]
```

Matched GLM measurements and physical smaller-memory qualification remain open.
