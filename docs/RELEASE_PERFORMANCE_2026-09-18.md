# Release performance experiments — September 18, 2026

Measurements and failed attempts, not performance ceilings. This record does
not establish a release recommendation or broad model-quality parity.

## Swift-Qwen: first correctness fix / community-protocol attempt

**Rejected as a completed-answer benchmark:** every warmup and measured run
reached `stop=maxTokens`, with zero visible output bytes. The source-default
`xhigh` reasoning did not produce a visible answer within the frozen allowance.
Do not report the decode rates below as successful task throughput, raise the
cap retroactively, or silently replace the community protocol with another
reasoning policy. These failures remain evidence against default promotion.

The attempt nevertheless exposed the first correctness fix's prefill cost:

| Case | Prompt tokens | Main-equivalent prefill median (range) | First exact-arithmetic fix median (range) |
| --- | ---: | ---: | ---: |
| Short explanation | 102 | 1.78 s (1.77–1.79) | 2.54 s (2.53–2.56) |
| Medium review | 466 | 3.12 s (3.12–3.13) | 7.38 s (7.36–7.38) |
| Long synthesis | 2,980 | 14.41 s (14.40–14.42) | 43.12 s (43.12–43.16) |

Three measured repetitions, excluding one warmup per case/arm. The main-equivalent
path had previously failed Swift's numerical gate; it is not an acceptable
correctness fallback. The exact fix passed that gate, but its one-token GPU
tiles repeatedly read the same weights. A subsequent small multi-token tile
candidate reuses each read while retaining decode's arithmetic. Its separate
qualification and measurements must be recorded before claiming improvement.

### Provenance and protocol

Mac Studio Mac15,14, M3 Ultra (32 CPU cores), 256 GiB RAM; macOS 26.3 (25D125);
Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`), AC power,
`lowpowermode=0`. No other model owner, downloads, builds, tests, profiling or
experimental controls during the runs. Light code/document editing and GitHub
checks continued without rebuilding either measured executable/resource bundle.
GLM installation was stopped with SIGINT at 48.40 GB verified payload and
resumed only after the final measured process exited. No caches were purged.

- Baseline: preserved release build `d769a7e`; `Sources/`, `Package.swift` and
  `Package.resolved` are identical to merged main `c67e857`. Every preserved
  Metal resource was checked against that main tree. CLI SHA-256:
  `621d9b41f09836e8d23fc231597f316b052b86b1420daa5deafea1341602dfb2`.
- Candidate: release build `048d04f`, including the first Swift numerical fix.
  CLI SHA-256: `1ecb776144498d65f4dd85e353d3703a28088a32a60e35809e8f156cddd8788f`.
  Preserved separately after measurement at
  `/tmp/mference-release-decode-order-baseline.1hckzf` (binaries/resources only).
- Existing strict-verified `scratch/swiftqwen38.gturbo`; source
  `ukisai/Swift-Qwen3.8-27b` at `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`.
- Source default `xhigh`; MTP off by default for Swift; no effort override.
  Explicit `resident` is immaterial for this dense checkpoint. Full-SHA default.
- Frozen `real-generation-v1` prompts, seeds `20260721`, `20260722`, `20260723`;
  temperature 0.2, top-k 64, top-p 0.95, context 4096, max-new 1024. Alternating
  baseline/candidate fresh processes, repetition 0 discarded, 1–3 measured.
- Per-run OS memory checks passed; no model/test/installer process before each
  launch. Initial free-memory percentage 97, available disk about 737 GiB.
- All 24 processes exited **0**, but none passed the end-of-turn/output gate.
  Equal total token counts do not prove equal hidden-token/routing workloads.

Exact command form, with each of the three case/seed pairs above:

```bash
/tmp/mference-release-baseline.olcDV9/MferenceCLI --model scratch/swiftqwen38.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots resident
/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI --model scratch/swiftqwen38.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots resident
```

Exact expanded commands, exit codes, empty stdout files, stderr, input hashes,
machine record and preflight checks are preserved at
`/tmp/mference-release-bench-swift.NapxLW`. The only initial dirty file was the
user-owned, untracked execution plan. The orchestration is
`/tmp/mference-release-compare.sh`; it never starts simultaneous model owners.

### Complete timing footers

Rows are case, repetition, arm, footer; repetition 0 is warmup.

```text
short 0 baseline [stop=maxTokens prefill=102tok/1.77s new=1024tok decode=29.64s tok/s=34.544]
short 0 candidate [stop=maxTokens prefill=102tok/2.55s new=1024tok decode=29.88s tok/s=34.265]
short 1 baseline [stop=maxTokens prefill=102tok/1.79s new=1024tok decode=29.13s tok/s=35.156]
short 1 candidate [stop=maxTokens prefill=102tok/2.56s new=1024tok decode=28.48s tok/s=35.952]
short 2 baseline [stop=maxTokens prefill=102tok/1.77s new=1024tok decode=28.93s tok/s=35.398]
short 2 candidate [stop=maxTokens prefill=102tok/2.54s new=1024tok decode=28.29s tok/s=36.198]
short 3 baseline [stop=maxTokens prefill=102tok/1.78s new=1024tok decode=28.28s tok/s=36.206]
short 3 candidate [stop=maxTokens prefill=102tok/2.53s new=1024tok decode=28.24s tok/s=36.259]
medium 0 baseline [stop=maxTokens prefill=466tok/3.13s new=1024tok decode=28.72s tok/s=35.656]
medium 0 candidate [stop=maxTokens prefill=466tok/7.36s new=1024tok decode=28.92s tok/s=35.410]
medium 1 baseline [stop=maxTokens prefill=466tok/3.12s new=1024tok decode=28.76s tok/s=35.601]
medium 1 candidate [stop=maxTokens prefill=466tok/7.38s new=1024tok decode=28.66s tok/s=35.733]
medium 2 baseline [stop=maxTokens prefill=466tok/3.13s new=1024tok decode=28.71s tok/s=35.673]
medium 2 candidate [stop=maxTokens prefill=466tok/7.38s new=1024tok decode=28.72s tok/s=35.649]
medium 3 baseline [stop=maxTokens prefill=466tok/3.12s new=1024tok decode=28.61s tok/s=35.790]
medium 3 candidate [stop=maxTokens prefill=466tok/7.36s new=1024tok decode=28.67s tok/s=35.721]
long 0 baseline [stop=maxTokens prefill=2980tok/14.45s new=1024tok decode=31.43s tok/s=32.581]
long 0 candidate [stop=maxTokens prefill=2980tok/43.11s new=1024tok decode=31.56s tok/s=32.445]
long 1 baseline [stop=maxTokens prefill=2980tok/14.41s new=1024tok decode=31.71s tok/s=32.295]
long 1 candidate [stop=maxTokens prefill=2980tok/43.12s new=1024tok decode=31.60s tok/s=32.410]
long 2 baseline [stop=maxTokens prefill=2980tok/14.40s new=1024tok decode=31.41s tok/s=32.604]
long 2 candidate [stop=maxTokens prefill=2980tok/43.12s new=1024tok decode=31.49s tok/s=32.521]
long 3 baseline [stop=maxTokens prefill=2980tok/14.42s new=1024tok decode=31.57s tok/s=32.432]
long 3 candidate [stop=maxTokens prefill=2980tok/43.16s new=1024tok decode=31.64s tok/s=32.366]
```

## Swift-Qwen: four-row exact-arithmetic weight reuse

**Again rejected as a completed-answer community benchmark.** All 24 runs
exited 0 but stopped at `maxTokens`, 1,024 generated tokens and zero visible
stdout bytes. No change to the fixed protocol, output gate or reasoning effort
was made. This is engineering evidence about prefill cost, not useful-answer
throughput or a reason to promote Swift's default.

Both compared versions pass the unchanged Swift numerical/state gate. This
comparison therefore does not use the earlier numerically failing main path
as a correctness fallback.

| Case | Prompt tokens | One-row exact prefill median (range) | Four-row exact prefill median (range) | Prefill time reduction |
| --- | ---: | ---: | ---: | ---: |
| Short explanation | 102 | 2.54 s (2.51–2.55) | 2.28 s (2.26–2.30) | 10.2% |
| Medium review | 466 | 7.39 s (7.38–7.39) | 6.10 s (6.08–6.12) | 17.5% |
| Long synthesis | 2,980 | 43.05 s (43.03–43.05) | 35.07 s (35.00–35.09) | 18.5% |

One warmup per case/arm, then three measured fresh-process repetitions. The
reduction exceeds the observed within-arm ranges, but does **not** recover
the original main path's speed: that path was faster and failed the numerical
gate. Decode arithmetic and the generation policy are unchanged.

### Provenance and protocol

Same Mac Studio Mac15,14 / M3 Ultra 32-core / 256 GiB, macOS 26.3 (25D125),
Swift 6.3.3 toolchain and AC / Low Power Mode off as the first comparison.
Initial memory-free check 97%, disk 721 GiB. Same strict-verified Swift install,
manifest and pinned source revision; same three frozen prompts and seeds,
temperature 0.2, top-k 64, top-p 0.95, context 4096, max-new 1024, MTP off and
source-default xhigh. No experimental controls or profiling.

- Baseline: release `048d04f`, preserved at
  `/tmp/mference-release-decode-order-baseline.1hckzf/MferenceCLI`, SHA-256
  `1ecb776144498d65f4dd85e353d3703a28088a32a60e35809e8f156cddd8788f`.
- Candidate: release `c9c3378`, including the four-row kernel from `ee0bfb9`,
  SHA-256 `fc9f5e66045002e3e8a3afa1cd045330e7b6d1e83148e63125871d8aead97bed`.
  Binaries/resources preserved **after** measurement at
  `/tmp/mference-release-four-row-baseline.HNSBdW`; no model copies.
- GLM installer paused at 990 verified ranges / 64,990,180,016 source bytes;
  no downloads, builds, package tests or other model owners during measurement.
  Light source/doc editing and GitHub checks continued; neither measured
  binary nor its resources were rebuilt. New cancellation edits in the working
  tree are not part of the measured candidate. GLM resumed after all 24 exits.
- The final CLI's empty-token-limit notice was observed in every candidate
  stderr; it changes neither stdout nor the timing footer.
- Exact expanded commands, input/system hashes, all stdout/stderr, exit codes
  and per-run safety checks:
  `/tmp/mference-release-bench-swift-tiled.vkh2D7`.

Command form (substitute each frozen case/seed pair as above):

```bash
/tmp/mference-release-decode-order-baseline.1hckzf/MferenceCLI --model scratch/swiftqwen38.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots resident
/tmp/mference-phase1-build.sXnNTs/release/MferenceCLI --model scratch/swiftqwen38.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 --expert-cache-slots resident
```

### Complete timing footers

Repetition 0 is warmup; 1–3 are measured. Every process exited 0.

```text
short-explanation 0 baseline [stop=maxTokens prefill=102tok/2.51s new=1024tok decode=29.36s tok/s=34.876]
short-explanation 0 candidate [stop=maxTokens prefill=102tok/2.33s new=1024tok decode=28.38s tok/s=36.088]
short-explanation 1 baseline [stop=maxTokens prefill=102tok/2.54s new=1024tok decode=28.49s tok/s=35.945]
short-explanation 1 candidate [stop=maxTokens prefill=102tok/2.26s new=1024tok decode=28.13s tok/s=36.408]
short-explanation 2 baseline [stop=maxTokens prefill=102tok/2.51s new=1024tok decode=28.24s tok/s=36.258]
short-explanation 2 candidate [stop=maxTokens prefill=102tok/2.30s new=1024tok decode=28.24s tok/s=36.266]
short-explanation 3 baseline [stop=maxTokens prefill=102tok/2.55s new=1024tok decode=28.23s tok/s=36.272]
short-explanation 3 candidate [stop=maxTokens prefill=102tok/2.28s new=1024tok decode=28.22s tok/s=36.286]
medium-review 0 baseline [stop=maxTokens prefill=466tok/7.39s new=1024tok decode=28.73s tok/s=35.646]
medium-review 0 candidate [stop=maxTokens prefill=466tok/6.09s new=1024tok decode=28.66s tok/s=35.732]
medium-review 1 baseline [stop=maxTokens prefill=466tok/7.39s new=1024tok decode=28.68s tok/s=35.699]
medium-review 1 candidate [stop=maxTokens prefill=466tok/6.12s new=1024tok decode=28.74s tok/s=35.625]
medium-review 2 baseline [stop=maxTokens prefill=466tok/7.39s new=1024tok decode=29.00s tok/s=35.310]
medium-review 2 candidate [stop=maxTokens prefill=466tok/6.08s new=1024tok decode=28.70s tok/s=35.685]
medium-review 3 baseline [stop=maxTokens prefill=466tok/7.38s new=1024tok decode=28.96s tok/s=35.360]
medium-review 3 candidate [stop=maxTokens prefill=466tok/6.10s new=1024tok decode=28.77s tok/s=35.597]
long-synthesis 0 baseline [stop=maxTokens prefill=2980tok/43.15s new=1024tok decode=31.46s tok/s=32.548]
long-synthesis 0 candidate [stop=maxTokens prefill=2980tok/35.09s new=1024tok decode=32.71s tok/s=31.306]
long-synthesis 1 baseline [stop=maxTokens prefill=2980tok/43.05s new=1024tok decode=31.32s tok/s=32.698]
long-synthesis 1 candidate [stop=maxTokens prefill=2980tok/35.07s new=1024tok decode=31.31s tok/s=32.703]
long-synthesis 2 baseline [stop=maxTokens prefill=2980tok/43.05s new=1024tok decode=31.41s tok/s=32.603]
long-synthesis 2 candidate [stop=maxTokens prefill=2980tok/35.09s new=1024tok decode=31.43s tok/s=32.579]
long-synthesis 3 baseline [stop=maxTokens prefill=2980tok/43.03s new=1024tok decode=31.38s tok/s=32.633]
long-synthesis 3 candidate [stop=maxTokens prefill=2980tok/35.00s new=1024tok decode=31.34s tok/s=32.676]
```
