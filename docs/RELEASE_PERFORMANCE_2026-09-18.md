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
