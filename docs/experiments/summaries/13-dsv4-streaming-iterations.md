# DSV4 streaming iterations — M5/24 GB, 2026-08-07

Greedy 64-token `MFERENCE_PHASES=1` diagnostics against the fresh
`scratch/dsv4.gturbo` install (90 GB pool), trusted-receipt verification,
`--rdadvise adaptive`. Diagnostic runs, not protocol benchmarks; every
comparison below flips exactly one variable against the 16-slot baseline in
[12-decode-attribution-qwen-dsv4.md](12-decode-attribution-qwen-dsv4.md)
(4.62 tok/s; exposed I/O 5,263 ms; GPU waits 6,648 ms).

## Iteration 1: `MFERENCE_SPEC_PREFETCH=pilot` — 4.19 tok/s (worse)

| Phase | baseline | pilot |
| --- | ---: | ---: |
| exposed I/O (GPU idle) | 5,263 ms | 1,891 ms |
| expert I/O await | 6,964 ms | 3,134 ms |
| GPU waits | 6,648 ms | 11,828 ms |
| spec prefetch | dormant | 15,876 predicted / 6,026 issued / 73.1% recall / 48.2 GB |

The predictor is good and the I/O-hiding thesis is validated: exposed I/O
fell 64%. But the transport spends more than the hiding saves: a per-layer
CPU↔GPU sync to read the prediction back, plus ~48 GB of unthrottled
speculative reads for 64 tokens — ~6 prefetches into 16-slot caches per
layer per token evict the LFU-hot set, and ~3 GB/s of speculative traffic
competes with demand misses. Matches the earlier Qwen-side rejection.

## Iteration 2: 32 slots (no pilot) — 3.81 tok/s (worse)

Exposed I/O 4,207 ms (better than baseline), GPU waits 9,245 ms (much
worse). ~12 GB of wired slot memory (32 × 43 × 8.6 MB) crowds the ~18 GB
GPU working-set cap — the same failure mode as Qwen resident and Qwen
128-slot. Third independent confirmation: on a 24 GB host, small wired sets
plus page cache beat large wired sets.

## Standing conclusions for the streaming rework

1. 16 slots + adaptive rdadvise remains the best measured DSV4 profile on
   this host (4.62 tok/s).
2. The win that survives all evidence: **pilot's information without
   pilot's costs** — an async prediction hand-off (GPU writes the L+1
   router result to shared memory; a background I/O thread issues reads;
   the compute path never waits) plus an issue policy throttled to
   non-resident predictions inside a strict per-layer slot budget (≤2), so
   speculation can never evict more than it earns. Exposed I/O headroom if
   transport is free: ~38% of DSV4 decode.
3. Hash-routed layers (3 of 43) get exact CPU-side prediction from the
   token id at sampling time — include them in the async issue path at
   zero GEMV cost, but alone they are a ~7% lever.
4. n-gram speculative decoding (Task 12b/14d) stays the orthogonal
   multiplier: k-token verification divides per-token expert reads by up
   to k on exactly this model class.
