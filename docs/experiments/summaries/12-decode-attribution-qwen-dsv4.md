# Decode attribution: Qwen 3.6 and DSV4-Flash — M5/24 GB, 2026-08-07

`MFERENCE_PHASES=1` greedy decode breakdowns on a quiet machine, production
profiles (Qwen: 32 slots; DSV4: 16 slots, `--rdadvise adaptive`,
trusted-receipt). Diagnostic runs, not protocol benchmarks.

## Qwen 3.6, 128 tokens, 24.33 tok/s (41.1 ms/token)

| Phase | total ms | share |
| --- | ---: | ---: |
| GPU compute (GPU waits) | 2,741 | 52% |
| expert I/O await | 2,370 | 45% |
| — overlapped with GPU | 1,154 | hidden |
| — exposed (GPU idle) | 1,215 | 23% |
| cb encode+commit | 150 | 3% |
| spec prefetch | predicted 0 | dormant |

Even with the page cache holding the whole 18 GB pool, ~2.4 s of a 5.3 s
decode is spent in `pread` copies out of the page cache into slots, a
quarter of the step fully exposing the GPU. Decode sits at ~27% of the
~90 tok/s bandwidth roofline.

## DSV4-Flash, 64 tokens, 4.62 tok/s (216 ms/token)

| Phase | total ms | share |
| --- | ---: | ---: |
| GPU compute (GPU waits) | 6,648 | 48% |
| expert I/O await | 6,964 | 50% |
| — overlapped with GPU | 1,702 | hidden (24% of I/O) |
| — exposed (GPU idle) | 5,263 | 38% |
| cb encode+commit | 236 | 2% |
| spec prefetch | predicted 0 | dormant |

## Consequences (ranking the candidate queues)

1. **Overlap is the fattest streaming target.** DSV4's exposed I/O is 38% of
   wall time; even Qwen exposes 23%. The speculative-prefetch machinery
   exists but is inert in production profiles. Perfect overlap ≈ +60% DSV4
   decode before any kernel work. → Task 14c ranks first, with 14b
   (queue-depth saturation of miss reads) feeding it.
2. **GPU compute is ~half of both models' steps** — the kernel program
   (batched multi-expert GEMV, tensor-ops paths) attacks the other half.
3. **Slot-copy overhead is real on warm hosts:** Qwen's 2.4 s of page-cache
   memcpy is exactly what resident mode eliminated on short prompts; a
   copy-free hot path for *cache hits* (serving hits from mapped pages while
   keeping slots for misses) is a measurable hybrid candidate.
