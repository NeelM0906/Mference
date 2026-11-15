# Pre-encoded decode command sequences: rejected on encode cost, M1 Max/32 GB, 2026-11-15

## Hypothesis

Moving per-call parameters into buffers would let each layer's decode command
sequence be encoded once into an indirect command buffer (ICB) and replayed
every token. That would remove CPU encode time from the critical path of a
decode step. This was the optional step D3 of the M1 Max performance program;
it was to be pursued only if CPU encode turned out to be the largest remaining
decode term.

## Measurement

These were standalone Metal probes, lead diagnostics rather than protocol
benchmarks. The host was a MacBook Pro M1 Max (24-core GPU), 32 GB, macOS
27.2.1. One "token" is 900 dispatches of a small kernel with four buffer
bindings and two constants, which is Qwen 3.6's documented dispatch count. The
probe ran 2 blocks of 500 tokens, after 30 warm-up tokens per mode, and reports
medians.

| Encoding | CPU encode µs / token | µs / dispatch | GPU µs / token |
| --- | ---: | ---: | ---: |
| One compute encoder per dispatch (the decode path's shape) | 540–548 | 0.60–0.61 | 2,717–3,049 |
| One encoder, `memoryBarrier(scope: .buffers)` between dispatches | 255–259 | 0.28–0.29 | 1,867–1,904 |
| Pre-encoded ICB, one `executeCommandsInBuffer` per token | 11–13 | 0.012–0.015 | 2,090–2,110 |

On the same host, Qwen 3.6 decodes at about 34 tok/s (about 29 ms per token)
once the M1 keep-warm fix is in (PR #54). The GPU is busy for 43–47% of the
decode span. Each synchronous command-buffer round trip costs 165–330 µs, of
which 4–90 µs is GPU work.

## Provenance

**Host and toolchain:**

- MacBook Pro (MacBookPro18,4), Apple M1 Max, 10-core CPU, 24-core GPU,
  32 GB.
- macOS 27.2.1 (26B131), on AC power, with no thermal warning before or after
  the run.
- Apple Swift 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1), Command Line Tools.

**Probe source:** a standalone program, not part of this repository's
`main`. It is kept on a local branch of the performance program, at
`Scripts/spikes/ws2/Common.swift` and `Scripts/spikes/ws2/S6/main.swift`
(commit `0446ab0`).

- Each token encodes 900 dispatches of
  `a[i] = b[i] * p1 + c[i] + d[i] + float(p0 & 1)`.
- Each dispatch uses 4 threadgroups of 64 threads, four buffer bindings and
  two constants.
- The ICB variant passes the constants through a buffer and sets a barrier on
  each command.
- CPU time is `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)` around encoding, from
  `makeCommandBuffer` up to but not including `commit`.
- GPU time is `gpuEndTime − gpuStartTime` after `waitUntilCompleted`.

**Commands:**

```bash
swiftc -O -framework Metal -framework Foundation \
  Scripts/spikes/ws2/Common.swift Scripts/spikes/ws2/S6/main.swift -o S6
S6 <out-dir>        # defaults: --n 900 --reps 500
```

**Exit code:** 0. The run lasted 2026-11-15 07:40:15–07:40:24Z, with no
other model or test process running.

**Complete output** (µs, 500 tokens per row; `_b1` and `_b2` are the two
blocks):

| Row | Metric | p10 | Median | p90 | p99 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| encoder_per_dispatch_b1 | CPU encode per token | 536.834 | 547.791 | 630.042 | 1670.500 | 604.254 |
| encoder_per_dispatch_b1 | CPU encode per dispatch | 0.596 | 0.609 | 0.700 | 1.856 | 0.671 |
| encoder_per_dispatch_b1 | GPU per token | 2499.750 | 3049.208 | 3993.542 | 5271.583 | 3187.031 |
| one_encoder_barriers_b1 | CPU encode per token | 245.208 | 259.292 | 280.250 | 296.750 | 260.395 |
| one_encoder_barriers_b1 | CPU encode per dispatch | 0.272 | 0.288 | 0.311 | 0.330 | 0.289 |
| one_encoder_barriers_b1 | GPU per token | 1530.250 | 1904.125 | 2447.208 | 2946.125 | 1962.444 |
| icb_execute_b1 | CPU encode per token | 8.416 | 13.334 | 24.458 | 39.750 | 15.229 |
| icb_execute_b1 | CPU encode per dispatch | 0.009 | 0.015 | 0.027 | 0.044 | 0.017 |
| icb_execute_b1 | GPU per token | 1794.958 | 2090.417 | 2397.208 | 2888.042 | 2123.749 |
| encoder_per_dispatch_b2 | CPU encode per token | 517.333 | 539.584 | 572.333 | 1346.250 | 571.031 |
| encoder_per_dispatch_b2 | CPU encode per dispatch | 0.575 | 0.600 | 0.636 | 1.496 | 0.634 |
| encoder_per_dispatch_b2 | GPU per token | 1865.583 | 2717.417 | 3810.625 | 5103.708 | 2813.984 |
| one_encoder_barriers_b2 | CPU encode per token | 246.292 | 254.500 | 276.667 | 292.417 | 258.191 |
| one_encoder_barriers_b2 | CPU encode per dispatch | 0.274 | 0.283 | 0.307 | 0.325 | 0.287 |
| one_encoder_barriers_b2 | GPU per token | 1569.250 | 1867.000 | 2198.125 | 2866.500 | 1886.645 |
| icb_execute_b2 | CPU encode per token | 6.542 | 10.541 | 19.459 | 25.792 | 12.004 |
| icb_execute_b2 | CPU encode per dispatch | 0.007 | 0.012 | 0.022 | 0.029 | 0.013 |
| icb_execute_b2 | GPU per token | 1792.750 | 2109.750 | 2356.250 | 2854.750 | 2124.775 |

**Deviations from the community benchmark protocol:**

- This is a synthetic probe, not a model run, so the protocol's prompts,
  sampling and timing footer do not apply.
- System swap stood at 4.58 GB, left by an unrelated earlier workload. It did
  not grow during the run, and memory pressure stayed normal.

**Sources of the decode-context figures:**

- The 34 tok/s figure and the 43–47% GPU-busy share come from Qwen 3.6
  protocol runs at runtime commit `edbbdd9` (PR #54), with the same host and
  toolchain.
- The 165–330 µs round trip comes from a companion standalone probe
  (`Scripts/spikes/ws2/S1/main.swift`, same branch).

Neither is recomputed here.

## Disposition

**Rejected.** Encoding is about 0.6 µs per dispatch, or about 0.5 ms of a
~29 ms token, roughly 2%. It is not the largest remaining term. The idle gaps
come from tens of synchronous CPU–GPU handoffs per token, so the work goes to
removing those handoffs (resident experts with GPU routing, and halt-and-resume
on the slot path) instead.

## Lesson

On this hardware CPU encode is cheap. The repository's earlier M5 estimate was
1.0–1.3 µs per dispatch, and the M1 Max measured 0.6. Measure encode on the
target host before restructuring how parameters are passed.

The probe also measured GPU-side overhead. One encoder per dispatch costs about
1 µs more GPU time per dispatch than a single encoder with barriers. That
matters only where dispatches are tiny.

An ICB is still worth keeping in mind for skipping work on the GPU: an ICB run
with a GPU-written execution range of length 0 skipped 800 dispatches in about
9 µs. That is a different use from pre-encoding, and it was not adopted here.
