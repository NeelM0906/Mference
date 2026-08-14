# Benchmarks

This page records Mference measurements on an 8 GB M2 MacBook Air and a
24 GB M5 Pro. Each number belongs to the workload shown. Prompt length,
generated length, cache state, and hardware all change throughput, so ranges
across workloads are not run-to-run variation.

Each table states its workload and decoding settings. Mference uses the
model installed by the [command-line instructions](../README.md#try-it).
Decode rate excludes model installation, model loading, and prompt prefill.

## Results at a glance

| Host and runtime | Decode rate | Reported memory |
| --- | ---: | ---: |
| 8 GB M2, Mference | 5.10-6.30 tok/s | ~1.9-2.1 GB footprint |
| 24 GB M5 Pro, Mference | 31-35 tok/s | ~2.1 GB footprint |
| 24 GB M5 Pro, mlx-lm | 76.33-82.07 tok/s | 8.3-9.8 GB RSS; 14.7-15.3 GB GPU allocation |
| M5, Mference, Qwen 3.6 35B-A3B | 23.5-29.3 tok/s | 32-slot Qwen profile (auto as of 2026-08-06) |

A separate report covers the four pre-Maple model families on a 256 GB M3 Ultra,
including the first measured DeepSeek-V4-Flash and Inkling-Small rows:
[M3 Ultra benchmarks](BENCHMARKS_M3_ULTRA.md).

## M2 measured decode

These rows ran on a `Mac14,15` M2 MacBook Air with 8 GB of memory. No
experiment, profiler, or trace mode was active.

| Prompt / generated tokens | Prefill | TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: | ---: |
| 6 / 32 | 7,025 ms | 7,979 ms | 6.30 tok/s | 1,304 / 1,791 MiB |
| 121 / 64 | 7,934 ms | 8,862 ms | 5.10 tok/s | 1,528 / 1,776 MiB |
| 527 / 64 | 21,736 ms | 22,649 ms | 5.90 tok/s | 1,535 / 1,886 MiB |
| 1,017 / 128 | 36,729 ms | 37,656 ms | 5.38 tok/s | 1,455 / 1,971 MiB |

Each workload ran once in a fresh process. The file cache was warm but
uncontrolled, and every row produced the same token IDs as its validation
control. These four points show the production path running under the 8 GB
rule; they do not form a confidence interval or describe sustained long
generation.

### Where the short M2 row spent its time

A separate diagnostic pass on the six-token prompt divided a 162.8 ms decode
step into four broad parts:

| Work | ms/token |
| --- | ---: |
| Expert reads | 83.1 |
| Waiting in the command-buffer pipeline | 55.6 |
| Tied output head | 14.2 |
| Other runtime work | 9.9 |

The diagnostic instrumentation disabled the normal command-buffer pipeline
and reduced throughput to 4.23 tok/s. The breakdown explains where that run
spent time; it does not describe independent speedups or a performance bound.

## M5 measured decode

These rows ran on 2026-07-20 on a 24 GB M5 Pro (`Mac17,8`) with macOS 26.5.1,
Xcode 26.6, and Swift 6.3.3. No profiler or trace mode was active.

The benchmark uses chat-framed prompts and fixed, non-repeating natural
continuations. This keeps the generated text and expert-routing workload stable
without rewarding a model repetition loop. The complete production sampling
and decode path still runs for every token.

One warmup preceded three fresh-process measurements per workload. The table
reports medians; the file cache was warm but uncontrolled. A separate
free-generation smoke reached the end of each model turn without a repetition
loop.

| Prompt / generated tokens | Prefill / TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: |
| 61 / 256 | 5,096 / 5,668 ms | 35.17 tok/s | 1,834 / 2,126 MiB |
| 430 / 256 | 6,762 / 7,325 ms | 34.72 tok/s | 1,851 / 2,142 MiB |
| 3,015 / 256 | 23,038 / 23,610 ms | 31.01 tok/s | 1,835 / 2,126 MiB |

## Qwen 3.6 35B-A3B measured decode

This optimized production run was recorded on 2026-08-06 on an M5
MacBook Pro (`Mac17,2`) with 24 GB, macOS 26.5, and Swift 6.3.3. The CLI's
automatic profile selected 32 Qwen expert-cache slots; no experimental flag or
profiler was active. One discarded warmup preceded one fresh-process measured
run per frozen community case, and every output ended normally.

| Case | Prompt / generated tokens | Prefill | Decode |
| --- | --- | ---: | ---: |
| short-explanation | 62 / 538 | 7.57 s | 29.293 tok/s |
| medium-review | 426 / 704 | 8.48 s | 27.460 tok/s |
| long-synthesis | 2,940 / 695 | 26.54 s | 23.470 tok/s |

The corresponding byte-identical 16-slot controls decoded at 21.487, 24.828,
and 21.490 tok/s. That is an 18.1% geometric-mean decode gain from the
model-aware default, on top of the custom Qwen decode kernels. Long-prompt
prefill is 2.20x faster than the original 58.45 s baseline. The original and
current generated token counts differ, so only the same-output controls are
used for the decode speedup claim.

Since this run, the 2026-08 performance campaign (PR #16, merged 2026-08-13)
changed the Qwen production defaults: the automatic profile now selects 96
expert-cache slots on hosts with at least 24 GiB (32 with at least 16 GiB,
else 16), with the GPU-resident slot map and eager routed commit on by
default. The rows above therefore describe the 2026-08-06 32-slot automatic
profile, not a current checkout.

### Historical 16-slot run

These rows ran on 2026-07-31 on an M5 with 24 GB of memory, macOS 26.5, and
Swift 6.2, against the experimental
[Qwen 3.6 35B-A3B](../docs/QWEN36_PERFORMANCE.md) path. They follow the
[community benchmark protocol](COMMUNITY_BENCHMARKS.md): the three frozen
`real-generation-v1` prompts with their fixed seeds, app sampling defaults
(temperature `0.2`, Top-K `64`, Top-P `0.95`), 4K context, 16 expert-cache
slots, one discarded warmup, then one measured run per case in a fresh
process. Every measured footer reported `stop=endOfTurn`.

| Case | Prompt / generated tokens | Prefill | Decode | Peak RSS / footprint |
| --- | --- | ---: | ---: | ---: |
| short-explanation | 62 / 493 | 7.74 s | 23.05 tok/s | 1,139 / 1,447 MiB |
| medium-review | 426 / 697 | 12.71 s | 21.20 tok/s | 1,142 / 1,448 MiB |
| long-synthesis | 2,940 / 700 | 59.16 s | 18.84 tok/s | 1,093 / 1,464 MiB |

Qwen 3.6 decodes slower than Gemma 4 on the same host while using about
0.7 GB less memory. The gap is expert-streaming I/O, not compute: Qwen's
18.1 GB expert pool does not fit the page cache, and its 16 slots cover 6.2%
of a layer's 256 experts against Gemma's 12.5% of 128. The
[performance notes](QWEN36_PERFORMANCE.md) break the token down phase by
phase.

### Under an 8 GB working set

**Correction (2026-08-08):** the table originally published here reported
throughput unchanged under an emulated 8 GB working set. That run's memory
pin did not hold — the ballast was reclaimable, so the page cache kept the
expert pool and the run measured the unconstrained machine twice. (The
original rationale, "the 18.1 GB pool does not fit the page cache on
either configuration," was also wrong for the 24 GB host, where it does.)
With a verified ballast (15.2 GiB `mlock`ed, free memory near zero), the
community cases at the 8 GB auto profile (16 slots) measure:

| Case | Decode, 24 GB | Decode, ~8 GB (verified) | Footprint |
| --- | ---: | ---: | ---: |
| short-explanation | 26.45 tok/s | 14.35 tok/s | ~1.1 GB |
| medium-review | 24.99 tok/s | 12.75 tok/s | ~1.1 GB |
| long-synthesis | 21.96 tok/s | 10.81 tok/s | ~1.1 GB |

Every run reached `stop=endOfTurn`. With the pool unable to cache, decode
is SSD-latency-bound; rerunning with the 2026-08-08 optimizations disabled
lands in the same regime (12.3/12.4 tok/s short/long), confirming the
correction concerns the pin, not the runtime changes. This remains
emulated pressure on M5 hardware: a physical 8 GB Mac has a slower SSD and
GPU and should be expected to decode more slowly still, as the M2 rows
above show for Gemma 4.

## Inkling-Small 276B-A12B measured decode

The native top-6 decode path was measured on 2026-08-06 on the same 24 GB M5
MacBook Pro (`Mac17,2`), with macOS 26.5 and Swift 6.3.3. The untouched base
was `6bf428f`; the optimized implementation was `485df08`. Both used release
builds, the production automatic 16-slot cache, strict full-SHA verification,
and the frozen community prompts and sampling settings. One discarded warmup
preceded one measured fresh process per case; every process exited 0 and
reported `stop=endOfTurn`.

| Case | Prompt / generated tokens | Base decode | Native top-6 decode | Gain |
| --- | --- | ---: | ---: | ---: |
| short-explanation | 59 / 469 | 2.909 tok/s | 3.434 tok/s | 18.0% |
| medium-review | 421 / 576 | 2.961 tok/s | 3.670 tok/s | 23.9% |
| long-synthesis | 2,785 / 372 | 2.819 tok/s | 3.038 tok/s | 7.8% |

The geometric-mean decode gain is **16.4%**. Each optimized output is
byte-identical to its base output, so the table compares the same generated
tokens and expert-routing workload. Complete measured timing footers, in
base/optimized order, were:

```text
[stop=endOfTurn prefill=59tok/64.89s new=469tok decode=161.22s tok/s=2.909]
[stop=endOfTurn prefill=59tok/64.25s new=469tok decode=136.57s tok/s=3.434]
[stop=endOfTurn prefill=421tok/79.61s new=576tok decode=194.53s tok/s=2.961]
[stop=endOfTurn prefill=421tok/76.41s new=576tok decode=156.95s tok/s=3.670]
[stop=endOfTurn prefill=2785tok/169.39s new=372tok decode=131.94s tok/s=2.819]
[stop=endOfTurn prefill=2785tok/180.48s new=372tok decode=122.46s tok/s=3.038]
```

Inkling routes six experts, but the old runtime padded them to the shared
top-8 INT4 contract with two duplicate buffers and zero weights. The optimized
path dispatches a dedicated six-SIMD-group down/reduce Metal kernel, runs only
six phase-1 expert slots, and submits resident phase-1 work while the CPU reads
cache misses. It preserves the FP32 shared-sink/residual path.

A production 24-slot warmup was also tried with the same short workload. It
produced byte-identical output but decoded at 0.710 tok/s while free-memory
pressure fell to 24%, versus 3.374 tok/s for the optimized 16-slot warmup.
Because the first larger cache already exceeded the useful memory envelope,
32 slots were not run and Inkling's automatic default remains 16.

The exact measured command was:

```text
.build/release/MferenceCLI --model scratch/inklingsmall.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/<case>.json --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 --seed <seed>
```

## Same-host MLX comparison

The same M5 Pro ran MLX 0.32.0 and mlx-lm 0.31.3 against the same checkpoint,
prompt-token IDs, and generated-token counts. MLX measured 82.07, 80.25, and
76.33 tok/s for the 121-, 527-, and 1,017-token prompts.

Treat this as throughput context, not a complete engine comparison:

- The engines ran in separate blocks rather than a balanced, interleaved order.
- Their first-token clocks started at different points, so TTFT is not comparable.
- Generated IDs matched for the shortest prompt but diverged for the two longer prompts.
- Mference recorded a 1.89-2.09 GiB physical footprint. MLX reported
  14.66-15.31 GB of peak GPU allocation and 8.27-9.79 GB of peak process RSS.
  Those counters measure different things and should not be compared as a
  direct memory ratio.

The MLX process required the larger host and is not an 8 GB Mference
deployment path.

## Reproduce and contribute a result

The [community benchmark guide](COMMUNITY_BENCHMARKS.md) uses short, medium,
and long chat-framed prompts with fixed seeds. It requires coherent output and
a normal end of turn, so a repetition loop cannot become a published speed
result. The public CLI's timing footer reports decode-only throughput without a
separate research harness.

Community runs generate their own output, while the reference table uses fixed
non-repeating continuations for token-for-token stability. Compare community
submissions only when their prompt and generated token counts match.

A current checkout may not reproduce a historical number after the runtime,
compiler, or operating system changes. Report the commit and all three rows
rather than presenting one run as a general hardware result.

Read [System design](SYSTEM_DESIGN.md) for the runtime and resource split,
[Experiments](OPTIMIZATION_JOURNEY.md) for the main wins and failures, and the
[measurement lessons](experiments/summaries/09-validation-and-measurement-lessons.md)
for the rules used to evaluate performance changes.
