# Benchmarks

This page records Mference measurements on an 8 GB M2 MacBook Air and a
24 GB M5 Pro. Each number belongs to the workload shown. Prompt length,
generated length, cache state, and hardware all change throughput, so ranges
across workloads are not run-to-run variation.

Each table states its workload and decoding settings. Mference uses the
model installed by the [command-line instructions](../README.md#command-line).
Decode rate excludes model installation, model loading, and prompt prefill.

## Results at a glance

| Host and runtime | Decode rate | Reported memory |
| --- | ---: | ---: |
| 8 GB M2, Mference | 5.10-6.30 tok/s | ~1.9-2.1 GB footprint |
| 24 GB M5 Pro, Mference | 31-35 tok/s | ~2.1 GB footprint |
| 24 GB M5 Pro, mlx-lm | 76.33-82.07 tok/s | 8.3-9.8 GB RSS; 14.7-15.3 GB GPU allocation |
| M5, Mference, Qwen 3.6 35B-A3B | 23.5-29.3 tok/s | 32-slot Qwen profile (auto as of 2026-08-06) |
| 256 GB M3 Ultra, Mference, Flash-Next 180B-A3.5B | 18.17-21.03 tok/s decode; 24.3-210.5 tok/s prefill | Resident mapped expert pool; 398-431 MiB reported RSS |
| 32 GB M1 Max, Mference, Qwen 3.6 35B-A3B (`cb99287`; ledger M23; `--verify full-sha256`) | 27.52-34.46 tok/s | 6.98-7.54 GB footprint |
| 32 GB M1 Max, Mference, Qwen3.8-Flash-Next Unsloth UD-IQ3_XXS (`54d658c`; ledger M32; `--verify full-sha256`) | 3.40-4.12 tok/s | 0.66-5.35 GB footprint |

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

## Qwen3.8-Flash-Next native prefill and resident GPU routing

These rows ran on 2026-09-10 on a 256 GB M3 Ultra (`Mac15,14`) with macOS
26.3 and Swift 6.2.4, against the INT8-router install. The automatic profile
selected the resident mapped routed-expert pool. One discarded warmup preceded
one fresh-process measured run per frozen community case; no experimental
control or profiler was active, every process exited 0, and all three outputs
reached `stop=endOfTurn` and were read for coherence.

| Case | Prompt / generated | Prefill | Prompt rate | Old scalar resident | Speedup | Decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| short-explanation | 62 / 521 | 2.55 s | 24.31 tok/s | 5.85 s | 2.29× | 20.109 tok/s |
| medium-review | 426 / 557 | 4.42 s | 96.38 tok/s | 29.03 s | 6.57× | 21.029 tok/s |
| long-synthesis | 2,940 / 635 | 13.97 s | 210.45 tok/s | 192.69 s | 13.79× | 18.168 tok/s |

The current rows are from `qwen38flashnext-prefill-final`, run with the exact
community command, one discarded warmup and one measured fresh process per
case. The old and new prefill columns use the same prompt IDs and warm resident
backend. Generated token counts differ because batched reduction order can move
stochastic sampling across a near tie, so the decode column is the autonomous
current run rather than a token-for-token speed A/B. Every output was coherent
and reached `stop=endOfTurn`. A current-build A/B on the real install kept the
prompt and next-token argmaxes and the 16-token greedy rollout exact; relative
head drift was 2.17% after the prompt and 1.71% after one continuation.

Long resident prompts now use a grouped routed-expert TensorOps kernel,
contiguous QSA rows attend in one causal batch, and FP32 TensorOps projections
feed the hyper-connection and indexer paths. Focused kernel tests compare the
new paths with their established GPU or CPU references. The resident decode
path separately keeps router ids on the GPU and addresses selected expert
records directly in the mapped layer slab. Against the immediately prior
CPU-routed resident run it improved decode by 29.9%, 33.3%, and 38.8% (33.9%
geometric mean), with byte-identical output in all three cases.

This closes host-side scalar prompt replay and adds a real expert matrix path,
but it does not clear the external performance gate. The public Rapid-MLX M3
Ultra record reports 262-875 prompt tok/s on different prompt lengths and
21-25 tok/s ordinary decode, so both remain directional targets. Mference's
next prefill bottleneck is the CPU grouping and layer synchronization between
GPU routing and the grouped expert dispatch; moving that handoff onto the GPU
is the next honest route toward the public low bar.

## M1 Max 32 GB measured (2026-11 performance program)

These rows ran between 2026-11-14 and 2026-11-17 on one MacBook Pro
(`MacBookPro18,4`): Apple M1 Max with a 24-core GPU, 32 GB, macOS 27.2.1
(26B131), and Swift 6.4.2. They cover two installs: Qwen 3.6 35B-A3B
(`qwen36.gturbo`, pinned `mlx-community/Qwen3.6-35B-A3B-4bit@38740b8`) and the
[Unsloth UD-IQ3_XXS Flash-Next GGUF](families/QWEN38_FLASH_NEXT_UNSLOTH.md)
(`qwen38flashnextudiq3xxs.gturbo`). Every run used a release build of the
named commit, on AC power, with no other model process running. Each figure
names the ledger entry that records it (M1, M7, ...), the commit measured,
and the `--verify` mode. The ledger is the program's measurement log, with
raw output per entry; it is kept outside the repository. Figures measured by a
workstream lead in its own gated hold (machine lock, quiet machine, AC power,
swap guard) name that hold's record instead, for example `ws1-b-mma`; those
records are kept with the ledger. Apart from the library smoke (ledger M41),
none of these figures was re-run on the final `main`, `48461a1`.

Three workloads appear:

- **P** is the [community protocol](COMMUNITY_BENCHMARKS.md): the three frozen
  `real-generation-v1` cases, a fresh CLI process per run, `--max-new 1024
  --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`, the case
  seeds, and `--verify full-sha256`. Several A/B comparisons below ran the P
  cases with a different verify mode; each row says which.
- **S** is a fresh `MferenceServer` on loopback answering a cold request (0
  cached tokens) with 32 generated tokens. "S 3K" is the 2,940-token
  long-synthesis prompt. TTFT is measured by a streaming client.
- **L** is a 31,480-token prompt with 32 generated tokens, through the CLI
  (`--max-context 32768`) or through a server (`--max-context 36864`) that
  then answers a follow-up turn reusing 31,511 cached tokens.

The S and L streaming client is
[`Scripts/server_stream_bench.py`](../Scripts/server_stream_bench.py), and the
32K prompt and its follow-up are the frozen
[long-context-v1](benchmark-prompts/long-context-v1/README.md) set.
Decode over 32 tokens is indicative only: after a 32K prefill it takes
1.2-1.7 s (ledger M31; `8177a1a` and `b9e7a48`; `--verify auto`).

Under `--verify full-sha256`, P prefill includes hashing the whole expert pool
on first touch. For Qwen 3.6 that is about 8.5-8.7 s of the short and medium
prefill times; the same prompts prefill in about 2-3 s with `--verify auto`
(ledger note after M7; `06c13ee`-era binaries).

### Qwen 3.6 35B-A3B: community protocol baselines

Mode P, `--verify full-sha256`, five measured runs per case after a discarded
warmup. All runs ended at `stop=endOfTurn`.

| Case | Commit (ledger) | Prompt / generated | Prefill median | Decode median (min-max) | Peak footprint |
| --- | --- | --- | ---: | ---: | ---: |
| short-explanation | `06c13ee` (M1) | 62 / 520 | 10.44 s | 2.956 tok/s, slow mode in 5 of 5 runs | ~6.7 GB |
| medium-review | `06c13ee` (M1) | 426 / 703 | 11.71 s | bimodal: 3.006-3.072 (3 runs), 7.656-7.897 (2 runs) | ~6.75 GB |
| long-synthesis | `06c13ee` (M1) | 2,940 / 626 | 24.51 s | bimodal: 2.404-2.550 (4 runs), 7.415 (1 run) | ~7.2 GB |
| short-explanation | `477122c` (M7) | 62 / 520 | 10.67 s | 33.51 (32.74-34.24) | 6,658-6,680 MB |
| medium-review | `477122c` (M7) | 426 / 703 | 11.90 s | 34.80 (34.64-35.22) | 6,729-6,741 MB |
| long-synthesis | `477122c` (M7) | 2,940 / 626 | 24.49 s | 27.67 (27.03-30.76) | 7,171-7,195 MB |
| short-explanation | `cb99287` (M23) | 62 / 513 | 10.45 s | 34.46 (33.81-34.56) | 6,979-6,992 MB |
| medium-review | `cb99287` (M23) | 426 / 695 | 11.85 s | 34.16 (33.39-34.44) | 7,044-7,049 MB |
| long-synthesis | `cb99287` (M23) | 2,940 / 565 | 21.71 s | 27.52 (26.73-27.61) | 7,522-7,538 MB |

At `06c13ee` decode was held down by the GPU keep-warm filler, whose command
buffers took about 2.9 ms of GPU time each on this GPU; `477122c` (#54) is the
first commit without it (ledger M1 and M7; the filler timing is from the M2
diagnostic on a `06c13ee` build). The `cb99287` outputs
differ from the earlier ones because #59 changed them within its tolerance
gate, so its decode column covers different token sequences (ledger M23).
Peak footprint at `cb99287` is 4.6-4.8% above `477122c` on every case; the
cause was not tested (ledger M23).

### Qwen 3.6 35B-A3B: server and 32K baselines

S and L ran with generation capped at 32 tokens, except the `06c13ee` S short
and medium rows, which decoded to the end of the turn (TTFT does not depend on
the cap). `cb99287` used 5 measured S runs and 3 measured L runs per row; the
older rows used 5.

| Workload | Earlier baseline | `cb99287` (M23) |
| --- | ---: | ---: |
| S short TTFT, 62 tokens | 0.78 s (0.71-1.14), `06c13ee` (M3) | 0.958 s (0.949-0.963) |
| S medium TTFT, 426 tokens | 2.66 s (2.51-2.73), `06c13ee` (M3) | 2.085 s (2.057-2.096) |
| S long TTFT, 2,940 tokens | 17.71 s (17.66-17.82), `06c13ee` (M3) | 12.89 s (12.45-13.00) |
| S decode, 32 tokens (short / medium / long) | filler-bound, not a baseline (M3) | 37.04 / 35.42 / 27.30 tok/s |
| L server cold TTFT, 31,480 tokens | 581.8 s (577.8-609.1), `06c13ee` (M3) | 229.60 s (229.42-231.68) |
| L server follow-up TTFT, 31,511 cached | 48.2 s (47.7-55.4), `06c13ee` (M3) | 16.19 s (16.07-16.51) |
| L server decode, 32 tokens (cold / follow-up) | filler-bound, not a baseline (M3) | 18.01 / 17.95 tok/s |
| L CLI cold prefill, 31,480 tokens | 573.0 s (555.7-579.4), `2361831` (M12) | 227.70 s (227.68-228.27) |
| L CLI decode, 32 tokens | 7.36 tok/s (6.93-10.43), `2361831` (M12) | 18.68 tok/s (16.97-19.08) |
| Peak footprint | server 7.0-7.1 GB (M3); L CLI 7.53 GB (M12) | server 7.35 / 7.33 GB (S / L); L CLI 7.50-7.52 GB |

Every S and L run above used `--verify auto`, the server and CLI default
(ledger M3, M12, M23). The S short TTFT at `cb99287` is 22.8% above
`06c13ee`'s median, with overlapping ranges and a server-side cause that was
not tested. The L CLI decode rise from 7.36 to 18.68 tok/s happened between the
M19 builds (`ff0fc32`, `68b0aeb`) and `f831b21` and was not bisected
(ledger M23).

### Per-change A/B results

Each merged change was measured against its own base `main` in one of three
ways:

- **Tier-1 pair:** one control run, then one candidate run, in one session.
  The claimed metric is fixed before measurement. A change is accepted when
  the claim improves by 10% or more and no tracked metric is more than
  7% worse; a smaller result goes to an A/B/B/A.
- **A/B/B/A:** control, candidate, candidate, control segments in one session,
  each a fresh process or server. An effect counts only when both comparisons
  agree in direction and the ranges do not overlap.
- **Tier 0:** no runtime change. A static diff check plus an output-identity
  run.

Class E changes must keep output byte-identical to the control; every class E
row below did. Class T changes may change output within a tolerance gate
recorded with the change. Percentages are those recorded with each
measurement.

**Qwen 3.6 35B-A3B**

| PR (merge) | Change, class | Ledger; candidate vs control; method; verify | Claimed metric: control → candidate (Δ) | Verdict |
| --- | --- | --- | --- | --- |
| #57 (`b6f60cc`) | Live-memory budget type, not yet called by the runtime | M10; `25ddcc1` vs `477122c`; A/B/B/A, P long; `full-sha256` | No claim. Decode +0.3% / -0.2%, prefill -0.8% / +0.2%; output identical | PASS |
| | | M15; `a831aa5` (merged head) vs `2361831`; tier 0; `full-sha256` | Long output identical | PASS |
| #58 (`ff0fc32`) | Prefill stage breakdown, a diagnostic toggle off by default | M11; `3cc33a8` vs `477122c`; A/B/B/A, P; `full-sha256` | No claim. All ranges overlap; medium prefill +0.9% / +0.3% and decode -0.7% / -1.2% are not a measured effect | PASS |
| | | M13; `2f9b8cd` vs `2361831`; A/B/B/A, P; `full-sha256` | Short decode -7.0% / -5.5%, ranges overlap | PASS, flagged |
| | | M13a; same arms; A/B/B/A, P short, 4 runs per segment; `full-sha256` | Short decode +6.9% / +3.1%, prefill -3.2% / -1.6%: the M13 direction reversed | PASS |
| #59 (`b834cf7`) | GQA prefill attention (P3a), class T | M18; `68b0aeb` vs `ff0fc32`; tier-1 pair; `auto` | S 3K TTFT 18.03 → 14.82 s (-17.8%); L CLI cold prefill 561.8 → 237.8 s (-57.7%) | Tier-1 ACCEPT |
| | | M19; same arms; A/B/B/A, L CLI; `auto` | L cold prefill -57.5% / -58.2%, measured effect. L decode -1.6% / -3.0%, ranges overlap | PASS; L decode recorded as a limit |
| #60 (`f831b21`) | Prefill expert sweeps bypass the page cache, class E | M16; `bd88b1d` vs `ff0fc32`; tier-1 pair; `trusted-receipt` | S 3K TTFT 18.21 → 16.94 s (-7.0%) | Ambiguous, below the 10% bar |
| | | M17; same arms; A/B/B/A, S 3K and P medium/long; `trusted-receipt` | S 3K TTFT +20.2% / -4.8%: no measured effect. S decode after the 3K prefill +156% / +176%, measured effect | TTFT claim unsupported; the maintainer chose to merge on the S decode claim |
| | | M20; `eef6ba1` (merged head) vs `b834cf7`; tier-1 pair; `trusted-receipt` | S decode after 3K, 32 tokens, 10.50 → 29.47 tok/s (+180.7%) | Tier-1 ACCEPT |
| #61 (`cb99287`) | Installer stall fix, installer only | M21; `6b9fd50` vs `f831b21`; tier 0; `full-sha256` | Long output identical | Tier-0 PASS |
| #62 (`4b62f16`) | Multi-prefix server prompt cache, class E | M22; `3a296fb` vs `f831b21` runtime; tier-1 pair, server sequence; `auto` | Follow-up TTFT after a 32K turn and an unrelated short turn: 247.71 → 16.48 s (-93.3%). Server peak footprint 7.33 → 8.82 GB (+20.4%), within the cache's 4 GiB budget | Claim confirmed, class E holds; ACCEPT subject to a ruling on whether footprint counts for a cache meant to hold memory |
| #63 (`9a67018`) | Grouped routed-expert prefill (P3b), class T | M24; `9829772` vs `4b62f16`; tier-1 pair; `auto` | S 3K TTFT 12.349 → 12.267 s (-0.7%) | Ambiguous |
| | | M25; same arms; A/B/B/A, S 3K and P medium/long; `auto` | S 3K TTFT -2.0% / -1.8%, measured effect (-1.9%). P long prefill -5.7% / -5.5%, measured effect | PASS: claim confirmed below the bar, no regression |
| #64 (`c12f01d`) | Live-memory governor for the prefix cache | M26 (`d26509a`), M27 (`076cd6b`, merged head) vs `9a67018`; tier 0, server sequence; `auto` | Both turns identical in content, cached tokens and cache outcome; no shed fired | Tier-0 PASS |
| #65 (`0a138de`) | Batched grouped decode attention, class T | M28; `a94a86d` vs `c12f01d`; tier-1 pair; `auto` | L CLI decode, 32 tokens, median of 2: 17.86 → 27.56 tok/s (+54.3%) | Tier-1 PASS |
| #66 (`8177a1a`) | INT4 grouped GEMM for routed prefill off Apple10, class T | M29; `9e1482d` vs `0a138de`; tier-1 pair; `auto` | S 3K TTFT 12.281 → 9.328 s (-24.0%). L CLI cold prefill 216.46 → 179.44 s (-17.1%), reported, not claimed | Tier-1 PASS |
| #67 (`f8143c9`) | GQA prefill attention with shared K/V staging, class E | M30; `b9e7a48` vs `8177a1a`; tier-1 pair; `auto` | L CLI cold prefill, median of 2: 183.58 → 90.77 s (-50.6%). Also: 32K follow-up TTFT 15.16 → 6.42 s (-57.6%); S 3K TTFT 9.618 → 8.443 s (-12.2%) | Claim passes; one L decode run (12.31 tok/s) tripped the 7% guard |
| | | M31; same arms; A/B/B/A, L CLI; `auto` | L decode +20.6% then -10.1%: opposite directions, no effect. L prefill 88.98-89.27 s against 187.09-215.23 s | Guard flag cleared |
| #72 (`80106b8`) | Shadow speculative prefetch at budget 8 on 32 GiB hosts, class E | M37; `5e1b4ef` vs `67951bd`; A/B/B/A, P long, 3 runs per segment; `auto` | P long decode +7.8% / +6.9%, ranges do not overlap; pooled 31.345 → 33.394 tok/s (+6.5%). Short +4.5%, medium +2.8% and L 32K (128 tokens) +2.0%, one run each; no footprint increase | Confirmed |

Notes on reading these rows:

- P prefill in M18, M24, M25, M28, M29 and M30 ran with `--verify auto`, and in
  M16, M17 and M20 with `--verify trusted-receipt`, so those P times are not
  comparable with the `full-sha256` baselines above. #60 used
  `trusted-receipt` on purpose: a first-touch hash would load the expert pool
  into the page cache and hide the effect under test.
- M17's TTFT disagreement comes from one slow candidate segment (19.9-22.0 s)
  with no recorded cause.
- M31 voided one control run whose swap-outs grew (about 514 MB). It was
  replaced, and no candidate run was voided.
- After #67, the last Qwen 3.6 numbers in this series are that change's
  candidate arm (ledger M30, M31; `b9e7a48`; `--verify auto`): L CLI cold
  prefill 89.70 / 91.83 s (M30) and 88.98-89.27 s (M31), 32K follow-up TTFT
  6.42 s, S 3K TTFT 8.443 s. #72's candidate arm decoded P long at
  32.81-33.59 tok/s (ledger M37; `5e1b4ef`; `--verify auto`). No full
  baseline was re-run on a later `main`.

**Qwen3.8-Flash-Next, Unsloth UD-IQ3_XXS**

| PR (merge) | Change, class | Ledger; candidate vs control; method; verify | Claimed metric: control → candidate (Δ) | Verdict |
| --- | --- | --- | --- | --- |
| #68 (`034757d`) | The family itself | M32; `54d658c`; community protocol (FAMILY_GATE step 5) and first baseline; `full-sha256` | See below | Gate 5 recorded |
| #69 (`ad99455`) | Decode MoE join after the router readback, class E | M34; `c16a9cf` (merged head) vs `034757d`; A/B/B/A, P short, 3 runs per segment; `full-sha256` | Short decode +8.2% / +8.2%, ranges do not overlap; pooled medians 4.10 → 4.39 tok/s (+7.1%) | Confirmed. 2% guard PASS: short prefill +3.1% / +0.2% |
| #70 (`67951bd`) | Batched per-token GEMV loops in GGUF prefill, class E | M35; `a68b588` vs `ad99455`; tier-1 pair, P, one run per case; `full-sha256` | Long prefill 179.11 → 120.81 s (-32.6%); medium 49.21 → 42.21 s (-14.2%); long peak footprint 5.35 → 3.39 GB | Tier-1 ACCEPT; decode -2.1% to -2.6% on all three cases flagged |
| | | M36; `1be8d3d` (final head) vs `ad99455`; A/B/B/A, P short, 3 runs per segment; `full-sha256` | Short decode -0.1% / +0.9% (pooled 4.238 → 4.256 tok/s). Short prefill -2.8% / -2.6%, ranges do not overlap (pooled 31.94 → 31.07 s) | 2% rule PASS; M35's decode flag resolved |
| #71 (`1a7b740`) | Server prefill chunk follows host RAM (4,096 tokens on this 32 GiB host), class T, gate exact | WS1 hold `ws1-server-chunk`; `0b86be7` (default 4,096) and `882fe49` (default 1,024); server, 1 warmup + 2 measured each; `auto` | S 3K TTFT 108.30 / 110.12 s at 1,024 → 87.32 / 87.71 s at 4,096; peak footprint 2,318 → 4,388 MB | Merged; the merged head was not re-measured |
| #77 (`d895cf5`) | Simdgroup-matrix GEMM for INT8-2L prefill projections, class T (pooled top-1 0.9875) | WS1 hold `ws1-b-mma`; `bb5c2a9` (Sources identical to the merged head); server, default chunk 4,096, prompt cache off, 1 warmup + 2 measured; `auto` | S 3K TTFT **72.11 / 72.42 s** (server prefill 72.05 s); peak footprint 4,449 MB. Against #71's 87.32 / 87.71 s that is -17.5%, measured in a different session, not interleaved | Merged |
| #75 (`48461a1`) | Flash-Next in the multi-prefix prompt cache (1.5 GiB automatic cap), class E at chunks 4,096 and 128 | WS3 hold `ws3-k4d-classE-4k-defaultchunk`; `6ce8deb` vs `80106b8`; server, `--max-context 4096`, greedy, 32 tokens, diagnostics on, one sample per arm; `auto` | After an unrelated 426-token turn, the 2,996-token follow-up restored 2,971 cached tokens and finished in 14.72 s (5.12 s prefill); `main`'s live continuation of the same turn took 13.77 s. The cold 2,940-token first turn took 100.55 s (`main`) and 122.56 s (candidate); later cold repeats put the candidate 2.9% slower on average (holds `ws3-k4-abba-x1-defaultchunk`, `ws3-k4-ba-x1-defaultchunk`), not attributed. Peak footprint 3.37 GB both | Merged |
| #78 (`97890dc`) | The expert slot slab is mapped, so a released model's pages leave at once, class E on all six cases of both installs | WS3 holds `ws3-slab-mmap-switch`, `ws3-slab-mmap-abba-long`, `ws3-slab-mmap-classE`; `8f76f61` (Sources identical to the merged head) vs `782715d`; library server and CLI; `auto` | Library switch Qwen 3.6 → Unsloth: footprint 6,986 MiB (last sample before the swap) → 271 MiB when the swap finished, then 2,851 MiB steady; process peak 7,047 MiB (7.39 GB); swap 0 → 0. Cold long prefill A/B/B/A: 93.63 / 91.13 s against 91.88 / 94.54 s, no effect | Merged |

The `ad99455` control in M35 and M36 was a build of `c16a9cf`, whose sources
are identical. The merged head of #70 differs from `1be8d3d` only in one test
file. #76 (`782715d`) adds a stderr notice when `--verify auto` falls back to a
full SHA-256 pass and has no figures.

**Unsloth gate 5 at `54d658c`** (ledger M32; `--verify full-sha256`; one
discarded warmup per hold and five measured fresh processes per case; 15 of
15 at `stop=endOfTurn`):

| Case | Prompt / generated | Prefill median | Decode median (min-max) |
| --- | --- | ---: | ---: |
| short-explanation | 62 / 547 | 31.96 s | 4.12 tok/s (4.10-4.14) |
| medium-review | 426 / 706 | 50.86 s | 3.83 tok/s (3.44-4.08) |
| long-synthesis | 2,940 / 569 | 178.59 s | 3.40 tok/s (3.33-3.48) |

One medium-review run was voided because the host ran on battery for 45 s of
it, and a run on mains power replaced it. Peak footprint was 0.66-0.68 GB on
short-explanation and 5.35 GB on long-synthesis. At the same commit a fresh
server (`--max-context 36864`, prompt cache off, `--verify auto` by default)
took 320.85 s (319.27-322.43) to first token on the 2,940-token prompt, about
1.8× the CLI prefill; the gap was not investigated. Its decode was 2.81-2.93
tok/s and its peak footprint 1.86 GB. No valid 32K run exists for this
install: the one CLI attempt completed but was void under the swap-out rule.
The [family page](families/QWEN38_FLASH_NEXT_UNSLOTH.md) carries the rest.

After #69 and #70, the measured short-explanation arm at `1be8d3d` decodes at
4.254-4.257 tok/s and prefills in 31.05-31.09 s (segment medians), and one
long-synthesis run prefilled in 120.78 s with a 3.38 GB peak footprint
(ledger M36; `--verify full-sha256`). These are A/B arms, not a new
five-run protocol baseline.

A library-mode server from `034757d` found both installs and served one
request to each, swapping from Qwen 3.6 to the Unsloth install in 4.7 s. The
process peak footprint was 7.67 GB, one model's worth (ledger M33;
`034757d`; `--verify auto` by default). There, the old model's memory left the
footprint 2-6 s after the swap finished.

On the final `main`, `48461a1` (ledger M41; the release build of that commit,
default controls), the same smoke answered both requests with "The capital of
France is Paris." (8 tokens each). Time to first token, including the load, was
5.76 s for Qwen 3.6 and 10.62 s for the Unsloth install after a 4.8 s swap.
With #78 the old model's memory now leaves before the new load: the footprint
went 7,064 MB → 313-321 MB during the load → 2,837 MB steady. The process peak
was 7.41 GB and swap stayed at 0. The 8-token decode rates, 14.6 and
2.65 tok/s, are indicative only.

The Unsloth install remains short of every program target on this host: about
72 s to first token on a 3K prompt against 11-16 s, about 3.4 tok/s decode
against 7-10 tok/s, and no valid 32K run. The
[family page](families/QWEN38_FLASH_NEXT_UNSLOTH.md#targets-and-measured-shortfall)
lists each target with its latest measurement.

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
