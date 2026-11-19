# Native engine benchmarks

This page is the benchmark protocol for the native engine
([architecture](NATIVE_ENGINE.md)), plus the llama.cpp baselines it is
measured against. The baselines are llama.cpp and ggml-metal at the pinned
commit (`b1b770d9`), running Quick (Qwen3.5-4B UD-Q4_K_XL) on the M1 Max.
Correctness is covered by [the parity bar](NATIVE_PARITY.md); this page is
about speed and memory. Results are measurements on one machine, not ceilings.

## What every result records

Every result records:
- the chip, GPU family and RAM;
- the macOS build and the toolchain (Swift and clang);
- the commit of the code measured, and the llama.cpp pin;
- the exact command and its exit code;
- the peak memory footprint (`/usr/bin/time -l`);
- swap at the start and the end;
- the power source.

A run whose swap grows is void, and it stops the program (the swap guard).

## Running

Every timing run is one command in its own quiet hold of the machine lock
(`lock.sh run --quiet`, so no build or test starts while it measures), run
through [`Scripts/native/guarded-run.sh`](../Scripts/native/guarded-run.sh)
with `--timing`. The wrapper:
- refuses unless it is on AC power, so every compared number shares a power
  source;
- waits for running build and test jobs to finish
  (`pgrep -f '/usr/bin/swift-(build|test) '`), since no new one can start
  inside the quiet hold, and refuses if they are still running after 30
  minutes;
- refuses without a passing `lock.sh preflight --gating` (no Low Power Mode,
  thermal state recorded), or while another model process or a bare `xctest`
  test bundle runs;
- stops the command and exits 70 on any swap growth;
- records swap, wall time, peak footprint and the power source in
  `<log-dir>/run.json`, with `"valid": true` only for a clean run.

The driver also waits for zero build and test jobs before it takes each hold,
and writes each step's `host.txt` (Mference commit, chip, GPU, macOS,
toolchain, power and thermal state, the tools' `BUILD_INFO`) into that step's
directory.

Build the tools from a clean checkout at the pin:

```bash
make -f Scripts/native/llama/Makefile LL=<llama.cpp> OUT=<build> -j8 all-tools
```

[`Scripts/native/bench/run_baselines.sh`](../Scripts/native/bench/run_baselines.sh)
runs every baseline below as its own guarded hold. It skips steps that
already finished, so a stopped run resumes:

```bash
LOCK=<program>/lock.sh BUILD=<build> GGUF=<Quick gguf> TOKENS=<goldens>/quick-v1/metal/long-32k/golden.json \
GGML_METAL_PATH_RESOURCES=<llama.cpp>/ggml/src/ggml-metal Scripts/native/bench/run_baselines.sh <out-dir>
```

Contexts stop at 8K tokens: every end-to-end and kernel measurement is at
512, 4K and 8K, and no evaluation or benchmark goes above 8K (maintainer
decision, 2026-11-19).

Nothing else may run during a timing run: no experimental controls, no
profiling. Numbers are medians over the stated repetitions, with min and max
where they are given.

## End to end (Quick)

There are two sources:
- **`llama-bench` at the pin** gives prefill tok/s (`pp512`, `pp4096`,
  `pp8192`) and decode tok/s (`tg128`), both from an empty context and after
  4K and 8K tokens of history (`-d`). Every layer runs on Metal, with
  llama.cpp's defaults for batch, ubatch and flash attention.
- **[`mfe_bench.cpp`](../Scripts/native/bench/mfe_bench.cpp)** uses the product
  `LlamaBackend` settings (`n_batch` 2048, `n_ubatch` 512, F16 KV, flash
  attention auto) and real text tokens (the first tokens of the `long-32k`
  golden's prompt). At 512, 4K and 8K tokens it measures:
  - **cold TTFT:** a fresh context, prefill, then the first token's logits;
  - **warm TTFT:** a 64-token follow-up appended to the cached history, then
    the first token (prefix reuse);
  - **decode tok/s:** at that depth.

The model load time is reported separately, and each context gets one untimed
warm-up before the measured repetitions.

### Baseline host

Measured on 2026-11-19 with `run_baselines.sh`, every step a quiet hold on
AC power, no thermal warning, every `run.json` valid with no swap growth:
- Apple M1 Max (24-core GPU, Apple7), 32 GB; macOS 27.2.1 (26B131);
  Apple clang 21.0.0, Swift 6.4.2.
- llama.cpp `b1b770d9`; the tools (`llama-bench`, `mfe-bench`, `mfe-kbench`)
  built by `Scripts/native/llama/Makefile` (Metal + CPU, `-O3
  -mcpu=native`) from Mference `7cad2990`.
- `run_baselines.sh` at `7cad2990` for `llama-bench-pp-tg`,
  `kbench-deep-iq` and `sched-bench`, and at `620e1fe1` (the 8K contexts)
  for the other four steps. The tools' sources differ between the two only
  in their default contexts, which these runs pass explicitly.
- Evidence: each step's `run.json`, `host.txt`, `stdout.log` and JSON
  output, kept with the program's results (`results/h/baselines/`).

### Baseline

#### llama-bench (Quick, every layer on Metal, llama.cpp defaults)

| Test | tok/s, median | min – max | reps |
|---|---:|---:|---:|
| `pp512` | 649.9 | 649.7 – 650.0 | 5 |
| `pp4096` | 561.4 | 556.4 – 585.0 | 5 |
| `pp8192` | 542.4 | 534.4 – 557.4 | 5 |
| `tg128` | 42.9 | 42.6 – 44.5 | 5 |
| `tg128 @ d4096` | 41.6 | 41.6 – 52.0 | 3 |
| `tg128 @ d8192` | 40.5 | 39.5 – 42.2 | 3 |

#### mfe-bench (product LlamaBackend settings; model load 1.51 s; medians of 3)

| Context | Cold TTFT | Prefill tok/s | Warm TTFT (+64 tokens) | Decode tok/s at depth |
|---:|---:|---:|---:|---:|
| 512 | 0.866 s (0.863 – 1.099) | 591.5 | 140 ms | 42.5 (40.3 – 44.4) |
| 4,096 | 8.891 s (8.543 – 8.906) | 460.7 | 168 ms | 36.8 (35.6 – 37.8) |
| 8,192 | 18.556 s (18.099 – 19.200) | 441.5 | 207 ms | 33.3 (33.1 – 36.4) |

#### Run records

| Step | Wall | Peak footprint | Swap delta | Power |
|---|---:|---:|---:|---|
| `llama-bench-pp-tg` | 72 s | 340 MiB | -24 MB | AC Power |
| `llama-bench-pp8k` | 91 s | 456 MiB | +0 MB | AC Power |
| `llama-bench-tg-depth` | 41 s | 964 MiB | +0 MB | AC Power |
| `mfe-bench-ttft` | 162 s | 495 MiB | +0 MB | AC Power |
| `kbench-quick` | 116 s | 3,631 MiB | +0 MB | AC Power |
| `kbench-deep-iq` | 212 s | 888 MiB | -24 MB | AC Power |
| `sched-bench` | 6 s | 28 MiB | +0 MB | AC Power |

`tg128` at depth varies more between repetitions than from an empty context
(3 repetitions; one 4K sample ran at 52 tok/s). `mfe-bench`'s decode at
depth is the second measurement of the same figure.

## Kernels (ggml-metal at Quick's shapes)

[`mfe_kbench.cpp`](../Scripts/native/bench/mfe_kbench.cpp) times ggml-metal's
own kernels at the pin, the way llama.cpp's graph calls them:
- Each measurement builds one graph of M independent copies of the op on the
  Metal backend, with concurrent dispatch off.
- The copies use distinct weight, KV or state buffers. M comes from each
  copy's own working set: enough that the graph's distinct bytes pass 256 MiB
  (`--min-mib`), so the system cache cannot serve them, and never more than
  2 GiB allocated in all (`--max-mib`). A GEMM's activation is one buffer
  shared by every copy.
- **Timing is host wall-clock** around ggml's synchronous graph compute;
  ggml-metal exposes no GPU timestamps. The time per op is the median graph
  time divided by M, over 20 runs, after 3 warm-ups. Each op therefore
  carries its share of dispatch cost, and the `dispatch` suite reports that
  floor (a one-element op, 1,024 copies).
- **W2 and W3 time their kernels the same way:** M copies in one synchronous
  submission, timed on the host clock, divided by M. Only then does a
  "≥ ggml-metal" gate compare like with like.

For each op the table reports:
- the bytes it must read;
- the bandwidth that implies;
- that bandwidth as a share of the memory-bandwidth floor, against both the
  nominal 400 GB/s and the run's measured 1 GiB copy bandwidth;
- for 2 to 16 rows, the cost per row relative to the 1-row op (a value
  below 1 means batching rows is cheaper per row).

These are the gates:
- **W2 (GEMM).** Quick's quant types at the shapes they have in Quick, for
  1, 2, 4, 8, 16 and 512 (prefill) rows: Q4_K, Q5_K, Q6_K (including the
  2560 → 248320 LM head), Q8_0, IQ4_XS, and F16 (the GDN gate projections).
- **W3 (attention, GDN).** The Quick attention shape is 16 query heads, 4 KV
  heads, head size 256, F16 KV, flash attention with a causal mask, and
  llama.cpp's head-interleaved, permuted layouts. It is measured at 512, 4K
  and 8K context for 1 to 16 rows and a 512-row prefill chunk. The GDN core
  is `ggml_gated_delta_net` (16 q/k heads, 32 v heads, d 128) plus
  `ggml_ssm_conv` (kernel 4, 8192 channels), measured for the same row counts.
- **Deep go/no-go (input to W2).** IQ3_S, IQ3_XXS and IQ4_XS GEMV and
  verify at Deep's shapes (Qwen3.8-27B: 5120 → 12288, 10240, 6144, 1024 and
  17408; 6144 → 5120; 17408 → 5120) for 1 to 16 rows.

### Baseline

Same host and build as [the end-to-end baseline](#baseline-host). Times are
per op; GB/s is the op's required bytes over that time.

The measured 1 GiB copy is ggml-metal's own `cpy` kernel, and it is slower
than the GEMVs (54 and 64 GB/s in these two runs), so it is no bandwidth
floor here: the share of the nominal 400 GB/s is the figure to compare. The
JSON output also has the share of the copy bandwidth.

#### Quick kernels

Dispatch floor 5.46 µs per op (1,024 one-element ops); measured 1 GiB copy 53.7 GB/s.

| GEMM (K×N) | 1 row | GB/s | % of 400 GB/s | per-row cost ×, 2 / 4 / 8 / 16 rows | 512 rows |
|---|---:|---:|---:|---|---:|
| q4_K 2560x8192 | 85.7 µs | 138 | 35% | 0.86 / 0.44 / 0.22 / 0.17 | 4.58 ms |
| q4_K 2560x1024 | 15.4 µs | 97 | 24% | 0.82 / 0.48 / 0.26 / 0.20 | 0.65 ms |
| q4_K 4096x2560 | 42.5 µs | 139 | 35% | 0.91 / 0.56 / 0.28 / 0.20 | 2.33 ms |
| q4_K 2560x9216 | 92.6 µs | 144 | 36% | 0.90 / 0.44 / 0.22 / 0.17 | 5.19 ms |
| q4_K 9216x2560 | 101.7 µs | 131 | 33% | 0.87 / 0.50 / 0.24 / 0.17 | 5.28 ms |
| q5_K 2560x8192 | 121.6 µs | 119 | 30% | 0.87 / 0.40 / 0.21 / 0.28 | 5.27 ms |
| q5_K 2560x1024 | 21.9 µs | 83 | 21% | 0.88 / 0.44 / 0.24 / 0.19 | 0.72 ms |
| q5_K 2560x4096 | 62.3 µs | 116 | 29% | 0.89 / 0.45 / 0.23 / 0.16 | 2.71 ms |
| q5_K 2560x9216 | 136.1 µs | 120 | 30% | 0.78 / 0.41 / 0.21 / 0.28 | 5.95 ms |
| q5_K 9216x2560 | 138.2 µs | 118 | 29% | 0.76 / 0.40 / 0.20 / 0.15 | 6.01 ms |
| q6_K 2560x1024 | 17.5 µs | 124 | 31% | 0.86 / 0.45 / 0.24 / 0.17 | 0.65 ms |
| q6_K 9216x2560 | 134.2 µs | 145 | 36% | 0.71 / 0.36 / 0.19 / 0.15 | 5.40 ms |
| q6_K 2560x248320 | 2528.5 µs | 207 | 52% | 0.90 / 0.46 / 0.23 / 0.22 | 140.00 ms |
| q8_0 4096x2560 | 62.7 µs | 178 | 45% | 0.55 / 0.28 / 0.15 / 0.13 | 2.01 ms |
| iq4_xs 2560x9216 | 83.7 µs | 150 | 38% | 0.84 / 0.45 / 0.23 / 0.21 | 4.99 ms |
| f16 2560x32 | 10.1 µs | 17 | 4% | 0.61 / 0.27 / 0.15 / 0.09 | 0.13 ms |

| Attention, context | 1 row | GB/s | % of 400 GB/s | per-row cost ×, 2 / 4 / 8 / 16 rows | 512 rows |
|---:|---:|---:|---:|---|---:|
| 512 | 63.6 µs | 33 | 8% | 0.77 / 0.53 / 0.39 / 0.36 | 3.72 ms |
| 4,096 | 206.4 µs | 81 | 20% | 0.90 / 0.70 / 0.63 / 0.53 | 45.78 ms |
| 8,192 | 259.6 µs | 129 | 32% | 0.74 / 0.67 / 0.65 / 0.59 | 21.79 ms |

| GDN op | 1 row | GB/s | % of 400 GB/s | per-row cost ×, 2 / 4 / 8 / 16 rows | 512 rows |
|---|---:|---:|---:|---|---:|
| gated_delta_net S128 HK16 HV32 | 27.0 µs | 157 | 39% | 0.56 / 0.38 / 0.25 / 0.20 | 2.17 ms |
| ssm_conv K4 C8192 | 12.5 µs | 24 | 6% | 0.56 / 0.45 / 0.15 / 0.08 | 0.31 ms |

The 512-row attention chunk at 4K context is slower than at 8K, in this run
and in the earlier one at 512 / 4K / 32K (42.6 ms at 4K). That is ggml-metal's
own time for the op, recorded as measured; W3 compares against it as it
stands.

#### Deep IQ GEMV and verify

Dispatch floor 5.39 µs per op (1,024 one-element ops); measured 1 GiB copy 63.5 GB/s.

| GEMM (K×N) | 1 row | GB/s | % of 400 GB/s | per-row cost ×, 2 / 4 / 8 / 16 rows |
|---|---:|---:|---:|---|
| iq3_s 5120x12288 | 211.0 µs | 128 | 32% | 0.91 / 0.47 / 0.26 / 0.20 |
| iq3_s 5120x10240 | 186.4 µs | 121 | 30% | 0.89 / 0.44 / 0.24 / 0.19 |
| iq3_s 5120x6144 | 113.6 µs | 119 | 30% | 0.87 / 0.44 / 0.23 / 0.17 |
| iq3_s 5120x1024 | 32.0 µs | 71 | 18% | 0.67 / 0.34 / 0.19 / 0.15 |
| iq3_s 6144x5120 | 115.9 µs | 117 | 29% | 0.88 / 0.44 / 0.23 / 0.17 |
| iq3_s 5120x17408 | 314.5 µs | 122 | 31% | 0.92 / 0.48 / 0.25 / 0.22 |
| iq3_s 17408x5120 | 324.9 µs | 118 | 30% | 0.98 / 0.49 / 0.27 / 0.18 |
| iq3_xxs 5120x12288 | 223.8 µs | 108 | 27% | 0.95 / 0.48 / 0.24 / 0.20 |
| iq3_xxs 5120x10240 | 189.9 µs | 106 | 26% | 0.98 / 0.48 / 0.23 / 0.21 |
| iq3_xxs 5120x6144 | 109.7 µs | 110 | 28% | 0.95 / 0.48 / 0.24 / 0.19 |
| iq3_xxs 5120x1024 | 35.1 µs | 58 | 14% | 0.68 / 0.34 / 0.18 / 0.15 |
| iq3_xxs 6144x5120 | 114.4 µs | 106 | 26% | 0.95 / 0.49 / 0.26 / 0.19 |
| iq3_xxs 5120x17408 | 301.8 µs | 113 | 28% | 0.93 / 0.50 / 0.26 / 0.21 |
| iq3_xxs 17408x5120 | 334.5 µs | 102 | 26% | 0.94 / 0.47 / 0.24 / 0.18 |
| iq4_xs 5120x12288 | 238.2 µs | 141 | 35% | 0.69 / 0.38 / 0.19 / 0.19 |
| iq4_xs 5120x10240 | 165.9 µs | 168 | 42% | 0.92 / 0.52 / 0.26 / 0.24 |
| iq4_xs 5120x6144 | 105.2 µs | 159 | 40% | 0.88 / 0.45 / 0.23 / 0.17 |
| iq4_xs 5120x1024 | 24.3 µs | 116 | 29% | 0.80 / 0.49 / 0.36 / 0.26 |
| iq4_xs 6144x5120 | 105.8 µs | 158 | 40% | 0.83 / 0.44 / 0.23 / 0.18 |
| iq4_xs 5120x17408 | 281.3 µs | 169 | 42% | 0.88 / 0.46 / 0.23 / 0.23 |
| iq4_xs 17408x5120 | 308.7 µs | 154 | 38% | 0.79 / 0.47 / 0.25 / 0.21 |

## Scheduler bound

`docs/NATIVE_ENGINE.md` sets W5's bound with a synthetic `ForwardModel`
whose step is a fixed GPU kernel of about 10 ms:
- median GPU idle gap between consecutive decode command buffers ≤ 50 µs;
- p99 gap ≤ 250 µs;
- median scheduler CPU time per step, excluding the model's encode, ≤ 50 µs.

The harness is
[`Tests/Mference/Core/Native/Harness/SchedulerBench.swift`](../Tests/Mference/Core/Native/Harness/SchedulerBench.swift):
- **`SyntheticForwardModel`:** each step is one fixed-cost kernel, calibrated
  to the requested duration on the running GPU. It writes logits whose argmax
  is a known function of the position. Its `SyntheticKVStore` only counts
  positions and keeps the contract's rollback rule.
- **`RecordingCommandQueue`:** hand it to a `StepSchedulerFactory` in place of
  the context's queue. It records every command buffer's GPU start and end.
- **`SchedulerBench.GapStats`:** turns those intervals into the median, p99
  and maximum idle gap.
- **`SchedulerBench.verdict`:** applies the bound, together with the
  scheduler's own `StepTimings.medianSchedulerCPUMicroseconds`.

`SchedulerBenchTests` checks the recorder with two reference loops, one
sequential and one pipelined (two steps in flight). The recorder waits for
every completion before it reads, and `GapStats` takes a window (the decode
steps alone) and measures each gap from the latest earlier end, so
overlapping buffers never count as idle.

With `MFERENCE_NATIVE_BENCH=1` it prints both loops at the bound's 10 ms
step. That is a timing run: the `sched-bench` step of `run_baselines.sh`
builds the test bundle first, outside the lock, then runs
`Scripts/test.sh --skip-build` inside a quiet hold through
`guarded-run.sh --timing`.

**Reference loops** (a timing run, with the baselines; 61 decode steps of
about 10 ms each):

| Loop | Step (GPU busy) | Gap median | Gap p99 | Gap max | Model encode, median |
|---|---:|---:|---:|---:|---:|
| sequential (one step in flight) | 10.24 ms | 391.1 µs | 588.9 µs | 588.9 µs | 7.8 µs |
| pipelined (two steps in flight) | 10.30 ms | 36.7 µs | 53.4 µs | 53.4 µs | 7.8 µs |

The pipelined loop is inside the bound's gap limits (50 µs median, 250 µs
p99); the sequential loop, which waits for each step before it encodes the
next, is not. The reference loops have no scheduler, so the CPU-time limit
applies to W5's scheduler alone.
