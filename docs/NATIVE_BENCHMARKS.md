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

Every timing run is one command in its own hold of the machine lock, run
through [`Scripts/native/guarded-run.sh`](../Scripts/native/guarded-run.sh)
with `--timing`. The wrapper:
- refuses unless it is on AC power, so every compared number shares a power
  source;
- refuses without a passing `lock.sh preflight`, or while another model
  process runs;
- stops the command and exits 70 on any swap growth;
- records swap, wall time, peak footprint and the power source in
  `<log-dir>/run.json`.

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

Nothing else may run during a timing run: no experimental controls, no
profiling. Numbers are medians over the stated repetitions, with min and max
where they are given.

## End to end (Quick)

There are two sources:
- **`llama-bench` at the pin** gives prefill tok/s (`pp512`, `pp4096`,
  `pp32768`) and decode tok/s (`tg128`), both from an empty context and after
  4K and 32K tokens of history (`-d`). Every layer runs on Metal, with
  llama.cpp's defaults for batch, ubatch and flash attention.
- **[`mfe_bench.cpp`](../Scripts/native/bench/mfe_bench.cpp)** uses the product
  `LlamaBackend` settings (`n_batch` 2048, `n_ubatch` 512, F16 KV, flash
  attention auto) and real text tokens (the `long-32k` golden's prompt). At
  512, 4K and 32K tokens it measures:
  - **cold TTFT:** a fresh context, prefill, then the first token's logits;
  - **warm TTFT:** a 64-token follow-up appended to the cached history, then
    the first token (prefix reuse);
  - **decode tok/s:** at that depth.

The model load time is reported separately.

**Baseline: pending.** Model and timing runs are held after the program's
swap-guard trips. The numbers are recorded with `run_baselines.sh` on the
M1 Max, on AC power, once the maintainer resumes them, and land here with
their `run.json` evidence.

## Kernels (ggml-metal at Quick's shapes)

[`mfe_kbench.cpp`](../Scripts/native/bench/mfe_kbench.cpp) times ggml-metal's
own kernels at the pin, the way llama.cpp's graph calls them:
- Each measurement builds one graph of M independent copies of the op on the
  Metal backend, with concurrent dispatch off.
- The copies use distinct weight or KV buffers totalling at least 256 MiB, so
  the system cache cannot serve them.
- The time per op is the median graph time divided by M, over 20 runs, after
  3 warm-ups.

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
  and 32K context for 1 to 16 rows and a 512-row prefill chunk. The GDN core
  is `ggml_gated_delta_net` (16 q/k heads, 32 v heads, d 128) plus
  `ggml_ssm_conv` (kernel 4, 8192 channels), measured for the same row counts.
- **Deep go/no-go (input to W2).** IQ3_S, IQ3_XXS and IQ4_XS GEMV and
  verify at Deep's shapes (Qwen3.8-27B: 5120 → 12288, 10240, 6144, 1024 and
  17408; 6144 → 5120; 17408 → 5120) for 1 to 16 rows.

**Baseline: pending.** Model and timing runs are held after the program's
swap-guard trips. The numbers are recorded with `run_baselines.sh` on the
M1 Max, on AC power, once the maintainer resumes them, and land here with
their `run.json` evidence.

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
sequential and one pipelined (two steps in flight). With
`MFERENCE_NATIVE_BENCH=1` it prints both at the bound's 10 ms step (a timing
run: AC power, under the lock):

```bash
MFERENCE_NATIVE_BENCH=1 Scripts/test.sh --filter SchedulerBenchTests
```

**Reference loops: pending** (a timing run). The bound's 10 ms measurement
of the sequential and pipelined reference loops is recorded with the
baselines.
