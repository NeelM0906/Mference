# Inkling Decode Acceleration Design

## Goal

Materially increase Inkling-Small decode throughput while preserving its
top-6 router, two shared-expert sinks, FP32 residual stream, and generated
output. The gain must hold in the public community generation protocol, not
only in an isolated kernel benchmark.

## Baseline

The untouched `main` build at `6bf428f` generated 469 tokens for the frozen
`short-explanation` case on a 24 GB M5 (`Mac17,2`). After one discarded
warmup, the measured run reported 2.909 tok/s. The command used the production
automatic cache choice (16 slots), strict verification, sampling defaults,
and no profiler or experimental control.

Inkling routes six experts per MoE layer. The current runtime pads that list to
the shared eight-expert contract, duplicates two expert buffers, and keeps the
two padding weights at zero. The duplicates avoid additional I/O but still run
gate/up/activation and down-projection work. The current Inkling loop also
waits for every expert-cache miss before launching phase 1, even when some of
the selected experts are already resident.

## Design

### Native top-6 routed MoE

`MoE` will accept a specialized decode top-k of either 6 or 8. Its existing
phase-1 kernels already consume a runtime/function-constant top-k, so Inkling
will dispatch exactly six slots. A dedicated INT4
`moe_phase2_down_reduce_k6` Metal kernel will launch six SIMD groups per output
row, evaluate only the six real down projections, reduce them in router order,
and seed the sum with Inkling's shared-expert result exactly as today.

Gemma and Qwen retain the current top-8 pipeline and pipeline states. The
argument-buffer layout remains eight entries wide, but a top-6 dispatch binds
and reads only entries 0 through 5.

### Cache-hit compute overlap

After the Inkling router signal completes, the runtime will create an expert
cache plan for the six selected experts. If the plan contains both hits and
misses, it will bind all six reserved slot buffers and immediately submit the
phase-1 subset for hit slots. The CPU then reads misses while that command
buffer and the shared-expert branch execute. Once reads finish, a second
phase-1 subset computes only the miss slots; the dedicated top-6 phase-2 kernel
then combines all six activations with the shared branch.

All-hit, all-miss, and planning-fallback cases use a single full phase-1
dispatch. Buffers are allocated during runner initialization, and no new
allocation occurs in the token loop. The existing command-buffer ordering and
final token drain remain the lifetime boundary for cache-slot contents.

### Cache capacity

The documented 16-, 24-, and 32-slot settings will be compared on the same
frozen prompt and generated workload. An Inkling-specific automatic default
will be added only if it materially improves end-to-end decode on the 24 GB
host without unacceptable memory pressure. Explicit user choices and the Mac
app's existing setting remain authoritative.

## Correctness

A Metal parity test will cover the native top-6 full and hit/miss-split paths
against the CPU routed-MoE reference. It must use six distinct expert blobs and
non-zero weights so any missing, duplicated, or reordered slot changes output.
Existing top-8 parity remains unchanged.

The env-gated Inkling real-model regression will compare token output after the
runtime integration. Final community A/B runs must have matching prompt and
generated token counts, `stop=endOfTurn`, coherent output, and no non-finite
logit error. Any cache-default claim additionally requires byte-identical
generated output at the compared slot counts.

## Measurement

Build release once per source state. Run one discarded warmup and then one
fresh-process measurement for each frozen community case, using temperature
0.2, Top-K 64, Top-P 0.95, the frozen seed, 4K context, and up to 1,024 new
tokens. Record commit, hardware/RAM, macOS, Swift, exact command, exit code,
complete timing footer, output quality, and every protocol deviation.

The optimization is successful only if the geometric-mean decode rate rises
meaningfully across the matched cases. Kernel-only timing is diagnostic and
will not be used as the headline result.

