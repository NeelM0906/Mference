# Qwen 3.6 Prefill and Decode Acceleration Design

Date: 2026-08-05
Status: approved for implementation

## Objective

Meaningfully reduce Qwen 3.6 35B-A3B prefill latency and increase sustained
decode throughput on Apple Silicon while preserving the runtime's
approximately 2 GB process-memory target, numerical correctness, and portable
fallbacks.

The implementation must do all of the following:

1. Make prompt-scale chunking the default for one-shot CLI generation, exposing
   the existing reduction in repeated expert I/O without requiring a flag.
2. Accelerate the remaining Gated-DeltaNet (GDN) compute with dedicated Apple
   TensorOps kernels, rather than claiming the chunk-size policy improvement as
   the kernel result.
3. Attack decode's two dominant costs together: routed-expert SSD stalls and
   the repeated fixed-shape Qwen GDN/gating dispatches. Prediction may change
   cache residency and scheduling, but never routing decisions or model math.

## Evidence and baseline

The local verified model is `scratch/qwen36.gturbo` (18 GB installed). It has 40
layers: 30 GDN linear-attention layers and 10 full-attention layers. The GDN
shape is fixed at 16 key heads, 32 value heads, and 128-dimensional key and
value heads.

Clean-base measurements use source commit
`aae03d4c59dc8167f8c1031c864e05afe9cc8b90` on an Apple M5 MacBook Pro with
24 GB RAM, macOS 26.5, and Swift 6.3.3. The saved release CLI has SHA-256
`44ef0c9bba1e9ffeb748b700f57a6a86fcc188e33ba092a53065409e36a816b9`.

The isolated-prefill command used the frozen `long-synthesis` messages, seed
`20260723`, 4,096-token context, deterministic sampling, trusted-receipt
verification, and one generated token. This is a targeted prefill A/B, not the
community full-generation protocol. Each row below is the measured run after
a discarded warmup of the same path.

| 2,940-token prompt path | Prefill |
| --- | ---: |
| Production default, 128-token chunks | 52.09 s |
| Existing `--prefill-chunk auto` path | 19.14 s |

Prompt-scale chunking is therefore 2.72x faster than the current default on
this clean-base case. The remaining 19.14 seconds are the stronger baseline
for judging the new kernels.

The exact three-case community generation protocol establishes the decode and
end-to-end baseline. Every run below exited zero with `stop=endOfTurn`.

| Case | Prefill | Generated | Decode | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Short | 62 tok / 7.60 s | 493 tok | 19.73 s | 24.992 tok/s |
| Medium | 426 tok / 12.04 s | 697 tok | 30.04 s | 23.204 tok/s |
| Long | 2,940 tok / 58.45 s | 700 tok | 34.45 s | 20.318 tok/s |

These figures are measurements, not ceilings. Final comparisons rebuild the
candidate from its revision and repeat the same warmup-plus-measured protocol.

## Success criteria

The optimized release must meet all of these requirements:

- Long isolated prefill is at least 2x faster than the 52.09-second production
  default baseline.
- Long isolated prefill is at least 20% faster than the 19.14-second existing
  prompt-scale path, requiring less than 15.31 seconds.
- Medium prefill does not regress relative to the matching prompt-scale
  baseline.
- Decode throughput improves by at least 15% on each of the short, medium, and
  long community cases, with a target of at least 20% geometric-mean gain.
- Each decode comparison has the same formatted prompt, generated-token count,
  stop reason, and output tokens as its clean-base counterpart. Performance
  gains cannot come from ending generation early or changing sampling.
- The new GDN path matches the current recurrent kernel within an explicitly
  tested numerical tolerance and produces a compatible final FP32 state.
- Prefill followed by decode matches the established runtime behavior on toy
  and real Qwen models.
- The complete package test suite and release build pass.
- The final report records revisions, hardware, operating system, Swift
  version, exact commands, exit codes, complete timing footers, and protocol
  deviations.

If the TensorOps kernel does not beat the 19.14-second prompt-scale baseline,
or if decode misses its per-case threshold, the work is not complete even if
changing the default alone produces a large headline gain.

## Design

### 1. Adaptive one-shot default

Change the CLI's default `PrefillChunkChoice` from fixed 128 to `auto`.
One-shot generation already resolves `auto` to the smallest supported chunk
that covers the formatted prompt, capped at 4,096 tokens. Interactive chat
continues using 128-token chunks so later turns do not permanently allocate
prompt-scale scratch for a growing conversation.

The help text and runtime documentation will describe the new one-shot default
and the interactive exception. Explicit `--prefill-chunk N` behavior remains
unchanged.

The server and Mac app retain their existing explicit fixed-size runtime
options in this change. Extending their user-facing configuration to represent
an adaptive choice is separate UI/API work and is not required for the kernel
speedup.

### 2. Apple TensorOps chunkwise GDN

The current `gdn_delta_step_prefill` kernel implements the exact recurrence by
assigning one SIMD group to each value row and looping over every token. This
keeps state in registers but leaves the sequence dimension serial and performs
two matrix-vector products per token.

The new path reformulates that recurrence into fixed-size internal sequence
tiles using the decay-aware WY/UT representation from Gated Delta Networks.
Within each tile, the work becomes matrix multiplication suitable for Apple
simdgroup matrix operations:

1. A preprocessing kernel computes the per-token decay and beta values and the
   tile-relative cumulative decay factors in FP32.
2. A WY-factor kernel constructs the triangular delta-rule transform for each
   key head and internal tile.
3. TensorOps kernels compute the initial-state contribution, causal
   within-tile contribution, output block, and next FP32 state using 8x8
   simdgroup matrix tiles.
4. The final tile writes the same persistent `[Hv, Dv, Dk]` FP32 state layout
   consumed by decode.

Internal tile size is selected from 32 and 64 tokens by measured performance.
The outer prompt remains one prompt-scale runtime chunk, so routed experts are
still swept once per layer. Internal GDN tiling does not cause expert rereads.

The production specialization is limited to Qwen's exact geometry:

- 16 key heads and 32 value heads;
- 128 key and value dimensions;
- FP16 q/k/v inputs and outputs;
- FP32 recurrent state; and
- Apple10 TensorOps availability.

Other shapes, unsupported devices, short inputs below the measured crossover,
and pipeline-compilation failures use the existing recurrent prefill kernel.
The chunkwise prefill path does not alter decode state layout; decode
specialization is selected independently as described below.

The implementation follows the equations in the ICLR 2025 Gated Delta
Networks paper and the operator decomposition published by Qwen's FlashQLA
project. It will be independently expressed in Metal for Apple hardware; no
CUDA, TileLang, or Python runtime dependency is introduced.

References:

- [Gated Delta Networks paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf)
- [Qwen FlashQLA design](https://qwen.ai/blog?id=flashqla)
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)

### 3. Batched Qwen shared gate

Prompt-scale Qwen prefill currently encodes the shared-expert scalar gate once
per row and then encodes a second per-row sigmoid multiplication. At a
2,940-token prompt this produces thousands of command encoders per layer.

Add a dedicated batched kernel that:

- evaluates the INT8 `[1, hidden]` scalar-gate dot product for every token;
- applies sigmoid in FP32; and
- multiplies the corresponding shared-expert output row in place.

One dispatch replaces both row loops. A CPU reference and the existing
row-wise kernels define correctness.

### 4. Scratch-memory discipline

Prompt-scale Qwen scratch must stay compatible with the documented memory
envelope. Buffers whose lifetimes do not overlap will alias storage rather
than adding a second prompt-sized allocation set. GDN tile intermediates are
bounded by the 32/64-token internal tile size and reused across layers.

The implementation must not increase expert-cache slots, duplicate the model,
retain per-layer prompt intermediates, or create prompt-length FP32 state
history. Only the final recurrent state remains persistent.

### 5. Dispatch and fallback

TensorOps pipelines are optional at runtime. Selection requires all production
shape checks, Apple10 support, a minimum measured row count, and successful
pipeline creation. Otherwise the existing GDN path runs unchanged.

Kernel failures are not silently ignored: command-buffer errors propagate
through the existing inference error path. Unsupported hardware is a normal
fallback, not an error.

### 6. Decode dependency attack

The generic decode runner already overlaps exact expert reads with resident
expert GPU work and pipelines routed compute into the following layer. Simply
adding more I/O concurrency would contend with foreground reads and repeat
experiments that the repository has already rejected. The remaining scheduling
opportunity is to begin useful next-layer reads before the exact next-layer
route becomes available.

Add a Qwen-specific current-token pilot router. While layer `L` executes, it
uses the actual INT8 router weights for layer `L+1` and the best safely
available approximation of that layer's router input—the normalized
post-attention activation already produced for layer `L`—to produce ranked
expert candidates and a confidence margin. The pilot GEMV is encoded before
the existing router shared-event signal, so its tiny result is CPU-visible at
the same wake without another command-buffer wait. This is deliberately
different from the rejected previous-token and copied-cross-layer predictors:
it predicts the current token's next route from current-token activations and
the target layer's own weights.

Only missing candidates are eligible for speculative reads. Exact demand reads
always have priority. Speculative reads begin after the current layer's demand
I/O has been issued and overlap routed GPU work and the next mixer rather than
competing with an outstanding demand batch. The exact next-layer route still
controls computation; a prediction can only make an exact expert resident
earlier.

### 7. Adaptive speculative expert engine

Speculation starts conservatively with at most two missing candidates per
layer. It reserves protected cache destinations without changing LFU hit
counters and always leaves at least `topK` slots available for the subsequent
exact route. Reservation, cancellation, short reads, and I/O errors unwind
without corrupting resident metadata or preventing the exact cache plan.

An online controller records confirmed candidates, reads that eliminated an
exact demand miss, wasted bytes, and elapsed speculative I/O. It increases the
budget only above a measured break-even hit rate, reduces it when benefit
falls, and disables speculation for the session when it produces sustained
waste or foreground delay. Predictor confidence supplies a second gate; close
router scores are not prefetched. Controller state resets with inference state
and is bounded independently per layer so one poorly predictable layer cannot
poison the whole model.

The production default enables this engine only for the pinned Qwen geometry
and a streamer/cache configuration with enough safe slack. All other models,
cache sizes, and failed pilot-pipeline creation use the exact existing path.
No environment variable or undocumented benchmark-only switch is required for
the optimized path.

### 8. Qwen decode kernels

Add dedicated Qwen decode specializations for the fixed 16-key-head,
32-value-head, 128-dimensional GDN geometry. Function constants and fixed
threadgroup mappings remove runtime division, head remapping, and shape
branches from the recurrent delta update and normalization path while
preserving the existing FP32 state representation.

Two adjacent decode dispatch boundaries are also candidates for measured
fusion:

1. fuse Q/K normalization, learned scale application, and layout preparation
   for the recurrent delta step; and
2. replace the shared-expert scalar-gate INT8 GEMV plus elementwise sigmoid
   multiply with one fixed-shape Qwen gate dispatch.

Each specialization is accepted only if an isolated kernel benchmark and the
real decode protocol improve. Ineffective variants are removed. Shape or
pipeline mismatches fall back to the existing kernels, and prefill/decode state
compatibility remains unchanged.

## Correctness strategy

Focused kernel tests will cover:

- zero and nonzero initial FP32 states;
- one internal tile, multiple tiles, and partial final tiles;
- prompt chunks that begin after an existing decoded/prefilled state;
- TensorOps output and final state against the current recurrent Metal kernel;
- batched shared-gate output against the row-wise reference; and
- Qwen decode specializations against the generic recurrent and gate kernels;
- deterministic repeated execution after state reset.

Runtime tests will cover Qwen toy prefill followed by decode, pure-decode versus
prefill/decode continuation, fallback selection, and adaptive CLI argument
resolution. They also cover prediction hit and miss paths, cache reservation
pressure, cancellation and I/O failure, controller disable/re-enable after
reset, and exact output equality with speculation enabled and disabled.
Real-model checks require finite logits, identical sampled tokens, and a
healthy short generation before performance claims are accepted.

Numerical tolerances will be derived from direct recurrent-versus-TensorOps
measurements. They must be tight enough to catch layout, gating, decay,
causality, and state-carry errors; a broad output-only tolerance is not
sufficient.

## Benchmark strategy

Development uses targeted isolated-prefill A/B runs with one generated token
to shorten the kernel loop and focused decode timings to reject weak variants.
Final evidence includes:

1. clean base revision with explicit 128-token chunks;
2. clean base revision with `--prefill-chunk auto`;
3. optimized revision using its new default; and
4. optimized revision with the TensorOps path forced only in a correctness or
   diagnostic test, never as an undocumented performance control.

The frozen medium and long prompts use their published seeds. Each final
binary gets the required discarded warmup and measured run. The complete
three-case community generation protocol is the authoritative decode and
end-to-end benchmark. Each optimized run must match its baseline token count,
stop reason, and output sequence. Isolated-prefill and focused decode numbers
remain explicitly labeled as protocol deviations. Final production benchmarks
use defaults only, with no experimental controls or profiling.

## Expected files

Implementation is expected to remain concentrated in:

- `Sources/MferenceCLI/Args.swift` and its argument tests;
- `Sources/Mference/Metal/GDN/gdn.metal` or a dedicated Qwen TensorOps Metal
  module;
- `Sources/Mference/Kernels/GDN/GDN.swift`;
- `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`;
- `Sources/Mference/Infrastructure/Streaming/PreadExpertStreamer.swift` and a focused
  Qwen pilot-router component;
- `Sources/Mference/Runtime/Prefill/PrefillChunkScratch.swift` if bounded tile
  scratch or aliasing is required;
- focused Qwen/GDN kernel, speculative-cache, and runtime tests; and
- `docs/QWEN36_PERFORMANCE.md` plus runtime-control documentation.

Unrelated model families, installer format, cache-capacity increases, and
changes to exact routing policy are outside scope.

## Delivery

Implementation proceeds in measured stages: adaptive default, batched prefill
gate, chunkwise TensorOps GDN, current-token pilot routing, adaptive speculative
I/O, and Qwen decode-kernel specialization. Each stage must preserve
correctness and is benchmarked independently. Regressing or ineffective
experimental variants are removed rather than shipped behind permanent flags.
