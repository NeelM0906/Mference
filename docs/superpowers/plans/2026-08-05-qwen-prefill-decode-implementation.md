# Qwen 3.6 Prefill and Decode Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose prompt-scale prefill by default, reduce the remaining Qwen GDN prefill compute by at least 20%, and improve decode throughput by at least 15% on every community benchmark case without changing generated tokens.

**Architecture:** The prefill track combines an adaptive CLI policy, a batched scalar-gate kernel, and a Qwen-only decay-aware chunkwise GDN kernel with the existing recurrent implementation as fallback. The decode track adds a Qwen pilot router that reuses the exact router kernels in private buffers, an adaptive cache-safe speculative reader, and fixed-shape GDN/shared-gate kernels. Both tracks meet at deterministic runtime tests and the frozen three-case benchmark.

**Tech Stack:** Swift 6.3, Metal 3.2/Apple10 TensorOps, Swift Testing, POSIX `pread`, existing `.gturbo` Qwen 3.6 35B-A3B model.

## Global Constraints

- Build and test on macOS 15+ with Swift 6.1+.
- Keep the 8 GB path at 16 expert-cache slots and do not retain prompt-length FP32 state history. A larger-host Qwen default may use 32 only if an exact-output production A/B proves the decode gain.
- Preserve FP32 recurrent-state layout `[Hv, Dv, Dk]` and all existing portable fallbacks.
- Exact routing remains authoritative; speculation changes residency only.
- Final performance runs use production defaults, one model process, no profiling, and the exact community protocol.
- Before each model run check disk, `memory_pressure -Q`, the complete model, and the required `pgrep` process list.
- Use the existing `scratch/qwen36.gturbo` installation in local commands; never duplicate or reinstall it.

---

## Track A: Prefill

### Task 1: Make prompt-scale chunking the one-shot default

**Files:**
- Modify: `Tests/Mference/Core/CLI/CLIArgumentsTests.swift`
- Modify: `Sources/MferenceCLI/Args.swift`
- Modify: `Sources/MferenceCLI/Run.swift`
- Modify: `docs/RUNTIME_CONTROLS.md`

**Interfaces:**
- Consumes: `PrefillChunkChoice.auto` and `RuntimeConfiguration.allowedPrefillChunkTokens`.
- Produces: `Args.prefillChunk == .auto` for one-shot defaults; interactive chat resolves the implicit default to 128 tokens.

- [x] **Step 1: Change the default assertion before production code**

```swift
@Test func prefillChunkDefaultsToAuto() throws {
    let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
    #expect(arguments.prefillChunk == .auto)
}
```

- [x] **Step 2: Run the focused test and confirm it fails**

Run: `Scripts/test.sh --filter CLIArgumentsTests.prefillChunkDefaultsToAuto`
Expected: FAIL because the parsed choice is `.fixed(128)`.

- [x] **Step 3: Change both initializer and parser defaults to `.auto`**

```swift
prefillChunk: PrefillChunkChoice = .auto
// parse local:
var prefillChunk = PrefillChunkChoice.auto
```

Update usage text to say `default auto` and state that interactive chat uses 128 unless explicitly overridden.

- [x] **Step 4: Pin one-shot and chat resolution**

Keep one-shot resolution as the smallest allowed size covering the formatted prompt, capped at 4,096. In `Run.swift`, resolve an implicit `.auto` to 128 inside the interactive loop while preserving explicit fixed values.

- [x] **Step 5: Run CLI tests**

Run: `Scripts/test.sh --filter CLIArgumentsTests`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add Sources/MferenceCLI/Args.swift Sources/MferenceCLI/Run.swift Tests/Mference/Core/CLI/CLIArgumentsTests.swift docs/RUNTIME_CONTROLS.md
git commit -m "Default one-shot prefill to adaptive chunks"
```

### Task 2: Batch the Qwen shared scalar gate

**Files:**
- Modify: `Sources/Mference/Metal/Prefill/prefill.metal`
- Modify: `Sources/Mference/Kernels/Prefill/MoE/PrefillSharedExpert.swift`
- Modify: `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`
- Modify: `Tests/Mference/Core/Kernels/Prefill/PrefillSharedExpertTests.swift`

**Interfaces:**
- Consumes: Qwen INT8 scalar projection (`SharedExpertInt8Proj`), FP16 normalized inputs, and FP16 shared-expert outputs.
- Produces: `PrefillSharedExpert.encodeQwenScalarGate(commandBuffer:x:xOffset:gate:y:yOffset:queryCount:d:xStrideElements:yStrideElements:) throws`.

- [x] **Step 1: Add a row-stride parity test**

Create four padded rows, compute the reference with `DequantInt8GEMV.encode` plus `Elementwise.encodeSigmoidScalarMul`, invoke the new block API once, and assert exact FP16 equality plus untouched padding.

- [x] **Step 2: Run the focused test and confirm the API is missing**

Run: `Scripts/test.sh --filter PrefillSharedExpertTests.qwenScalarGateBlockMatchesRepeatedRows`
Expected: compilation FAIL because `encodeQwenScalarGate` is absent.

- [x] **Step 3: Add `prefill_qwen_scalar_gate`**

Use one 256-thread threadgroup per token. Each thread accumulates its strided INT8 dot-product elements with the tensor's per-32-element BF16 scale and bias, reduce the scalar in FP32 through `simd_sum` plus an eight-entry threadgroup array, compute sigmoid once, then multiply that row of `y` in place. Bind explicit row counts and input/output strides so padding tests exercise the contract.

- [x] **Step 4: Add the Swift encoder and replace row loops**

Build the pipeline in `PrefillSharedExpert.init`, validate `gate.rows == 1` and `gate.cols == d`, bind the projection's weight/scale/bias offsets, and issue exactly one dispatch. Replace the Qwen prefill loop that currently emits one scalar GEMV and one sigmoid multiply per row.

- [x] **Step 5: Run gate and Qwen runner tests**

Run: `Scripts/test.sh --filter PrefillSharedExpertTests`
Expected: PASS.

Run: `Scripts/test.sh --filter QwenRunnerTests`
Expected: PASS with prefill/decode equivalence intact.

- [x] **Step 6: Commit**

```bash
git add Sources/Mference/Metal/Prefill/prefill.metal Sources/Mference/Kernels/Prefill/MoE/PrefillSharedExpert.swift Sources/Mference/Runtime/Inference/RealForwardRunner.swift Tests/Mference/Core/Kernels/Prefill/PrefillSharedExpertTests.swift
git commit -m "Batch Qwen prefill scalar gating"
```

### Task 3: Add the Qwen chunkwise GDN TensorOps path

> **Execution outcome (supersedes the proposed steps below):** An exact
> one-SIMD/four-value-row Qwen GDN specialization was implemented and tested,
> but rejected after its measured 17.72 s prefill regressed the accepted
> scalar-gate build's 17.28 s. The larger measured bottleneck was the INT4
> shared expert's per-token dispatch loop. Its replacement batches gate, up,
> activation, and down across the chunk with Apple TensorOps, overlaps the
> shared branch with CPU route grouping/SSD binding, and retains the scalar
> fallback. Focused parity, scratch-layout, and Qwen runner tests pass. The
> accepted measured long prefill is 15.25 s versus the 19.14 s auto baseline
> (20.3% faster) and the 52.09 s fixed-128 baseline (3.42x faster).

**Files:**
- Create: `Sources/Mference/Metal/GDN/gdn_chunkwise.metal`
- Create: `Sources/Mference/Kernels/GDN/QwenChunkwiseGDN.swift`
- Modify: `Sources/Mference/Infrastructure/Metal/MetalContext.swift`
- Modify: `Sources/Mference/Runtime/Prefill/PrefillChunkScratch.swift`
- Modify: `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`
- Modify: `Tests/Mference/Core/Kernels/GDN/GDNKernelTests.swift`
- Modify: `Tests/Mference/Core/Runtime/Prefill/PrefillChunkScratchTests.swift`

**Interfaces:**
- Consumes: normalized FP16 q/k, raw FP16 v, FP16 a/b, BF16 `A_log`/`dt_bias`, and persistent FP32 state.
- Produces: `QwenChunkwiseGDN.encode(commandBuffer:convOut:aProj:bProj:aLog:aLogOffset:dtBias:dtBiasOffset:state:y:rows:scratch:) -> Bool`; returns `false` for unsupported geometry or row counts so the recurrent path runs.

- [ ] **Step 1: Extend GDN parity cases before adding the path**

For row counts `32, 63, 64, 65, 127, 128`, generate deterministic nonzero initial state and inputs. Run the current recurrent prefill kernel as reference, run `QwenChunkwiseGDN`, and compare every output and final state with separately reported maximum absolute and relative errors.

- [ ] **Step 2: Run the new tests and confirm the type is missing**

Run: `Scripts/test.sh --filter GDNKernelTests.chunkwiseQwenMatchesRecurrent`
Expected: compilation FAIL because `QwenChunkwiseGDN` is absent.

- [ ] **Step 3: Add bounded tile scratch**

Allocate reusable buffers for 64-token tiles only: FP32 decay/beta, FP32 triangular WY coefficients per key head, and FP16 matrix tiles. Alias non-overlapping tile products. Add a scratch-size test proving allocation is independent of outer prompt length.

- [ ] **Step 4: Implement decay-aware preprocessing and WY construction**

For each tile, compute `g_t = exp(-exp(A_log) * softplus(a_t + dt_bias))` and `beta_t = sigmoid(b_t)` in FP32. Build the strictly causal triangular transform in increasing token order, including cross-token decay products, without storing a prompt-length state history. Clamp only at the same representable boundaries used by the recurrent FP16 inputs; keep recurrence coefficients FP32.

- [ ] **Step 5: Implement 8x8 simdgroup matrix stages**

Use Apple `simdgroup_float8x8` accumulation for the initial-state projection, triangular within-tile correction, output projection, and final-state update. Process 32-token internal tiles first; enable 64 only when its focused timing wins. Write final state into the unchanged `[Hv, Dv, Dk]` layout.

- [ ] **Step 6: Dispatch only the pinned Qwen geometry**

Construct the optional pipeline only for 16 key heads, 32 value heads, 128 key/value dimensions, Apple10 support, and at least the measured crossover row count. A `false` result immediately invokes `GDN.encodeDeltaStepPrefill` with the original buffers.

- [ ] **Step 7: Run correctness and package tests**

Run: `Scripts/test.sh --filter GDNKernelTests`
Expected: PASS with the documented tolerance and finite final state.

Run: `Scripts/test.sh`
Expected: PASS.

- [ ] **Step 8: Run the isolated long prefill A/B**

After the model-safety checks and one release build, run discarded warmup and measured `long-synthesis` with `--prefill-chunk auto --max-new 1 --temperature 0 --seed 20260723 --verify trusted-receipt`. Require measured prefill below 15.31 seconds; otherwise retain measurements, remove the ineffective production selection, and continue optimizing the kernel before committing.

- [ ] **Step 9: Commit the accepted kernel**

```bash
git add Sources/Mference/Metal/GDN/gdn_chunkwise.metal Sources/Mference/Kernels/GDN/QwenChunkwiseGDN.swift Sources/Mference/Infrastructure/Metal/MetalContext.swift Sources/Mference/Runtime/Prefill/PrefillChunkScratch.swift Sources/Mference/Runtime/Inference/RealForwardRunner.swift Tests/Mference/Core/Kernels/GDN/GDNKernelTests.swift Tests/Mference/Core/Runtime/Prefill/PrefillChunkScratchTests.swift
git commit -m "Add TensorOps Qwen GDN prefill"
```

---

## Track B: Decode

> **Execution outcome:** Current-token pilot routing was implemented with
> cache-safe reservations and exact-output tests, then removed after it slowed
> medium decode to 16.117 tok/s through SSD contention. Accepted dedicated work
> instead fuses the Qwen shared expert, specializes the GDN input/delta/gated
> norm and full-attention paths, and removes redundant dispatches without a
> state-layout change. A byte-identical 16-vs-32 cache A/B then proved that the
> larger Qwen working set is worthwhile on this 24 GB host: final decode is
> 29.293/27.460/23.470 tok/s, an 18.1% geometric gain over the same-output
> 16-slot controls. Auto therefore selects 32 for Qwen only on hosts with at
> least 16 GiB; smaller hosts and other families retain 16.

### Task 4: Add a private Qwen pilot router

**Files:**
- Create: `Sources/Mference/Kernels/MoE/SpeculativeRouterQwen.swift`
- Modify: `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`
- Modify: `Tests/Mference/Core/Runtime/DecodeOverlapTests.swift`

**Interfaces:**
- Consumes: next layer's INT8 router weights/scales/biases, effective-scale buffer, per-expert scale, and current layer's normalized post-attention activation.
- Produces: `SpeculativeRouterQwen.encodePrediction(...)`, `predictedIndices: MTLBuffer`, and FP16 `predictedWeights: MTLBuffer`; buffers are private to the pilot and never alias the real router.

- [ ] **Step 1: Add Qwen pilot determinism and issuance tests**

Use a Qwen synthetic fixture with more experts than cache slots. Assert pilot mode issues at least one nonresident read, speculation-off and pilot logits are byte-identical, and reset clears pending prediction state.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `Scripts/test.sh --filter DecodeOverlapTests`
Expected: the new Qwen pilot issuance assertion FAILS because only the DSV4 path encodes a pilot.

- [ ] **Step 3: Implement `SpeculativeRouterQwen` using exact router pipelines**

Compile `router_gemv_gemma4_r4` and `router_topk_select_k8_par` with Qwen's 256-expert, 2,048-hidden, top-8 function constants. Allocate private FP32 logits, UInt32 indices, and FP16 weights. Encode both kernels before the shared-event signal.

- [ ] **Step 4: Encode and capture Qwen lookahead**

In the generic Qwen layer loop, apply layer `L+1`'s router to layer `L`'s `routedX`, using `effectiveScaleBuffers[L + 1]` and the next layer's per-expert scale. At the existing router wake, copy top-8 ids and weights into `pilotPrediction`; do not add a command buffer or wait.

- [ ] **Step 5: Make pilot the Qwen production default only**

When `MFERENCE_SPEC_PREFETCH` is absent, initialize `.pilot` for `cfg.family == .qwen36` and `.off` for other families. Preserve explicit `off`, `prefetch`, `advise`, and `pilot` environment spellings as diagnostic/test seams.

- [ ] **Step 6: Run decode overlap and Qwen runner tests**

Run: `Scripts/test.sh --filter DecodeOverlapTests`
Expected: PASS.

Run: `Scripts/test.sh --filter QwenRunnerTests`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Sources/Mference/Kernels/MoE/SpeculativeRouterQwen.swift Sources/Mference/Runtime/Inference/RealForwardRunner.swift Tests/Mference/Core/Runtime/DecodeOverlapTests.swift
git commit -m "Add current-token Qwen pilot routing"
```

### Task 5: Make speculation adaptive and cache-safe

**Files:**
- Create: `Sources/Mference/Runtime/Inference/SpeculativePrefetchController.swift`
- Modify: `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`
- Modify: `Sources/Mference/Infrastructure/Streaming/PreadExpertStreamer.swift`
- Modify: `Tests/Mference/Core/Infrastructure/Streaming/PreadExpertStreamerTests+Speculative.swift`
- Modify: `Tests/Mference/Core/Runtime/DecodeOverlapTests.swift`

**Interfaces:**
- Consumes: pilot candidates/weights plus confirmed exact routes.
- Produces: `SpeculativePrefetchController.budget(layer:confidence:) -> Int`, `record(layer:issued:confirmed:bytes:)`, and `reset()`.

- [ ] **Step 1: Add pure controller tests**

Assert an initial budget of two, growth after two windows above 60% confirmed/issued, shrinkage below 30%, per-layer isolation, a zero budget after sustained waste, confidence gating, and restoration to two after reset.

- [ ] **Step 2: Add reservation failure tests**

Inject a short read, assert every reservation's in-flight bit is cleared, failed slots remain empty, and a subsequent exact top-8 plan succeeds. Assert speculative reservation always leaves at least `topK` evictable slots.

- [ ] **Step 3: Run the focused tests and confirm missing behavior**

Run: `Scripts/test.sh --filter SpeculativePrefetchController`
Expected: compilation FAIL because the controller is absent.

- [ ] **Step 4: Implement bounded adaptive accounting**

Store fixed per-layer counters for issued, confirmed, bytes, and evaluation windows. Begin at two candidates; permit budgets 0, 2, 4, and 8. Increase only after two qualifying windows, decrease immediately on a poor window, and disable a layer after three consecutive poor windows. Rank candidates by the pilot's selected weights and stop before a candidate whose weight-to-maximum ratio is below the tested confidence threshold.

- [ ] **Step 5: Score issued experts rather than resident predictions**

Extend `SpeculativeExpertPrefetch` to retain the reserved expert set. On exact-route settlement, count only reserved experts that the route used as confirmed saved misses. Feed that result to the controller, cap candidate lists to its budget, and reset the controller in both `reset()` and decode-window initialization.

- [ ] **Step 6: Preserve exact-demand priority**

Keep one outstanding speculative batch, issue it only after current demand reads and routed command-buffer commit, join before exact planning, and retain `keepEvictable: cfg.topKExperts`. Propagate exact-read errors; speculative failures publish empty slots and reduce the controller budget.

- [ ] **Step 7: Run streamer, overlap, and full tests**

Run: `Scripts/test.sh --filter PreadExpertStreamerTests`
Expected: PASS.

Run: `Scripts/test.sh --filter DecodeOverlapTests`
Expected: PASS with identical logits.

Run: `Scripts/test.sh`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Sources/Mference/Runtime/Inference/SpeculativePrefetchController.swift Sources/Mference/Runtime/Inference/RealForwardRunner.swift Sources/Mference/Infrastructure/Streaming/PreadExpertStreamer.swift Tests/Mference/Core/Infrastructure/Streaming/PreadExpertStreamerTests+Speculative.swift Tests/Mference/Core/Runtime/DecodeOverlapTests.swift
git commit -m "Adapt Qwen expert speculation online"
```

### Task 6: Specialize Qwen decode kernels

**Files:**
- Modify: `Sources/Mference/Metal/GDN/gdn.metal`
- Modify: `Sources/Mference/Kernels/GDN/GDN.swift`
- Modify: `Sources/Mference/Metal/Primitives/utility.metal`
- Modify: `Sources/Mference/Kernels/Primitives/Elementwise.swift`
- Modify: `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`
- Modify: `Tests/Mference/Core/Kernels/GDN/GDNKernelTests.swift`
- Modify: `Tests/Mference/Core/Runtime/QwenRunnerTests.swift`

**Interfaces:**
- Consumes: pinned Qwen GDN geometry and the Qwen INT8 scalar gate.
- Produces: optional fixed-shape Q/K norm and delta PSOs plus `Elementwise.encodeQwenSharedScalarGate(...)`.

- [ ] **Step 1: Add generic-versus-specialized parity tests**

Run both paths from identical nonzero state for 32 decode steps. Require exact FP16 outputs where operation order is unchanged and tight FP32 state tolerance where constant folding changes instruction selection. Add a scalar-gate test against the existing two-dispatch reference.

- [ ] **Step 2: Run tests and confirm specialized APIs are absent**

Run: `Scripts/test.sh --filter GDNKernelTests.qwenDecodeSpecializedMatchesGeneric`
Expected: compilation FAIL.

- [ ] **Step 3: Compile fixed-shape GDN PSOs**

Add Metal function constants for 16 key heads, 32 value heads, 128 key/value dimensions, and the two-to-one value/key head map. Use them to remove dynamic division and dimension branches in `gdn_qk_norm` and `gdn_delta_step_decode`; select them only when the `LinearAttentionConfig` exactly matches Qwen.

- [ ] **Step 4: Fuse the scalar gate decode dispatches**

Add a one-threadgroup kernel that computes the 2,048-element INT8 dot with per-group BF16 affine parameters, reduces in FP32, applies sigmoid once, and multiplies the 2,048-element shared output in place. Replace the current `DequantInt8GEMV` plus `encodeSigmoidScalarMul` sequence only for the exact Qwen shape.

- [ ] **Step 5: Run kernel and runtime tests**

Run: `Scripts/test.sh --filter GDNKernelTests`
Expected: PASS.

Run: `Scripts/test.sh --filter QwenRunnerTests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Sources/Mference/Metal/GDN/gdn.metal Sources/Mference/Kernels/GDN/GDN.swift Sources/Mference/Metal/Primitives/utility.metal Sources/Mference/Kernels/Primitives/Elementwise.swift Sources/Mference/Runtime/Inference/RealForwardRunner.swift Tests/Mference/Core/Kernels/GDN/GDNKernelTests.swift Tests/Mference/Core/Runtime/QwenRunnerTests.swift
git commit -m "Specialize Qwen decode kernels"
```

---

## Integration and Release

### Task 7: Validate, document, review, and publish

**Files:**
- Modify: `docs/QWEN36_PERFORMANCE.md`
- Modify: `docs/RUNTIME_CONTROLS.md`
- Modify: `docs/BENCHMARKS.md`

**Interfaces:**
- Consumes: accepted prefill and decode paths from Tasks 1–6.
- Produces: reproducible measurements and a ready Qwen optimization PR.

- [ ] **Step 1: Run static and package verification**

Run: `git diff --check`
Expected: exit 0.

Run: `Scripts/test.sh`
Expected: exit 0.

Run: `swift build -c release --product MferenceCLI`
Expected: exit 0.

- [ ] **Step 2: Record the candidate system manifest**

Capture `git status --short`, `git rev-parse HEAD`, `sw_vers`, `swift --version`, filtered hardware/RAM, model manifest SHA-256, prompt SHA-256 values, power source, Low Power Mode, disk space, memory pressure, and the process check under `benchmark-results/qwen-optimized/system/`.

- [ ] **Step 3: Re-run isolated prefill evidence**

Run fixed-128 and default-auto warmup/measured pairs with the frozen medium and long prompts and one generated token. Require long default auto below 15.31 seconds and no medium regression against its clean prompt-scale baseline. Label trusted-receipt and one-token generation as protocol deviations, and retain complete stderr footers.

- [ ] **Step 4: Run the exact three-case community protocol**

Use temperature 0.2, Top-K 64, Top-P 0.95, max context 4,096, max new 1,024, seeds `20260721`, `20260722`, and `20260723`, one discarded warmup per case, then fresh measured processes. Require `stop=endOfTurn` and the clean-base generated counts 493, 697, and 700.

- [ ] **Step 5: Enforce decode gates**

Require at least 28.741 tok/s short, 26.685 tok/s medium, and 23.366 tok/s long, with at least 20% geometric-mean target. Compare stdout and token sequences to clean base; reject any run with looping, truncation, changed token counts, or changed output.

- [ ] **Step 6: Document exact evidence**

Record candidate commit, hardware/RAM, macOS, Swift, exact commands, exit codes, every complete timing footer, warmup policy, and deviations in `docs/QWEN36_PERFORMANCE.md`; update runtime/default documentation and the benchmark table.

- [ ] **Step 7: Request mandatory code review and address findings**

Run the repository-required Swift/code review over every changed source and test file. Fix actionable correctness, safety, and maintainability findings, then rerun focused tests and `Scripts/test.sh`.

- [ ] **Step 8: Commit documentation and push**

```bash
git add docs/QWEN36_PERFORMANCE.md docs/RUNTIME_CONTROLS.md docs/BENCHMARKS.md
git commit -m "Document Qwen prefill and decode gains"
git push -u origin codex/qwen-prefill-speedup
```

- [ ] **Step 9: Open the pull request**

Create a ready PR against `main` summarizing the two independent speedups, correctness evidence, exact benchmark table, memory behavior, fallbacks, and all protocol deviations.
