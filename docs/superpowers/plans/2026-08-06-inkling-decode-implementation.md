# Inkling Decode Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Inkling's padded top-8 decode work with a native top-6 Metal path, overlap resident-expert phase 1 with cache-miss reads, and prove an end-to-end decode gain.

**Architecture:** Generalize the shared INT4 routed-MoE wrapper only at its top-k boundary while retaining the existing top-8 kernels for Gemma and Qwen. Inkling selects a dedicated six-SIMD-group down/reduce kernel and uses the existing cache-plan API to split phase 1 into resident and missing slot subsets.

**Tech Stack:** Swift 6.3, Metal Shading Language, Swift Testing, Mference's pread expert cache and community benchmark CLI.

## Global Constraints

- Preserve Inkling top-6 routing order, two shared-expert sinks, and FP32 residual semantics.
- Preserve the existing top-8 Gemma and Qwen paths.
- Allocate no buffers in the per-token decode loop.
- Run real-model tests only after the macOS, Swift, disk, memory-pressure, install, and process checks in `AGENTS.md` pass.
- Publish performance only from release builds following `docs/COMMUNITY_BENCHMARKS.md`, with deviations stated.

---

### Task 1: Native top-6 routed-MoE kernel

**Files:**
- Modify: `Tests/Mference/Core/Kernels/MoE/MoEFusedFFNTests.swift`
- Modify: `Sources/Mference/Kernels/MoE/MoE.swift`
- Modify: `Sources/Mference/Metal/MoE/moe.metal`

**Interfaces:**
- Consumes: `MoEExpertOffsets`, eight-entry `RoutedBlobs` argument-buffer layout, FP16 routed activations and weights.
- Produces: `MoE.init(..., specializedTopK: UInt32)` supporting 6 or 8 and `encodeRoutedPersistentPhase2Reduce(..., topK:)` selecting the matching reduction kernel.

- [ ] **Step 1: Write the failing top-6 parity test**

  Add a six-distinct-expert case that constructs `MoE(context:siluActivation:specializedD:specializedF:specializedNumExperts:specializedTopK:)`, runs both full phase 1 and a 3-hit/3-miss subset split, then compares both outputs to `MoeRef.applyStreamedRouted`.

- [ ] **Step 2: Run the focused test and verify RED**

  Run `Scripts/test.sh --filter MoEFusedFFNTests`. Expect compilation to fail because `specializedTopK` and a top-6 reduction path do not exist.

- [ ] **Step 3: Implement the native top-6 kernels**

  Add `moe_phase2_down_reduce_k6` with six SIMD groups and ordered FP32 accumulation. Store `specializedTopK` in `MoE`, use it in phase-1 function constants, load the top-6 PSO, allow only top-k 6 or 8, and bind only the requested number of expert buffers.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run `Scripts/test.sh --filter MoEFusedFFNTests` and require both the new top-6 parity case and existing top-8 case to pass.

### Task 2: Inkling cache-hit phase-1 overlap

**Files:**
- Modify: `Sources/Mference/Runtime/Inference/RealForwardRunner.swift`
- Test: `Tests/Mference/Core/Kernels/MoE/MoEFusedFFNTests.swift`
- Test: `Tests/Mference/Core/Runtime/InklingGenerationRegressionTests.swift`

**Interfaces:**
- Consumes: `Model.planRoutedExperts`, `Model.routedExpertBuffers(for:)`, `Model.fetchRoutedExperts(plan:)`, `MoE.encodeRoutedPersistentPhase1SubsetU16Load`, and runner-owned active-slot buffers.
- Produces: Inkling decode that submits hit phase 1 before awaiting misses and computes each of six expert slots exactly once.

- [ ] **Step 1: Verify the subset test catches a missing slot**

  Temporarily omit one of the six active slots in the test's split dispatch and run `Scripts/test.sh --filter MoEFusedFFNTests`; require an output mismatch, then restore the complete split.

- [ ] **Step 2: Integrate planned fetch and hit/miss dispatch**

  Initialize `MoE` with `specializedTopK: UInt32(cfg.topKExperts)`. In `produceTokenInkling`, plan the six selected experts, submit the hit subset when the plan is mixed, await only planned misses, submit the miss subset, and reduce with `topK == 6`. Retain the full-dispatch fallback for all-hit, all-miss, or unavailable planning.

- [ ] **Step 3: Run focused unit and real-model regression tests**

  Run `Scripts/test.sh --filter MoEFusedFFNTests`, then run the env-gated Inkling generation regression with `MFERENCE_INKLING_GTURBO=scratch/inklingsmall.gturbo`. Require matching expected output and no non-finite logit failure.

### Task 3: Select the production cache default by measurement

**Files:**
- Modify if justified: `Tests/Mference/Core/Runtime/Configuration/RuntimeConfigurationTests.swift`
- Modify if justified: `Sources/Mference/Runtime/Configuration/RuntimeConfiguration.swift`
- Modify if justified: `docs/RUNTIME_CONTROLS.md`

**Interfaces:**
- Consumes: `RuntimeConfiguration.defaultExpertCacheSlots(for:physicalMemoryBytes:)` and CLI `--expert-cache-slots` controls.
- Produces: an Inkling-aware automatic slot count only if measured benefit and memory pressure justify it.

- [ ] **Step 1: Run matched 16-, 24-, and 32-slot controls**

  Use the same release binary, frozen prompt, seed, sampling settings, generated-token limit, and warm-cache procedure. Capture complete footers, output hashes, peak memory, and `memory_pressure -Q` after each run.

- [ ] **Step 2: Add a failing automatic-policy test if a larger cache wins**

  Add literal expectations at memory boundaries that select the smallest winning Inkling cache size while leaving Qwen and other families unchanged. Run `Scripts/test.sh --filter RuntimeConfigurationTests` and require RED before editing production policy.

- [ ] **Step 3: Implement and document the measured policy**

  Change only `defaultExpertCacheSlots`, its tests, and the runtime-controls table. Explicit CLI and app settings remain unchanged.

### Task 4: Full verification and publication

**Files:**
- Modify: `docs/INKLING_SMALL.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: final release binary and frozen community prompts.
- Produces: reproducible end-to-end performance evidence and user-facing numbers.

- [ ] **Step 1: Run full static and package verification**

  Run `swift build -c release --product MferenceCLI`, `Scripts/test.sh`, `git diff --check origin/main`, and `LANG=en_US.UTF-8 ruby Scripts/check_markdown_links.rb`. Require zero failures.

- [ ] **Step 2: Run the final community benchmark**

  Run one discarded warmup and one fresh-process measurement for short, medium, and long cases with the final production defaults. Require `stop=endOfTurn`, coherent complete output, and matched generated tokens for every A/B claim.

- [ ] **Step 3: Update performance documentation**

  Record baseline/final footers, speedups, hardware, RAM, macOS, Swift, commit, exact generic command, exit codes, output comparison, and protocol deviations. Update landing-page numbers only with verified final results.

- [ ] **Step 4: Review, commit, push, and raise the PR**

  Inspect the complete diff, run the verification commands again after documentation edits, commit the implementation and evidence, push `codex/inkling-decode-speedup`, and open a ready pull request against `main`.

