# Qwen 3.6 Residency Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full-residency expert-cache rung for Qwen 3.6 35B-A3B and drive its decode rate to meet or beat mlx-lm on the same 24 GB M5 host, without regressing the 16/32-slot low-memory rungs.

**Architecture:** A new `ExpertStreamingMode.resident` maps each `packed_experts/layer_XX.bin` once via the existing `ResidentBuffer` (mmap + `makeBuffer(bytesNoCopy:)`) and serves expert `TensorView`s by subregion offset — no slot copies, no LRU, no I/O workers on the hot path. The CLI auto profile picks `resident` when the expert pool plus core weights and headroom fit physical memory. With I/O gone, a measurement-gated kernel program attacks the remaining GPU/orchestration time.

**Tech Stack:** Swift 6.1+, Metal 3+, Swift Testing (`import Testing`), `run-benchmark.sh` community protocol runner, mlx-lm (Python, baseline only).

**Spec:** `docs/superpowers/specs/2026-08-07-qwen-residency-ladder-design.md`

## Global Constraints

- Host for all measured runs: the 24 GB M5 (`Mac17,2`), macOS 26.5, no other model process running (`run-benchmark.sh` preflight enforces this).
- Benchmark protocol: three frozen `real-generation-v1` community prompts, app sampling defaults (temperature 0.2, Top-K 64, Top-P 0.95), 4K context, one discarded warmup, fresh-process measured runs (`./run-benchmark.sh`), per `docs/COMMUNITY_BENCHMARKS.md`.
- Acceptance discipline (from `docs/OPTIMIZATION_JOURNEY.md`): exact transformations require byte-identical output vs control; float-reordering kernels pass reference-output quality gates; a candidate ships only on a repeatable end-to-end gain with alternating control/candidate runs; inconclusive results do not ship.
- No `.gturbo` format changes, no installer changes, no new model families, no server/batching work.
- `--expert-cache-slots` numeric values (8/16/24/32) keep their current behavior everywhere.
- Commit after every green task; conventional commit format (`feat:`, `perf:`, `test:`, `docs:`).

---

### Task 1: Reinstall the Qwen model (prerequisite; long-running)

The previous `qwen36.gturbo` was deleted in a disk cleanup (only `scratch/qwen36.gturbo.install.lock` remains). ~19.55 GB streamed install; 606 GiB free disk confirmed.

**Files:** none (model install only).

- [ ] **Step 1: Build release binaries**

Run: `swift build -c release`
Expected: `Compiling` output ending in `Build complete!`

- [ ] **Step 2: Start the streaming install (background; hours-scale)**

```bash
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo
```

Expected: progress output; on completion the directory contains `manifest.json` and `verified-install.json`. The installer resumes if interrupted — rerun the same command.

- [ ] **Step 3: Verify the install loads**

```bash
.build/release/MferenceCLI --model scratch/qwen36.gturbo --prompt "The capital of France is" --max-tokens 8
```

Expected: a completion mentioning Paris and a stats footer. No commit (no source change).

### Task 2: mlx-lm Qwen baseline on the same host

**Files:**
- Create: `benchmark-results/mlx-qwen36-baseline.md`

- [ ] **Step 1: Install mlx-lm in an isolated venv**

```bash
python3 -m venv scratch/bench/mlx-venv
scratch/bench/mlx-venv/bin/pip install --upgrade pip mlx-lm
scratch/bench/mlx-venv/bin/python -c "import mlx_lm; print(mlx_lm.__version__)"
```

Expected: a version string prints.

- [ ] **Step 2: Run the three frozen community prompts through mlx-lm**

Use the same checkpoint family Mference repacks (`Qwen/Qwen3.6-35B-A3B` 4-bit MLX conversion; confirm the exact repo id in `Sources/MferenceRepack` pinned sources and use its mlx-community equivalent). For each of the three frozen prompt files used by `run-benchmark.sh` (see `docs/COMMUNITY_BENCHMARKS.md` for their text), run:

```bash
scratch/bench/mlx-venv/bin/mlx_lm.generate \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt "<frozen prompt text>" \
  --max-tokens 700 --temp 0.2 --top-k 64 --top-p 0.95
```

One discarded warmup per case, then three measured fresh-process runs; record the reported prompt tok/s, generation tok/s, and peak memory from `mlx_lm.generate`'s footer, plus `footprint` via `/usr/bin/time -l`.

- [ ] **Step 3: Write the baseline note**

`benchmark-results/mlx-qwen36-baseline.md` records: exact model repo id and revision, mlx-lm version, per-case medians (prefill tok/s, decode tok/s, memory), and the current Mference 32-slot numbers (29.293 / 27.460 / 23.470 tok/s) for the gap statement.

- [ ] **Step 4: Commit**

```bash
git add benchmark-results/mlx-qwen36-baseline.md
git commit -m "docs: record mlx-lm Qwen 3.6 baseline on M5/24GB"
```

### Task 3: Attribution profile of the current 32-slot decode step

**Files:**
- Create: `benchmark-results/qwen36-decode-attribution.md`

- [ ] **Step 1: Measure the production 32-slot decode**

```bash
./run-benchmark.sh qwen36-slot32-control scratch/qwen36.gturbo 3
```

Expected: three passing cases with medians near the recorded 29.3/27.5/23.5 tok/s.

- [ ] **Step 2: Split the token step into I/O, GPU, orchestration**

Use the same diagnostic approach as the existing phase breakdown in `docs/QWEN36_PERFORMANCE.md` (greedy 128-token decode from a 10-token prompt). Instrument with signposts or the existing diagnostics used for that table (search `Sources/MferenceCLI` and `Sources/Mference/Runtime/Generation` for the timing hooks that produced it) and report ms/token for: expert misses + cache bookkeeping, GPU command-buffer time, CPU orchestration, sampling/other.

- [ ] **Step 3: Compute the bandwidth roofline**

State the physics ceiling alongside the measurements: active bytes per decoded token (shared core + attention + 8 routed experts + head at their manifest quantizations, computed from `packed_experts/layout.json` strides and the manifest tensor sizes) divided by the host's measured memory bandwidth (`sysctl` reports the part; use a simple Metal copy microbenchmark or published M5 figures, stated as such). Report Mference and mlx-lm as percentages of that ceiling, per the roofline method in the RunInfra B200 write-up.

- [ ] **Step 4: Write the attribution note and commit**

The note states: measured gap vs mlx-lm (from Task 2), the roofline percentage for both engines, and how much of the step residency can remove (I/O + bookkeeping share) vs what kernels must earn (GPU + orchestration share).

```bash
git add benchmark-results/qwen36-decode-attribution.md
git commit -m "docs: attribute Qwen 32-slot decode time ahead of resident rung"
```

### Task 4: `ResidentExpertStreamer` (TDD)

**Files:**
- Create: `Sources/Mference/Infrastructure/Streaming/ResidentExpertStreamer.swift`
- Test: `Tests/Mference/ResidentExpertStreamerTests.swift`

**Interfaces:**
- Consumes: `StreamLayout` (`ExpertStreamer.swift`), `ResidentBuffer` (`Infrastructure/ModelIO/ResidentBuffer.swift`).
- Produces: `public final class ResidentExpertStreamer` with `init(layout:device:) throws`, `func expertBuffer(layer: Int, expert: Int) throws -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)`, and `func warmUp()`. Task 5 depends on these exact signatures.

- [ ] **Step 1: Write the failing test**

```swift
import Foundation
import Metal
import Testing
@testable import Mference

@Test("Resident streamer serves expert bytes identical to the file")
func residentStreamerServesFileBytes() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let pageSize = Int(getpagesize())
    let expertsPerLayer = 4
    let stride = UInt64(pageSize) // page-aligned stride, like production
    var fileBytes = [UInt8]()
    for expert in 0..<expertsPerLayer {
        fileBytes.append(contentsOf: [UInt8](repeating: UInt8(expert &+ 1),
                                             count: Int(stride)))
    }
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("resident-test-\(UUID().uuidString).bin")
    try Data(fileBytes).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }

    let layout = StreamLayout(path: url.path,
                              streamOffset: 0,
                              streamSize: UInt64(fileBytes.count),
                              expertsPerLayer: expertsPerLayer,
                              expertStride: stride)
    let streamer = try ResidentExpertStreamer(layout: layout, device: device)
    for expert in 0..<expertsPerLayer {
        let view = try streamer.expertBuffer(layer: 0, expert: expert)
        #expect(view.size == stride)
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        let bytes = UnsafeRawBufferPointer(start: base, count: Int(stride))
        #expect(bytes.allSatisfy { $0 == UInt8(expert + 1) })
    }
}

@Test("Resident streamer rejects out-of-range experts")
func residentStreamerRejectsOutOfRange() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let pageSize = Int(getpagesize())
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("resident-test-\(UUID().uuidString).bin")
    try Data(count: pageSize).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    let layout = StreamLayout(path: url.path,
                              streamOffset: 0,
                              streamSize: UInt64(pageSize),
                              expertsPerLayer: 1,
                              expertStride: UInt64(pageSize))
    let streamer = try ResidentExpertStreamer(layout: layout, device: device)
    #expect(throws: (any Error).self) {
        _ = try streamer.expertBuffer(layer: 0, expert: 1)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter ResidentExpertStreamerTests`
Expected: FAIL — `ResidentExpertStreamer` not found.

- [ ] **Step 3: Implement**

```swift
import Darwin
import Foundation
import Metal

/// All-resident routed-expert backend. Maps the entire layer file once and
/// serves page-aligned subregion views of one shared MTLBuffer. No slots,
/// no bookkeeping, no reads on the hot path.
public final class ResidentExpertStreamer: @unchecked Sendable {
    public let layout: StreamLayout
    private let resident: ResidentBuffer

    public init(layout: StreamLayout, device: MTLDevice) throws {
        self.layout = layout
        self.resident = try ResidentBuffer(
            fileURL: URL(fileURLWithPath: layout.path),
            fileOffset: layout.streamOffset,
            residentSize: layout.streamSize,
            device: device)
    }

    public func expertBuffer(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard expert >= 0, expert < layout.expertsPerLayer else {
            throw StreamerError.slotOutOfRange(expert)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return (resident.buffer, regionOffset, layout.expertStride)
    }

    /// Touch the mapping sequentially so first-token decode does not pay
    /// the page-in cost. Called at load time; counts as model load, not decode.
    public func warmUp() {
        let contents = resident.buffer.contents()
        let pageSize = Int(getpagesize())
        var checksum: UInt8 = 0
        var offset = 0
        while offset < Int(layout.streamSize) {
            checksum ^= contents.load(fromByteOffset: offset, as: UInt8.self)
            offset += pageSize
        }
        _ = checksum
    }
}
```

Note: `ResidentBuffer` is `internal` to the module and currently applies `POSIX_MADV_RANDOM`; that is acceptable for the first cut (warmUp touches pages explicitly). If profiling in Task 8 shows page-in stalls, revisit with `POSIX_MADV_WILLNEED` as a measured candidate — do not pre-optimize here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter ResidentExpertStreamerTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/Mference/Infrastructure/Streaming/ResidentExpertStreamer.swift Tests/Mference/ResidentExpertStreamerTests.swift
git commit -m "feat: add all-resident routed-expert streamer"
```

### Task 5: Wire `.resident` mode through `Model` and `ModelExpertIO` (TDD)

**Files:**
- Modify: `Sources/Mference/Runtime/Inference/Model.swift` (enum at ~line 20, `ensureLayerOpened` switch at ~line 553)
- Modify: `Sources/Mference/Runtime/Inference/ModelExpertIO.swift`
- Test: `Tests/Mference/ResidentModeModelTests.swift`

**Interfaces:**
- Consumes: `ResidentExpertStreamer` from Task 4.
- Produces: `ExpertStreamingMode.resident` enum case; existing `Model` expert-IO methods (`planRoutedExperts`, `fetchRoutedExperts`, `adviseRoutedExperts`, `routedExpertBuffers`) work unchanged for callers in resident mode. Task 6 depends on `.resident` existing.

Design: `Model` stores per-layer backends as an enum so existing `PreadExpertStreamer`-typed paths stay untouched in slot mode:

```swift
enum ExpertBackend {
    case pread(PreadExpertStreamer)
    case resident(ResidentExpertStreamer)
}
```

In resident mode:
- `planRoutedExperts` / `planRoutedExpertsIfPossible` return a plan with `hits == experts.count`, empty `misses`, empty `assignedSlots`.
- `fetchRoutedExperts` / `routedExpertBuffers` build `TensorView`s directly from `expertBuffer(layer:expert:)` — synchronously, no dispatch to a background queue.
- `adviseRoutedExperts` returns `ExpertIOAdviceResult.skipped(requested:)` (nothing to advise).
- `routedExpertCacheSlotCount` returns `nil` (callers already handle `nil` as "no slot cache"); the speculative-prefetch path keys off it and must be verified inert in resident mode.
- `Model.load` calls `warmUp()` on each backend at open when mode is `.resident` and records the time in `ModelLoadStats`.

- [ ] **Step 1: Write the failing test**

```swift
import Metal
import Testing
@testable import Mference

// Uses the same synthetic-layer fixture approach as ResidentExpertStreamerTests:
// build a minimal fake packed_experts directory (1 layer, 4 experts,
// page-aligned stride) via the test helpers in Tests/Mference (see existing
// streamer tests for the fixture pattern), then:

@Test("Resident mode plans are all-hit and fetch without I/O workers")
func residentModePlansAllHit() throws {
    let model = try makeSyntheticModel(streamingMode: .resident) // fixture helper
    let plan = try #require(try model.planRoutedExperts(layer: 0, experts: [2, 0, 3]))
    #expect(plan.hits == 3)
    #expect(plan.misses.isEmpty)
    let views = try model.routedExpertBuffers(for: plan)
    #expect(views.count == 3)
}

@Test("Resident and pread modes serve identical expert bytes")
func residentMatchesPreadBytes() throws {
    let resident = try makeSyntheticModel(streamingMode: .resident)
    let pread = try makeSyntheticModel(streamingMode: .pread(slotCount: 8))
    // fetch the same experts through both models and compare bytes
    // (loop over layer 0 experts, compare TensorView contents)
}
```

Adapt `makeSyntheticModel` to whatever fixture the existing `Tests/Mference` streamer/model tests use; if none exists at model granularity, test at the `ExpertBackend` seam instead and add one end-to-end CLI parity check in Task 7.

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter ResidentModeModelTests`
Expected: FAIL — `.resident` case does not exist.

- [ ] **Step 3: Implement the enum case, backend storage, and IO fast paths**

Mechanical changes: add `case resident` to `ExpertStreamingMode`; change `streamersBox.streamers` element type to `ExpertBackend?`; in `ensureLayerOpened`, construct the backend per mode; in each `ModelExpertIO` method, switch on the backend (pread arm keeps today's code verbatim; resident arm implements the fast-path semantics above). `routedExpertStreamer(layer:)` (speculative prefetch hook) keeps its `PreadExpertStreamer` return type and `throw`s in resident mode; its only caller must guard via `routedExpertCacheSlotCount != nil` — verify and adjust that call site.

- [ ] **Step 4: Run the full test suite**

Run: `swift test`
Expected: PASS — new tests green, zero regressions in existing streamer/model tests.

- [ ] **Step 5: Commit**

```bash
git add -A Sources/Mference Tests/Mference
git commit -m "feat: route expert IO through resident backend in resident mode"
```

### Task 6: Config, CLI, and auto-rung selection (TDD)

**Files:**
- Modify: `Sources/Mference/Runtime/Configuration/RuntimeConfiguration.swift` (`allowedExpertCacheSlots` at ~line 25, `defaultExpertCacheSlots` at ~line 60)
- Modify: `Sources/MferenceCLI/Args.swift`, `Sources/MferenceCLI/Run.swift`, `Sources/MferenceServer/Core/ServerInference.swift`
- Test: `Tests/Mference/ResidencyAutoProfileTests.swift`

**Interfaces:**
- Consumes: `ExpertStreamingMode.resident` from Task 5.
- Produces: `RuntimeConfiguration.defaultExpertStreamingMode(for:physicalMemoryBytes:expertPoolBytes:coreWeightsBytes:) -> ExpertStreamingMode`; CLI `--expert-cache-slots` additionally accepts the literal `resident`.

Selection rule (explicit, testable): choose `.resident` iff
`physicalMemoryBytes >= expertPoolBytes + coreWeightsBytes + 4 GiB headroom`;
otherwise fall back to today's `defaultExpertCacheSlots` rule. On the 24 GB M5 with Qwen (pool ≈ 18.1 GB decimal, core ≈ 1.4 GB) this selects resident; on 16 GiB hosts it selects 32 slots as today. The mmap-backed pool is clean file-backed memory, so an aggressive-but-fitting rung degrades toward page-cache streaming under pressure rather than failing.

- [ ] **Step 1: Write the failing test**

```swift
import Testing
@testable import Mference

@Test("Auto profile picks resident only when pool, core, and headroom fit")
func autoProfilePicksResident() {
    let gib = UInt64(1) << 30
    let qwenPool = UInt64(18_100_000_000)
    let qwenCore = UInt64(1_450_000_000)
    let m5 = 24 * gib
    let smallHost = 16 * gib

    let top = RuntimeConfiguration.defaultExpertStreamingMode(
        for: .qwen36, physicalMemoryBytes: m5,
        expertPoolBytes: qwenPool, coreWeightsBytes: qwenCore)
    guard case .resident = top else { Issue.record("expected resident"); return }

    let mid = RuntimeConfiguration.defaultExpertStreamingMode(
        for: .qwen36, physicalMemoryBytes: smallHost,
        expertPoolBytes: qwenPool, coreWeightsBytes: qwenCore)
    guard case .pread(let slots) = mid, slots == 32 else {
        Issue.record("expected 32-slot fallback"); return
    }
}
```

- [ ] **Step 2: Run test to verify it fails** — `swift test --filter ResidencyAutoProfileTests`, expected FAIL.

- [ ] **Step 3: Implement** the selection function; extend CLI/server arg parsing so `--expert-cache-slots resident` maps to `.resident`, numeric values map to `.pread`, and `auto` calls the new function with pool/core sizes computed from the manifest + `packed_experts/layout.json` before `Model.load`. The Mac app keeps its explicit slot setting (no app UI change in Phase 1).

- [ ] **Step 4: Run the full suite** — `swift test`, expected PASS.

- [ ] **Step 5: Commit**

```bash
git add -A Sources Tests
git commit -m "feat: auto-select resident expert rung by host memory"
```

### Task 7: Resident-rung end-to-end acceptance

**Files:**
- Create: `benchmark-results/qwen36-resident-ab.md`

- [ ] **Step 1: Byte-identical gate.** Run one community case with `--expert-cache-slots 32` and again with `--expert-cache-slots resident` (same seed). Diff the generated token files: must be byte-identical (resident is an exact transformation).

- [ ] **Step 2: Full A/B.**

```bash
./run-benchmark.sh qwen36-slot32-control scratch/qwen36.gturbo 3 --expert-cache-slots 32
./run-benchmark.sh qwen36-resident scratch/qwen36.gturbo 3 --expert-cache-slots resident
```

Alternate the two labels run-for-run (control, candidate, control, …) to neutralize cache/thermal drift. Record medians, footprint, and model-load (incl. warmup) time.

- [ ] **Step 3: Accept or fall back.** Accept if resident shows a repeatable decode gain on all three cases and footprint stays within the host (no memory-pressure kills). If page-in thrash makes it lose, implement the near-resident fallback documented in the spec (bounded hot set: extend `allowedExpertCacheSlots` with 96 and 128 and re-run this A/B at 128 slots) before proceeding.

- [ ] **Step 4: Commit the A/B note.**

```bash
git add benchmark-results/qwen36-resident-ab.md
git commit -m "docs: record resident-rung A/B on M5/24GB"
```

### Task 8: Post-residency decode profile and kernel-queue re-rank

**Files:**
- Create: `benchmark-results/qwen36-resident-profile.md`

- [ ] **Step 1:** Repeat the Task 3 attribution on the resident rung (greedy 128-token decode). Expected shape: expert I/O ≈ 0; the step is GPU compute + CPU orchestration.
- [ ] **Step 2:** Rank the kernel candidates (Tasks 9–12) by measured share; record the ranking and the mlx-lm gap remaining. Re-rank after every accepted candidate.
- [ ] **Step 3:** Commit the note: `git add benchmark-results/qwen36-resident-profile.md && git commit -m "docs: profile resident decode step and rank kernel candidates"`

### Tasks 9–12: Kernel supremacy iterations (measurement-gated)

Each candidate below follows the same six-step protocol; a candidate that fails its gate is reverted and recorded in `docs/OPTIMIZATION_JOURNEY.md`. Implementation detail is deliberately deferred to the profile evidence from Task 8 — the contract, files, and gates are fixed here.

**Iteration protocol (applies to each of Tasks 9–12):**
1. Confirm from the latest profile that this candidate targets the largest (or next largest) measured share.
2. Implement behind a runtime flag (`--experiment <name>`, following the existing experiment-flag pattern in `Sources/MferenceCLI/Args.swift`).
3. Parity gate: byte-identical output for exact transforms; reference-output quality gate for float-reordering kernels (same harness as prior Qwen kernel work, see `docs/QWEN36_PERFORMANCE.md`).
4. A/B: alternating `./run-benchmark.sh` control/candidate runs, 3 reps each.
5. Accept (make default, remove flag) only on a repeatable all-case gain; otherwise revert.
6. Commit either `perf: <candidate>` with the A/B numbers, or the journey-doc failure record.

- [ ] **Task 9: Batched multi-expert GEMV.** One dispatch computes all top-k routed experts for a token from the resident pool (expert weight base offsets passed as a per-expert argument table), replacing per-expert kernel launches. Files: new kernel in `Sources/Mference/Kernels/MoE/`, host code in `Sources/Mference/Metal/MoE/`, wired via the Qwen decode path in `Sources/Mference/Runtime/Inference/`. Float-reordering: quality gate.

- [ ] **Task 10: GPU router top-k.** Keep routing logits, top-k selection, and gate weights on-GPU, eliminating the per-layer CPU round-trip that exists to build the fetch plan — in resident mode no fetch plan is needed, which is what makes this candidate newly viable (carried over from the ds4/colibri port plan). Files: `Sources/Mference/Kernels/MoE/`, `Sources/Mference/Metal/MoE/`, decode loop in `Sources/Mference/Runtime/Generation/`. Exact transform for argmax ties → byte-identical gate.

- [ ] **Task 11: Command-buffer consolidation.** Encode the full token step (all 40 layers) into one command buffer where hazards allow, cutting per-layer encode/commit overhead. Caution: a prior argument-buffer reuse experiment lost 9% on prefill — decode-only first, A/B gated. Exact transform → byte-identical gate.

- [ ] **Task 12: M5 Neural Accelerator (MPP tensor_ops) decode paths.** Where group-64 affine INT4 shapes allow, route the hot GEMVs through `mpp::tensor_ops::matmul2d` (the staged affine MPP prefill path is precedent — see `docs/IMPLEMENTATION_REFERENCES.md` Apple platform contracts). If MPP cannot consume the affine format without a dequant stage that eats the gain, record the failure and stop this line. Float-reordering: quality gate.

- [ ] **Task 12b: N-gram speculative decoding with batched MoE verification.** Draft tokens from an n-gram trie over the already-generated text (no draft model); verify k drafted tokens in one batched forward pass, reusing the chunked-prefill execution shape. The batched pass reads the shared core once per k tokens and the union of the k tokens' routed experts — this amortizes expert I/O on slot rungs exactly like prefill chunking, and cuts bytes/token at the resident rung. Exactness: greedy verification is exact by construction; the production sampled path (temp 0.2/Top-K/Top-P) must use standard speculative rejection sampling to keep the output distribution exact, and the A/B compares token ids with an explicit length check (never `zip`-style truncation). Tune trie depth / draft breadth as a measured knob — the B200 write-up showed a sharp peak, so sweep it. Files: new `Sources/Mference/Runtime/Generation/NgramSpeculator.swift`, decode-loop integration in `Sources/Mference/Runtime/Generation/`, batched verify via the existing prefill chunk path. Gates per the iteration protocol; expected to help every rung, so also A/B at 16 slots before accepting as a default.

- [ ] **Exit check:** after each accepted candidate, re-run Task 8's profile and the mlx-lm comparison. The program ends when Mference ≥ mlx-lm on the three-case median, or when the remaining candidates are exhausted (then re-plan with the new profile).

### Task 14 (streaming workstream): make the SSD path latency-hiding, not latency-bound

Owner-added scope (2026-08-07): meaningfully improve expert streaming itself.
Evidence base: the 8 GB M2 spent 83 ms/token (51%) on expert reads; DSV4
(91 GB pool) and Inkling (145 GB pool) are streaming-bound on every host; and
the achieved read rate during streaming-bound decode (~360 MB/s) is far below
the SSD's 5–7 GB/s — the path is **latency-bound and serialized**, not
bandwidth-bound. Testbed: DeepSeek-V4-Flash on the M5/24 GB (pool ≫ RAM, so
the page cache cannot hide it, unlike Qwen).

Candidates, each run under the same iteration protocol as Tasks 9–12
(profile → flag → parity → alternating A/B → accept/revert):

- [ ] **14a: I/O attribution on DSV4.** Split a decode step into read
  latency, queue depth achieved, hit rate, and stall time; measure achieved
  MB/s vs the SSD ceiling. This ranks 14b–14f.
- [ ] **14b: Queue-depth saturation.** Issue all of a layer's expert-miss
  `pread`s concurrently (and misses across the in-flight layer window), so
  the NVMe queue sees 8–16 outstanding requests instead of ~1. Latency per
  expert is hidden behind its siblings.
- [ ] **14c: Cross-layer prefetch overlap.** While the GPU computes layer L,
  fetch layer L+1's predicted experts (extend the existing
  `SpeculativeExpertPrefetch`; measure its recall first). Router-driven
  perfection is impossible (L+1 routing needs L's output), so this trades
  recall for overlap — accept only on end-to-end gain.
- [ ] **14d: n-gram speculation as an I/O amplifier** (shared with Task 12b):
  k-token batched verification reads each layer's expert union once per k
  tokens — on streaming-bound models this divides read count by up to k.
  Prioritize its A/B on DSV4/Inkling, not just Qwen.
- [ ] **14e: Read coalescing.** When a fetch plan's missed experts are
  adjacent on disk, merge them into one larger `pread`. (The earlier
  co-activation *layout* experiment failed at long context — coalescing
  changes read issue, not file layout, so it dodges that failure mode.)
- [ ] **14f: DSV4 chunked prefill** (carried from the standing perf plan):
  DSV4 currently prefills through the decode path, re-reading experts per
  token; routing it through the chunked-prefill machinery is the single
  largest known streaming win for that family.

### Task 13: Documentation and the public claim

**Files:**
- Modify: `docs/BENCHMARKS.md`, `docs/RUNTIME_CONTROLS.md`, `docs/QWEN36_PERFORMANCE.md`, `README.md`, `docs/OPTIMIZATION_JOURNEY.md`

- [ ] **Step 1:** `RUNTIME_CONTROLS.md`: document `resident` as an accepted `--expert-cache-slots` value and the auto rule (with the exact memory inequality).
- [ ] **Step 2:** `BENCHMARKS.md` + `QWEN36_PERFORMANCE.md`: add the resident-rung table, the mlx-lm same-host comparison (both sides' exact commands), and keep the 16/32-slot rows as the low-memory rungs.
- [ ] **Step 3:** `README.md`: update the headline line only if the measured claim supports it; both halves of the claim must be reproducible from documented commands.
- [ ] **Step 4:** `OPTIMIZATION_JOURNEY.md`: record accepted and failed candidates.
- [ ] **Step 5:** Commit: `git add docs README.md && git commit -m "docs: publish Qwen residency-ladder results"`
