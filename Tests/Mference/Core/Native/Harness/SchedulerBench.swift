import Foundation
import Metal
@testable import Mference

/// The W5 scheduler-overhead measurement (`docs/NATIVE_BENCHMARKS.md`,
/// "Scheduler bound"): a synthetic fixed-cost `ForwardModel`, a command queue
/// that records every command buffer's GPU start and end, and the gap
/// statistics checked against the bound in `docs/NATIVE_ENGINE.md` (median
/// GPU idle gap ≤ 50 µs, p99 ≤ 250 µs, median scheduler CPU per step ≤ 50 µs).
/// No model is loaded.
enum SchedulerBench {
    static let maxMedianGapMicroseconds = 50.0
    static let maxP99GapMicroseconds = 250.0
    static let maxMedianSchedulerCPUMicroseconds = 50.0

    /// Idle time between consecutive command buffers on the GPU timeline.
    struct GapStats: Equatable {
        var count: Int
        var medianMicroseconds: Double
        var p99Microseconds: Double
        var maxMicroseconds: Double
        /// Median GPU busy time per command buffer.
        var medianBusyMicroseconds: Double

        /// `intervals` in any order. `window`, if given, selects the command
        /// buffers (by start order) to measure: the decode steps alone, say,
        /// without the prefill or a trailing rollback; otherwise the first
        /// `skip` are dropped as warm-up. A gap is the idle time before a
        /// buffer starts, measured from the latest end of any earlier buffer
        /// (a running max), so overlapping buffers never count as idle.
        init(intervals: [(start: CFTimeInterval, end: CFTimeInterval)], skip: Int = 2, window: Range<Int>? = nil) {
            let all = intervals.sorted { $0.start < $1.start }
            let range = (window ?? min(skip, all.count)..<all.count).clamped(to: 0..<all.count)
            // latestEnd[i]: the latest end among buffers 0..<i (a running max)
            var latestEnd = [Double](repeating: -.infinity, count: all.count)
            for i in all.indices.dropFirst() { latestEnd[i] = max(latestEnd[i - 1], all[i - 1].end) }
            var gaps: [Double] = []
            for i in range where i > range.lowerBound {
                gaps.append(max(0, all[i].start - latestEnd[i]) * 1e6)
            }
            let busy = all[range].map { ($0.end - $0.start) * 1e6 }
            let d = ParityMetrics.Distribution(gaps)
            count = gaps.count
            medianMicroseconds = d.median
            p99Microseconds = d.p99
            maxMicroseconds = d.max
            medianBusyMicroseconds = ParityMetrics.Distribution(busy).median
        }
    }

    struct Verdict: Equatable {
        var failures: [String]
        var passed: Bool { failures.isEmpty }
    }

    /// The bound for a scheduler's decode loop.
    static func verdict(gaps: GapStats, medianSchedulerCPUMicroseconds cpu: Double) -> Verdict {
        var f: [String] = []
        if gaps.medianMicroseconds > maxMedianGapMicroseconds {
            f.append(String(format: "median GPU gap %.1f µs > %.0f", gaps.medianMicroseconds, maxMedianGapMicroseconds))
        }
        if gaps.p99Microseconds > maxP99GapMicroseconds {
            f.append(String(format: "p99 GPU gap %.1f µs > %.0f", gaps.p99Microseconds, maxP99GapMicroseconds))
        }
        if cpu > maxMedianSchedulerCPUMicroseconds {
            f.append(String(format: "median scheduler CPU %.1f µs > %.0f", cpu, maxMedianSchedulerCPUMicroseconds))
        }
        return Verdict(failures: f)
    }
}

// MARK: - A command queue that records GPU timestamps

/// Wraps a real queue; every command buffer it makes reports its GPU start
/// and end when it completes. Hand it to a `StepSchedulerFactory` in place of
/// the context's queue.
final class RecordingCommandQueue: NSObject, MTLCommandQueue, @unchecked Sendable {
    let base: MTLCommandQueue
    private let lock = NSLock()
    private var spans: [(start: CFTimeInterval, end: CFTimeInterval)] = []

    init(_ base: MTLCommandQueue) { self.base = base }

    var label: String? {
        get { base.label }
        set { base.label = newValue }
    }
    var device: MTLDevice { base.device }

    private func record(_ cb: MTLCommandBuffer?) -> MTLCommandBuffer? {
        cb?.addCompletedHandler { [weak self] done in
            guard let self, done.status == .completed else { return }
            self.lock.lock()
            self.spans.append((done.gpuStartTime, done.gpuEndTime))
            self.lock.unlock()
        }
        return cb
    }

    func makeCommandBuffer() -> MTLCommandBuffer? { record(base.makeCommandBuffer()) }
    func makeCommandBuffer(descriptor: MTLCommandBufferDescriptor) -> MTLCommandBuffer? {
        record(base.makeCommandBuffer(descriptor: descriptor))
    }
    func makeCommandBufferWithUnretainedReferences() -> MTLCommandBuffer? {
        record(base.makeCommandBufferWithUnretainedReferences())
    }
    func insertDebugCaptureBoundary() {}
    func addResidencySet(_ residencySet: MTLResidencySet) { base.addResidencySet(residencySet) }
    func removeResidencySet(_ residencySet: MTLResidencySet) { base.removeResidencySet(residencySet) }
    // the SDK's raw-pointer forms (the array overloads are extensions over them)
    func __addResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
        base.addResidencySets(Array(UnsafeBufferPointer(start: residencySets, count: count)))
    }
    func __removeResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
        base.removeResidencySets(Array(UnsafeBufferPointer(start: residencySets, count: count)))
    }

    /// Recorded (start, end) GPU times so far.
    var intervals: [(start: CFTimeInterval, end: CFTimeInterval)] {
        lock.lock(); defer { lock.unlock() }
        return spans
    }

    func reset() {
        lock.lock(); spans.removeAll(); lock.unlock()
    }

    /// Waits until at least `count` intervals are recorded (completion
    /// handlers run asynchronously, after `waitUntilCompleted` returns).
    /// False on timeout.
    @discardableResult
    func waitForIntervals(_ count: Int, timeout: TimeInterval = 5) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while intervals.count < count {
            if Date() >= deadline { return false }
            Thread.sleep(forTimeInterval: 0.001)
        }
        return true
    }
}

// MARK: - The fixed-cost kernel

/// A GPU kernel that runs for a fixed time (one SIMD group spinning on
/// dependent arithmetic), then writes logits whose argmax is
/// `(step * 7 + 3) % vocab` so the sampled sequence is known.
final class FixedCostKernel {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void mfe_fixed_cost(device float *logits [[buffer(0)]],
                               constant uint &iterations [[buffer(1)]],
                               constant uint &vocab [[buffer(2)]],
                               constant uint &choice [[buffer(3)]],
                               uint tid [[thread_position_in_grid]]) {
        float x = float(tid) * 1e-7f;
        for (uint i = 0; i < iterations; ++i) { x = fma(x, 0.999999f, 1e-7f); }
        for (uint v = tid; v < vocab; v += 32) {
            logits[v] = (v == choice ? 1.0f : 0.0f) + x * 1e-30f;
        }
    }
    """
    let pipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: Self.source, options: nil)
        guard let function = library.makeFunction(name: "mfe_fixed_cost") else {
            throw Native.NativeError.resource("mfe_fixed_cost")
        }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(_ encoder: MTLComputeCommandEncoder, logits: MTLBuffer, offset: Int, iterations: UInt32,
                vocab: UInt32, choice: UInt32) {
        var it = iterations, v = vocab, c = choice
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logits, offset: offset, index: 0)
        encoder.setBytes(&it, length: 4, index: 1)
        encoder.setBytes(&v, length: 4, index: 2)
        encoder.setBytes(&c, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Iterations that take about `milliseconds` on this GPU (measured).
    func calibrate(queue: MTLCommandQueue, milliseconds: Double) throws -> UInt32 {
        let buffer = try harnessSharedBuffer(queue.device, bytes: 64 * 4)
        var iterations: UInt32 = 1 << 16
        for _ in 0..<12 {
            let ms: Double = try autoreleasepool {
                guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                    throw Native.NativeError.resource("command buffer")
                }
                encode(enc, logits: buffer, offset: 0, iterations: iterations, vocab: 64, choice: 0)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
            }
            if ms <= 0 { iterations *= 2; continue }
            let next = Double(iterations) * milliseconds / ms
            if abs(ms - milliseconds) / milliseconds < 0.05 { break }
            iterations = UInt32(min(Double(UInt32.max / 2), max(1024, next)))
        }
        return iterations
    }
}

/// A shared buffer, or a thrown resource error.
func harnessSharedBuffer(_ device: MTLDevice, bytes: Int) throws -> MTLBuffer {
    guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw Native.NativeError.resource("buffer \(bytes)")
    }
    return b
}

// MARK: - Synthetic model and store

/// A `ForwardModel` whose every step is one fixed-cost kernel (about 10 ms by
/// default) writing logits whose argmax is `(position * 7 + 3) % vocab`. It
/// uses no weights; its `KVStore` only counts positions. For measuring a
/// scheduler's overhead, never its numerics.
final class SyntheticForwardModel: Native.ForwardModel {
    static let toyConfig = Native.Qwen35Config(
        blockCount: 4, hiddenSize: 64, ffnSize: 128, headCount: 2, kvHeadCount: 1,
        keyLength: 32, valueLength: 32, rmsNormEpsilon: 1e-6, ropeFreqBase: 1e7,
        ropeDimensionCount: 16, ropeSections: [3, 3, 2, 0], fullAttentionInterval: 4,
        ssmConvKernel: 4, ssmStateSize: 32, ssmGroupCount: 2, ssmTimeStepRank: 2,
        ssmInnerSize: 64, contextLength: 1 << 20, nextnPredictLayers: 0,
        vocabSize: 64, tiedEmbeddings: true)

    let config = SyntheticForwardModel.toyConfig
    let kv: any Native.KVStore
    let maxPrefillChunk = 512
    let maxVerifyRows = 1
    var tap: (any Native.TensorTap)?
    let kernel: FixedCostKernel
    let iterations: UInt32
    private let logits: MTLBuffer
    private var ring = 0
    /// Wall time spent inside `encodeStep`, per call, in µs.
    private(set) var encodeMicroseconds: [Double] = []

    init(device: MTLDevice, kernel: FixedCostKernel, iterations: UInt32) throws {
        self.kernel = kernel
        self.iterations = iterations
        self.kv = try SyntheticKVStore(device: device, config: Self.toyConfig)
        self.logits = try harnessSharedBuffer(device, bytes: 2 * Self.toyConfig.vocabSize * 4)
    }

    /// The token the model "predicts" after position `p`.
    static func choice(after position: Int) -> Int32 { Int32((position * 7 + 3) % toyConfig.vocabSize) }

    func encodeStep(_ plan: Native.StepPlan, encoder: MTLComputeCommandEncoder) throws -> Native.TensorRef? {
        let t0 = DispatchTime.now().uptimeNanoseconds
        defer { encodeMicroseconds.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3) }
        let slot = ring
        ring ^= 1
        let offset = slot * config.vocabSize * 4
        let last = plan.startPosition + plan.rows - 1
        kernel.encode(encoder, logits: logits, offset: offset, iterations: iterations,
                      vocab: UInt32(config.vocabSize), choice: UInt32(Self.choice(after: last)))
        guard plan.logits != .none else { return nil }
        return Native.TensorRef(buffer: logits, offset: offset, dtype: .f32, rows: 1, cols: config.vocabSize)
    }
}

/// Counts positions; holds no real KV. Rollback and reset are supported (no
/// GPU work); the prefix cache is empty.
final class SyntheticKVStore: Native.KVStore {
    let geometry: Native.KVPageGeometry
    let recurrentLayout: Native.RecurrentLayout
    private(set) var length = 0
    let capacity: Int
    private var lastStepStart = 0
    private let dummy: MTLBuffer

    init(device: MTLDevice, config: Native.Qwen35Config) throws {
        geometry = Native.KVPageGeometry(attentionLayers: config.fullAttentionLayers.count,
                                         kvHeads: config.kvHeadCount, headDim: config.keyLength)
        recurrentLayout = Native.RecurrentLayout(convKernel: config.ssmConvKernel, convChannels: config.ssmConvChannels,
                                                 valueHeads: config.ssmTimeStepRank, keyDim: config.ssmStateSize,
                                                 valueDim: config.ssmStateSize)
        capacity = config.contextLength
        dummy = try harnessSharedBuffer(device, bytes: 4096)
    }

    private var lastStepKind: Native.StepKind = .prefill

    func beginStep(rows: Int, kind: Native.StepKind) throws {
        guard length + rows <= capacity else { throw Native.NativeError.stateOutOfRange("synthetic capacity") }
        lastStepStart = length
        lastStepKind = kind
        length += rows
    }

    /// The contract's deprecated form (removed at the freeze).
    func beginStep(rows: Int) throws { try beginStep(rows: rows, kind: .decode) }

    func pageTable() -> Native.KVPageTable {
        Native.KVPageTable(geometry: geometry, arena: dummy, pageTable: dummy, pageTableOffset: 0, pageCount: 0,
                           tokenCount: length)
    }

    func recurrentState(gdnLayer: Int) -> (input: Native.RecurrentStateRef, output: Native.RecurrentStateRef) {
        let r = Native.RecurrentStateRef(layout: recurrentLayout, buffer: dummy, offset: 0)
        return (r, r)
    }

    func verifyArena(gdnLayer: Int) -> Native.RecurrentVerifyArena {
        Native.RecurrentVerifyArena(layout: recurrentLayout, buffer: dummy, offset: 0, slotStride: 0, slots: 0)
    }

    func rollback(to position: Int, commandBuffer: MTLCommandBuffer) throws {
        // the start of the most recent step, or inside it only after .verify
        guard position == lastStepStart || (lastStepKind == .verify && position > lastStepStart && position <= length) else {
            throw Native.NativeError.stateOutOfRange("synthetic rollback to \(position)")
        }
        length = position
    }

    func reset(commandBuffer: MTLCommandBuffer) throws { length = 0; lastStepStart = 0 }
    func lookupPrefix(_ tokens: [Int32]) -> Native.PrefixMatch? { nil }
    func attachPrefix(_ match: Native.PrefixMatch, commandBuffer: MTLCommandBuffer) throws {
        throw Native.NativeError.unsupported("synthetic store has no prefix cache")
    }
    func insertPrefix(_ tokens: [Int32], commandBuffer: MTLCommandBuffer) throws -> Native.PrefixHandle {
        Native.PrefixHandle(id: 0)
    }
    var footprint: Native.MemoryPlan.Footprint { .init() }
}
