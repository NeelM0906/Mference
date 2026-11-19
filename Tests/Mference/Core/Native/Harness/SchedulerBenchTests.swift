import Foundation
import Metal
import Testing
@testable import Mference

/// The scheduler-overhead harness, model-free. The GPU loops are short (a
/// ~2 ms fixed-cost step) and assert only orderings, never the bound itself;
/// `MFERENCE_NATIVE_BENCH=1` prints the full report at the 10 ms step the
/// bound is defined with.
@Suite("SchedulerBenchTests")
struct SchedulerBenchTests {
    @Test func gapStatistics() {
        // three 1 ms buffers, gaps 0.010 and 0.050 ms after the skipped head
        let s = SchedulerBench.GapStats(intervals: [(0, 0.001), (0.00101, 0.00201), (0.00206, 0.00306)], skip: 0)
        #expect(s.count == 2)
        #expect(abs(s.medianMicroseconds - 10) < 1e-6)
        #expect(abs(s.maxMicroseconds - 50) < 1e-6)
        #expect(abs(s.medianBusyMicroseconds - 1000) < 1e-6)
        #expect(SchedulerBench.GapStats(intervals: [], skip: 2).count == 0)
    }

    @Test func gapsUseARunningMaxAndAWindow() {
        // buffer 0 runs long; 1 and 2 start inside it: no idle time at all
        let overlap = SchedulerBench.GapStats(intervals: [(0, 0.010), (0.001, 0.002), (0.003, 0.004)], skip: 0)
        #expect(overlap.count == 2 && overlap.maxMicroseconds == 0)
        // a window measures only the chosen buffers (by start order), each
        // gap still taken from every earlier buffer's end
        let w = SchedulerBench.GapStats(intervals: [(0, 0.001), (0.002, 0.003), (0.0031, 0.004), (0.0045, 0.005)],
                                        window: 1..<3)
        #expect(w.count == 1)
        #expect(abs(w.medianMicroseconds - 100) < 1e-6)
    }

    @Test func verdict() {
        var s = SchedulerBench.GapStats(intervals: [], skip: 0)
        s.medianMicroseconds = 20; s.p99Microseconds = 100
        #expect(SchedulerBench.verdict(gaps: s, medianSchedulerCPUMicroseconds: 30).passed)
        s.p99Microseconds = 300
        let v = SchedulerBench.verdict(gaps: s, medianSchedulerCPUMicroseconds: 60)
        #expect(v.failures.count == 2)
    }

    /// The synthetic store keeps the contract's rollback rule: back to the
    /// last step's start always, inside it only after a verify step.
    @Test func syntheticStoreRollbackRule() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer() else { return }
        let kv = try SyntheticKVStore(device: device, config: SyntheticForwardModel.toyConfig)
        try kv.beginStep(rows: 4, kind: .prefill)
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 3, commandBuffer: cb) }
        try kv.rollback(to: 4, commandBuffer: cb)
        try kv.beginStep(rows: 3, kind: .verify)
        try kv.rollback(to: 6, commandBuffer: cb)
        #expect(kv.length == 6)
    }

    struct Rig {
        let queue: RecordingCommandQueue
        let model: SyntheticForwardModel
    }

    static func rig(milliseconds: Double) throws -> Rig? {
        guard let device = MTLCreateSystemDefaultDevice(), let base = device.makeCommandQueue() else { return nil }
        let kernel = try FixedCostKernel(device: device)
        let queue = RecordingCommandQueue(base)
        let iterations = try kernel.calibrate(queue: base, milliseconds: milliseconds)
        return Rig(queue: queue, model: try SyntheticForwardModel(device: device, kernel: kernel, iterations: iterations))
    }

    /// One decode step through the contract: beginStep, encode, commit.
    static func step(_ rig: Rig, position: Int) throws -> (MTLCommandBuffer, Native.TensorRef) {
        try rig.model.kv.beginStep(rows: 1, kind: .decode)
        let cb = try #require(rig.queue.makeCommandBuffer())
        let enc = try #require(cb.makeComputeCommandEncoder())
        let plan = Native.StepPlan(kind: .decode, startPosition: position, rows: 1, input: .host([0]), logits: .last)
        let logits = try #require(try rig.model.encodeStep(plan, encoder: enc))
        enc.endEncoding()
        cb.commit()
        return (cb, logits)
    }

    static func argmax(_ t: Native.TensorRef) -> Int32 {
        let p = t.buffer.contents().advanced(by: t.offset).assumingMemoryBound(to: Float.self)
        return Int32(ParityMetrics.argmax(UnsafeBufferPointer(start: p, count: t.cols)))
    }

    /// Sequential (wait after every step) versus pipelined (step t+1 committed
    /// before waiting on t): the recorder sees the difference, and the model's
    /// logits carry the expected choices.
    static func loops(_ rig: Rig, steps: Int) throws -> (sequential: SchedulerBench.GapStats, pipelined: SchedulerBench.GapStats) {
        // Every step's command buffer and encoder are autoreleased Objective-C
        // objects: drain them per step, or a long loop grows the heap.
        rig.queue.reset()
        for p in 0..<steps {
            try autoreleasepool {
                let (cb, logits) = try step(rig, position: p)
                cb.waitUntilCompleted()
                #expect(argmax(logits) == SyntheticForwardModel.choice(after: p))
            }
        }
        #expect(rig.queue.waitForIntervals(steps), "recorder missed completions")
        let sequential = SchedulerBench.GapStats(intervals: rig.queue.intervals)

        rig.queue.reset()
        var pending: MTLCommandBuffer?
        for p in steps..<(2 * steps) {
            try autoreleasepool {
                let (cb, _) = try step(rig, position: p)
                pending?.waitUntilCompleted()
                pending = cb
            }
        }
        pending?.waitUntilCompleted()
        #expect(rig.queue.waitForIntervals(steps), "recorder missed completions")
        let pipelined = SchedulerBench.GapStats(intervals: rig.queue.intervals)
        return (sequential, pipelined)
    }

    @Test func recorderSeesPipelining() throws {
        guard let rig = try Self.rig(milliseconds: 2) else { return }
        let (sequential, pipelined) = try Self.loops(rig, steps: 16)
        #expect(sequential.count == 13 && pipelined.count == 13)
        #expect(sequential.medianBusyMicroseconds > 500)
        #expect(pipelined.medianMicroseconds <= sequential.medianMicroseconds)
        #expect(rig.model.kv.length == 32)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_NATIVE_BENCH"] == "1"))
    func reportAtTenMilliseconds() throws {
        guard let rig = try Self.rig(milliseconds: 10) else { return }
        let (sequential, pipelined) = try Self.loops(rig, steps: 64)
        let encode = ParityMetrics.Distribution(rig.model.encodeMicroseconds).median
        for (name, s) in [("sequential", sequential), ("pipelined", pipelined)] {
            print(String(format: "scheduler-bench %@: busy %.0f µs, gap median %.1f µs, p99 %.1f µs, max %.1f µs (n=%d); model encode median %.1f µs",
                         name, s.medianBusyMicroseconds, s.medianMicroseconds, s.p99Microseconds, s.maxMicroseconds, s.count, encode))
        }
    }
}
