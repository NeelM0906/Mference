import Foundation
import Metal
import Testing
@testable import Mference

/// W5's L5 gate (`docs/NATIVE_BENCHMARKS.md`, "Scheduler bound"): the
/// pipelined scheduler over H's synthetic ~10 ms model and recording queue,
/// against the bound (median GPU gap ≤ 50 µs, p99 ≤ 250 µs, median scheduler
/// CPU ≤ 50 µs per step), with the reference scheduler measured the same way
/// for comparison. A timing run: skipped unless `MFERENCE_NATIVE_BENCH=1`
/// (AC, quiet lock hold, `guarded-run.sh --timing`).
/// `MFERENCE_NATIVE_BENCH_OUT=<dir>` also writes `scheduler-bench.json`.
@Suite("SchedulerBenchGateTests", .serialized)
struct SchedulerBenchGateTests {
    static let env = ProcessInfo.processInfo.environment

    struct Report: Codable {
        var scheduler: String
        var stepMilliseconds: Double
        var decodeSteps: Int
        var gapCount: Int
        var medianGapMicroseconds: Double
        var p99GapMicroseconds: Double
        var maxGapMicroseconds: Double
        var medianBusyMicroseconds: Double
        var medianSchedulerCPUMicroseconds: Double
        var failures: [String]
    }

    /// One generation of `steps` tokens over a fresh synthetic model: the
    /// decode command buffers' gaps (setup and prefill excluded) and the
    /// scheduler's own CPU figure.
    static func measure(_ factory: any Native.StepSchedulerFactory, name: String, fast: Bool, milliseconds: Double,
                        steps: Int) async throws -> Report {
        let ctx = try MetalContext()
        let kernel = try FixedCostKernel(device: ctx.device)
        let iterations = try kernel.calibrate(queue: ctx.queue, milliseconds: milliseconds)
        let queue = RecordingCommandQueue(ctx.queue)
        let model = try SyntheticForwardModel(device: ctx.device, kernel: kernel, iterations: iterations)
        let sampler: any Native.Sampler = fast
            ? try Native.ChainSampler(context: ctx) : try Native.ReferenceSampler(context: ctx)
        let strategy: any Native.DecodingStrategy = fast ? Native.PlainDecoding() : Native.ReferencePlainDecoding()
        let scheduler = try factory.make(model: model, sampler: sampler, strategy: strategy, queue: queue)
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let outcome = try await scheduler.run(Native.GenerationJob(
            promptTokens: prompt, params: SchedulerPipelinedTests.greedy, maxNewTokens: steps + 1, stopTokens: [],
            reusePrefix: false, prefillChunk: 8)) { _ in }
        #expect(outcome.tokens == (0...steps).map { SyntheticForwardModel.choice(after: prompt.count - 1 + $0) })
        // setup, prefill, then `steps` decode buffers
        #expect(queue.waitForIntervals(2 + steps, timeout: 10), "recorder missed completions")
        let gaps = SchedulerBench.GapStats(intervals: queue.intervals, window: 2 ..< 2 + steps)
        let cpu = outcome.timings.medianSchedulerCPUMicroseconds
        let verdict = SchedulerBench.verdict(gaps: gaps, medianSchedulerCPUMicroseconds: cpu)
        return Report(scheduler: name, stepMilliseconds: milliseconds, decodeSteps: outcome.timings.decodeSteps,
                      gapCount: gaps.count, medianGapMicroseconds: gaps.medianMicroseconds,
                      p99GapMicroseconds: gaps.p99Microseconds, maxGapMicroseconds: gaps.maxMicroseconds,
                      medianBusyMicroseconds: gaps.medianBusyMicroseconds,
                      medianSchedulerCPUMicroseconds: cpu, failures: verdict.failures)
    }

    static func line(_ r: Report) -> String {
        String(format: "| %@ | %.0f | %d | %.1f | %.1f | %.1f | %.1f | %@ |", r.scheduler, r.medianBusyMicroseconds,
               r.gapCount, r.medianGapMicroseconds, r.p99GapMicroseconds, r.maxGapMicroseconds,
               r.medianSchedulerCPUMicroseconds, r.failures.isEmpty ? "pass" : r.failures.joined(separator: "; "))
    }

    /// A short, model-free smoke of the same path (2 ms steps): asserts only
    /// that the pipelined scheduler's gaps are no worse than the sequential
    /// reference's, never the bound itself.
    @Test func pipelinedGapsAreNoWorseThanSequential() async throws {
        let fast = try await Self.measure(Native.PipelinedStepSchedulerFactory(), name: "pipelined", fast: true,
                                          milliseconds: 2, steps: 16)
        let reference = try await Self.measure(Native.ReferenceStepSchedulerFactory(), name: "reference",
                                               fast: false, milliseconds: 2, steps: 16)
        #expect(fast.decodeSteps == 16 && fast.gapCount == 15)
        #expect(fast.medianGapMicroseconds <= reference.medianGapMicroseconds)
    }

    @Test(.enabled(if: env["MFERENCE_NATIVE_BENCH"] == "1"))
    func gateAtTenMilliseconds() async throws {
        let steps = 128
        let fast = try await Self.measure(Native.PipelinedStepSchedulerFactory(), name: "pipelined", fast: true,
                                          milliseconds: 10, steps: steps)
        let reference = try await Self.measure(Native.ReferenceStepSchedulerFactory(), name: "reference",
                                               fast: false, milliseconds: 10, steps: steps)
        print("scheduler-bench | scheduler | busy µs | gaps | median gap µs | p99 gap µs | max gap µs | median CPU µs | bound |")
        print("scheduler-bench " + Self.line(fast))
        print("scheduler-bench " + Self.line(reference))
        if let out = Self.env["MFERENCE_NATIVE_BENCH_OUT"] {
            let url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode([fast, reference]).write(to: url.appendingPathComponent("scheduler-bench.json"))
        }
        #expect(fast.failures.isEmpty, "\(fast.failures)")
    }
}
