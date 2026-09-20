import Foundation
import Metal
import Testing
@testable import Mference

@Suite(.serialized) struct InklingPrefillContractTests {
    private func run(resident: Bool) async throws -> [[UInt16]] {
        let directory = try InklingToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let cfg = InklingToySynthetic.config
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: cfg, streamingMode: resident ? .resident : .pread(slotCount: 8))
        let runtime = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8,
                prefillChunkTokens: 16, forceLogitsHead: true))
        let runner = try #require(runtime.producer as? RealForwardRunner)
        let output = try #require(context.device.makeBuffer(length: cfg.vocabSize * 2,
                                                            options: .storageModeShared))
        let tokens: [Int32] = (0..<83).map { Int32(4 + ($0 * 17) % 239) }
        func row() -> [UInt16] {
            let bits = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt16.self),
                                                 count: cfg.vocabSize))
            #expect(bits.prefix(250).allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(bits.suffix(6).allSatisfy { Float16(bitPattern: $0) == -.infinity })
            #expect(Set(bits.prefix(250)).count > 16, "nonconstant full-runner logits")
            return bits
        }
        func append(_ start: Int, _ end: Int) async throws {
            let result = try await runner.prefillChunked(tokens: tokens[start..<end],
                startPosition: start, outputMode: .logits, config: runtime.prefillConfig,
                into: output, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == end - start)
            #expect(result.execution?.replayedTokens == 0)
            #expect(result.execution?.batchedChunkSizes.reduce(0, +) == end - start)
            #expect(result.newPosition == end)
            try runner.prepareForContinuation(expectedPosition: end)
        }
        let boundaries = [3, 31, 35, 67, 83]
        var sequential: [[UInt16]] = []
        if resident {
            for (position, token) in tokens.enumerated() {
                try await runner.produce(token: token, position: position, into: output)
                if boundaries.contains(position + 1) { sequential.append(row()) }
            }
            for step in 0..<4 {
                try await runner.produce(token: Int32(7 + 11 * step), position: 83 + step, into: output)
                sequential.append(row())
            }
        }
        var expected: [[UInt16]] = []
        for pass in 0..<2 {
            runner.reset()
            var rows: [[UInt16]] = []
            var start = 0
            // Ragged warm chunks cross the local ring, relative-bias extent
            // and the deliberately small global log-scaling threshold.
            for end in boundaries {
                try await append(start, end)
                rows.append(row())
                start = end
            }
            for step in 0..<4 {
                try await runner.produce(token: Int32(7 + 11 * step), position: 83 + step, into: output)
                rows.append(row())
            }
            if pass == 0 { expected = rows }
            else { #expect(rows == expected, "reset restores all four convolution and KV states") }
        }
        for (index, pair) in zip(expected, sequential).enumerated() {
            var maxError: Float = 0
            var dot: Double = 0, aa: Double = 0, bb: Double = 0
            for (a, b) in zip(pair.0.prefix(250), pair.1.prefix(250)) {
                let actual = Float(Float16(bitPattern: a))
                let reference = Float(Float16(bitPattern: b))
                maxError = max(maxError, abs(actual - reference))
                dot += Double(actual) * Double(reference)
                aa += Double(actual) * Double(actual)
                bb += Double(reference) * Double(reference)
            }
            let cosine = dot / max((aa * bb).squareRoot(), 1e-20)
            // Different routed reductions have FP16 stores in different places.
            // Fixed pre-run bounds: absolute error plus direction, not a
            // tolerance inferred from whichever result this invocation gives.
            #expect(maxError <= 0.002)
            #expect(cosine >= 0.999)
            print("[Inkling CI prefill] row=\(index) maxAbs=\(maxError) cosine=\(cosine)")
        }
        runner.reset()
        try await append(0, 3)
        #expect(row() == expected[0])
        runner.prefillWillEncodeLayer = { if $0 == 3 { throw CancellationError() } }
        do {
            try await append(3, 19)
            Issue.record("expected cancellation after first routed layer")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { try runner.prepareForContinuation(expectedPosition: 3) }
        do {
            try await runner.produce(token: tokens[3], position: 3, into: output)
            Issue.record("dirty state must reject decode")
        } catch is PrefillError { }
        runner.prefillWillEncodeLayer = nil
        runner.reset()
        try await append(0, 3)
        #expect(row() == expected[0], "reset after interrupted warm append")
        return expected
    }

    @Test func nonzeroFixturePreservesMemoryProfileAndRecoveryContract() async throws {
        let streamed = try await run(resident: false)
        let resident = try await run(resident: true)
        #expect(streamed == resident, "all boundary and continuation logits match exactly")
    }
}
