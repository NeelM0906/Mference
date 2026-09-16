import Foundation
import Metal
import Testing
@testable import Mference

/// Opt-in correctness screen, not a throughput benchmark. Uses one installed
/// model and one runner, resetting between alternatives; never copies weights.
/// Requires MFERENCE_SWIFT_QWEN_GTURBO and explicit MFERENCE_MTP=1.
@Suite(.serialized) struct SwiftQwenInstalledQualificationTests {
    @Test func prefillAppendAndSpeculativeContinuation() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MFERENCE_SWIFT_QWEN_GTURBO"] else { return }
        let url = URL(fileURLWithPath: path)
        #expect(try ManifestReader.peekModelID(directoryURL: url) == CheckpointIdentity.swiftQwen38)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: url, device: ctx.device)
        let tok = try await MFTokenizer.load(forModelDirectory: url)
        let runner = try Qwen38ForwardRunner(model: model, context: ctx, maxContext: 2048)
        let speculator = try #require(runner.mtp, "Run qualification with MFERENCE_MTP=1")
        let vocab = model.config.vocabSize
        let logits = try #require(ctx.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        let document = (0..<300).map { "Record \($0): item \($0 * 7 + 3) was delivered to shelf \($0 % 11)." }
            .joined(separator: "\n")
        let source = tok.encode(document, addBOS: false)
        #expect(source.count > 1025)

        func readLogits() -> [Float16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float16.self), count: vocab))
        }
        func compare(_ want: [Float16], _ got: [Float16], label: String) {
            let differences = zip(want, got).map { abs(Float($0) - Float($1)) }
            let maximum = differences.max() ?? .infinity
            let mean = differences.reduce(0, +) / Float(vocab)
            let left = want.indices.max(by: { want[$0] < want[$1] })
            let right = got.indices.max(by: { got[$0] < got[$1] })
            let mismatch = zip(want, got).filter { $0.bitPattern != $1.bitPattern }.count
            print("[swift-qualification] \(label) max_abs=\(maximum) mean_abs=\(mean) bit_mismatches=\(mismatch) top1_equal=\(left == right)")
            // Predeclared FP16 screen limits, not a claim of bit-exact kernels.
            #expect(want.allSatisfy { $0.isFinite } && got.allSatisfy { $0.isFinite })
            #expect(maximum <= 0.25 && mean <= 0.01, "\(label): numerical screen")
            #expect(left == right, "\(label): top-1 differs")
        }
        func probes(at position: Int, first: Int32) async throws -> [[Float16]] {
            var rows: [[Float16]] = []
            for (offset, token) in [first, Int32(100), 500, 37, 19].enumerated() {
                try await runner.produceExactPrefill(token: token, position: position + offset, into: logits)
                rows.append(readLogits())
            }
            return rows
        }
        func batch(_ tokens: ArraySlice<Int32>, at position: Int) async throws -> PrefillResult {
            let result = try await runner.prefillChunked(tokens: tokens, startPosition: position,
                outputMode: .logits, config: .production(chunkTokens: 64), into: logits, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == tokens.count)
            #expect(result.execution?.replayedTokens == 0)
            return result
        }

        runner.mtp = nil
        for count in [65, 257, 1025] {
            let prompt = Array(source.prefix(count))
            runner.reset()
            for (position, token) in prompt.enumerated() {
                if position == count - 1 {
                    try await runner.produceExactPrefill(token: token, position: position, into: logits)
                } else {
                    try await runner.produceWithoutLogits(token: token, position: position)
                }
            }
            let head = readLogits()
            let reference = try await probes(at: count, first: 42)
            runner.reset()
            _ = try await batch(prompt.prefix(33), at: 0)
            try runner.prepareForContinuation(expectedPosition: 33)
            _ = try await batch(prompt.dropFirst(33), at: 33)
            compare(head, readLogits(), label: "prefill-\(count)-append33-head")
            let candidate = try await probes(at: count, first: 42)
            for index in reference.indices {
                compare(reference[index], candidate[index], label: "prefill-\(count)-state-probe\(index)")
            }
        }

        let prompt = try tok.encodeChat(messages: [.init(role: .user,
            content: "List the integers 1 through 30 in order, separated by spaces, and nothing else.")],
            reasoningEffort: .off)
        func generate(count: Int, speculative: Bool) async throws -> ([Int32], [[Float16]]) {
            runner.mtp = speculative ? speculator : nil
            runner.reset()
            let result = try await runner.prefillChunked(tokens: prompt[...], startPosition: 0,
                outputMode: .greedyIfAvailable, config: .production(chunkTokens: 64),
                into: logits, onProgress: { _ in })
            guard case .greedyToken(let first) = result.seed else {
                throw PrefillError.chunkedUnsupported("expected fused greedy seed")
            }
            var output = [Int32(bitPattern: first)]
            while output.count < count {
                try await runner.produce(token: output.last!, position: prompt.count + output.count - 1, into: logits)
                output.append(Int32(bitPattern: runner.lastGreedyToken))
            }
            let cursor = prompt.count + count - 1
            try runner.prepareForContinuation(expectedPosition: cursor)
            #expect(runner.continuationPosition == cursor)
            if let stats = runner.mtpSpecStats {
                print("[swift-qualification] mtp-stop\(count) rounds=\(stats.rounds) drafted=\(stats.draftedTokens) accepted=\(stats.acceptedTokens) rollbacks=\(stats.rollbacks)")
                #expect(stats.rounds > 0)
            }
            // Disable speculation after cursor reconciliation, then probe the
            // actual target KV/GDN/conv state via full-vocabulary logits.
            runner.mtp = nil
            return (output, try await probes(at: cursor, first: output.last!))
        }
        for count in [2, 7, 32, 64] {
            let plain = try await generate(count: count, speculative: false)
            let speculative = try await generate(count: count, speculative: true)
            #expect(plain.0 == speculative.0, "MTP stop at \(count)")
            for index in plain.1.indices {
                compare(plain.1[index], speculative.1[index], label: "mtp-stop\(count)-state-probe\(index)")
            }
        }
    }
}
