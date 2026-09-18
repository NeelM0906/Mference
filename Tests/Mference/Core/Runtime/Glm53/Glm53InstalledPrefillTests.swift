import Foundation
import Metal
import Testing
@testable import Mference

/// Real-checkpoint correctness only. Run after AGENTS.md's preflight with
/// MFERENCE_GLM53_GTURBO pointing at a completed install. Each arm releases its
/// model before the next loads; no weight copies or model downloads occur here.
@Suite(.serialized) struct Glm53InstalledPrefillTests {
    private struct Result {
        var heads: [[UInt16]]
        var continuation: [Int32]
        var tail: [[UInt16]]
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data(("[glm installed prefill] " + message + "\n").utf8))
    }

    private static func runArm(path: String, resident: Bool) async throws -> Result {
        let ctx = try MetalContext()
        let config = try #require(ArchConfig.knownArchitectures[.glm53Flash])
        let url = URL(fileURLWithPath: path)
        let model = try Model.load(directoryURL: url, device: ctx.device,
            expecting: config, streamingMode: resident ? .resident : .pread(slotCount: 16))
        let tokenizer = try await MFTokenizer.load(forModelDirectory: url)
        let runner = try Glm53ForwardRunner(model: model, context: ctx, maxContext: 2112,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 16,
                prefillChunkTokens: 128, forceLogitsHead: true))
        #expect(runner.expertsResident == resident)
        let out = try #require(ctx.device.makeBuffer(length: config.vocabSize * 2,
                                                     options: .storageModeShared))
        let text = String(repeating: "A small town measures river levels every morning. Compare weekly averages, explain uncertainty, and preserve original observations.\n", count: 300)
        let source = tokenizer.encode(text, addBOS: true)
        try #require(source.count >= 2083)
        let tokens = Array(source.prefix(2083))
        let validVocab = config.unpaddedVocabSize > 0 ? config.unpaddedVocabSize : config.vocabSize
        func snapshot() -> [UInt16] {
            let bits = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: UInt16.self),
                                                count: config.vocabSize))
            #expect(bits.prefix(validVocab).allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(bits.dropFirst(validVocab).allSatisfy { Float16(bitPattern: $0) == -.infinity })
            return bits
        }
        func greedy() -> Int32 {
            let values = out.contents().assumingMemoryBound(to: Float16.self)
            var best = 0
            for i in 1..<validVocab where values[i] > values[best] { best = i }
            return Int32(best)
        }
        let label = resident ? "resident" : "16 slots"
        log("strict-verified \(label); chunk=128; sparse cutover=\(runner.idxTopK)")
        var result = Result(heads: [], continuation: [], tail: [])
        var position = 0
        for end in [33, 2047, 2051, 2083] {
            try runner.prepareForContinuation(expectedPosition: position)
            let prefill = try await runner.prefillChunked(tokens: tokens[position..<end],
                startPosition: position, outputMode: .logits,
                config: .production(chunkTokens: 128), into: out, onProgress: { _ in })
            #expect(prefill.execution?.batchedTokens == end - position)
            #expect(prefill.execution?.replayedTokens == 0)
            #expect(prefill.newPosition == end)
            result.heads.append(snapshot())
            position = end
            log("\(label) head at \(end), replay=\(prefill.execution?.replayedTokens ?? -1)")
        }
        for i in 0..<8 {
            let token = greedy()
            result.continuation.append(token)
            try await runner.produce(token: token, position: position + i, into: out)
            result.tail.append(snapshot())
        }

        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens.prefix(33), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 128), into: out, onProgress: { _ in })
        #expect(snapshot() == result.heads[0], "reset reproduces clean prefix")
        runner.prefillDidCompleteLayer = { layer in
            if layer == 1 { throw CancellationError() }
        }
        do {
            _ = try await runner.prefillChunked(tokens: tokens[33..<65], startPosition: 33,
                outputMode: .logits, config: .production(chunkTokens: 128), into: out, onProgress: { _ in })
            Issue.record("expected partial-chunk cancellation")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { try runner.prepareForContinuation(expectedPosition: 33) }
        do {
            try await runner.produce(token: tokens[33], position: 33, into: out)
            Issue.record("dirty state must reject decode")
        } catch let error as PrefillError {
            guard case .chunkedRunnerDirty = error else { throw error }
        }
        runner.prefillDidCompleteLayer = nil
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens.prefix(33), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 128), into: out, onProgress: { _ in })
        #expect(snapshot() == result.heads[0], "reset after cancellation restores the clean result")
        log("\(label) completed boundary, continuation, cancellation and reset checks")
        return result
    }

    @Test func streamedMatchesResidentAcrossSparseCutover() async throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_GLM53_GTURBO"] else { return }
        let reference = try await Self.runArm(path: path, resident: true)
        let streamed = try await Self.runArm(path: path, resident: false)
        #expect(streamed.continuation == reference.continuation, "eight greedy choices")
        for i in reference.heads.indices {
            let mismatches = zip(reference.heads[i], streamed.heads[i]).filter { $0 != $1 }.count
            #expect(mismatches == 0, "boundary \(i): full logits")
            Self.log("boundary \(i): streamed/resident mismatches=\(mismatches)")
        }
        for i in reference.tail.indices {
            #expect(streamed.tail[i] == reference.tail[i], "continuation \(i): full logits")
        }
    }
}
