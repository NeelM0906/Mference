import Foundation
import Metal
import Testing
@testable import Mference

/// On the Qwen 3.6 toy runner, a prefix restored through
/// `PrefixStateSnapshotting` continues with logits bit-identical to
/// continuing the live runner: after another conversation, after a branch on
/// the same lineage, and after a cancelled prefill plus reset.
@Suite struct QwenRunnerPrefixSnapshotTests {

    // MARK: Toy Qwen 3.6 runner, full-logit bit identity

    private struct Toy {
        let dir: URL
        let context: MetalContext
        let runner: RealForwardRunner
        let logits: MTLBuffer
    }

    private func makeToy() throws -> Toy {
        let dir = try QwenToySynthetic.write()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: context.device, expecting: .qwen36Toy())
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 64,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 32, forceLogitsHead: true,
                                                       kvGrowthTokens: 16))
        let logits = try #require(context.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.stride, options: .storageModeShared))
        return Toy(dir: dir, context: context, runner: runner, logits: logits)
    }

    private func bits(_ buffer: MTLBuffer, vocab: Int = 1024) -> [UInt16] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt16.self), count: vocab))
    }

    /// Prefills `tokens` from `start` in 32-token chunks and returns the
    /// final row's logits bits.
    private func prefill(_ toy: Toy, _ tokens: [Int32], from start: Int) async throws -> [UInt16] {
        if start > 0 { try toy.runner.prepareForContinuation(expectedPosition: start) }
        let result = try await toy.runner.prefillChunked(
            tokens: tokens[...], startPosition: start, outputMode: .logits,
            config: .production(chunkTokens: 32), into: toy.logits, onProgress: { _ in })
        #expect(result.newPosition == start + tokens.count)
        #expect(result.seed == .logitsWritten)
        return bits(toy.logits)
    }

    /// Decodes `tokens` one per step and returns every step's logits bits.
    private func decode(_ toy: Toy, _ tokens: [Int32], from start: Int) async throws -> [[UInt16]] {
        var out: [[UInt16]] = []
        for (i, token) in tokens.enumerated() {
            try await toy.runner.produce(token: token, position: start + i, into: toy.logits)
            out.append(bits(toy.logits))
        }
        return out
    }

    private let prefixA: [Int32] = (0..<37).map { Int32(($0 * 29 + 3) % 1000 + 1) }
    private let suffixA: [Int32] = (0..<9).map { Int32(($0 * 17 + 5) % 1000 + 1) }
    private let decodeA: [Int32] = [11, 400, 7, 931]
    private let otherB: [Int32] = (0..<23).map { Int32(($0 * 41 + 9) % 1000 + 1) }

    /// Control: A's prefix, then its suffix and decode, on the live runner.
    private func control(_ toy: Toy) async throws -> (suffix: [UInt16], decode: [[UInt16]]) {
        toy.runner.reset()
        _ = try await prefill(toy, prefixA, from: 0)
        let suffix = try await prefill(toy, suffixA, from: prefixA.count)
        let steps = try await decode(toy, decodeA, from: prefixA.count + suffixA.count)
        return (suffix, steps)
    }

    @Test func restoredImageContinuesBitIdenticallyAfterAnotherConversation() async throws {
        let toy = try makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        #expect(toy.runner.supportsPrefixStateSnapshots)
        let expected = try await control(toy)

        toy.runner.reset()
        _ = try await prefill(toy, prefixA, from: 0)
        let p = prefixA.count
        let rows = try PrefixSnapshotRegion(byteCount: toy.runner.prefixKVRowBytes(tokens: p))
        let state = try PrefixSnapshotRegion(byteCount: toy.runner.prefixRecurrentStateBytes)
        try toy.runner.capturePrefixKVRows(count: p, into: rows)
        try toy.runner.capturePrefixRecurrentState(into: state)

        // Another conversation overwrites every row, the recurrent state and
        // the expert slots, then a reset releases the grown KV regions.
        toy.runner.reset()
        _ = try await prefill(toy, otherB, from: 0)
        _ = try await decode(toy, [3, 5, 8], from: otherB.count)
        toy.runner.reset()

        try toy.runner.restorePrefixState(rows: (rows, p), position: p, recurrentState: state)
        #expect(toy.runner.continuationPosition == p)
        let suffix = try await prefill(toy, suffixA, from: p)
        let steps = try await decode(toy, decodeA, from: p + suffixA.count)
        #expect(suffix == expected.suffix)
        #expect(steps == expected.decode)
    }

    @Test func liveRewindToACheckpointContinuesBitIdenticallyAfterABranch() async throws {
        let toy = try makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let expected = try await control(toy)

        toy.runner.reset()
        _ = try await prefill(toy, prefixA, from: 0)
        let p = prefixA.count
        let state = try PrefixSnapshotRegion(byteCount: toy.runner.prefixRecurrentStateBytes)
        try toy.runner.capturePrefixRecurrentState(into: state)

        // A branch continues the same conversation differently, writing rows
        // at p and beyond and advancing the recurrent state.
        _ = try await prefill(toy, otherB, from: p)
        _ = try await decode(toy, [3, 5], from: p + otherB.count)

        try toy.runner.restorePrefixState(rows: nil, position: p, recurrentState: state)
        #expect(toy.runner.continuationPosition == p)
        let suffix = try await prefill(toy, suffixA, from: p)
        let steps = try await decode(toy, decodeA, from: p + suffixA.count)
        #expect(suffix == expected.suffix)
        #expect(steps == expected.decode)
    }

    @Test func restoreAfterACancelledPrefillContinuesBitIdentically() async throws {
        let toy = try makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let expected = try await control(toy)

        toy.runner.reset()
        _ = try await prefill(toy, prefixA, from: 0)
        let p = prefixA.count
        let rows = try PrefixSnapshotRegion(byteCount: toy.runner.prefixKVRowBytes(tokens: p))
        let state = try PrefixSnapshotRegion(byteCount: toy.runner.prefixRecurrentStateBytes)
        try toy.runner.capturePrefixKVRows(count: p, into: rows)
        try toy.runner.capturePrefixRecurrentState(into: state)

        // Cancel the suffix prefill partway through the layer stack; the
        // runner is dirty and refuses snapshots until reset.
        toy.runner.prefillWillEncodeLayer = { layer in if layer == 2 { throw CancellationError() } }
        await #expect(throws: (any Error).self) {
            _ = try await prefill(toy, suffixA, from: p)
        }
        toy.runner.prefillWillEncodeLayer = nil
        #expect(throws: (any Error).self) {
            try toy.runner.restorePrefixState(rows: (rows, p), position: p, recurrentState: state)
        }
        toy.runner.reset()

        try toy.runner.restorePrefixState(rows: (rows, p), position: p, recurrentState: state)
        let suffix = try await prefill(toy, suffixA, from: p)
        let steps = try await decode(toy, decodeA, from: p + suffixA.count)
        #expect(suffix == expected.suffix)
        #expect(steps == expected.decode)
    }
}
