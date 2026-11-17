import Foundation
import Metal
import Testing
@testable import Mference

/// K4: the multi-prefix snapshot conformance on the Flash-Next runner. The
/// reference is always the live continuation computed on the same runner from
/// a reset, through the same prefill chunks and decode steps, so every
/// comparison is bitwise. The toy install takes the shipped GDN state path,
/// the one the Unsloth install uses, and has a PLE layer and a QSA indexer
/// whose 32-token budget selects once a prefix passes 32 tokens.
@Suite(.serialized) struct FlashNextPrefixSnapshotTests {
    private struct Harness {
        let directory: URL
        let context: MetalContext
        let model: Model
        let runner: FlashNextForwardRunner
        let output: MTLBuffer

        var vocab: Int { model.config.vocabSize }

        func bits() -> [UInt16] {
            let values = Array(UnsafeBufferPointer(
                start: output.contents().bindMemory(to: UInt16.self, capacity: vocab), count: vocab))
            #expect(values.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(values.contains { Float16(bitPattern: $0) != 0 })
            return values
        }
    }

    /// How a prefix is built: a chunked prefill, then single decode steps.
    private struct Plan {
        let prompt: [Int32]
        let decoded: [Int32]
        var length: Int { prompt.count + decoded.count }
    }

    private func make() throws -> Harness {
        let directory = try FlashNextToySynthetic.write()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: 96)
        let output = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        return Harness(directory: directory, context: context, model: model, runner: runner, output: output)
    }

    private func tokens(_ count: Int, seed: Int, vocab: Int) -> [Int32] {
        let base: Int = seed * 13 + 11
        let span: Int = vocab - 4
        return (0..<count).map { (index: Int) -> Int32 in
            let offset: Int = (index * 37 + base) % span
            return Int32(4 + offset)
        }
    }

    private func plan(prompt: Int, decoded: Int, seed: Int, vocab: Int) -> Plan {
        let all = tokens(prompt + decoded, seed: seed, vocab: vocab)
        return Plan(prompt: Array(all.prefix(prompt)), decoded: Array(all.dropFirst(prompt)))
    }

    /// Builds `plan` from the runner's current position 0.
    private func build(_ plan: Plan, on h: Harness) async throws {
        _ = try await h.runner.prefillChunked(tokens: plan.prompt[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output,
            onProgress: { _ in })
        try h.runner.prepareForContinuation(expectedPosition: plan.prompt.count)
        for (i, token) in plan.decoded.enumerated() {
            try await h.runner.produce(token: token, position: plan.prompt.count + i, into: h.output)
        }
        #expect(h.runner.continuationPosition == plan.length)
    }

    /// Decodes `continuation` from the runner's position and returns every
    /// step's logits.
    private func run(_ continuation: [Int32], on h: Harness) async throws -> [[UInt16]] {
        var out: [[UInt16]] = []
        let start = h.runner.continuationPosition
        for (i, token) in continuation.enumerated() {
            try await h.runner.produce(token: token, position: start + i, into: h.output)
            out.append(h.bits())
        }
        return out
    }

    /// The live reference: reset, build `plan`, decode `continuation`.
    private func reference(_ plan: Plan, _ continuation: [Int32], on h: Harness) async throws -> [[UInt16]] {
        h.runner.reset()
        try await build(plan, on: h)
        return try await run(continuation, on: h)
    }

    private func recurrentRegion(_ h: Harness) throws -> PrefixSnapshotRegion {
        let region = try PrefixSnapshotRegion(byteCount: h.runner.prefixRecurrentStateBytes)
        try h.runner.capturePrefixRecurrentState(into: region)
        return region
    }

    private func rowRegion(_ h: Harness, rows: Int) throws -> PrefixSnapshotRegion {
        let region = try PrefixSnapshotRegion(byteCount: h.runner.prefixKVRowBytes(tokens: rows))
        try h.runner.capturePrefixKVRows(count: rows, into: region)
        return region
    }

    @Test func toyRunnerSupportsSnapshots() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        #expect(h.runner.supportsPrefixStateSnapshots)
        #expect(h.runner.prefixKVRowBytes(tokens: 0) == 0)
        #expect(h.runner.prefixKVRowBytes(tokens: 8) > h.runner.prefixKVRowBytes(tokens: 7),
                "a completed block adds a block-key row")
        #expect(h.runner.prefixRecurrentStateBytes > 0)
    }

    /// Rewind within the live lineage: capture the recurrent state at P, let
    /// the lineage run past P (completing P's partial block), rewind to P and
    /// take a different continuation. P covers every block phase, prefill- and
    /// decode-ended prefixes, and prefixes past the indexer budget.
    @Test(arguments: [(40, 0), (41, 0), (35, 2), (38, 5), (20, 3)])
    func rewindMatchesTheLiveContinuation(prompt: Int, decoded: Int) async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let prefix = plan(prompt: prompt, decoded: decoded, seed: 1, vocab: h.vocab)
        let first = tokens(9, seed: 2, vocab: h.vocab)
        let second = tokens(7, seed: 3, vocab: h.vocab)
        let expected = try await reference(prefix, second, on: h)

        h.runner.reset()
        try await build(prefix, on: h)
        let state = try recurrentRegion(h)
        _ = try await run(first, on: h)
        try h.runner.restorePrefixState(rows: nil, position: prefix.length, recurrentState: state)
        #expect(h.runner.continuationPosition == prefix.length)
        #expect(try await run(second, on: h) == expected)
    }

    /// Restore from a row image after another conversation overwrote the
    /// rows, longer and shorter than the restored prefix.
    @Test(arguments: [60, 20])
    func imageRestoreMatchesTheLiveContinuation(otherLength: Int) async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let prefix = plan(prompt: 37, decoded: 2, seed: 4, vocab: h.vocab)
        let other = plan(prompt: otherLength, decoded: 1, seed: 5, vocab: h.vocab)
        let continuation = tokens(8, seed: 6, vocab: h.vocab)
        let expected = try await reference(prefix, continuation, on: h)

        h.runner.reset()
        try await build(prefix, on: h)
        let rows = try rowRegion(h, rows: prefix.length)
        let state = try recurrentRegion(h)
        h.runner.reset()
        try await build(other, on: h)
        try h.runner.restorePrefixState(rows: (rows, prefix.length), position: prefix.length,
                                        recurrentState: state)
        #expect(try await run(continuation, on: h) == expected)
    }

    /// K2 shares one image between the checkpoints of a lineage: restore an
    /// earlier checkpoint from an image taken further along.
    @Test func earlierCheckpointRestoresFromALongerImage() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let early = plan(prompt: 33, decoded: 2, seed: 7, vocab: h.vocab)
        let later = tokens(10, seed: 8, vocab: h.vocab)
        let continuation = tokens(6, seed: 9, vocab: h.vocab)
        let expected = try await reference(early, continuation, on: h)

        h.runner.reset()
        try await build(early, on: h)
        let earlyState = try recurrentRegion(h)
        _ = try await run(later, on: h)
        let rows = try rowRegion(h, rows: h.runner.continuationPosition)
        let imageRows = h.runner.continuationPosition
        h.runner.reset()
        try await build(plan(prompt: 50, decoded: 0, seed: 10, vocab: h.vocab), on: h)
        try h.runner.restorePrefixState(rows: (rows, imageRows), position: early.length,
                                        recurrentState: earlyState)
        #expect(try await run(continuation, on: h) == expected)
    }

    @Test func refusalsLeaveTheRunnerUsable() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let prefix = plan(prompt: 24, decoded: 1, seed: 11, vocab: h.vocab)
        try await build(prefix, on: h)
        let state = try recurrentRegion(h)

        // The recurrent state records where it was taken.
        #expect(throws: PrefixSnapshotError.self) {
            try h.runner.restorePrefixState(rows: nil, position: prefix.length - 1, recurrentState: state)
        }
        // A rewind cannot move forward.
        _ = try await run(tokens(2, seed: 12, vocab: h.vocab), on: h)
        let ahead = try recurrentRegion(h)
        h.runner.reset()
        try await build(prefix, on: h)
        #expect(throws: PrefixSnapshotError.self) {
            try h.runner.restorePrefixState(rows: nil, position: prefix.length + 2, recurrentState: ahead)
        }
        // Too small a region.
        let small = try PrefixSnapshotRegion(byteCount: h.runner.prefixKVRowBytes(tokens: 8) - 4)
        #expect(throws: PrefixSnapshotError.self) { try h.runner.capturePrefixKVRows(count: 8, into: small) }
        // Rows past the cursor.
        let big = try PrefixSnapshotRegion(byteCount: h.runner.prefixKVRowBytes(tokens: 40))
        #expect(throws: PrefixSnapshotError.self) { try h.runner.capturePrefixKVRows(count: 40, into: big) }
        // None of that dirtied the runner.
        try h.runner.prepareForContinuation(expectedPosition: prefix.length)
        _ = try await run(tokens(1, seed: 13, vocab: h.vocab), on: h)
    }

    @Test func diagnosticsAndDraftConsumersOptOut() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        h.runner.capture = .init()
        #expect(!h.runner.supportsPrefixStateSnapshots)
        h.runner.capture = nil
        h.runner.consumeTargetHiddenRows = { _ in }
        #expect(!h.runner.supportsPrefixStateSnapshots)
        h.runner.consumeTargetHiddenRows = nil
        #expect(h.runner.supportsPrefixStateSnapshots)
    }

    /// A restore makes every older decode checkpoint stale.
    @Test func restoreInvalidatesDecodeCheckpoints() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        try await build(plan(prompt: 18, decoded: 1, seed: 14, vocab: h.vocab), on: h)
        let state = try recurrentRegion(h)
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        try h.runner.restorePrefixState(rows: nil, position: 19, recurrentState: state)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restoreDecodeCheckpoint(checkpoint) }
    }
}
