import Darwin
import Foundation
import Metal
import Testing
@testable import Mference

/// On the Mini-width toy, a prefix restored through `PrefixStateSnapshotting`
/// continues with logits bit-identical to continuing the live runner: after
/// another conversation and a reset (the global rows from an image), after a
/// branch on the same lineage (a rewind), after the rings wrapped over the
/// prefix's window, and after a prefill cancelled between two spans or left
/// dirty; then the same through `PrefixStateCache`. Full tier: each toy
/// builds about 0.9 GB of experts.
@Suite("Inkling2ForwardRunner prefix snapshots on the toy", .serialized)
struct Inkling2PrefixSnapshotTests {

    /// The toy's experts, with a visit that throws on demand (`run()` then
    /// throws mid-chunk and leaves the chunk state dirty, the source sound),
    /// and abandons counted, or failing on demand.
    private final class ThrowingExperts: Inkling2ExpertSource {
        let base: any Inkling2ExpertSource
        var throwAtLayer: Int?
        var abandons = 0
        var abandonThrows = false
        init(_ base: any Inkling2ExpertSource) { self.base = base }
        var shape: Inkling2MoEShape { base.shape }
        var boundExpertBytes: Int { base.boundExpertBytes }
        func beginVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws -> Inkling2ExpertPool {
            if layer == throwAtLayer { throw CancellationError() }
            return try base.beginVisit(layer: layer, commandBuffer: commandBuffer)
        }
        func endVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws {
            try base.endVisit(layer: layer, commandBuffer: commandBuffer)
        }
        func decodeNeedsRoutes(layer: Int) -> Bool { base.decodeNeedsRoutes(layer: layer) }
        func beginDecodeVisit(layer: Int, expertIDs: [Int]) throws -> Inkling2DecodeBinding {
            try base.beginDecodeVisit(layer: layer, expertIDs: expertIDs)
        }
        func endDecodeVisit(_ binding: Inkling2DecodeBinding, commandBuffer: MTLCommandBuffer) {
            base.endDecodeVisit(binding, commandBuffer: commandBuffer)
        }
        func abandonOpenVisits() throws {
            abandons += 1
            if abandonThrows { throw CancellationError() }
            try base.abandonOpenVisits()
        }
        func recoverAfterFailedCommands() { base.recoverAfterFailedCommands() }
        func prefetchFirstStagedVisit() throws { try base.prefetchFirstStagedVisit() }
        func cancelPrefetch() throws { try base.cancelPrefetch() }
        var stagingFailure: String? { base.stagingFailure }
    }

    private final class Toy: @unchecked Sendable {
        let runner: Inkling2ForwardRunner
        let logits: MTLBuffer
        let experts: ThrowingExperts
        /// The runner's own chunk splits the spans; the config asks for at
        /// least 64, the smallest production chunk the toy tests use.
        let prefillConfig: PrefillRuntimeConfig
        init(runner: Inkling2ForwardRunner, logits: MTLBuffer, experts: ThrowingExperts, chunk: Int) {
            self.runner = runner
            self.logits = logits
            self.experts = experts
            prefillConfig = .production(chunkTokens: max(64, chunk))
        }
    }

    /// By default chunks of 8: the 13-token prefix runs as two spans, a
    /// 20-token prompt as three.
    private func makeToy(maxContext: Int = 64, chunk: Int = 8) throws -> Toy {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        var weights = toy.runnerWeights
        let experts = ThrowingExperts(weights.experts)
        weights.experts = experts
        let runner = try Inkling2ForwardRunner(weights: weights, context: context, maxContext: maxContext,
                                               maxChunkTokens: chunk)
        let logits = try #require(context.device.makeBuffer(length: Inkling2RunnerToy.config.vocabPadded * 2,
                                                            options: .storageModeShared))
        return Toy(runner: runner, logits: logits, experts: experts, chunk: chunk)
    }

    private func bits(_ toy: Toy) -> [UInt16] {
        Array(UnsafeBufferPointer(start: toy.logits.contents().assumingMemoryBound(to: UInt16.self),
                                  count: Inkling2RunnerToy.config.vocabPadded))
    }

    /// Prefills `tokens` from `start` and returns the last row's logits bits.
    private func prefill(_ toy: Toy, _ tokens: [Int32], from start: Int) async throws -> [UInt16] {
        if start > 0 { try toy.runner.prepareForContinuation(expectedPosition: start) }
        let result = try await toy.runner.prefillChunked(
            tokens: tokens[...], startPosition: start, outputMode: .logits,
            config: toy.prefillConfig, into: toy.logits, onProgress: { _ in })
        #expect(result.newPosition == start + tokens.count)
        return bits(toy)
    }

    /// Decodes `tokens` one per step and returns every step's logits bits.
    private func decode(_ toy: Toy, _ tokens: [Int32], from start: Int) async throws -> [[UInt16]] {
        var out: [[UInt16]] = []
        for (i, token) in tokens.enumerated() {
            try await toy.runner.produce(token: token, position: start + i, into: toy.logits)
            out.append(bits(toy))
        }
        return out
    }

    /// Prefills `tokens` (more than one span) from `start` and cancels after
    /// the first span: the next span's check throws, the first stays
    /// committed and the chunk state clean. Returns the cursor.
    private func prefillCancelledAfterFirstSpan(_ toy: Toy, _ tokens: [Int32], from start: Int) async throws -> Int {
        if start > 0 { try toy.runner.prepareForContinuation(expectedPosition: start) }
        let task = Task {
            _ = try await toy.runner.prefillChunked(
                tokens: tokens[...], startPosition: start, outputMode: .logits,
                config: toy.prefillConfig, into: toy.logits,
                onProgress: { _ in _ = withUnsafeCurrentTask { $0?.cancel() } })
        }
        let result = await task.result
        #expect(throws: CancellationError.self) { _ = try result.get() }
        try toy.runner.chunkState.requireClean(operation: "test")
        return toy.runner.continuationPosition
    }

    private let prefixA: [Int32] = (0..<13).map { Int32(($0 * 29 + 3) % 499 + 1) }
    private let suffixA: [Int32] = (0..<6).map { Int32(($0 * 17 + 5) % 499 + 1) }
    private let decodeA: [Int32] = [11, 400, 7, 431]
    private let otherB: [Int32] = (0..<20).map { Int32(($0 * 41 + 9) % 499 + 1) }

    /// Control: A's prefix, then its suffix and decode, on the live runner.
    private func control(_ toy: Toy) async throws -> (suffix: [UInt16], decode: [[UInt16]]) {
        toy.runner.reset()
        _ = try await prefill(toy, prefixA, from: 0)
        let suffix = try await prefill(toy, suffixA, from: prefixA.count)
        let steps = try await decode(toy, decodeA, from: prefixA.count + suffixA.count)
        return (suffix, steps)
    }

    /// Restores A's prefix and checks that its continuation matches the control.
    private func expectContinuation(_ toy: Toy, matches expected: (suffix: [UInt16], decode: [[UInt16]]),
                                    sourceLocation: SourceLocation = #_sourceLocation) async throws {
        let p = prefixA.count
        #expect(toy.runner.continuationPosition == p, sourceLocation: sourceLocation)
        let suffix = try await prefill(toy, suffixA, from: p)
        let steps = try await decode(toy, decodeA, from: p + suffixA.count)
        #expect(suffix == expected.suffix, sourceLocation: sourceLocation)
        #expect(steps == expected.decode, sourceLocation: sourceLocation)
    }

    /// Captures A's prefix: its global rows and its light snapshot.
    private func captureA(_ toy: Toy) async throws -> (rows: PrefixSnapshotRegion, state: PrefixSnapshotRegion) {
        toy.runner.reset()
        _ = try await prefill(toy, prefixA, from: 0)
        let p = prefixA.count
        #expect(toy.runner.supportsPrefixStateSnapshots)
        #expect(toy.runner.prefixKVRowBytes(tokens: p) == 8 * 2 * p * 2048)
        #expect(toy.runner.prefixRecurrentStateBytes == 64 + 48 * 19_968 * 4 + 40 * 2 * p * 2048)
        let rows = try PrefixSnapshotRegion(byteCount: toy.runner.prefixKVRowBytes(tokens: p))
        let state = try PrefixSnapshotRegion(byteCount: toy.runner.prefixRecurrentStateBytes)
        try toy.runner.capturePrefixKVRows(count: p, into: rows)
        try toy.runner.capturePrefixRecurrentState(into: state)
        return (rows, state)
    }

    @Test(.fullTier) func imageAfterAnotherConversationThenRewindsContinueBitIdentically() async throws {
        let toy = try makeToy()
        let expected = try await control(toy)
        let a = try await captureA(toy)
        let p = prefixA.count

        // Another conversation rewrites every ring, tail and global row; the
        // reset returns the global pages, as the cache does before an image.
        toy.runner.reset()
        _ = try await prefill(toy, otherB, from: 0)
        _ = try await decode(toy, [3, 5, 8], from: otherB.count)
        toy.runner.reset()
        try toy.runner.restorePrefixState(rows: (a.rows, p), position: p, recurrentState: a.state)
        try await expectContinuation(toy, matches: expected)

        // A branch on the restored lineage, then a rewind to A's prefix (no
        // rows): the rows below it are the image's, the rings and tails the
        // snapshot's.
        try toy.runner.restorePrefixState(rows: nil, position: p, recurrentState: a.state)
        _ = try await prefill(toy, Array(otherB[..<10]), from: p)
        _ = try await decode(toy, [3, 5], from: p + 10)
        try toy.runner.restorePrefixState(rows: nil, position: p, recurrentState: a.state)
        try await expectContinuation(toy, matches: expected)
    }

    /// K4's cases: a prefill cancelled between spans (each span committed,
    /// the chunk clean, the early read given back by the prefill; the restore
    /// still abandons, after its checks), a chunk left dirty by a throw inside
    /// `run()` (refused until the reset), and an abandon that fails (the
    /// source then refuses every call, and stays failed).
    @Test(.fullTier) func restoresAfterCancelledOrFailedPrefillsContinueBitIdentically() async throws {
        let toy = try makeToy()
        let expected = try await control(toy)
        let a = try await captureA(toy)
        let p = prefixA.count

        // Cancelled after its first span: the runner holds A's prefix plus 7
        // rows of B; a rewind restores the prefix without a reset.
        #expect(try await prefillCancelledAfterFirstSpan(toy, otherB, from: p) == p + 7)
        var abandons = toy.experts.abandons
        try toy.runner.restorePrefixState(rows: nil, position: p, recurrentState: a.state)
        #expect(toy.experts.abandons == abandons + 1)
        try await expectContinuation(toy, matches: expected)

        // The same after a reset (the server's path after a failed request):
        // only the image restores.
        let at = toy.runner.continuationPosition
        #expect(try await prefillCancelledAfterFirstSpan(toy, otherB, from: at) == at + 7)
        toy.runner.reset()
        abandons = toy.experts.abandons
        #expect(throws: PrefixSnapshotError.self) {
            try toy.runner.restorePrefixState(rows: nil, position: p, recurrentState: a.state)
        }
        #expect(toy.experts.abandons == abandons, "a refused restore abandons nothing")
        try toy.runner.restorePrefixState(rows: (a.rows, p), position: p, recurrentState: a.state)
        try await expectContinuation(toy, matches: expected)

        // run() throws mid-chunk (a visit fails to begin): the chunk is dirty,
        // the source sound, and every snapshot call refuses until the reset.
        let scratch = try PrefixSnapshotRegion(byteCount: toy.runner.prefixRecurrentStateBytes)
        let before = toy.runner.continuationPosition
        toy.experts.throwAtLayer = 2
        await #expect(throws: CancellationError.self) { _ = try await prefill(toy, otherB, from: before) }
        toy.experts.throwAtLayer = nil
        #expect(throws: (any Error).self) {
            try toy.runner.restorePrefixState(rows: (a.rows, p), position: p, recurrentState: a.state)
        }
        #expect(throws: (any Error).self) { try toy.runner.capturePrefixRecurrentState(into: scratch) }
        #expect(throws: (any Error).self) { try toy.runner.capturePrefixKVRows(count: p, into: scratch) }
        #expect(toy.runner.expertsFailed == nil)
        #expect(toy.runner.continuationPosition == before)
        toy.runner.reset()
        try toy.runner.restorePrefixState(rows: (a.rows, p), position: p, recurrentState: a.state)
        try await expectContinuation(toy, matches: expected)

        // An abandon that fails marks the source failed: the restore throws
        // before copying, and every later call refuses.
        let position = toy.runner.continuationPosition
        toy.experts.abandonThrows = true
        #expect(throws: (any Error).self) {
            try toy.runner.restorePrefixState(rows: (a.rows, p), position: p, recurrentState: a.state)
        }
        toy.experts.abandonThrows = false
        #expect(toy.runner.expertsFailed?.contains("abandoning open expert visits failed") == true)
        #expect(toy.runner.continuationPosition == position)
        abandons = toy.experts.abandons
        #expect(throws: (any Error).self) { try toy.runner.capturePrefixKVRows(count: p, into: a.rows) }
        #expect(throws: (any Error).self) { try toy.runner.capturePrefixRecurrentState(into: scratch) }
        #expect(throws: (any Error).self) {
            try toy.runner.restorePrefixState(rows: (a.rows, p), position: p, recurrentState: a.state)
        }
        #expect(toy.experts.abandons == abandons)
        #expect(toy.runner.expertsFailed != nil)
    }

    /// A prefix of 1,030 tokens keeps the last 1,023 ring rows (the window
    /// minus one); a branch of 300 tokens wraps the rings (1,024 + 256 slots)
    /// over the oldest of them and over the row just outside the window. A
    /// rewind restores exactly those rows: the continuation matches the
    /// control only if the snapshot holds every row the window reads and the
    /// kernels read none older.
    @Test(.fullTier) func aRewindAfterTheRingsWrappedContinuesBitIdentically() async throws {
        let toy = try makeToy(maxContext: 1_400, chunk: 256)
        let long: [Int32] = (0..<1_030).map { Int32(($0 * 29 + 3) % 499 + 1) }
        let branch: [Int32] = (0..<300).map { Int32(($0 * 53 + 11) % 499 + 1) }
        let p = long.count

        toy.runner.reset()
        _ = try await prefill(toy, long, from: 0)
        let suffix = try await prefill(toy, suffixA, from: p)
        let steps = try await decode(toy, decodeA, from: p + suffixA.count)

        toy.runner.reset()
        _ = try await prefill(toy, long, from: 0)
        #expect(toy.runner.prefixRecurrentStateBytes == 64 + 48 * 19_968 * 4 + 40 * 2 * 1_023 * 2048)
        let state = try PrefixSnapshotRegion(byteCount: toy.runner.prefixRecurrentStateBytes)
        try toy.runner.capturePrefixRecurrentState(into: state)
        _ = try await prefill(toy, branch, from: p)
        _ = try await decode(toy, [3, 5], from: p + branch.count)
        #expect(toy.runner.continuationPosition > 1_024 + 256 + p - 1_023)

        try toy.runner.restorePrefixState(rows: nil, position: p, recurrentState: state)
        #expect(toy.runner.continuationPosition == p)
        #expect(try await prefill(toy, suffixA, from: p) == suffix)
        #expect(try await decode(toy, decodeA, from: p + suffixA.count) == steps)
    }

    // MARK: Through `PrefixStateCache`

    private static func tokenizer() async throws -> MFTokenizer {
        let folder = try #require(Bundle.module.url(
            forResource: "ChatMLTokenizer", withExtension: nil, subdirectory: "Fixtures"))
        return try await MFTokenizer.load(from: folder)
    }

    private static let domain = PromptPrefixCacheDomain(
        modelID: "inkling2-toy", sourceSnapshotHash: nil, runtimeProfileHash: "runtime",
        maximumContext: 64, kvStorage: "fp16", fp16RingEnabled: false, templateSHA256: "template")

    /// Exact-prefix hits only: with no messages a turn continues an entry
    /// exactly when its rendered prompt extends the entry's KV-backed tokens.
    private static let turn = PromptPrefixCacheTurn(messages: [])

    private static func entry(kvBacked: [Int32], boundary: Int32) -> PromptPrefixCacheEntry? {
        PromptPrefixCache.makeEntry(
            domain: domain, turn: turn, content: "answer", calls: [],
            result: RawDecodeResult(prefillTokens: kvBacked.count, cachedPromptTokens: 0,
                                    computedPrefillTokens: kvBacked.count, prefillSeconds: 0,
                                    newTokens: 1, decodeSeconds: 0, reason: .endOfTurn,
                                    kvPosition: kvBacked.count, kvBackedTokenIDs: kvBacked,
                                    uncommittedBoundaryTokenIDs: [boundary], prefillExecution: nil))
    }

    /// One turn through `cache`: prefill what `begin` left, decode
    /// `generated` teacher-forced, publish. Returns the KV-backed tokens and
    /// every logits row (prompt end, then each decode step).
    private func turn(_ toy: Toy, _ cache: PrefixStateCache, _ tokenizer: MFTokenizer, _ prompt: [Int32],
                      generated: [Int32], expect outcome: PrefixStateCache.Outcome,
                      sourceLocation: SourceLocation = #_sourceLocation) async throws
        -> (kv: [Int32], logits: [[UInt16]]) {
        let begun = try cache.begin(domain: Self.domain, turn: Self.turn, renderedPromptIDs: prompt,
                                    tokenizer: tokenizer, maxContext: 64, runner: toy.runner)
        #expect(begun.outcome == outcome, sourceLocation: sourceLocation)
        let start: Int
        switch begun.start {
        case .reset:
            toy.runner.reset()
            start = 0
        case .resume(let cached):
            start = cached
        }
        var rows = [try await prefill(toy, Array(prompt[start...]), from: start)]
        rows += try await decode(toy, generated, from: prompt.count)
        let kv = prompt + generated
        cache.didFinish(entry: Self.entry(kvBacked: kv, boundary: 7), runner: toy.runner)
        return (kv, rows)
    }

    private let a1Prompt: [Int32] = (0..<13).map { Int32(($0 * 29 + 3) % 499 + 1) }
    private let a1Generated: [Int32] = [11, 400, 7, 431]
    private let a2User: [Int32] = (0..<6).map { Int32(($0 * 17 + 5) % 499 + 1) }
    private let a2Generated: [Int32] = [12, 401, 8, 432]
    private let bPrompt: [Int32] = (0..<20).map { Int32(($0 * 41 + 9) % 499 + 1) }

    @Test(.fullTier) func throughTheCacheRestoredRewoundAndAfterACancelMatchTheControl() async throws {
        let toy = try makeToy()
        let tokenizer = try await Self.tokenizer()
        func freshCache() -> PrefixStateCache { PrefixStateCache(ceilingBytes: 256 << 20) }

        // Control: the single-prefix view, A1 then A2 on a live runner.
        var cache = freshCache()
        let a1 = try await turn(toy, cache, tokenizer, a1Prompt, generated: a1Generated, expect: .miss)
        let a2Prompt = a1.kv + [7] + a2User
        let expected = try await turn(toy, cache, tokenizer, a2Prompt, generated: a2Generated, expect: .live).logits

        // Another conversation, then back: A1's rows come from an image.
        cache = freshCache()
        _ = try await turn(toy, cache, tokenizer, a1Prompt, generated: a1Generated, expect: .miss)
        _ = try await turn(toy, cache, tokenizer, bPrompt, generated: [3, 5, 8], expect: .miss)
        #expect(try await turn(toy, cache, tokenizer, a2Prompt, generated: a2Generated,
                               expect: .restored).logits == expected)

        // A branch, then A2: a rewind to A1's end.
        cache = freshCache()
        _ = try await turn(toy, cache, tokenizer, a1Prompt, generated: a1Generated, expect: .miss)
        _ = try await turn(toy, cache, tokenizer, a1.kv + [7] + [44, 45, 46, 47, 48], generated: [3, 5],
                           expect: .live)
        #expect(try await turn(toy, cache, tokenizer, a2Prompt, generated: a2Generated,
                               expect: .rewound).logits == expected)

        // B cancelled between two spans; the server resets and drops the
        // live prefix, and A2 restores A1 from its image.
        cache = freshCache()
        _ = try await turn(toy, cache, tokenizer, a1Prompt, generated: a1Generated, expect: .miss)
        let begun = try cache.begin(domain: Self.domain, turn: Self.turn, renderedPromptIDs: bPrompt,
                                    tokenizer: tokenizer, maxContext: 64, runner: toy.runner)
        #expect(begun.outcome == .miss)
        toy.runner.reset()
        #expect(try await prefillCancelledAfterFirstSpan(toy, bPrompt, from: 0) == 7)
        toy.runner.reset()
        cache.didFail()
        #expect(try await turn(toy, cache, tokenizer, a2Prompt, generated: a2Generated,
                               expect: .restored).logits == expected)
    }
}
