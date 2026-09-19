import Foundation
import Metal
import Testing
@testable import Mference

/// Same bit-exact full-logit contract as ProductionPrefillContractTests:
/// identical input and execution schedule before/after state restoration.
@Suite(.serialized) struct GemmaPrefixRecoveryTests {
    struct Harness {
        let directory: URL
        let context: MetalContext
        let runner: RealForwardRunner
        let logits: MTLBuffer
        let config = ArchConfig.gemma4Toy(topKExperts: 8)
        init() throws {
            directory = try ModelLoaderTests.writeToySynthetic(config: config, finiteNorms: true)
            context = try MetalContext()
            let model = try Model.load(directoryURL: directory, device: context.device,
                expecting: config, streamingMode: .pread(slotCount: 8))
            runner = try RealForwardRunner(model: model, context: context, maxContext: 768,
                runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8,
                    prefillChunkTokens: 64, forceLogitsHead: true))
            logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2,
                options: .storageModeShared))
        }
        func row() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self),
                count: config.vocabSize))
        }
        func prefill(_ tokens: ArraySlice<Int32>, at position: Int) async throws {
            _ = try await runner.prefillChunked(tokens: tokens, startPosition: position,
                outputMode: .logits, config: .production(chunkTokens: 64), into: logits, onProgress: { _ in })
        }
    }

    static let tokens: [Int32] = (0..<768).map { Int32(4 + ($0 * 37 + 11) % 1000) }

    @Test(arguments: [128, 380], [false, true])
    func recoveredStateReproducesEveryLogit(boundary: Int, rollover: Bool) async throws {
        let h = try Harness()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let r = h.runner
        try await h.prefill(Self.tokens[..<boundary], at: 0)
        #expect(try r.captureGemmaPrefix())
        let bound = UInt64(2 * min(boundary, 255) * 2 * 16 * 2)
        #expect(r.gemmaRecoveryBytes == bound)
        try await h.prefill(Self.tokens[boundary..<(boundary + 31)], at: boundary)
        let expected = h.row()
        try await r.produce(token: Self.tokens[boundary + 31], position: boundary + 31, into: h.logits)
        let expectedNext = h.row()
        let end = rollover ? 740 : boundary + 64
        try await h.prefill(Self.tokens[(boundary + 32)..<end], at: boundary + 32)
        #expect(r.gemmaRecoverablePrefix(upTo: boundary + 10) == (rollover ? boundary : boundary + 10))
        #expect(try r.recoverGemmaPrefix(to: boundary) == (rollover ? .snapshot : .current))
        #expect(r.continuationPosition == boundary)
        try r.prepareForContinuation(expectedPosition: boundary)
        try await h.prefill(Self.tokens[boundary..<(boundary + 31)], at: boundary)
        #expect(h.row() == expected, "all prompt logits match the clean execution")
        try await r.produce(token: Self.tokens[boundary + 31], position: boundary + 31, into: h.logits)
        #expect(h.row() == expectedNext, "all decode logits match the clean execution")
        #expect(h.row().allSatisfy { Float16(bitPattern: $0).isFinite })
        if rollover && boundary > 256 {
            #expect(r.gemmaRecoverablePrefix(upTo: boundary - 1) == 0,
                "restoring the cursor does not invent older SWA rows")
        }
        r.reset()
        #expect(r.gemmaRecoveryBytes == 0)
        #expect(r.gemmaRecoverablePrefix(upTo: boundary) == 0)
    }

    @Test(arguments: [GemmaPrefixRecoverySource.current, .snapshot], [false, true])
    func interruptedCopyRequiresReset(phase: GemmaPrefixRecoverySource, cancellation: Bool) async throws {
        let h = try Harness()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let r = h.runner
        try await h.prefill(Self.tokens[..<128], at: 0)
        let expected = h.row()
        #expect(try r.captureGemmaPrefix())
        if phase == .snapshot { try await h.prefill(Self.tokens[128..<740], at: 128) }
        enum Injected: Error { case copyFailed }
        r.gemmaRecoveryWillCopyLayer = { source, _ in
            if source == phase {
                if cancellation { withUnsafeCurrentTask { $0?.cancel() } }
                else { throw Injected.copyFailed }
            }
        }
        let interrupted = Task { @Sendable in
            if phase == .snapshot { _ = try r.recoverGemmaPrefix(to: 128) }
            else { _ = try r.captureGemmaPrefix() }
        }
        do { try await interrupted.value; Issue.record("injected copy must fail") }
        catch is CancellationError { #expect(cancellation) }
        catch is Injected { #expect(!cancellation) }
        #expect(r.gemmaRecoverablePrefix(upTo: 128) == 0)
        #expect(throws: (any Error).self) { try r.prepareForContinuation(expectedPosition: r.continuationPosition) }
        r.gemmaRecoveryWillCopyLayer = nil
        r.reset()
        #expect(r.gemmaRecoveryBytes == 0)
        try await h.prefill(Self.tokens[..<128], at: 0)
        #expect(h.row() == expected)
    }

    @Test func snapshotHasSeparateKVAndOnlyOneBoundedImage() throws {
        let context = try MetalContext()
        let config = ArchConfig.gemma4Toy(topKExperts: 8)
        let kv = try KVCacheManager(device: context.device, config: config, maxContext: 768,
            fp16RingEnabled: true, maxPrefillChunkTokens: 64)
        func fill(to end: Int, salt: Int) {
            while kv.position < end {
                let p = kv.position
                for layer in 0..<config.numLayers {
                    let k = kv.kSlot(layer: layer, position: p)
                    let v = kv.vSlot(layer: layer, position: p)
                    memset(k.buffer.contents().advanced(by: k.offset), Int32((p + salt) % 251), kv.stride(layer: layer))
                    memset(v.buffer.contents().advanced(by: v.offset), Int32((p + salt + 97) % 251), kv.stride(layer: layer))
                }
                kv.advance()
            }
        }
        for turn in 0..<8 {
            kv.reset()
            fill(to: 380, salt: turn)
            #expect(try kv.captureGemmaRecovery())
            let bytes = UInt64(2 * 255 * kv.stride(layer: 0))
            #expect(kv.gemmaRecoveryBytes == bytes)
            fill(to: 740, salt: turn + 31)
            #expect(kv.gemmaRecoverablePrefix(upTo: 379) == 0)
            #expect(try kv.recoverGemmaPrefix(to: 380) == .snapshot)
            for p in 125..<380 {
                for layer in 0..<config.numLayers {
                    let k = kv.kSlot(layer: layer, position: p)
                    let v = kv.vSlot(layer: layer, position: p)
                    #expect(k.buffer.contents().advanced(by: k.offset).load(as: UInt8.self) == UInt8((p + turn) % 251))
                    #expect(v.buffer.contents().advanced(by: v.offset).load(as: UInt8.self) == UInt8((p + turn + 97) % 251))
                }
            }
            #expect(kv.gemmaRecoveryBytes == bytes)
        }
        kv.discardGemmaRecovery()
        #expect(kv.gemmaRecoveryBytes == 0)
    }

    @Test func interruptedSuffixPrefillCannotReuseRestoredState() async throws {
        let h = try Harness()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let r = h.runner
        try await h.prefill(Self.tokens[..<128], at: 0)
        let expected = h.row()
        #expect(try r.captureGemmaPrefix())
        try await h.prefill(Self.tokens[128..<740], at: 128)
        #expect(try r.recoverGemmaPrefix(to: 128) == .snapshot)
        r.prefillWillEncodeLayer = { layer in
            if layer == 1 { withUnsafeCurrentTask { $0?.cancel() } }
        }
        struct Output: @unchecked Sendable { let buffer: MTLBuffer }
        let output = Output(buffer: h.logits)
        let interrupted = Task { @Sendable in
            _ = try await r.prefillChunked(tokens: Self.tokens[128..<160], startPosition: 128,
                outputMode: .logits, config: .production(chunkTokens: 64),
                into: output.buffer, onProgress: { _ in })
        }
        do { try await interrupted.value; Issue.record("suffix prefill must be interrupted") }
        catch is CancellationError {}
        #expect(r.gemmaRecoverablePrefix(upTo: 128) == 0)
        #expect(throws: (any Error).self) { try r.prepareForContinuation(expectedPosition: 128) }
        r.prefillWillEncodeLayer = nil
        r.reset()
        #expect(r.gemmaRecoveryBytes == 0)
        try await h.prefill(Self.tokens[..<128], at: 0)
        #expect(h.row() == expected)
    }

    @Test func qwenRecurrentStateNeverOptsIntoGemmaRewind() async throws {
        let directory = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let config = ArchConfig.qwen36Toy()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: config, streamingMode: .pread(slotCount: 8))
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 128,
            runtimeConfiguration: .init(expertCacheSlots: 8, prefillChunkTokens: 64, forceLogitsHead: true))
        let output = try #require(context.device.makeBuffer(length: config.vocabSize * 2, options: .storageModeShared))
        let tokens = (0..<33).map { Int32(4 + ($0 * 37) % (config.vocabSize - 4)) }
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 64), into: output, onProgress: { _ in })
        #expect(!runner.supportsGemmaPrefixRecovery)
        #expect(try runner.captureGemmaPrefix() == false)
        #expect(runner.gemmaRecoveryBytes == 0)
        #expect(runner.gemmaRecoverablePrefix(upTo: 16) == 0)
        #expect(throws: (any Error).self) { try runner.recoverGemmaPrefix(to: 16) }
        try runner.prepareForContinuation(expectedPosition: 33)
        #expect(runner.continuationPosition == 33)
    }
}
