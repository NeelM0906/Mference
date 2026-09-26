import Foundation
import Metal
import Testing
@testable import Mference

/// Full-attention KV that starts at one growth step and grows with the
/// conversation: capacity becomes what is needed plus one more step over the
/// same reserved pages, so rows stay in place and nothing is copied, and a new
/// conversation starts again from one step.
@Suite struct KVCacheGrowthTests {
    private let gemma = ArchConfig.gemma4_26B_A4B
    private let qwen = ArchConfig.qwen36_35B_A3B

    private func make(_ config: ArchConfig, maxContext: Int, step: Int?) throws -> KVCacheManager {
        try KVCacheManager(device: MetalContext().device, config: config, maxContext: maxContext,
                           fp16RingEnabled: true, slidingWindow: config.slidingWindow,
                           maxPrefillChunkTokens: 128, fullAttentionGrowthStep: step)
    }

    private func fullLayer(_ config: ArchConfig) throws -> Int {
        try #require(config.fullAttentionLayerMask.firstIndex(of: 1))
    }

    @Test func fullLayersStartAtOneStepAndSlidingRingsKeepTheirSize() throws {
        let kv = try make(gemma, maxContext: 262_144, step: 16_384)
        let full = try fullLayer(gemma)
        let sliding = try #require(gemma.fullAttentionLayerMask.firstIndex(of: 0))
        #expect(kv.capacity(layer: full) == 16_384)
        #expect(kv.capacity(layer: sliding) == gemma.slidingWindow + 128)
        let whole = try make(gemma, maxContext: 262_144, step: nil)
        #expect(whole.capacity(layer: full) == 262_144)
    }

    @Test func aContextWithinOneStepIsAllocatedWhole() throws {
        let kv = try make(gemma, maxContext: 8_192, step: 16_384)
        #expect(kv.capacity(layer: try fullLayer(gemma)) == 8_192)
        #expect(try !kv.ensureCapacity(for: 8_192))
    }

    @Test func growthAddsOneStepAndStopsAtMaxContext() throws {
        let kv = try make(qwen, maxContext: 40_000, step: 16_384)
        let full = try fullLayer(qwen)
        #expect(try !kv.ensureCapacity(for: 16_384))
        #expect(try kv.ensureCapacity(for: 16_385))
        #expect(kv.capacity(layer: full) == 16_385 + 16_384)
        #expect(try kv.ensureCapacity(for: 33_000))
        #expect(kv.capacity(layer: full) == 40_000)
    }

    /// Decode grows by the smaller headroom it passes, until the context is full.
    @Test func decodeHeadroomGrowsInSmallerStepsUpToMaxContext() throws {
        let kv = try make(qwen, maxContext: 40_000, step: 16_384)
        let full = try fullLayer(qwen)
        #expect(try kv.ensureCapacity(for: 16_385, headroom: 8_192))
        #expect(kv.capacity(layer: full) == 16_385 + 8_192)
        #expect(try !kv.ensureCapacity(for: 24_577, headroom: 8_192))
        #expect(try kv.ensureCapacity(for: 24_578, headroom: 8_192))
        #expect(kv.capacity(layer: full) == 24_578 + 8_192)
        #expect(try kv.ensureCapacity(for: 39_000, headroom: 8_192))
        #expect(kv.capacity(layer: full) == 40_000)
    }

    @Test func rowsWrittenBeforeGrowthSurviveIt() throws {
        let kv = try make(qwen, maxContext: 4_096, step: 64)
        let layer = try fullLayer(qwen)
        let stride = kv.stride(layer: layer)
        func byte(_ slot: (buffer: MTLBuffer, offset: Int), _ index: Int) -> UInt8 {
            slot.buffer.contents().load(fromByteOffset: slot.offset + index, as: UInt8.self)
        }
        for position in 0..<64 {
            let k = kv.kSlot(layer: layer, position: position)
            let v = kv.vSlot(layer: layer, position: position)
            memset(k.buffer.contents() + k.offset, Int32(position % 251), stride)
            memset(v.buffer.contents() + v.offset, Int32((position + 97) % 251), stride)
            kv.advance()
        }
        let before = kv.keyBuffer(layer: layer, validTokenCount: 64).contents()
        #expect(try kv.ensureCapacity(for: 65))
        let grown = kv.keyBuffer(layer: layer, validTokenCount: 64)
        #expect(kv.capacity(layer: layer) == 65 + 64)
        #expect(grown.length >= (65 + 64) * stride)
        // Growing re-wraps the same pages: nothing is copied and no second
        // buffer holds the rows.
        #expect(grown.contents() == before)
        for position in 0..<64 {
            let k = kv.kSlot(layer: layer, position: position)
            let v = kv.vSlot(layer: layer, position: position)
            for index in [0, stride - 1] {
                #expect(byte(k, index) == UInt8(position % 251), "K row \(position)")
                #expect(byte(v, index) == UInt8((position + 97) % 251), "V row \(position)")
            }
        }
    }

    @Test func resetShrinksBackToOneStep() throws {
        let kv = try make(qwen, maxContext: 4_096, step: 64)
        let full = try fullLayer(qwen)
        let original = kv.keyBuffer(layer: full, validTokenCount: 0).contents()
        _ = try kv.ensureCapacity(for: 1_000)
        let grown = kv.diagnosticBufferBytes
        kv.reset()
        #expect(kv.capacity(layer: full) == 64)
        #expect(kv.diagnosticBufferBytes < grown)
        // A fresh region: the grown one is unmapped once its wrapper goes.
        #expect(kv.keyBuffer(layer: full, validTokenCount: 0).contents() != original)
    }
}

/// A runner whose KV grows produces exactly the logits of one that reserved
/// the whole context (`--kv-reserve`): growth during prefill, during decode,
/// and again after reset.
@Suite(.serialized) struct KVGrowthRunnerParityTests {
    private static let tokens: [Int32] = (0..<400).map { Int32(4 + ($0 * 37 + 11) % 1000) }

    /// The logits of every step, and the KV the runner started with.
    private func rollout(directory: URL, config: ArchConfig, step: Int?, maxContext: Int,
                         chunk: Int, prompt: Int, decode: Int,
                         tokens: [Int32] = Self.tokens) async throws -> (rows: [[UInt16]], startingKV: UInt64) {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: config, streamingMode: .pread(slotCount: 8))
        let runner = try RealForwardRunner(model: model, context: context, maxContext: maxContext,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8, prefillChunkTokens: chunk,
                                                       forceLogitsHead: true, kvGrowthTokens: step))
        let startingKV = try #require(runner.diagnosticKVStateBytes)
        let logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2,
                                                            options: .storageModeShared))
        func row() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self),
                                      count: config.vocabSize))
        }
        var rows: [[UInt16]] = []
        for _ in 0..<2 {  // the second pass runs after reset(), from the shrunk cache
            runner.reset()
            _ = try await runner.prefillChunked(tokens: tokens[..<prompt], startPosition: 0,
                                                outputMode: .logits, config: .production(chunkTokens: chunk),
                                                into: logits, onProgress: { _ in })
            rows.append(row())
            for position in prompt..<(prompt + decode) {
                try await runner.produce(token: tokens[position], position: position, into: logits)
                rows.append(row())
            }
        }
        return (rows, startingKV)
    }

    @Test func gemmaGrowingKVMatchesTheWholeReservation() async throws {
        let config = ArchConfig.gemma4Toy(topKExperts: 8)
        let directory = try ModelLoaderTests.writeToySynthetic(config: config, finiteNorms: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        func run(_ step: Int?) async throws -> (rows: [[UInt16]], startingKV: UInt64) {
            try await rollout(directory: directory, config: config, step: step, maxContext: 768,
                              chunk: 64, prompt: 200, decode: 60)
        }
        let whole = try await run(nil)
        let growing = try await run(48)
        #expect(growing.startingKV < whole.startingKV)
        #expect(growing.rows == whole.rows)
    }

    @Test func qwenGrowingKVMatchesTheWholeReservation() async throws {
        let directory = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        func run(_ step: Int?) async throws -> (rows: [[UInt16]], startingKV: UInt64) {
            try await rollout(directory: directory, config: .qwen36Toy(), step: step, maxContext: 256,
                              chunk: 32, prompt: 40, decode: 60)
        }
        let whole = try await run(nil)
        let growing = try await run(16)
        #expect(growing.startingKV < whole.startingKV)
        #expect(growing.rows == whole.rows)
    }

    /// Inkling's global layers grow; its local layers keep their rings.
    @Test func inklingGrowingKVMatchesTheWholeReservation() async throws {
        let directory = try InklingToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokens: [Int32] = (0..<400).map { Int32(4 + ($0 * 17) % 239) }
        func run(_ step: Int?) async throws -> (rows: [[UInt16]], startingKV: UInt64) {
            try await rollout(directory: directory, config: InklingToySynthetic.config, step: step,
                              maxContext: 256, chunk: 32, prompt: 70, decode: 60, tokens: tokens)
        }
        let whole = try await run(nil)
        let growing = try await run(16)
        #expect(growing.startingKV < whole.startingKV)
        #expect(growing.rows == whole.rows)
    }
}
