import Foundation
import Metal
import Testing
@testable import Mference

/// WS4 U4: the GGUF install's chunked prefill (Q6_K chunk gather, batched
/// INT8-2L projections, LUT4 routed tiles) against the same prompt run as
/// decode steps, on the real install. Env-gated: set `MFERENCE_UNSLOTH_GTURBO`
/// to the `qwen38flashnextudiq3xxs.gturbo` directory. A model run: one at a
/// time, under the program lock.
///
/// The two paths reduce in different orders (the LUT4 prefill row form and
/// batched GEMMs vs the decode kernels), so they agree to rounding, not bit
/// for bit. The gate is the decision (argmax) and a bound at the logits' scale,
/// for the prompt's last row and for the first decode step after it.
@Suite struct FlashNextGGUFRealPrefillTests {
    static var install: URL? {
        ProcessInfo.processInfo.environment["MFERENCE_UNSLOTH_GTURBO"].map { URL(fileURLWithPath: $0) }
    }

    static func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float16.self, capacity: count),
                                  count: count)).map(Float.init)
    }

    static func argmax(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] }! }

    static func compare(_ label: String, _ chunked: [Float], _ sequential: [Float]) {
        #expect(chunked.allSatisfy(\.isFinite) && sequential.allSatisfy(\.isFinite), "\(label): non-finite logits")
        let error = zip(chunked, sequential).map { abs($0 - $1) }.max() ?? .infinity
        let scale = sequential.map(abs).max() ?? 0
        print("GGUF \(label): max |chunked - sequential| \(error) at logit scale \(scale); "
              + "argmax chunked \(argmax(chunked)) sequential \(argmax(sequential))")
        #expect(argmax(chunked) == argmax(sequential), "\(label): decisions differ")
        #expect(error <= scale / 32, "\(label): \(error) exceeds scale/32 = \(scale / 32)")
    }

    @Test(.enabled(if: FlashNextGGUFRealPrefillTests.install != nil, "set MFERENCE_UNSLOTH_GTURBO"))
    func chunkedPrefillMatchesDecodeSteps() async throws {
        let dir = try #require(Self.install)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: context.device,
                                   streamingMode: .pread(slotCount: 16))
        #expect(model.config.family == .qwen38flashnextudiq3xxs)
        let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: 128)
        let vocab = model.config.vocabSize
        let logits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        // 40 ordinary token ids: two chunks at 32 (32 + 8), so the second
        // chunk's PLE history, GDN state, KV and indexer blocks carry over.
        let prompt: [Int32] = (0..<40).map { Int32(1_000 + ($0 * 7_919) % 150_000) }
        let next: Int32 = 13

        for (p, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: p, into: logits)
        }
        let sequentialPrompt = Self.floats(logits, count: vocab)
        try await runner.produce(token: next, position: prompt.count, into: logits)
        let sequentialNext = Self.floats(logits, count: vocab)

        runner.reset()
        let result = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits) { _ in }
        #expect(result.newPosition == prompt.count && result.seed == .logitsWritten)
        #expect(result.execution?.replayedTokens ?? 0 == 0, "prefill fell back to decode steps")
        let chunkedPrompt = Self.floats(logits, count: vocab)
        try await runner.produce(token: next, position: prompt.count, into: logits)
        let chunkedNext = Self.floats(logits, count: vocab)

        Self.compare("prompt row", chunkedPrompt, sequentialPrompt)
        Self.compare("next decode step", chunkedNext, sequentialNext)
    }
}
