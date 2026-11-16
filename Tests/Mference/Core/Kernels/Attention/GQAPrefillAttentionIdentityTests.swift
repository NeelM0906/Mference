import Foundation
import Metal
import Testing
@testable import Mference

/// The staged GQA prefill attention kernel (`attention_prefill_gqa8_d256_simd`,
/// 8 tokens per threadgroup sharing staged K/V blocks) against the unstaged
/// P3a form it replaced (`GQAPrefillAttentionUnstagedReference`): byte for byte
/// at Qwen 3.6's full-attention shape (headDim 256, 16/2 heads, KV stride 512,
/// Q/O stride 4096), including token counts that leave a partial threadgroup.
@Suite(.serialized) struct GQAPrefillAttentionIdentityTests {
    struct Chunk: CustomStringConvertible {
        let start: Int
        let count: Int
        var description: String { "start=\(start) count=\(count)" }
    }

    static func params(_ c: Chunk) -> PrefillAttentionParams {
        PrefillAttentionParams(
            startPosition: UInt32(c.start), queryCount: UInt32(c.count),
            headDim: 256, numQHeads: 16, numKVHeads: 2,
            kvValidCount: UInt32(c.start + c.count), slidingWindow: UInt32(c.start + c.count),
            kvTokenStrideElements: 512, qTokenStrideElements: 4096,
            oTokenStrideElements: 4096, scale: 0.0625)
    }

    /// Seeded values in [-1.5, 1.5): scores spread over several units, so the
    /// online softmax rescales often and outputs are far from zero.
    static func halfBuffer(_ device: MTLDevice, _ count: Int, seed: UInt64) -> MTLBuffer {
        let buffer = device.makeBuffer(length: max(1, count) * 2, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        var state = seed
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            pointer[i] = Float16(3 * (Float(state >> 40) / Float(1 << 24) - 0.5))
        }
        return buffer
    }

    struct Run {
        let mismatchedChunks: [String]
        let maxAbsOutput: Float
        let stagedSeconds: Double
        let unstagedSeconds: Double
    }

    /// Runs every chunk through both kernels on the same Q/K/V.
    static func compare(_ chunks: [Chunk], timed: Bool = false) throws -> Run {
        let ctx = try MetalContext()
        let device = ctx.device
        let maxKV = chunks.map { $0.start + $0.count }.max()!
        let maxQ = chunks.map(\.count).max()!
        let q = halfBuffer(device, maxQ * 4096, seed: 11)
        let k = halfBuffer(device, maxKV * 512, seed: 12)
        let v = halfBuffer(device, maxKV * 512, seed: 13)
        let staged = try PrefillAttention(context: ctx, gqaSimdgroup: true)
        let unstaged = try PrefillAttention(context: ctx, gqaSimdgroup: true)
        unstaged.useGQAPipelineForTesting(try GQAPrefillAttentionUnstagedReference.pipeline(device: device),
                                          tokensPerGroup: GQAPrefillAttentionUnstagedReference.tokensPerGroup)
        let a = try #require(device.makeBuffer(length: maxQ * 4096 * 2, options: .storageModeShared))
        let b = try #require(device.makeBuffer(length: maxQ * 4096 * 2, options: .storageModeShared))
        var mismatched: [String] = []
        var maxAbs: Float = 0
        var stagedSeconds = 0.0, unstagedSeconds = 0.0
        for c in chunks {
            memset(a.contents(), 0x7F, maxQ * 4096 * 2)
            memset(b.contents(), 0x3C, maxQ * 4096 * 2)
            for (attention, out) in [(staged, a), (unstaged, b)] {
                let before = attention.gqaSimdDispatches
                let cb = try #require(ctx.queue.makeCommandBuffer())
                attention.encodeCausal(commandBuffer: cb, q: q, k: k, v: v, out: out, params: Self.params(c))
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil, "\(String(describing: cb.error))")
                #expect(attention.gqaSimdDispatches == before + 1, "\(c) did not run the GQA kernel")
                if timed {
                    if attention === staged { stagedSeconds += cb.gpuEndTime - cb.gpuStartTime }
                    else { unstagedSeconds += cb.gpuEndTime - cb.gpuStartTime }
                }
            }
            if memcmp(a.contents(), b.contents(), c.count * 4096 * 2) != 0 { mismatched.append(c.description) }
            let out = a.contents().bindMemory(to: Float16.self, capacity: c.count * 4096)
            for i in 0..<(c.count * 4096) { maxAbs = max(maxAbs, abs(Float(out[i]))) }
        }
        return Run(mismatchedChunks: mismatched, maxAbsOutput: maxAbs,
                   stagedSeconds: stagedSeconds, unstagedSeconds: unstagedSeconds)
    }

    /// Small shapes, always run: one token, partial threadgroups (counts not
    /// divisible by 8), a block-straddling start, and a full 64-token chunk.
    @Test func stagedMatchesUnstagedBitForBitAtSmallShapes() throws {
        let run = try Self.compare([Chunk(start: 0, count: 1), Chunk(start: 0, count: 9),
                                    Chunk(start: 37, count: 70), Chunk(start: 200, count: 33),
                                    Chunk(start: 5, count: 64), Chunk(start: 1_000, count: 37)])
        #expect(run.maxAbsOutput > 0.1, "outputs too small to compare: \(run.maxAbsOutput)")
        #expect(run.mismatchedChunks.isEmpty, "differs at \(run.mismatchedChunks)")
    }

    private static var fullShapesEnabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_ATTN_B"] == "1"
    }

    /// Production shapes, env-gated (`MFERENCE_PREFILL_ATTN_B=1`): every
    /// 4,096-token chunk of the 31,480-token cold prompt (first and last chunk
    /// edges included), WS3's warm follow-up (1,446 queries at a 31,480-key
    /// offset), the 2,940-token prompt, and 512- and 64-token prompts. Prints
    /// GPU time; asserts bit equality.
    @Test(.enabled(if: fullShapesEnabled), arguments: ["cold32k", "warm1446", "3k", "512", "64"])
    func stagedMatchesUnstagedBitForBitAtProductionShapes(shape: String) throws {
        let chunks: [Chunk]
        switch shape {
        case "cold32k": chunks = stride(from: 0, to: 31_480, by: 4096).map { Chunk(start: $0, count: min(4096, 31_480 - $0)) }
        case "warm1446": chunks = [Chunk(start: 31_480, count: 1_446)]
        case "3k": chunks = [Chunk(start: 0, count: 2_940)]
        case "512": chunks = [Chunk(start: 0, count: 512)]
        default: chunks = [Chunk(start: 0, count: 64)]
        }
        let run = try Self.compare(chunks, timed: true)
        print(String(format: "[attn-b] %@ chunks=%d unstaged %.1f ms staged %.1f ms (%.2fx) identical=%@ max|out|=%.3f",
                     shape, chunks.count, run.unstagedSeconds * 1000, run.stagedSeconds * 1000,
                     run.unstagedSeconds / run.stagedSeconds,
                     run.mismatchedChunks.isEmpty ? "yes" : "NO", run.maxAbsOutput))
        // Softmax over ~31K keys averages the values down to a few 1e-2;
        // still far above FP16 resolution, so equality is not vacuous.
        #expect(run.maxAbsOutput > 0.01, "outputs too small to compare: \(run.maxAbsOutput)")
        #expect(run.mismatchedChunks.isEmpty, "differs at \(run.mismatchedChunks)")
    }
}
