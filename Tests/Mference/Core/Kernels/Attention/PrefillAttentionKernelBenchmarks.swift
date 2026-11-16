import Foundation
import Metal
import Testing
@testable import Mference

/// GPU time of the tiled and GQA-shared prefill attention kernels at Qwen 3.6's
/// full-attention shape (headDim 256, 16/2 heads, KV stride 512, Q/O stride
/// 4096) on synthetic data. Env-gated: `MFERENCE_PREFILL_ATTN_KBENCH=1`.
/// Prints one line per (shape, kernel); asserts nothing about speed.
@Suite(.serialized) struct PrefillAttentionKernelBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_ATTN_KBENCH"] == "1"
    }

    private static func halfBuffer(_ device: MTLDevice, _ count: Int, salt: Int) -> MTLBuffer {
        let buffer = device.makeBuffer(length: max(1, count) * 2, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { pointer[i] = Float16(Float((i &* 31 &+ salt) % 97) / 97 * 0.7 - 0.35) }
        return buffer
    }

    @Test(.enabled(if: enabled),
          arguments: [(0, 2940), (28672, 512), (28672, 4096)])
    func qwenFullAttention(_ shape: (start: Int, chunk: Int)) throws {
        let ctx = try MetalContext()
        let kvValid = shape.start + shape.chunk
        let q = Self.halfBuffer(ctx.device, shape.chunk * 4096, salt: 1)
        let k = Self.halfBuffer(ctx.device, kvValid * 512, salt: 2)
        let v = Self.halfBuffer(ctx.device, kvValid * 512, salt: 3)
        let out = ctx.device.makeBuffer(length: shape.chunk * 4096 * 2, options: .storageModeShared)!
        let params = PrefillAttentionParams(
            startPosition: UInt32(shape.start), queryCount: UInt32(shape.chunk),
            headDim: 256, numQHeads: 16, numKVHeads: 2,
            kvValidCount: UInt32(kvValid), slidingWindow: UInt32(kvValid),
            kvTokenStrideElements: 512, qTokenStrideElements: 4096,
            oTokenStrideElements: 4096, scale: 0.0625)
        for (label, gqa) in [("tiled", false), ("gqa-simd", true)] {
            let attention = try PrefillAttention(context: ctx, gqaSimdgroup: gqa)
            var best = Double.infinity
            for _ in 0..<2 {
                let cb = ctx.queue.makeCommandBuffer()!
                attention.encodeCausal(commandBuffer: cb, q: q, k: k, v: v, out: out, params: params)
                cb.commit()
                cb.waitUntilCompleted()
                #expect(cb.status == .completed)
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            print(String(format: "[attn-kbench] start=%d chunk=%d %@ %.1f ms",
                         shape.start, shape.chunk, label, best * 1000))
        }
    }
}
