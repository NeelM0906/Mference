import Foundation
import Metal
import Testing
@testable import Mference

/// GPU time of Qwen 3.6's decode full-attention passes
/// (`attention_decode_full_grouped_vec256_partial` + `attention_decode_combine`,
/// headDim 256, 16/2 heads) on synthetic data, at several context lengths and
/// split-KV chunk counts. Ten layers, each with its own K/V, run in one command
/// buffer, as in one decode token. Env-gated: `MFERENCE_DECODE_ATTN_KBENCH=1`.
/// Prints one line per (seqLen, chunks); asserts nothing about speed.
@Suite(.serialized) struct DecodeAttentionKernelBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_DECODE_ATTN_KBENCH"] == "1"
    }

    private static let headDim = 256
    private static let numQHeads = 16
    private static let numKVHeads = 2
    private static let layers = 10
    private static let reps = 7

    private static func halfBuffer(_ device: MTLDevice, _ count: Int, salt: Int) -> MTLBuffer {
        let buffer = device.makeBuffer(length: max(1, count) * 2, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { pointer[i] = Float16(Float((i &* 31 &+ salt) % 97) / 97 * 0.7 - 0.35) }
        return buffer
    }

    private static func pipeline(_ ctx: MetalContext, _ name: String,
                                 numChunks: Int?) throws -> MTLComputePipelineState {
        var constants = [
            MetalFunctionConstant(index: 60, value: .uint32(UInt32(headDim))),
            MetalFunctionConstant(index: 61, value: .uint32(UInt32(numQHeads))),
            MetalFunctionConstant(index: 62, value: .uint32(UInt32(numKVHeads))),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        if let numChunks {
            constants.append(MetalFunctionConstant(index: 65, value: .uint32(UInt32(numChunks))))
        }
        return try ctx.pipeline(name, constants: constants)
    }

    @Test(.enabled(if: enabled), arguments: [4096, 16384, 32768])
    func qwenGroupedFullDecode(seqLen: Int) throws {
        let ctx = try MetalContext()
        let device = ctx.device
        let rowElems = Self.numKVHeads * Self.headDim
        let q = Self.halfBuffer(device, Self.numQHeads * Self.headDim, salt: 1)
        let ks = (0..<Self.layers).map { Self.halfBuffer(device, seqLen * rowElems, salt: 2 + $0) }
        let vs = (0..<Self.layers).map { Self.halfBuffer(device, seqLen * rowElems, salt: 40 + $0) }
        let maxChunks = 64
        let partialElems = Self.numQHeads * maxChunks
        let m = device.makeBuffer(length: partialElems * 4, options: .storageModeShared)!
        let d = device.makeBuffer(length: partialElems * 4, options: .storageModeShared)!
        let o = device.makeBuffer(length: partialElems * Self.headDim * 4, options: .storageModeShared)!
        let outs = (0..<Self.layers).map { _ in
            device.makeBuffer(length: Self.numQHeads * Self.headDim * 2, options: .storageModeShared)!
        }
        // Production: the partial pass is specialized on the shape only; the
        // combine is specialized on 16 chunks when there are 16.
        let partialPSO = try Self.pipeline(ctx, "attention_decode_full_grouped_vec256_partial",
                                           numChunks: nil)
        let kvBytesPerToken = Double(Self.layers * 2 * seqLen * rowElems * 2)
        var reference: [Float16]?

        for chunks in [16, 32, 64] {
            let combinePSO = try Self.pipeline(ctx, "attention_decode_combine",
                                               numChunks: chunks == 16 ? 16 : nil)
            let chunkLen = (seqLen + chunks - 1) / chunks
            var times: [Double] = []
            for _ in 0..<Self.reps {
                let cb = ctx.queue.makeCommandBuffer()!
                for layer in 0..<Self.layers {
                    let p1 = cb.makeComputeCommandEncoder()!
                    p1.setComputePipelineState(partialPSO)
                    p1.setBuffer(q, offset: 0, index: 0)
                    p1.setBuffer(ks[layer], offset: 0, index: 1)
                    p1.setBuffer(vs[layer], offset: 0, index: 2)
                    p1.setBuffer(m, offset: 0, index: 3)
                    p1.setBuffer(d, offset: 0, index: 4)
                    p1.setBuffer(o, offset: 0, index: 5)
                    var hd = UInt32(Self.headDim), nq = UInt32(Self.numQHeads)
                    var nkv = UInt32(Self.numKVHeads), sl = UInt32(seqLen), kst = UInt32(0)
                    var cl = UInt32(chunkLen), nc = UInt32(chunks), sc = Float(0.0625)
                    p1.setBytes(&hd, length: 4, index: 6)
                    p1.setBytes(&nq, length: 4, index: 7)
                    p1.setBytes(&nkv, length: 4, index: 8)
                    p1.setBytes(&sl, length: 4, index: 9)
                    p1.setBytes(&kst, length: 4, index: 10)
                    p1.setBytes(&cl, length: 4, index: 11)
                    p1.setBytes(&nc, length: 4, index: 12)
                    p1.setBytes(&sc, length: 4, index: 13)
                    p1.dispatchThreadgroups(MTLSize(width: Self.numKVHeads * chunks, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    p1.endEncoding()

                    let p2 = cb.makeComputeCommandEncoder()!
                    p2.setComputePipelineState(combinePSO)
                    p2.setBuffer(m, offset: 0, index: 0)
                    p2.setBuffer(d, offset: 0, index: 1)
                    p2.setBuffer(o, offset: 0, index: 2)
                    p2.setBuffer(outs[layer], offset: 0, index: 3)
                    var hd2 = UInt32(Self.headDim), nc2 = UInt32(chunks)
                    p2.setBytes(&hd2, length: 4, index: 4)
                    p2.setBytes(&nc2, length: 4, index: 5)
                    let w = min(256, combinePSO.maxTotalThreadsPerThreadgroup)
                    p2.dispatchThreadgroups(MTLSize(width: Self.numQHeads, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
                    p2.endEncoding()
                }
                cb.commit()
                cb.waitUntilCompleted()
                #expect(cb.status == .completed)
                times.append(cb.gpuEndTime - cb.gpuStartTime)
            }
            times.sort()
            let best = times[0], median = times[times.count / 2]
            // Last layer's output against the 16-chunk result: chunking only
            // changes partial-sum rounding.
            let n = Self.numQHeads * Self.headDim
            let row = Array(UnsafeBufferPointer(
                start: outs[Self.layers - 1].contents().bindMemory(to: Float16.self, capacity: n),
                count: n))
            var maxDiff: Float = 0
            if let reference {
                for i in 0..<n { maxDiff = max(maxDiff, abs(Float(row[i]) - Float(reference[i]))) }
            } else {
                reference = row
            }
            print(String(format: "[decode-attn-kbench] seq=%d chunks=%d tgs=%d chunkLen=%d "
                         + "10 layers best %.3f ms median %.3f ms; per layer %.3f ms; "
                         + "KV %.0f MB/token, %.1f GB/s; max|d| vs 16 = %g",
                         seqLen, chunks, Self.numKVHeads * chunks, chunkLen,
                         best * 1000, median * 1000, best * 1000 / Double(Self.layers),
                         kvBytesPerToken / 1e6, kvBytesPerToken / best / 1e9, Double(maxDiff)))
        }
    }
}
