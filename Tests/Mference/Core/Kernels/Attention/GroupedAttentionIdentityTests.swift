import Foundation
import Metal
import Testing
@testable import Mference

/// The grouped full-attention decode kernel must reproduce the per-head
/// split-KV kernel bit for bit: same chunks, same score reduction order, same
/// fused multiply-add contraction in the running output.
@Suite(.serialized)
struct GroupedAttentionIdentityTests {
    // The old contiguous reduction passes tolerance checks while changing scores.
    // One-key chunks expose scores directly as m; longer chunks pin recurrence state.
    /// Gemma's 512/16/2 and Qwen 3.6's 256/16/2 full attention.
    static let headDims: [Int] = [512, 256]

    @Test(arguments: headDims, [1, 3, 16, 17, 33, 65, 257, 16_384, 65_536])
    func exactOrder(headDim: Int, seqLen: Int) throws {
        try compare(seqLen: seqLen, pattern: "random", headDim: headDim)
    }

    @Test(arguments: headDims, ["cancellation", "lateMaximum", "equal", "wide"])
    func adversarial(headDim: Int, pattern: String) throws {
        try compare(seqLen: 257, pattern: pattern, headDim: headDim)
    }

    // Fusing weight * V instead of old * alpha changes component 3 by one ULP.
    @Test(arguments: headDims)
    func twoKeyRecurrence(headDim: Int) throws {
        try compare(seqLen: 2, pattern: "random", chunkLength: 2, headDim: headDim)
    }

    @Test(arguments: headDims, [UInt64(0), 1, 0x12345678, UInt64.max])
    func variedSeeds(headDim: Int, seed: UInt64) throws {
        try compare(seqLen: 257, pattern: "random", seed: seed, headDim: headDim)
    }

    /// QAT keeps MLX's three-pass arithmetic; its grouped kernels only move the
    /// eight heads of a K/V head into one threadgroup. 32 keys and fewer keep
    /// the source's column-group scores, so the boundary is covered on both sides.
    @Test(arguments: [1, 5, 32, 33, 64, 257, 4_097, 16_384, 65_536])
    func qatGroupedKernelsMatchTheSourceMapping(seqLen: Int) throws {
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx, gemmaQATMaxContext: 65_536)
        func buffer(_ count: Int) throws -> MTLBuffer {
            try #require(ctx.device.makeBuffer(length: count * 2, options: .storageModeShared))
        }
        let q = try buffer(8192), k = try buffer(seqLen * 1024), v = try buffer(seqLen * 1024)
        var state: UInt64 = 0x51A7 &+ UInt64(seqLen)
        for target in [q, k, v] {
            let count = target.length / 2
            let values = target.contents().bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let unit: Float = Float(Int32(truncatingIfNeeded: state >> 32)) / Float(Int32.max)
                values[i] = Float16(unit * 0.5)
            }
        }
        let outputs = try [buffer(8192), buffer(8192)]
        for index in 0..<2 {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            attention.encodeFull(commandBuffer: cb, q: q, k: k, v: v, out: outputs[index],
                                 headDim: 512, numQHeads: 16, numKVHeads: 2,
                                 seqLen: UInt32(seqLen), scale: 1, useGroupedVector: index == 1)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            try #require(cb.status == .completed)
        }
        #expect(memcmp(outputs[0].contents(), outputs[1].contents(), outputs[0].length) == 0,
                "QAT grouped attention differs at length=\(seqLen)")
    }

    @Test func geometrySelectsTheGroupedKernelOnlyForGemmaAndQwen36FullAttention() {
        let gemma = Attention.splitGeometry(headDim: 512, numQHeads: 16, numKVHeads: 2, seqLen: 4096,
                                            kvStart: 0, preferGQASWA: false, forceGroupedVector: true)
        #expect(gemma.useFullGroupedVectorPartial)
        #expect(gemma.partialThreadgroups == 2 * gemma.numChunks)
        let qwen = Attention.splitGeometry(headDim: 256, numQHeads: 16, numKVHeads: 2, seqLen: 4096,
                                           kvStart: 0, preferGQASWA: false, forceGroupedVector: true)
        #expect(qwen.useFullGroupedVectorPartial)
        #expect(qwen.partialThreadgroups == 2 * qwen.numChunks)
        #expect(qwen.numChunks == gemma.numChunks)
        let qwen38 = Attention.splitGeometry(headDim: 256, numQHeads: 24, numKVHeads: 4, seqLen: 4096,
                                             kvStart: 0, preferGQASWA: false, forceGroupedVector: true)
        #expect(!qwen38.useFullGroupedVectorPartial)
        let sliding = Attention.splitGeometry(headDim: 512, numQHeads: 16, numKVHeads: 2, seqLen: 4096,
                                              kvStart: 0, preferGQASWA: true, forceGroupedVector: true)
        #expect(!sliding.useFullGroupedVectorPartial)
    }

    private func compare(seqLen: Int, pattern: String, chunkLength: Int? = nil,
                         seed: UInt64 = 0x9E571, headDim: Int = 512) throws {
        let ctx = try MetalContext()
        func buffer(_ count: Int, _ stride: Int) throws -> MTLBuffer {
            try #require(ctx.device.makeBuffer(length: count * stride, options: .storageModeShared))
        }
        let q = try buffer(8192, 2)
        let k = try buffer(seqLen * 1024, 2)
        let v = try buffer(seqLen * 1024, 2)
        var state: UInt64 = seed
        func random() -> Float16 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit: Float = Float(Int32(truncatingIfNeeded: state >> 32)) / Float(Int32.max)
            return Float16(unit * 0.5)
        }
        let cancellation: [Float16] = [32, 0.03125, -32, -0.015625]
        let qp = q.contents().bindMemory(to: Float16.self, capacity: 8192)
        for i in 0..<8192 {
            switch pattern {
            case "wide": qp[i] = Float16(Float(random()) * 6)
            case "random": qp[i] = random()
            default: qp[i] = 1
            }
        }
        let kp = k.contents().bindMemory(to: Float16.self, capacity: seqLen * 1024)
        let vp = v.contents().bindMemory(to: Float16.self, capacity: seqLen * 1024)
        for i in 0..<(seqLen * 1024) {
            switch pattern {
            case "cancellation":
                kp[i] = cancellation[i % 4]
            case "lateMaximum":
                // Every chunk sees a new maximum after weights have underflowed.
                kp[i] = (i / (2 * headDim)) % 17 == 16 ? Float16(2) : Float16(-2)
            case "equal": kp[i] = 0.125
            case "wide": kp[i] = Float16(Float(random()) * 6)
            default: kp[i] = random()
            }
            vp[i] = random()
        }
        let constants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 60, value: .uint32(UInt32(headDim))),
            MetalFunctionConstant(index: 61, value: .uint32(16)),
            MetalFunctionConstant(index: 62, value: .uint32(2)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        let chunks16 = MetalFunctionConstant(index: 65, value: .uint32(16))
        let exact = try ctx.pipeline("attention_decode_partial", constants: constants + [chunks16])
        let grouped = try ctx.pipeline(headDim == 256
                                           ? "attention_decode_full_grouped_vec256_partial"
                                           : "attention_decode_full_grouped_vec_partial",
                                       constants: constants)
        let combine = try ctx.pipeline("attention_decode_combine", constants: constants + [chunks16])
        let chunk: Int = chunkLength ?? (seqLen + 15) / 16
        let dispatchValues: [Int] = [headDim, 16, 2, seqLen, 0, chunk, 16]
        func run(_ pso: MTLComputePipelineState, grouped: Bool) throws -> [MTLBuffer] {
            let outputs = try [buffer(256, 4), buffer(256, 4), buffer(256 * 512, 4), buffer(8192, 2)]
            let cb = try #require(ctx.queue.makeCommandBuffer())
            let enc = try #require(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            for (index, b) in ([q, k, v] + Array(outputs.prefix(3))).enumerated() {
                enc.setBuffer(b, offset: 0, index: index)
            }
            for (offset, value) in dispatchValues.enumerated() {
                var value = UInt32(value)
                enc.setBytes(&value, length: 4, index: 6 + offset)
            }
            var scale: Float = 1
            enc.setBytes(&scale, length: 4, index: 13)
            enc.dispatchThreadgroups(MTLSize(width: grouped ? 32 : 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            let merge = try #require(cb.makeComputeCommandEncoder())
            merge.setComputePipelineState(combine)
            for (index, b) in outputs.enumerated() { merge.setBuffer(b, offset: 0, index: index) }
            var hd = UInt32(headDim), nc: UInt32 = 16
            merge.setBytes(&hd, length: 4, index: 4)
            merge.setBytes(&nc, length: 4, index: 5)
            merge.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            merge.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            try #require(cb.status == .completed)
            return outputs
        }
        // The production entry point, with and without the grouped kernel.
        if seqLen >= 17, chunkLength == nil {
            let outputs = try [buffer(8192, 2), buffer(8192, 2)]
            for index in 0..<2 {
                let attention = try Attention(context: ctx)
                let cb = try #require(ctx.queue.makeCommandBuffer())
                attention.encodeFull(commandBuffer: cb, q: q, k: k, v: v, out: outputs[index],
                                     headDim: UInt32(headDim), numQHeads: 16, numKVHeads: 2,
                                     seqLen: UInt32(seqLen), scale: 1, useGroupedVector: index == 1)
                cb.commit()
                cb.waitUntilCompleted()
                if let error = cb.error { throw error }
                try #require(cb.status == .completed)
            }
            #expect(memcmp(outputs[0].contents(), outputs[1].contents(), outputs[0].length) == 0,
                    "runtime output mismatch length=\(seqLen) pattern=\(pattern) seed=\(seed) headDim=\(headDim)")
        }
        let names = ["m/score", "d", "o", "FP16"]
        let control = try run(exact, grouped: false)
        for repetition in 0..<2 {
            let candidate = try run(grouped, grouped: true)
            for index in 0..<4 {
                let stride = index == 3 ? 2 : 4
                let a = control[index].contents()
                let b = candidate[index].contents()
                var changed = 0
                var first = -1
                for element in 0..<(control[index].length / stride) {
                    let differs = stride == 4
                        ? a.load(fromByteOffset: element * 4, as: UInt32.self)
                            != b.load(fromByteOffset: element * 4, as: UInt32.self)
                        : a.load(fromByteOffset: element * 2, as: UInt16.self)
                            != b.load(fromByteOffset: element * 2, as: UInt16.self)
                    if differs {
                        changed += 1
                        if first < 0 { first = element }
                    }
                }
                #expect(changed == 0, """
                    headDim=\(headDim) length=\(seqLen) pattern=\(pattern) seed=\(seed) repetition=\(repetition) \
                    state=\(names[index]) first=\(first) changed=\(changed)
                    """)
            }
        }
    }
}
