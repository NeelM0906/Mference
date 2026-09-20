import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATKernelTests {
    @Test func sourceFP16EmbeddingAndNormalizationBoundaries() throws {
        let context = try MetalContext()
        let rms = try RMSNorm(context: context, sourceFP16: true)
        let prefillRMS = try PrefillRMSNorm(context: context, sourceFP16: true)
        for d in [256, 512, 2816] {
            let input = (0..<d).map { Float16(Float(($0 * 11) % 97 - 48) / 64) }
            let weights = (0..<d).map { Quantization.bf16Bits(Float(($0 % 11) + 2) / 8) }
            let sum = input.reduce(0.0) { $0 + Double($1) * Double($1) }
            let inv = 1 / sqrt(sum / Double(d) + 1e-6)
            // MLX 0.32.2 rms_norm.metal casts x*inv to the activation type
            // before multiplying by the learned weight. This is independent
            // of whether that weight is stored as BF16 on disk.
            let expected = zip(input, weights).map { x, w in
                Float16(Double(Float16(Double(x) * inv)) * Double(Quantization.bf16ToFloat(w)))
            }
            let fused = zip(input, weights).map { x, w in
                Float16(Double(x) * inv * Double(Quantization.bf16ToFloat(w)))
            }
            #expect(expected != fused, "fixture must detect the missing FP16 boundary")
            let x = try Self.buffer(input, context: context)
            let w = try Self.buffer(weights, context: context)
            for batched in [false, true] {
                let out = try Self.buffer([Float16](repeating: .nan, count: d), context: context)
                let cb = try #require(context.queue.makeCommandBuffer())
                if batched {
                    prefillRMS.encodeBF16W(commandBuffer: cb, x: x, weight: w, out: out,
                                           t: 1, d: UInt32(d), eps: 1e-6)
                } else {
                    rms.encodeBF16W(commandBuffer: cb, x: x, weight: w, out: out,
                                   d: UInt32(d), eps: 1e-6)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                let actual = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float16.self), count: d))
                #expect(actual == expected, "source norm D=\(d), batched=\(batched)")
            }
        }
        let d = 2816
        let matrix = Matrix(rows: 2, columns: d, groupSize: 32)
        let table = try Self.buffer(matrix.packed, context: context)
        let scales = try Self.buffer(matrix.scales, context: context)
        let biases = try Self.buffer(matrix.biases, context: context)
        let tokens = try Self.buffer([UInt32(1)], context: context)
        let scale = Float(d).squareRoot()
        let expected = (0..<d).map { Float16(matrix.value(row: 1, column: $0)) * Float16(scale) }
        let fused = (0..<d).map { Float16(matrix.value(row: 1, column: $0) * Double(scale)) }
        #expect(expected != fused, "fixture must detect source embedding scale precision")
        let embed = try EmbedLookupInt4(context: context, groupSize: 32, sourceFP16: true)
        let prefillEmbed = try PrefillEmbedLookupInt4(context: context, groupSize: 32, sourceFP16: true)
        for batched in [false, true] {
            let out = try Self.buffer([Float16](repeating: .nan, count: d), context: context)
            let cb = try #require(context.queue.makeCommandBuffer())
            if batched {
                prefillEmbed.encode(commandBuffer: cb, table: table, scales: scales, biases: biases,
                    tokens: tokens, out: out, t: 1, d: UInt32(d), outScale: scale)
            } else {
                embed.encode(commandBuffer: cb, table: table, scales: scales, biases: biases,
                    out: out, tokenId: 1, d: UInt32(d), outScale: scale)
            }
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            let actual = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float16.self), count: d))
            #expect(actual == expected, "source embedding batched=\(batched)")
        }
    }
}
