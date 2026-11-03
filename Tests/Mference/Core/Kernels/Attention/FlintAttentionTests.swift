import Foundation
import Metal
import Testing
@testable import Mference

@Suite
struct FlintAttentionTests {
    @Test func sinkAndSlidingRingMatchReference() throws {
        let context = try MetalContext()
        let attention = try FlintAttention(context: context)
        let qHeads = FlintAttention.numQHeads
        let kvHeads = FlintAttention.numKVHeads
        let headDim = FlintAttention.headDim
        let ring = 3

        let q = (0..<(qHeads * headDim)).map {
            Float16(Float(($0 * 7) % 29 - 14) / 32)
        }
        var k = [Float16](repeating: 0, count: ring * kvHeads * headDim)
        var v = [Float16](repeating: 0, count: ring * kvHeads * headDim)
        for logical in 2..<5 {
            let slot = logical % ring
            for index in 0..<(kvHeads * headDim) {
                k[slot * kvHeads * headDim + index] =
                    Float16(Float((logical * 11 + index * 3) % 31 - 15) / 32)
                v[slot * kvHeads * headDim + index] =
                    Float16(Float((logical * 17 + index * 5) % 37 - 18) / 32)
            }
        }
        let sinks = (0..<qHeads).map {
            Quantization.bf16Bits(Float(($0 * 5) % 13 - 6) / 16)
        }

        let qBuffer = try buffer(context, q)
        let kBuffer = try buffer(context, k)
        let vBuffer = try buffer(context, v)
        let sinkStorage = sinks + [0x7BAD, 0x7BAD]
        let sinkBuffer = try buffer(context, sinkStorage)
        let outStorage = [Float16](repeating: 99, count: 3 + qHeads * headDim)
        let outBuffer = try buffer(context, outStorage)

        let cb = try #require(context.queue.makeCommandBuffer())
        attention.encode(
            commandBuffer: cb,
            q: qBuffer,
            k: kBuffer,
            v: vBuffer,
            sinks: sinkBuffer,
            out: outBuffer,
            outOffset: 3 * MemoryLayout<Float16>.stride,
            start: 2,
            end: 5,
            ringCapacity: 3,
            scale: Float(1.0 / Double(headDim).squareRoot()))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let actual = output(from: outBuffer, count: outStorage.count)
        #expect(actual.prefix(3).allSatisfy { $0 == 99 })
        let expected = reference(
            q: q, k: k, v: v, sinks: sinks,
            first: 2, last: 5, ring: ring,
            scale: Float(1.0 / Double(headDim).squareRoot()))
        var maxDifference: Float = 0
        for i in 0..<expected.count {
            maxDifference = max(
                maxDifference,
                abs(Float(actual[3 + i]) - expected[i]))
        }
        #expect(maxDifference <= 0.005,
                "Flint sink attention max abs diff \(maxDifference)")
    }

    @Test func pagedSinkAttentionMatchesReference() throws {
        let context = try MetalContext()
        let pagedAttention = try Attention(context: context)
        let qHeads = FlintAttention.numQHeads
        let kvHeads = FlintAttention.numKVHeads
        let headDim = FlintAttention.headDim
        let tokenCount = 3

        let q = (0..<(qHeads * headDim)).map {
            Float16(Float(($0 * 5) % 23 - 11) / 32)
        }
        var k = [Float16](repeating: 0, count: tokenCount * kvHeads * headDim)
        var v = [Float16](repeating: 0, count: tokenCount * kvHeads * headDim)
        for logical in 0..<tokenCount {
            for index in 0..<(kvHeads * headDim) {
                k[logical * kvHeads * headDim + index] =
                    Float16(Float((logical * 13 + index * 3) % 31 - 15) / 32)
                v[logical * kvHeads * headDim + index] =
                    Float16(Float((logical * 19 + index * 5) % 37 - 18) / 32)
            }
        }
        let sinks = (0..<qHeads).map {
            Quantization.bf16Bits(Float(($0 * 3) % 11 - 5) / 16)
        }

        let qBuffer = try buffer(context, q)
        let kPool = try buffer(context, k)
        let vPool = try buffer(context, v)
        let sinkBuffer = try buffer(context, sinks + [0x7BAD, 0x7BAD])
        let outBuffer = try buffer(
            context,
            [Float16](repeating: 99, count: qHeads * headDim))
        let table: [UInt32] = [0]
        let tableBuffer = try buffer(context, table)

        let cb = try #require(context.queue.makeCommandBuffer())
        pagedAttention.encodeFullPaged(
            commandBuffer: cb,
            q: qBuffer,
            kPool: kPool,
            vPool: vPool,
            pageTable: tableBuffer,
            out: outBuffer,
            headDim: UInt32(headDim),
            numQHeads: UInt32(qHeads),
            numKVHeads: UInt32(kvHeads),
            selTokens: UInt32(tokenCount),
            scale: Float(1.0 / Double(headDim).squareRoot()),
            sinks: sinkBuffer,
            numSinks: UInt32(qHeads))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let actual = output(from: outBuffer, count: qHeads * headDim)
        let expected = reference(
            q: q, k: k, v: v, sinks: sinks,
            first: 0, last: tokenCount, ring: tokenCount,
            scale: Float(1.0 / Double(headDim).squareRoot()))
        var maxDifference: Float = 0
        for i in 0..<expected.count {
            maxDifference = max(
                maxDifference,
                abs(Float(actual[i]) - expected[i]))
        }
        #expect(maxDifference <= 0.005,
                "Flint paged sink attention max abs diff \(maxDifference)")
    }

    private func reference(q: [Float16],
                           k: [Float16],
                           v: [Float16],
                           sinks: [UInt16],
                           first: Int,
                           last: Int,
                           ring: Int,
                           scale: Float) -> [Float] {
        let qHeads = FlintAttention.numQHeads
        let kvHeads = FlintAttention.numKVHeads
        let headDim = FlintAttention.headDim
        let gqa = qHeads / kvHeads
        var out = [Float](repeating: 0, count: qHeads * headDim)
        for head in 0..<qHeads {
            let kvHead = head / gqa
            var scores: [Float] = []
            for logical in first..<last {
                let slot = logical % ring
                var score: Float = 0
                for d in 0..<headDim {
                    score += Float(q[head * headDim + d])
                        * Float(k[(slot * kvHeads + kvHead) * headDim + d])
                }
                scores.append(score * scale)
            }
            let sink = Quantization.bf16ToFloat(sinks[head])
            let maximum = max(sink, scores.max() ?? -Float.infinity)
            let weights = scores.map { exp($0 - maximum) }
            let denom = exp(sink - maximum) + weights.reduce(0, +)
            for d in 0..<headDim {
                var value: Float = 0
                for (offset, logical) in (first..<last).enumerated() {
                    let slot = logical % ring
                    value += weights[offset]
                        * Float(v[(slot * kvHeads + kvHead) * headDim + d])
                }
                out[head * headDim + d] = value / denom
            }
        }
        return out
    }

    private func buffer<T>(_ context: MetalContext, _ values: [T]) throws -> MTLBuffer {
        let byteCount = values.count * MemoryLayout<T>.stride
        let buffer = try #require(
            context.device.makeBuffer(length: byteCount, options: .storageModeShared))
        _ = values.withUnsafeBytes {
            memcpy(buffer.contents(), $0.baseAddress!, byteCount)
        }
        return buffer
    }

    private func output(from buffer: MTLBuffer, count: Int) -> [Float16] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float16.self, capacity: count),
            count: count))
    }
}
